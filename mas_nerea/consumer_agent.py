"""
consumer_agent.py
=================
Agente Consumidor (AC) — coordinador del Contract Net Protocol (CNP).

Es el único agente que conduce la simulación: itera sobre los timesteps,
emite CFPs, evalúa propuestas y decide la asignación de energía.

Decisión económica (cheapest-first):
    1. Ordena propuestas de AS y AE por precio ascendente.
    2. Solo acepta propuestas cuyo precio <= precio_de_red (racional: no paga
       más que lo que le costaría comprar a la red directamente).
    3. Compra de la fuente más barata hasta cubrir la demanda o agotar su oferta.
    4. El resto se cubre con la red eléctrica al precio del mercado (fallback).

Esto resuelve el "problema del déficit" sin pymgrid:
    Si solar + eolico < demanda → compra el resto a red. Sin agente extra, sin simulador.
    Si hide_information sube el precio sobre mercado → el agente es rechazado → red lo cubre.
    Si deception promete más de lo que entrega → INFORM reporta entrega real → coste extra en red.

CSV esperados
-------------
    load_csv  : una columna sin timestamps. Primera fila = timestep 0.
                Columna: "Electricity:Facility [kW](Hourly)"
    price_csv : separado por ';'. Columna 'value' en €/MWh → se convierte a €/kWh.

Columnas del DataFrame de salida (por timestep)
------------------------------------------------
    timestep, demand_kw, import_price_eur_kwh,
    solar_declared_kw, solar_price, solar_allocated_kw, solar_delivered_kw,
    wind_declared_kw,  wind_price,  wind_allocated_kw,  wind_delivered_kw,
    grid_purchased_kw,
    solar_cost_eur, wind_cost_eur, grid_cost_eur, total_cost_eur,
    renewable_coverage_pct,
    solar_shortfall_kw, wind_shortfall_kw    ← 0 si no hay engaño
"""

import socket
import time
import json
import pandas as pd
from fipa_acl import create_message, parse_message

HOST = "127.0.0.1"
PORT_SOLAR = 5001
PORT_WIND  = 5002

LOAD_COL   = "Electricity:Facility [kW](Hourly)"
PRICE_COL  = "value"


class ConsumerAgent:
    """
    Agente Consumidor (AC).

    Parámetros
    ----------
    load_csv_path  : str  CSV de demanda horaria [kW].
    price_csv_path : str  CSV de precios PVPC en €/MWh (sep=';').
    n_steps        : int  Número de timesteps a simular (None = todos).
    """

    def __init__(self, load_csv_path: str, price_csv_path: str,
                 n_steps: int = None):
        # Demanda horaria
        df_load = pd.read_csv(load_csv_path)
        self.load_series = df_load[LOAD_COL].astype(float).values

        # Precio €/MWh → €/kWh
        df_price = pd.read_csv(price_csv_path, sep=";")
        self.price_series = df_price[PRICE_COL].astype(float).values / 1000.0

        # Alinear longitudes
        n = min(len(self.load_series), len(self.price_series))
        if n_steps is not None:
            n = min(n, n_steps)
        self.n_steps = n
        self.load_series  = self.load_series[:n]
        self.price_series = self.price_series[:n]

        self.history: list[dict] = []
        print(f"[AgenteConsumidor] Listo | {n} timesteps")

    # ------------------------------------------------------------------ #
    # Utilidades de socket
    # ------------------------------------------------------------------ #

    @staticmethod
    def _connect(port: int) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((HOST, port))
        return s

    @staticmethod
    def _send(sock: socket.socket, performative: str, content: dict) -> None:
        msg = create_message(performative, "AgenteConsumidor", "unknown", content)
        sock.sendall(msg.encode())

    @staticmethod
    def _recv(sock: socket.socket) -> dict | None:
        try:
            data = sock.recv(4096).decode()
            if not data:
                return None
            return parse_message(data)
        except (json.JSONDecodeError, OSError):
            return None

    # ------------------------------------------------------------------ #
    # Lógica de decisión (cheapest-first, precio <= mercado)
    # ------------------------------------------------------------------ #

    def _decide_allocation(self, demand_kw: float,
                           proposals: list[tuple[str, float, float]],
                           import_price: float) -> dict[str, dict]:
        """
        Asigna energía siguiendo la política cheapest-first con techo de precio.

        Solo compra de una fuente renovable si su precio es menor que el precio
        de red (decisión racional). Cubre el resto con la red.

        Parámetros
        ----------
        demand_kw     : float   Demanda del timestep actual [kW].
        proposals     : list    [(source, declared_kw, price), ...]
        import_price  : float   Precio de la red en €/kWh.

        Returns
        -------
        dict {source: {"kw": float, "price": float}}
        La clave "grid" siempre está presente (puede ser 0 kW).
        """
        # Filtrar fuentes más caras que la red (agente racional)
        viable = [(src, kw, p) for src, kw, p in proposals if p < import_price]

        # Ordenar por precio ascendente
        viable.sort(key=lambda x: x[2])

        allocation: dict[str, dict] = {}
        remaining = demand_kw

        for source, declared_kw, price in viable:
            if remaining <= 0:
                break
            purchase = min(remaining, declared_kw)
            allocation[source] = {"kw": purchase, "price": price}
            remaining = max(0.0, remaining - purchase)

        # Red cubre el resto (incluye déficit por engaño o precio alto)
        allocation["grid"] = {"kw": remaining, "price": import_price}
        return allocation

    # ------------------------------------------------------------------ #
    # Bucle de simulación principal
    # ------------------------------------------------------------------ #

    def run(self) -> pd.DataFrame:
        """
        Ejecuta la simulación completa iterando sobre n_steps timesteps.

        Para cada timestep:
            1. Lee demanda y precio del CSV.
            2. Envía CFP a AS y AE.
            3. Recibe PROPOSE de ambos.
            4. Decide asignación (cheapest-first, techo precio red).
            5. Envía ACCEPT/REJECT; recibe INFORM de los aceptados.
            6. Calcula compra real a red (cubre déficits y demanda residual).
            7. Registra métricas del paso.

        Returns
        -------
        pd.DataFrame con una fila por timestep.
        """
        print("[AgenteConsumidor] Conectando con productores...")
        time.sleep(1.0)  # esperar a que los servidores estén listos

        sock_solar = self._connect(PORT_SOLAR)
        sock_wind  = self._connect(PORT_WIND)
        print("[AgenteConsumidor] Conexiones establecidas. Iniciando negociación...\n")

        try:
            for t in range(self.n_steps):
                demand       = float(self.load_series[t])
                import_price = float(self.price_series[t])

                # ── Fase 1: CFP ──────────────────────────────────────── #
                cfp_content = {
                    "demand_kw":            round(demand, 4),
                    "import_price_eur_kwh": round(import_price, 6),
                    "timestep":             t,
                }
                self._send(sock_solar, "cfp", cfp_content)
                self._send(sock_wind,  "cfp", cfp_content)

                msg_solar = self._recv(sock_solar)
                msg_wind  = self._recv(sock_wind)

                solar_prop = msg_solar["content"] if msg_solar else None
                wind_prop  = msg_wind["content"]  if msg_wind  else None

                # ── Fase 2: Decisión ─────────────────────────────────── #
                proposals = []
                if solar_prop:
                    proposals.append(("solar",
                                      float(solar_prop["declared_energy_kw"]),
                                      float(solar_prop["price_eur_kwh"])))
                if wind_prop:
                    proposals.append(("wind",
                                      float(wind_prop["declared_energy_kw"]),
                                      float(wind_prop["price_eur_kwh"])))

                allocation = self._decide_allocation(demand, proposals, import_price)

                # ── Fase 3: ACCEPT / REJECT ───────────────────────────── #
                solar_delivered = 0.0
                wind_delivered  = 0.0
                solar_cost      = 0.0
                wind_cost       = 0.0

                # Solar
                if "solar" in allocation and allocation["solar"]["kw"] > 0:
                    self._send(sock_solar, "accept-proposal", {
                        "purchased_kw":    round(allocation["solar"]["kw"], 4),
                        "price_eur_kwh":   round(allocation["solar"]["price"], 6),
                        "timestep":        t,
                    })
                    inform = self._recv(sock_solar)
                    if inform and inform["performative"] == "inform":
                        solar_delivered = float(inform["content"]["actual_delivered_kw"])
                    solar_cost = solar_delivered * allocation["solar"]["price"]
                else:
                    self._send(sock_solar, "reject-proposal", {"timestep": t})

                # Eólico
                if "wind" in allocation and allocation["wind"]["kw"] > 0:
                    self._send(sock_wind, "accept-proposal", {
                        "purchased_kw":    round(allocation["wind"]["kw"], 4),
                        "price_eur_kwh":   round(allocation["wind"]["price"], 6),
                        "timestep":        t,
                    })
                    inform = self._recv(sock_wind)
                    if inform and inform["performative"] == "inform":
                        wind_delivered = float(inform["content"]["actual_delivered_kw"])
                    wind_cost = wind_delivered * allocation["wind"]["price"]
                else:
                    self._send(sock_wind, "reject-proposal", {"timestep": t})

                # ── Fase 4: Compra real a red ─────────────────────────── #
                # Incluye: demanda residual + déficits por engaño
                renewable_total = solar_delivered + wind_delivered
                grid_purchased  = max(0.0, demand - renewable_total)
                grid_cost       = grid_purchased * import_price
                total_cost      = solar_cost + wind_cost + grid_cost

                # Shortfalls (evidencia de engaño si > 0)
                solar_shortfall = max(0.0,
                    allocation.get("solar", {}).get("kw", 0.0) - solar_delivered)
                wind_shortfall  = max(0.0,
                    allocation.get("wind",  {}).get("kw", 0.0) - wind_delivered)

                self.history.append({
                    "timestep":               t,
                    "demand_kw":              round(demand, 4),
                    "import_price_eur_kwh":   round(import_price, 6),
                    # Solar
                    "solar_declared_kw":      round(solar_prop["declared_energy_kw"] if solar_prop else 0, 4),
                    "solar_price_eur_kwh":    round(solar_prop["price_eur_kwh"] if solar_prop else 0, 6),
                    "solar_allocated_kw":     round(allocation.get("solar", {}).get("kw", 0), 4),
                    "solar_delivered_kw":     round(solar_delivered, 4),
                    "solar_shortfall_kw":     round(solar_shortfall, 4),
                    "solar_cost_eur":         round(solar_cost, 6),
                    # Eólico
                    "wind_declared_kw":       round(wind_prop["declared_energy_kw"] if wind_prop else 0, 4),
                    "wind_price_eur_kwh":     round(wind_prop["price_eur_kwh"] if wind_prop else 0, 6),
                    "wind_allocated_kw":      round(allocation.get("wind", {}).get("kw", 0), 4),
                    "wind_delivered_kw":      round(wind_delivered, 4),
                    "wind_shortfall_kw":      round(wind_shortfall, 4),
                    "wind_cost_eur":          round(wind_cost, 6),
                    # Red
                    "grid_purchased_kw":      round(grid_purchased, 4),
                    "grid_cost_eur":          round(grid_cost, 6),
                    # Totales
                    "total_cost_eur":         round(total_cost, 6),
                    "renewable_coverage_pct": round(
                        100.0 * renewable_total / demand if demand > 0 else 0.0, 2),
                })

                if (t + 1) % 200 == 0:
                    print(f"[AgenteConsumidor] Paso {t+1}/{self.n_steps} | "
                          f"Coste acum.: {sum(r['total_cost_eur'] for r in self.history):.2f} EUR")

        finally:
            sock_solar.close()
            sock_wind.close()

        df = pd.DataFrame(self.history)

        # Resumen final
        print(f"\n[AgenteConsumidor] Simulación completada:")
        print(f"  Coste total:              {df['total_cost_eur'].sum():.4f} EUR")
        print(f"  Cobertura renovable media: {df['renewable_coverage_pct'].mean():.1f}%")
        print(f"  Compra a red total:        {df['grid_purchased_kw'].sum():.1f} kWh")
        print(f"  Shortfall solar total:     {df['solar_shortfall_kw'].sum():.4f} kWh")
        print(f"  Shortfall eólico total:    {df['wind_shortfall_kw'].sum():.4f} kWh")
        return df

    # ------------------------------------------------------------------ #
    # Acceso al histórico
    # ------------------------------------------------------------------ #

    def get_history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)
