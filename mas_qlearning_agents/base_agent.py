"""
base_agent.py
=============
Clase base para agentes productores de energía (Solar y Eólico).

Encapsula toda la lógica de servidor TCP/FIPA-ACL y el ciclo de negociación
por timestep, dejando a las subclases solo la configuración de nombre, puerto
y columna del CSV.

Protocolo FIPA-ACL por timestep
--------------------------------
← CFP              {"demand_kw": X, "import_price_eur_kwh": Y, "timestep": t}
→ PROPOSE          {"declared_energy_kw": A, "price_eur_kwh": B, "timestep": t}
← ACCEPT-PROPOSAL  {"purchased_kw": C, "price_eur_kwh": B, "timestep": t}
  o REJECT-PROPOSAL {"timestep": t}
→ INFORM           {"actual_delivered_kw": D, "revenue_eur": E, "timestep": t}
   (solo si accept)

El INFORM reporta la entrega REAL, que puede ser menor que C si el agente
usó estrategia de engaño (deception). Esta discrepancia es el dato clave
para el análisis comparativo de estrategias.
"""

import socket
import threading
import json
import pandas as pd
from strategies import NegotiationStrategies, EnergyProposal
from strategy_qlearning import StrategyQLearning
from fipa_acl import create_message, parse_message

HOST = "127.0.0.1"


class ProducerAgent:
    """
    Agente productor de energía con servidor FIPA-ACL sobre TCP.

    Parámetros
    ----------
    name           : str    Nombre del agente (p.ej. "AgenteSolar").
    port           : int    Puerto TCP donde escucha.
    csv_path       : str    Ruta al CSV con predicciones de potencia.
    power_column   : str    Nombre de la columna de potencia en el CSV.
    strategy_name  : str    "honest" | "deception" | "hide_information".
    bluff_factor   : float  Para estrategia deception (default 1.3).
    hide_factor    : float  Para estrategia hide_information (default 0.70).
    """

    def __init__(self, name: str, port: int, csv_path: str, power_column: str,
                 strategy_name: str = "honest",
                 bluff_factor: float = 1.3, hide_factor: float = 0.70):
        self.name = name
        self.port = port
        self.strategy_name = strategy_name
        self.learner = StrategyQLearning()
        self.last_state = None
        self.last_action = None
        self.last_demand = 40.0
        self.last_price = 0.15
        self.bluff_factor = bluff_factor
        self.hide_factor = hide_factor

        # Carga de serie temporal de potencia
        df = pd.read_csv(csv_path)
        if "Date" in df.columns:
            df = df.sort_values("Date").reset_index(drop=True)
        self.power_series = df[power_column].astype(float).clip(lower=0.0).values

        # Estado interno
        self.current_step: int = 0
        self.total_revenue_eur: float = 0.0
        self.history: list[dict] = []
        self._current_proposal: EnergyProposal | None = None

        # Control del servidor
        self.running = True
        self.server_socket = None

        print(f"[{self.name}] Listo | {len(self.power_series)} timesteps | "
            f"Estrategia: {strategy_name}")

    # ------------------------------------------------------------------ #
    # Acceso a datos del timestep actual
    # ------------------------------------------------------------------ #

    def _get_real_power(self) -> float:
        """Potencia real disponible en el timestep actual [kW]."""
        if self.current_step < len(self.power_series):
            return float(self.power_series[self.current_step])
        return 0.0

    def _build_proposal(self, import_price: float) -> EnergyProposal:
        """
        Genera una propuesta usando Q-Learning para seleccionar
        la estrategia de negociación.
        """

        real_power = self._get_real_power()

        # Estado simplificado temporal:
        # demanda ficticia = 40 kW
        # precio real
        # potencia propia
        # potencia rival ficticia

        state = self.learner.get_state(
            demand_kw=self.last_demand,
            price_eur_kwh=self.last_price,
            own_power=real_power
        )

        action = self.learner.choose_action(
            state
        )

        strategy_name = (
            self.learner.action_to_strategy(
                action
            )
        )

        print(
            f"[{self.name}] "
            f"t={self.current_step} "
            f"estrategia={strategy_name}"
        )

        self.last_state = state
        self.last_action = action

        kwargs = {}

        if strategy_name == "deception":
            kwargs["bluff_factor"] = self.bluff_factor

        elif strategy_name == "hide_information":
            kwargs["hide_factor"] = self.hide_factor

        proposal = NegotiationStrategies.apply(
            strategy_name,
            real_power,
            import_price,
            **kwargs
        )

        return proposal

    # ------------------------------------------------------------------ #
    # Handlers de performativas FIPA-ACL
    # ------------------------------------------------------------------ #

    def _on_cfp(self, msg: dict, conn: socket.socket) -> None:
        """
        Responde a un CFP con una PROPOSE.
        Calcula la propuesta según la estrategia y la envía al consumidor.
        """
        content = msg["content"]

        self.last_demand = float(
            content["demand_kw"]
        )

        self.last_price = float(
            content["import_price_eur_kwh"]
        )

        self._current_proposal = self._build_proposal(
            self.last_price
        )

        reply = create_message(
            performative="propose",
            sender=self.name,
            receiver=msg["sender"],
            content={
                "declared_energy_kw": round(self._current_proposal.declared_energy_kw, 4),
                "price_eur_kwh":      round(self._current_proposal.price_eur_kwh, 6),
                "timestep":           self.current_step,
            }
        )
        conn.sendall(reply.encode())
        import time
        time.sleep(0.01)

    def _on_accept(self, msg: dict, conn: socket.socket) -> None:
        """
        Gestiona ACCEPT-PROPOSAL: envía INFORM con la entrega real.

        La entrega real = min(purchased_kw, real_energy_kw).
        Si el agente usó deception, purchased_kw puede ser > real → shortfall.
        """
        content = msg["content"]
        purchased_kw = float(content["purchased_kw"])
        agreed_price  = float(content["price_eur_kwh"])

        real_power      = self._current_proposal.real_energy_kw
        actual_delivered = min(purchased_kw, real_power)
        shortfall        = max(0.0, purchased_kw - actual_delivered)
        revenue          = actual_delivered * agreed_price

        # ==========================================
        # REWARD Q-LEARNING
        # ==========================================

        reward = (
            revenue
            - 2 * shortfall
        )

        if self.last_state is not None:

            next_state = self.last_state

            self.learner.update(
                self.last_state,
                self.last_action,
                reward,
                next_state
            )

        self.total_revenue_eur += revenue

        self.history.append({
            "timestep":            self.current_step,
            "strategy":            self.strategy_name,
            "real_power_kw":       round(real_power, 4),
            "declared_power_kw":   round(self._current_proposal.declared_energy_kw, 4),
            "purchased_kw":        round(purchased_kw, 4),
            "actual_delivered_kw": round(actual_delivered, 4),
            "shortfall_kw":        round(shortfall, 4),
            "price_eur_kwh":       round(agreed_price, 6),
            "revenue_eur":         round(revenue, 6),
            "accepted":            True,
        })

        confirm = create_message(
            performative="inform",
            sender=self.name,
            receiver=msg["sender"],
            content={
                "actual_delivered_kw": round(actual_delivered, 4),
                "revenue_eur":         round(revenue, 6),
                "timestep":            self.current_step,
            }
        )
        conn.sendall(confirm.encode())
        import time
        time.sleep(0.01)
        self.current_step += 1

    def _on_reject(self, msg: dict) -> None:
        """Gestiona REJECT-PROPOSAL: registra el paso sin ingresos."""
        self.history.append({
            "timestep":            self.current_step,
            "strategy":            self.strategy_name,
            "real_power_kw":       round(self._get_real_power(), 4),
            "declared_power_kw":   round(
                self._current_proposal.declared_energy_kw
                if self._current_proposal else 0.0, 4),
            "purchased_kw":        0.0,
            "actual_delivered_kw": 0.0,
            "shortfall_kw":        0.0,
            "price_eur_kwh":       0.0,
            "revenue_eur":         0.0,
            "accepted":            False,
        })

        reward = -1.0

        if self.last_state is not None:

            self.learner.update(
                self.last_state,
                self.last_action,
                reward,
                self.last_state
            )

        self.current_step += 1

    # ------------------------------------------------------------------ #
    # Gestión de la conexión TCP
    # ------------------------------------------------------------------ #

    def _handle_connection(self, conn: socket.socket, addr) -> None:
        """Bucle de mensajería para una conexión activa."""
        print(f"[{self.name}] Conexión aceptada desde {addr}")
        while True:
            try:
                data = conn.recv(4096).decode()
                if not data:
                    break
                msg = parse_message(data)
                perf = msg["performative"]

                if perf == "cfp":
                    self._on_cfp(msg, conn)
                elif perf == "accept-proposal":
                    self._on_accept(msg, conn)
                elif perf == "reject-proposal":
                    self._on_reject(msg)
                else:
                    print(f"[{self.name}] Performativa no reconocida: {perf}")

            except Exception as e:
                print(f"[{self.name}] ERROR: {e}")
                import traceback
                traceback.print_exc()
                break

        conn.close()
        print(f"[{self.name}] Conexión cerrada | "
              f"Ingresos totales: {self.total_revenue_eur:.4f} EUR | "
              f"Steps procesados: {self.current_step}")

    # ------------------------------------------------------------------ #
    # Arranque del servidor
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        self.server_socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )

        self.server_socket.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1
        )

        self.server_socket.bind((HOST, self.port))
        self.server_socket.listen(1)

        print(f"[{self.name}] Servidor activo en {HOST}:{self.port}")

        self.server_socket.settimeout(1.0)

        while self.running:
            try:
                conn, addr = self.server_socket.accept()

                t = threading.Thread(
                    target=self._handle_connection,
                    args=(conn, addr),
                    daemon=True
                )
                t.start()

            except socket.timeout:
                continue

            except OSError:
                break

        
    def stop(self):
        self.running = False

        if self.server_socket is not None:
            try:
                self.server_socket.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Acceso al histórico
    # ------------------------------------------------------------------ #

    def get_history_df(self) -> pd.DataFrame:
        """
        Devuelve el historial completo del agente como DataFrame.
        Útil para análisis en notebooks y comparación de estrategias.
        """
        return pd.DataFrame(self.history)
