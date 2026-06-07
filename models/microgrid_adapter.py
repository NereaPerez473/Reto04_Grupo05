"""
microgrid_adapter.py
====================
Adaptador del mundo físico de la microred para el sistema multiagente.

Rol en la arquitectura MAS
--------------------------
Este módulo es el ÚNICO punto de contacto con pymgrid. Ningún agente
importa pymgrid directamente; todos leen el estado y ejecutan acciones
a través de esta clase. Eso garantiza:
  - Un único estado consistente de la microred en cada instante.
  - Trazabilidad completa: cada step queda registrado en self.history.
  - Desacoplamiento: si pymgrid cambia de API, solo se toca aquí.

Flujo de datos externos
-----------------------
A diferencia del entorno RL original (donde load y PV eran internos de
pymgrid), aquí los cuatro datos de entrada vienen de CSVs externos:

    data/processed/solar_predictions.csv   → columna 'pv_kw'
    data/processed/wind_predictions.csv    → columna 'wind_kw'
    data/processed/load_data.csv           → columna 'load_kw'
    data/processed/price_data.csv          → columnas 'import_price', 'export_price'

Uso esperado desde los agentes
-------------------------------
    adapter = MicrogridAdapter(pymgrid_network, data_dir="data/processed")
    adapter.reset()

    # Lectura del estado (sin coste, sin efectos secundarios)
    state = adapter.get_state()

    # Ejecución de la acción negociada (llamado por el Coordinador)
    result = adapter.step(battery_kw=10.0)

    # Acceso al histórico completo
    df = adapter.get_history_df()
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Dataclass de estado: lo que los agentes "ven" del mundo físico
# ---------------------------------------------------------------------------

@dataclass
class MicrogridState:
    """
    Snapshot del estado físico de la microred en el step actual.
    Es la única información que los agentes productores (AS, AE) y el
    consumidor (AC) reciben del mundo antes de negociar.

    Atributos
    ---------
    step : int
        Índice temporal actual (0-indexed).
    hour : int
        Hora del día [0-23], derivada de step % 24.
    day_of_year : int
        Día del año [0-364], derivado de step // 24.

    pv_kw : float
        Potencia fotovoltaica disponible en este step [kW].
        Proviene del modelo solar ya inferido.
    wind_kw : float
        Potencia eólica disponible en este step [kW].
        Proviene del modelo eólico ya inferido.
    load_kw : float
        Demanda energética del consumidor en este step [kW].
    net_load_kw : float
        Carga neta = load_kw - pv_kw - wind_kw.
        Valor positivo → falta energía (batería/red deben aportar).
        Valor negativo → hay excedente (se puede exportar o cargar batería).

    soc : float
        Estado de carga de la batería [0.0 – 1.0].
    import_price : float
        Precio de importación de red en este step [€/kWh o unidad del CSV].
    export_price : float
        Precio de exportación a red en este step [€/kWh o unidad del CSV].
    """
    step: int
    hour: int
    day_of_year: int

    pv_kw: float
    wind_kw: float
    load_kw: float
    net_load_kw: float

    soc: float
    import_price: float
    export_price: float


@dataclass
class StepResult:
    """
    Resultado de ejecutar una acción en la microred.
    Lo devuelve MicrogridAdapter.step() al Coordinador tras aplicar
    la decisión negociada.

    Atributos
    ---------
    state_before : MicrogridState
        Estado antes de ejecutar la acción (para logging y xAI).
    state_after : MicrogridState
        Estado tras ejecutar la acción (siguiente observación).
    battery_kw : float
        Potencia de batería realmente aplicada [kW].
        Positivo → descarga (batería aporta energía).
        Negativo → carga (batería almacena energía).
    grid_kw : float
        Potencia de red resultante [kW].
        Positivo → importación.
        Negativo → exportación.
    cost : float
        Coste económico del step (positivo = gasto, negativo = ingreso).
    reward : float
        Recompensa normalizada (compatible con la escala del entorno RL).
    terminated : bool
        True si pymgrid señala fin de episodio.
    truncated : bool
        True si se alcanzó el horizonte temporal configurado.
    info : dict
        Diccionario extendido con métricas adicionales para logging.
    """
    state_before: MicrogridState
    state_after: MicrogridState
    battery_kw: float
    grid_kw: float
    cost: float
    reward: float
    terminated: bool
    truncated: bool
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class MicrogridAdapter:
    """
    Adaptador entre pymgrid y el sistema multiagente.

    Gestiona:
      - La carga y sincronización de los datos externos (solar, eólico,
        load, precios) con el estado interno de pymgrid.
      - La exposición del estado actual a los agentes (solo lectura).
      - La ejecución de la acción negociada sobre pymgrid.
      - El registro histórico de cada step para análisis posterior.

    Parámetros
    ----------
    pymgrid_network : pymgrid.Microgrid
        Instancia de microred ya configurada (misma que el entorno RL).
    data_dir : str | Path
        Directorio donde están los CSVs procesados tras inferencia.
        Estructura esperada:
          <data_dir>/solar_predictions.csv  → columna 'pv_kw'
          <data_dir>/wind_predictions.csv   → columna 'wind_kw'
          <data_dir>/load_data.csv          → columna 'load_kw'
          <data_dir>/price_data.csv         → columnas 'import_price', 'export_price'
    horizon : int
        Número máximo de steps por episodio. Por defecto 24*365 (un año horario).
    reward_scale_C : float
        Factor de normalización del reward (mismo que en el entorno RL).
    battery_capacity_kw : float
        Capacidad máxima de la batería en kW (para escalar acciones).
        Debe coincidir con la configuración de pymgrid.
    low_soc_threshold : float
        Umbral de SoC bajo bajo el que se aplica penalización.
    low_soc_penalty : float
        Peso de la penalización por SoC bajo.

    Notas sobre la integración con pymgrid
    ---------------------------------------
    pymgrid internamente avanza su propio puntero de datos en cada .run().
    Para asegurarnos de que el step i de pymgrid coincide con la fila i
    de nuestros CSVs externos, el adapter:
      1. Lee los valores de los CSVs en self.current_step.
      2. Inyecta esos valores en pymgrid ANTES de llamar a .run()
         sobreescribiendo los módulos load y pv con set_current_step si
         la API lo permite, o dejando que pymgrid avance en paralelo.
    
    IMPORTANTE: Si los CSVs externos tienen un número de filas distinto
    al horizonte de pymgrid, se usará el mínimo como horizonte efectivo.
    """

    # Nombres de columna esperados en los CSVs
    COL_PV = "pv_kw"
    COL_WIND = "wind_kw"
    COL_LOAD = "load_kw"
    COL_IMPORT_PRICE = "import_price"
    COL_EXPORT_PRICE = "export_price"

    def __init__(
        self,
        pymgrid_network,
        data_dir: str = "data/processed",
        horizon: int = 24 * 365,
        reward_scale_C: float = 91.88,
        battery_capacity_kw: float = 50.0,
        low_soc_threshold: float = 0.2,
        low_soc_penalty: float = 1.0,
    ):
        self.mg = pymgrid_network
        self.data_dir = Path(data_dir)
        self.reward_scale_C = float(reward_scale_C)
        self.battery_capacity_kw = float(battery_capacity_kw)
        self.low_soc_threshold = float(low_soc_threshold)
        self.low_soc_penalty = float(low_soc_penalty)

        # Carga de datos externos
        self._data = self._load_external_data()

        # El horizonte efectivo es el mínimo entre lo configurado y los datos
        self.horizon = min(horizon, len(self._data))

        # Estado interno
        self.current_step: int = 0
        self._history: list[dict] = []

        print(
            f"[MicrogridAdapter] Inicializado. "
            f"Horizonte efectivo: {self.horizon} steps "
            f"({self.horizon // 24} días)."
        )

    # ------------------------------------------------------------------
    # Carga de datos externos
    # ------------------------------------------------------------------

    def _load_external_data(self) -> pd.DataFrame:
        """
        Carga y une los cuatro CSVs externos en un único DataFrame indexado
        por step (0, 1, 2, …).

        El índice temporal se asume implícito: la fila 0 es la hora 0,
        la fila 1 es la hora 1, etc. Si los CSVs tienen columna de
        timestamp se ignora (solo importa el orden).

        Returns
        -------
        pd.DataFrame con columnas: pv_kw, wind_kw, load_kw,
                                   import_price, export_price
        """
        solar_path  = self.data_dir / "solar_predictions.csv"
        wind_path   = self.data_dir / "wind_predictions.csv"
        load_path   = self.data_dir / "load_data.csv"
        price_path  = self.data_dir / "price_data.csv"

        missing = [p for p in [solar_path, wind_path, load_path, price_path]
                   if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"[MicrogridAdapter] Archivos no encontrados en {self.data_dir}:\n"
                + "\n".join(f"  - {p.name}" for p in missing)
                + "\nEjecuta primero models/inference.py para generar los datos procesados."
            )

        df_solar = pd.read_csv(solar_path)[[self.COL_PV]].reset_index(drop=True)
        df_wind  = pd.read_csv(wind_path)[[self.COL_WIND]].reset_index(drop=True)
        df_load  = pd.read_csv(load_path)[[self.COL_LOAD]].reset_index(drop=True)
        df_price = pd.read_csv(price_path)[
            [self.COL_IMPORT_PRICE, self.COL_EXPORT_PRICE]
        ].reset_index(drop=True)

        # Verificación de longitud
        lengths = {
            "solar": len(df_solar),
            "wind":  len(df_wind),
            "load":  len(df_load),
            "price": len(df_price),
        }
        if len(set(lengths.values())) > 1:
            print(
                f"[MicrogridAdapter] ADVERTENCIA: los CSVs tienen distinto número "
                f"de filas: {lengths}. Se usará el mínimo."
            )
        min_len = min(lengths.values())

        data = pd.concat(
            [df_solar[:min_len], df_wind[:min_len],
             df_load[:min_len], df_price[:min_len]],
            axis=1
        )

        # Aseguramos tipos numéricos y rellenamos NaN con 0 (con warning)
        n_nan = data.isna().sum().sum()
        if n_nan > 0:
            print(
                f"[MicrogridAdapter] ADVERTENCIA: {n_nan} valores NaN encontrados "
                f"en los datos. Se reemplazarán por 0."
            )
            data = data.fillna(0.0)

        data = data.astype(float)
        print(f"[MicrogridAdapter] Datos externos cargados: {len(data)} steps.")
        return data

    # ------------------------------------------------------------------
    # Acceso al estado actual (solo lectura, sin efectos secundarios)
    # ------------------------------------------------------------------

    def get_state(self) -> MicrogridState:
        """
        Devuelve el estado actual de la microred SIN avanzar el tiempo.
        
        Es la llamada que los agentes hacen al inicio de cada ronda de
        negociación para conocer la situación antes de formular propuestas.
        Es segura llamarla múltiples veces en el mismo step.
        """
        row = self._data.iloc[self.current_step]

        pv_kw   = float(row[self.COL_PV])
        wind_kw = float(row[self.COL_WIND])
        load_kw = float(row[self.COL_LOAD])
        net_load_kw = load_kw - pv_kw - wind_kw

        soc = self._get_soc()
        import_price = float(row[self.COL_IMPORT_PRICE])
        export_price = float(row[self.COL_EXPORT_PRICE])

        return MicrogridState(
            step=self.current_step,
            hour=self.current_step % 24,
            day_of_year=(self.current_step // 24) % 365,
            pv_kw=pv_kw,
            wind_kw=wind_kw,
            load_kw=load_kw,
            net_load_kw=net_load_kw,
            soc=soc,
            import_price=import_price,
            export_price=export_price,
        )

    # ------------------------------------------------------------------
    # Ejecución de la acción negociada
    # ------------------------------------------------------------------

    def step(self, battery_kw: float) -> StepResult:
        """
        Aplica la acción de batería negociada y avanza la simulación un step.

        Este método lo llama ÚNICAMENTE el Coordinador (AG) tras resolver
        la negociación entre AS y AE. Los agentes productores NO llaman
        step() directamente.

        Parámetros
        ----------
        battery_kw : float
            Potencia de batería a aplicar [kW].
            Positivo → descarga (la batería aporta energía al sistema).
            Negativo → carga (la batería almacena energía).
            Se aplica clipping a [-battery_capacity_kw, +battery_capacity_kw].

        Returns
        -------
        StepResult
            Resultado completo del step, incluyendo coste, reward,
            estados antes/después y flags de terminación.
        """
        # Captura del estado antes de la acción (para logging y xAI)
        state_before = self.get_state()

        # Clipping de la acción a los límites físicos de la batería
        battery_kw = float(np.clip(
            battery_kw,
            -self.battery_capacity_kw,
            self.battery_capacity_kw
        ))

        # Balance físico: la red cubre el resto que batería+renovables no cubren
        # Grid = (Load - PV - Wind) - Battery
        # Positivo → importamos de red; negativo → exportamos a red
        grid_kw = state_before.net_load_kw - battery_kw

        control_dict = {
            "battery": [battery_kw],
            "grid":    [grid_kw],
        }

        soc_before = state_before.soc

        # Ejecución en pymgrid
        mg_obs, mg_reward, mg_done, mg_info = self.mg.run(
            control_dict,
            normalized=False
        )

        raw_reward = float(mg_reward)
        cost = -raw_reward
        reward = raw_reward / self.reward_scale_C

        terminated = bool(mg_done)
        self.current_step += 1
        truncated = bool(self.current_step >= self.horizon)

        # SoC tras la acción
        soc_after = self._get_soc() if not terminated else soc_before

        # Penalización por SoC bajo (mismo criterio que entorno RL)
        low_soc_penalty_applied = 0.0
        if soc_after < self.low_soc_threshold:
            soc_deficit_ratio = (
                (self.low_soc_threshold - soc_after) / self.low_soc_threshold
            )
            low_soc_penalty_applied = self.low_soc_penalty * soc_deficit_ratio
            reward -= low_soc_penalty_applied

        # Estado tras la acción
        if not terminated and not truncated:
            state_after = self.get_state()
        else:
            # En terminación usamos el último estado conocido
            state_after = state_before

        # Construcción del resultado
        info = {
            "step":                    self.current_step - 1,
            "battery_kw":              battery_kw,
            "grid_kw":                 grid_kw,
            "grid_import_kw":          max(0.0, grid_kw),
            "grid_export_kw":          max(0.0, -grid_kw),
            "cost":                    cost,
            "raw_reward":              raw_reward,
            "reward_normalized":       reward,
            "low_soc_penalty_applied": low_soc_penalty_applied,
            "soc_before":              soc_before,
            "soc_after":               soc_after,
            "pv_kw":                   state_before.pv_kw,
            "wind_kw":                 state_before.wind_kw,
            "load_kw":                 state_before.load_kw,
            "net_load_kw":             state_before.net_load_kw,
            "import_price":            state_before.import_price,
            "export_price":            state_before.export_price,
            "mg_info":                 mg_info,
            "terminated":              terminated,
            "truncated":               truncated,
        }

        result = StepResult(
            state_before=state_before,
            state_after=state_after,
            battery_kw=battery_kw,
            grid_kw=grid_kw,
            cost=cost,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

        # Registro en histórico
        self._history.append(info)

        return result

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> MicrogridState:
        """
        Reinicia la simulación al step 0.
        Llama a mg.reset() para que pymgrid vuelva a su estado inicial
        y limpia el histórico.

        Returns
        -------
        MicrogridState
            Estado inicial tras el reset.
        """
        self.mg.reset()
        self.current_step = 0
        self._history = []
        initial_state = self.get_state()
        print(
            f"[MicrogridAdapter] Reset. "
            f"SoC inicial: {initial_state.soc:.3f} | "
            f"Net load inicial: {initial_state.net_load_kw:.2f} kW"
        )
        return initial_state

    # ------------------------------------------------------------------
    # Propiedades de conveniencia
    # ------------------------------------------------------------------

    @property
    def done(self) -> bool:
        """True si la simulación ha terminado (por truncado o terminación)."""
        return self.current_step >= self.horizon

    @property
    def remaining_steps(self) -> int:
        """Steps que quedan hasta el horizonte."""
        return max(0, self.horizon - self.current_step)

    # ------------------------------------------------------------------
    # Acceso al histórico
    # ------------------------------------------------------------------

    def get_history_df(self) -> pd.DataFrame:
        """
        Devuelve el histórico completo de steps como DataFrame.
        Útil para análisis post-simulación, generación de métricas
        y como input para xAI.

        Columnas principales del DataFrame resultante:
          step, battery_kw, grid_kw, grid_import_kw, grid_export_kw,
          cost, reward_normalized, soc_before, soc_after,
          pv_kw, wind_kw, load_kw, net_load_kw,
          import_price, export_price, low_soc_penalty_applied,
          terminated, truncated
        """
        if not self._history:
            return pd.DataFrame()
        return pd.DataFrame(self._history).drop(columns=["mg_info"], errors="ignore")

    def get_summary(self) -> dict:
        """
        Resumen estadístico del episodio actual.
        Pensado para logging rápido al final de una simulación.
        """
        df = self.get_history_df()
        if df.empty:
            return {}
        return {
            "total_steps":        len(df),
            "total_cost":         df["cost"].sum(),
            "mean_reward":        df["reward_normalized"].mean(),
            "total_grid_import":  df["grid_import_kw"].sum(),
            "total_grid_export":  df["grid_export_kw"].sum(),
            "mean_soc":           df["soc_after"].mean(),
            "min_soc":            df["soc_after"].min(),
            "n_low_soc_steps":    (df["soc_after"] < self.low_soc_threshold).sum(),
            "total_pv":           df["pv_kw"].sum(),
            "total_wind":         df["wind_kw"].sum(),
            "total_load":         df["load_kw"].sum(),
        }

    # ------------------------------------------------------------------
    # Helpers privados de acceso a pymgrid
    # ------------------------------------------------------------------

    def _get_soc(self) -> float:
        """Lee el SoC actual de la batería desde pymgrid."""
        return float(np.clip(self.mg.battery.item().soc, 0.0, 1.0))
