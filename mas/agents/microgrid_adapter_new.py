"""
microgrid_adapter.py
====================
Adaptador del mundo físico de la microred para el sistema multiagente.

Cambios respecto a la versión anterior
---------------------------------------
1. SIN normalización del reward. En un MAS de negociación el reward no
   alimenta redes neuronales, así que trabajamos en €/step directamente.
   Esto hace que las métricas del paper sean interpretables.

2. Carga de CSVs adaptada a los archivos reales del proyecto:
     data/results/Predicciones_Eolico.csv   → columnas: Date, Power_AE
     data/results/Predicciones_Solar.csv    → columnas: Date, SystemProduction_AS
     data/raw/RefBldgFullService...csv      → columna: Electricity:Facility [kW](Hourly)
     data/raw/precio2025-peninsula.csv      → separador ';', columna: value (€/MWh)

3. Conversión de unidades:
     Precio CSV: €/MWh  →  se divide entre 1000 → €/kWh
     Potencia:   kW     →  sin cambio
     Energía/step: kW × 1h = kWh (horizonte horario)

4. Reward con tres componentes (en €, sin normalizar):
     reward(t) = -coste_economico(t) - pen_soc(t) - pen_deficit(t)

   coste_economico(t) [€]:
     = grid_import_kw × import_price_€/kWh
     - grid_export_kw × export_price_€/kWh

   pen_soc(t) [€ equivalentes]:
     Penaliza SoC < umbral. Proporcional al déficit.
     pen = λ_soc × (threshold - soc) / threshold
     λ_soc = 5 € (mismo orden de magnitud que el coste horario típico ~1-4 €)

   pen_deficit(t) [€ equivalentes]:
     Penaliza que la demanda del AC no quede cubierta.
     pen = λ_deficit × max(0, grid_import_kw - grid_max_kw)
     λ_deficit = 10 € >> λ_soc para que la cobertura sea prioritaria.
     Captura la restricción cooperativa del enunciado:
     "evitar que el AC quede sin energía".
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Dataclasses de estado y resultado
# ---------------------------------------------------------------------------

@dataclass
class MicrogridState:
    """
    Snapshot del estado físico de la microred en el step actual.
    Es la información que los agentes reciben antes de negociar.

    Atributos
    ---------
    step : int          Índice temporal actual (0-indexed).
    hour : int          Hora del día [0-23].
    day_of_year : int   Día del año [0-364].
    pv_kw : float       Potencia fotovoltaica disponible [kW].
    wind_kw : float     Potencia eólica disponible [kW].
    load_kw : float     Demanda del Agente Consumidor [kW].
    net_load_kw : float load_kw - pv_kw - wind_kw.
                        > 0 → necesita batería o red.
                        < 0 → excedente renovable.
    soc : float         Estado de carga de la batería [0.0 – 1.0].
    import_price_eur_kwh : float  Precio de importación [€/kWh].
    export_price_eur_kwh : float  Precio de exportación [€/kWh].
    """
    step: int
    hour: int
    day_of_year: int
    pv_kw: float
    wind_kw: float
    load_kw: float
    net_load_kw: float
    soc: float
    import_price_eur_kwh: float
    export_price_eur_kwh: float


@dataclass
class StepResult:
    """
    Resultado de ejecutar una acción negociada en la microred.

    Atributos
    ---------
    state_before : MicrogridState   Estado antes de la acción (para xAI).
    state_after  : MicrogridState   Estado tras la acción.
    battery_kw : float    Potencia de batería aplicada [kW].
                          > 0 descarga, < 0 carga.
    grid_kw : float       Potencia de red resultante [kW].
                          > 0 importación, < 0 exportación.
    reward : float        Reward económico [€]. Sin normalizar.
    economic_cost : float Coste económico puro [€].
    pen_soc : float       Penalización por SoC bajo [€ eq.].
    pen_deficit : float   Penalización por demanda no cubierta [€ eq.].
    terminated : bool     pymgrid señaló fin natural del episodio.
    truncated : bool      Se alcanzó el horizonte configurado.
    info : dict           Métricas extendidas para logging y Prefect.
    """
    state_before: MicrogridState
    state_after: MicrogridState
    battery_kw: float
    grid_kw: float
    reward: float
    economic_cost: float
    pen_soc: float
    pen_deficit: float
    terminated: bool
    truncated: bool
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class MicrogridAdapter:
    """
    Adaptador entre pymgrid y el sistema multiagente.

    Parámetros
    ----------
    pymgrid_network       : pymgrid.Microgrid  Instancia ya configurada.
    data_dir_results      : str  Directorio con predicciones (inference.py).
    data_dir_raw          : str  Directorio con load y precio crudos.
    load_filename         : str  Nombre del CSV de load en data_dir_raw.
    price_filename        : str  Nombre del CSV de precio en data_dir_raw.
    horizon               : int  Steps máximos (se recorta al mínimo con datos).
    battery_capacity_kw   : float  Capacidad máxima de batería [kW].
    grid_max_kw           : float  Potencia máxima importable de red [kW].
    low_soc_threshold     : float  Umbral de SoC [0-1].
    lambda_soc            : float  Peso penalización SoC [€].
    lambda_deficit        : float  Peso penalización déficit [€/kW].
    export_price_ratio    : float  export_price = import_price × ratio.
    """

    # Nombres de columna en los CSVs (ajustar si cambia la fuente)
    COL_WIND_DATE  = "Date"
    COL_WIND_POWER = "Power_AE"                          # kW
    COL_SOLAR_DATE  = "Date"
    COL_SOLAR_POWER = "SystemProduction_AS"              # kW
    COL_LOAD        = "Electricity:Facility [kW](Hourly)"  # kW
    COL_PRICE_VALUE = "value"                            # €/MWh → se convierte

    def __init__(
        self,
        pymgrid_network,
        data_dir_results: str = "data/results",
        data_dir_raw: str = "data/raw",
        load_filename: str = (
            "RefBldgFullServiceRestaurantNew2004_v1.3_7.1_6A_USA_MN_MINNEAPOLIS.csv"
        ),
        price_filename: str = "precio2025-peninsula.csv",
        horizon: int = 24 * 365,
        battery_capacity_kw: float = 50.0,
        grid_max_kw: float = 100.0,
        low_soc_threshold: float = 0.2,
        lambda_soc: float = 5.0,
        lambda_deficit: float = 10.0,
        export_price_ratio: float = 0.6,
    ):
        self.mg = pymgrid_network
        self.data_dir_results = Path(data_dir_results)
        self.data_dir_raw     = Path(data_dir_raw)
        self.load_filename    = load_filename
        self.price_filename   = price_filename
        self.battery_capacity_kw = float(battery_capacity_kw)
        self.grid_max_kw         = float(grid_max_kw)
        self.low_soc_threshold   = float(low_soc_threshold)
        self.lambda_soc          = float(lambda_soc)
        self.lambda_deficit      = float(lambda_deficit)
        self.export_price_ratio  = float(export_price_ratio)

        self._data = self._load_and_align_data()
        self.horizon = min(horizon, len(self._data))

        self.current_step: int = 0
        self._history: list[dict] = []

        print(
            f"[MicrogridAdapter] Listo. "
            f"Horizonte efectivo: {self.horizon} steps "
            f"({self.horizon // 24} dias).\n"
            f"  Rango net_load: "
            f"[{self._data['net_load_kw'].min():.2f}, "
            f"{self._data['net_load_kw'].max():.2f}] kW\n"
            f"  Rango precio import: "
            f"[{self._data['import_price_eur_kwh'].min():.4f}, "
            f"{self._data['import_price_eur_kwh'].max():.4f}] EUR/kWh"
        )

    # ------------------------------------------------------------------
    # Carga y alineación de datos externos
    # ------------------------------------------------------------------

    def _load_and_align_data(self) -> pd.DataFrame:
        """
        Carga los cuatro CSVs y los une en un DataFrame indexado por step.

        Las fechas de los distintos CSVs NO coinciden entre si ni con
        pymgrid (datos de anios y ubicaciones distintas). Se ignoran
        deliberadamente: solo importa el orden temporal relativo.

        Conversiones:
          Precio: EUR/MWh / 1000 → EUR/kWh
          net_load: load - pv - wind
          Potencias negativas → clip a 0 (ruido numérico del modelo)
        """
        # 1. Eólico
        wind_path = self.data_dir_results / "Predicciones_Eolico.csv"
        self._check_file(wind_path)
        df_wind = (
            pd.read_csv(wind_path)[[self.COL_WIND_POWER]]
            .rename(columns={self.COL_WIND_POWER: "wind_kw"})
            .reset_index(drop=True)
        )
        df_wind["wind_kw"] = pd.to_numeric(df_wind["wind_kw"], errors="coerce")

        # 2. Solar
        solar_path = self.data_dir_results / "Predicciones_Solar.csv"
        self._check_file(solar_path)
        df_solar = (
            pd.read_csv(solar_path)[[self.COL_SOLAR_POWER]]
            .rename(columns={self.COL_SOLAR_POWER: "pv_kw"})
            .reset_index(drop=True)
        )
        df_solar["pv_kw"] = pd.to_numeric(df_solar["pv_kw"], errors="coerce")

        # 3. Load (sin columna de fecha, solo potencia)
        load_path = self.data_dir_raw / self.load_filename
        self._check_file(load_path)
        df_load = (
            pd.read_csv(load_path)[[self.COL_LOAD]]
            .rename(columns={self.COL_LOAD: "load_kw"})
            .reset_index(drop=True)
        )
        df_load["load_kw"] = pd.to_numeric(df_load["load_kw"], errors="coerce")

        # 4. Precio (separador ';', columna 'value' en EUR/MWh)
        price_path = self.data_dir_raw / self.price_filename
        self._check_file(price_path)
        df_price = (
            pd.read_csv(price_path, sep=";")[[self.COL_PRICE_VALUE]]
            .rename(columns={self.COL_PRICE_VALUE: "import_price_eur_mwh"})
            .reset_index(drop=True)
        )
        df_price["import_price_eur_mwh"] = pd.to_numeric(
            df_price["import_price_eur_mwh"], errors="coerce"
        )
        # Conversion EUR/MWh → EUR/kWh
        df_price["import_price_eur_kwh"] = df_price["import_price_eur_mwh"] / 1000.0
        df_price["export_price_eur_kwh"] = (
            df_price["import_price_eur_kwh"] * self.export_price_ratio
        )
        df_price = df_price[["import_price_eur_kwh", "export_price_eur_kwh"]]

        # 5. Longitud minima y concatenacion
        lengths = {
            "wind":  len(df_wind),
            "solar": len(df_solar),
            "load":  len(df_load),
            "price": len(df_price),
        }
        min_len = min(lengths.values())
        if len(set(lengths.values())) > 1:
            print(
                f"[MicrogridAdapter] ADVERTENCIA: longitudes distintas: "
                f"{lengths}. Se usan las primeras {min_len} filas de cada uno."
            )

        data = pd.concat(
            [df_wind[:min_len], df_solar[:min_len],
             df_load[:min_len], df_price[:min_len]],
            axis=1,
        ).reset_index(drop=True)

        # 6. Limpieza
        n_nan = data.isna().sum().sum()
        if n_nan > 0:
            print(
                f"[MicrogridAdapter] ADVERTENCIA: {n_nan} NaN encontrados. "
                f"Se imputan con la media de cada columna."
            )
            data = data.fillna(data.mean(numeric_only=True))

        # Clip a 0: potencias no pueden ser negativas
        data["wind_kw"] = data["wind_kw"].clip(lower=0.0)
        data["pv_kw"]   = data["pv_kw"].clip(lower=0.0)
        data["load_kw"] = data["load_kw"].clip(lower=0.0)

        # 7. Net load
        data["net_load_kw"] = data["load_kw"] - data["pv_kw"] - data["wind_kw"]

        data = data.astype(float).reset_index(drop=True)
        print(
            f"[MicrogridAdapter] Datos cargados: {len(data)} steps.\n"
            f"{data[['pv_kw','wind_kw','load_kw','net_load_kw','import_price_eur_kwh']].describe().round(3).to_string()}"
        )
        return data

    # ------------------------------------------------------------------
    # Estado actual (solo lectura, sin efectos secundarios)
    # ------------------------------------------------------------------

    def get_state(self) -> MicrogridState:
        """
        Devuelve el estado actual SIN avanzar el tiempo.
        Seguro llamarlo multiples veces en el mismo step.
        Los agentes productores llaman esto para formular sus propuestas.
        """
        row = self._data.iloc[self.current_step]
        return MicrogridState(
            step=self.current_step,
            hour=self.current_step % 24,
            day_of_year=(self.current_step // 24) % 365,
            pv_kw=float(row["pv_kw"]),
            wind_kw=float(row["wind_kw"]),
            load_kw=float(row["load_kw"]),
            net_load_kw=float(row["net_load_kw"]),
            soc=self._get_soc(),
            import_price_eur_kwh=float(row["import_price_eur_kwh"]),
            export_price_eur_kwh=float(row["export_price_eur_kwh"]),
        )

    # ------------------------------------------------------------------
    # Ejecución de la acción negociada
    # ------------------------------------------------------------------

    def step(self, battery_kw: float) -> StepResult:
        """
        Aplica la acción de batería negociada y avanza la simulacion un step.
        Llamado UNICAMENTE por el Coordinador tras resolver la negociacion.

        Parametros
        ----------
        battery_kw : float
            Potencia de bateria a aplicar [kW].
            > 0 → descarga (aporta energia).
            < 0 → carga (almacena energia).

        Returns
        -------
        StepResult con reward en EUR y metricas completas.
        """
        state_before = self.get_state()
        soc_before   = state_before.soc

        # Clipping fisico a capacidad de bateria
        battery_kw = float(np.clip(
            battery_kw, -self.battery_capacity_kw, self.battery_capacity_kw
        ))

        # Balance: Grid cubre lo que bateria + renovables no cubren
        # Grid > 0 → importamos | Grid < 0 → exportamos
        grid_kw = state_before.net_load_kw - battery_kw

        control_dict = {"battery": [battery_kw], "grid": [grid_kw]}

        # Ejecucion en pymgrid (ignoramos mg_reward, usamos el nuestro)
        _mg_obs, _mg_reward, mg_done, mg_info = self.mg.run(
            control_dict, normalized=False
        )

        terminated = bool(mg_done)
        self.current_step += 1
        truncated = bool(self.current_step >= self.horizon)

        soc_after = self._get_soc() if not terminated else soc_before

        # --- Componentes del reward ---

        grid_import_kw = max(0.0, grid_kw)
        grid_export_kw = max(0.0, -grid_kw)

        # 1. Coste economico real [EUR]
        #    Importar cuesta; exportar genera ingreso.
        economic_cost = (
            grid_import_kw * state_before.import_price_eur_kwh
            - grid_export_kw * state_before.export_price_eur_kwh
        )

        # 2. Penalizacion por SoC bajo [EUR equivalentes]
        #    λ_soc = 5 EUR esta en el mismo orden que el coste horario tipico
        #    (~1-4 EUR), por lo que evitar SoC bajo es relevante pero no domina.
        pen_soc = 0.0
        if soc_after < self.low_soc_threshold:
            soc_deficit_ratio = (
                (self.low_soc_threshold - soc_after) / self.low_soc_threshold
            )
            pen_soc = self.lambda_soc * soc_deficit_ratio

        # 3. Penalizacion por demanda no cubierta [EUR equivalentes]
        #    Si la importacion supera grid_max_kw, hay deficit de suministro.
        #    λ_deficit = 10 EUR >> λ_soc → la cobertura de AC es prioritaria.
        #    Captura: "evitar que el AC quede sin energia" (enunciado).
        pen_deficit = 0.0
        if grid_import_kw > self.grid_max_kw:
            uncovered_kw = grid_import_kw - self.grid_max_kw
            pen_deficit = self.lambda_deficit * uncovered_kw

        # Reward: maximizar = minimizar costes y penalizaciones
        reward = -economic_cost - pen_soc - pen_deficit

        # Estado tras la accion
        state_after = self.get_state() if not (terminated or truncated) else state_before

        # Info completa para logging, Prefect y xAI
        info = {
            "step":                  self.current_step - 1,
            "hour":                  state_before.hour,
            "day_of_year":           state_before.day_of_year,
            "pv_kw":                 state_before.pv_kw,
            "wind_kw":               state_before.wind_kw,
            "load_kw":               state_before.load_kw,
            "net_load_kw":           state_before.net_load_kw,
            "battery_kw":            battery_kw,
            "grid_kw":               grid_kw,
            "grid_import_kw":        grid_import_kw,
            "grid_export_kw":        grid_export_kw,
            "import_price_eur_kwh":  state_before.import_price_eur_kwh,
            "export_price_eur_kwh":  state_before.export_price_eur_kwh,
            "soc_before":            soc_before,
            "soc_after":             soc_after,
            "reward":                reward,
            "economic_cost_eur":     economic_cost,
            "pen_soc_eur":           pen_soc,
            "pen_deficit_eur":       pen_deficit,
            "terminated":            terminated,
            "truncated":             truncated,
            "mg_info":               mg_info,
        }

        result = StepResult(
            state_before=state_before,
            state_after=state_after,
            battery_kw=battery_kw,
            grid_kw=grid_kw,
            reward=reward,
            economic_cost=economic_cost,
            pen_soc=pen_soc,
            pen_deficit=pen_deficit,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        self._history.append(info)
        return result

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> MicrogridState:
        """Reinicia la simulacion al step 0 y limpia el historico."""
        self.mg.reset()
        self.current_step = 0
        self._history = []
        s = self.get_state()
        print(
            f"[MicrogridAdapter] Reset.\n"
            f"  SoC inicial:       {s.soc:.3f}\n"
            f"  Net load inicial:  {s.net_load_kw:.2f} kW\n"
            f"  Precio import:     {s.import_price_eur_kwh:.4f} EUR/kWh\n"
            f"  Steps disponibles: {self.horizon}"
        )
        return s

    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------

    @property
    def done(self) -> bool:
        return self.current_step >= self.horizon

    @property
    def remaining_steps(self) -> int:
        return max(0, self.horizon - self.current_step)

    # ------------------------------------------------------------------
    # Historico y metricas
    # ------------------------------------------------------------------

    def get_history_df(self) -> pd.DataFrame:
        """
        Historico completo como DataFrame (sin mg_info).
        Input para notebooks de analisis, xAI y Prefect.
        """
        if not self._history:
            return pd.DataFrame()
        return pd.DataFrame(self._history).drop(columns=["mg_info"], errors="ignore")

    def get_episode_summary(self) -> dict:
        """
        Resumen del episodio. Llamado por el Coordinador al finalizar
        la simulacion para logging y metricas del paper.
        """
        df = self.get_history_df()
        if df.empty:
            return {}

        total_load  = df["load_kw"].sum()
        total_renew = df["pv_kw"].sum() + df["wind_kw"].sum()

        return {
            "total_economic_cost_eur":   df["economic_cost_eur"].sum(),
            "total_reward":              df["reward"].sum(),
            "total_pen_soc_eur":         df["pen_soc_eur"].sum(),
            "total_pen_deficit_eur":     df["pen_deficit_eur"].sum(),
            "mean_import_price_eur_kwh": df["import_price_eur_kwh"].mean(),
            "total_load_kwh":            total_load,
            "total_pv_kwh":              df["pv_kw"].sum(),
            "total_wind_kwh":            df["wind_kw"].sum(),
            "total_grid_import_kwh":     df["grid_import_kw"].sum(),
            "total_grid_export_kwh":     df["grid_export_kw"].sum(),
            "renewable_coverage_pct":    round(
                100 * total_renew / total_load, 2
            ) if total_load > 0 else 0.0,
            "mean_soc":                  df["soc_after"].mean(),
            "min_soc":                   df["soc_after"].min(),
            "n_low_soc_steps":           int((df["soc_after"] < self.low_soc_threshold).sum()),
            "n_deficit_steps":           int((df["pen_deficit_eur"] > 0).sum()),
            "total_steps":               len(df),
        }

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _get_soc(self) -> float:
        """Lee el SoC actual de la bateria desde pymgrid [0, 1]."""
        return float(np.clip(self.mg.battery.item().soc, 0.0, 1.0))

    @staticmethod
    def _check_file(path: Path) -> None:
        """Lanza FileNotFoundError con mensaje util si el archivo no existe."""
        if not path.exists():
            raise FileNotFoundError(
                f"[MicrogridAdapter] Archivo no encontrado: {path}\n"
                f"  Comprueba la ruta y ejecuta models/inference.py antes "
                f"de arrancar el sistema multiagente."
            )
