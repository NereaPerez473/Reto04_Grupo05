"""
optimizacion_tasks.py
=====================
Celdas de Optimizacion.ipynb convertidas a tareas Prefect.

La busqueda de hiperparametros con Optuna (seccion 6 del notebook) se OMITE:
los best_params se leen directamente de optuna_microred.db, ya generado.

Tareas expuestas:
  tarea_setup_datos          - secciones 1-5 del notebook (datos + problema + cotas)
  tarea_run_nsgaii           - seccion 6a (ejecucion final NSGA-II)
  tarea_run_spea2            - seccion 6b (ejecucion final SPEA2)
  tarea_analizar_resultados  - secciones 7-12 (metricas, plots, guardado)
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # sin display (Docker headless)
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.spea2 import SPEA2
from jmetal.core.problem import FloatProblem
from jmetal.core.quality_indicator import (
    EpsilonIndicator,
    GenerationalDistance,
    HyperVolume,
    InvertedGenerationalDistance,
)
from jmetal.core.solution import FloatSolution
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations
from prefect import task

warnings.filterwarnings("ignore")
logging.getLogger("jmetal").setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Parametros globales (pueden sobreescribirse via argumentos de las tasks)
# ---------------------------------------------------------------------------
BASE_SEED = 42
T_HOURS = 168
START_HOUR = 0
PRED_YEAR = 2020
BATTERY_HOURS = 3.0
BATTERY_EFF_C = 0.95
BATTERY_EFF_D = 0.95
BATTERY_SOC_MIN = 0.10
BATTERY_SOC_MAX = 0.90
BATTERY_SOC_INI = 0.50
POPULATION_SIZE = 200
GENERATIONS_FINAL = 500
MARGIN = 0.1


# ---------------------------------------------------------------------------
# Clase del problema (seccion 3 del notebook)
# ---------------------------------------------------------------------------
class Microgrid(FloatProblem):
    """Despacho bi-objetivo de una microred con bateria (coste vs energia de red)."""

    def __init__(self, demand, p_solar, p_wind, a_solar, a_wind, a_grid,
                 bat_capacity, bat_soc_ini, bat_soc_min, bat_soc_max,
                 bat_eff_c, bat_eff_d):
        super().__init__()
        self.d  = np.asarray(demand,  dtype=float)
        self.ps = np.asarray(p_solar, dtype=float)
        self.pw = np.asarray(p_wind,  dtype=float)
        self.a1 = np.asarray(a_solar, dtype=float)
        self.a2 = np.asarray(a_wind,  dtype=float)
        self.a3 = np.asarray(a_grid,  dtype=float)
        self.T          = len(self.d)
        self.bat_cap    = bat_capacity
        self.bat_soc_ini = bat_soc_ini
        self.bat_soc_min = bat_soc_min
        self.bat_soc_max = bat_soc_max
        self.eff_c      = bat_eff_c
        self.eff_d      = bat_eff_d

        self.lower_bound = [0.0] * self.T + [0.0] * self.T + [-self.bat_cap] * self.T
        self.upper_bound = list(self.ps) + list(self.pw) + [self.bat_cap] * self.T
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels     = ["Coste (€)", "Energia red (kWh)"]

    def number_of_variables(self)  -> int: return 3 * self.T
    def number_of_objectives(self) -> int: return 2
    def number_of_constraints(self)-> int: return 0
    def name(self)                 -> str: return "Microgrid Coordination"

    def _repair(self, variables):
        x  = np.asarray(variables, dtype=float)
        x1 = np.clip(x[:self.T],         0.0, self.ps)
        x2 = np.clip(x[self.T:2*self.T], 0.0, self.pw)
        b  = np.clip(x[2*self.T:],       -self.bat_cap, self.bat_cap)

        soc   = self.bat_soc_ini
        b_rep = np.empty(self.T)
        for t in range(self.T):
            bt = b[t]
            if bt >= 0:
                max_charge = (self.bat_soc_max - soc) / self.eff_c
                bt = min(bt, max_charge)
                surplus = max(0.0, x1[t] + x2[t] - self.d[t])
                bt = min(bt, surplus)
                soc += bt * self.eff_c
            else:
                max_discharge = (soc - self.bat_soc_min) * self.eff_d
                bt = max(bt, -max_discharge)
                soc += bt / self.eff_d
            b_rep[t] = bt
        return x1, x2, b_rep

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x1, x2, b = self._repair(solution.variables)
        solution.variables = np.concatenate([x1, x2, b]).tolist()
        deficit = np.maximum(0.0, self.d - x1 - x2 + b)
        solution.objectives[0] = float(self.a1 @ x1 + self.a2 @ x2 + self.a3 @ deficit)
        solution.objectives[1] = float(deficit.sum())
        return solution


# ---------------------------------------------------------------------------
# Helpers de modulo
# ---------------------------------------------------------------------------
def _objective_bounds(d, ps, pw, a_s, a_w, a_g):
    """Cotas analiticas del espacio objetivo (seccion 4 del notebook)."""
    f1_min = f1_max = 0.0
    for t in range(len(d)):
        srcs = [(a_s[t], ps[t]), (a_w[t], pw[t])]
        rem, c = d[t], 0.0
        for price, cap in sorted(srcs, key=lambda z: z[0]):
            if price < a_g[t]:
                u = min(cap, rem); c += price * u; rem -= u
        c += a_g[t] * rem; f1_min += c
        rem, c = d[t], 0.0
        for price, cap in sorted(srcs, key=lambda z: -z[0]):
            if price > a_g[t]:
                u = min(cap, rem); c += price * u; rem -= u
        c += a_g[t] * rem; f1_max += c
    f2_min = float(np.maximum(0.0, d - ps - pw).sum())
    f2_max = float(d.sum())
    return f1_min, f1_max, f2_min, f2_max


def _run_algorithm(algo_name: str, problem: Microgrid, population_size: int,
                   eta_c: float, eta_m: float, crossover_prob: float,
                   seed: int, generations: int) -> dict:
    """Lanza un algoritmo jMetal y devuelve front como lista de dicts serializables."""
    random.seed(seed)
    np.random.seed(seed)
    mutation_prob   = 1.0 / problem.number_of_variables()
    max_evaluations = population_size * generations

    crossover_op = SBXCrossover(probability=crossover_prob, distribution_index=eta_c)
    mutation_op  = PolynomialMutation(probability=mutation_prob, distribution_index=eta_m)
    termination  = StoppingByEvaluations(max_evaluations=max_evaluations)

    t0 = time.time()
    if algo_name == "NSGAII":
        algo = NSGAII(
            problem=problem, population_size=population_size,
            offspring_population_size=population_size,
            mutation=mutation_op, crossover=crossover_op,
            termination_criterion=termination,
        )
    elif algo_name == "SPEA2":
        algo = SPEA2(
            problem=problem, population_size=population_size,
            offspring_population_size=population_size,
            mutation=mutation_op, crossover=crossover_op,
            termination_criterion=termination,
        )
    else:
        raise ValueError(f"Algoritmo desconocido: {algo_name}")

    algo.run()
    elapsed = time.time() - t0

    front = get_non_dominated_solutions(algo.result())
    # Serializar a dicts puros para que Prefect no dependa de clases jMetal
    front_data = [{"objectives": list(s.objectives), "variables": list(s.variables)}
                  for s in front]
    return {"front": front_data, "n_solutions": len(front_data), "elapsed_s": elapsed}


def _spread(front: np.ndarray, reference_front: np.ndarray) -> float:
    """Indicador Spread (diversidad del frente)."""
    front = front[np.argsort(front[:, 0])]
    reference_front = reference_front[np.argsort(reference_front[:, 0])]
    d_f = np.linalg.norm(front[0]  - reference_front[0])
    d_l = np.linalg.norm(front[-1] - reference_front[-1])
    distances = np.linalg.norm(np.diff(front, axis=0), axis=1)
    if len(distances) == 0:
        return 0.0
    d_mean = np.mean(distances)
    return (d_f + d_l + np.sum(np.abs(distances - d_mean))) / (
        d_f + d_l + len(distances) * d_mean)


def _plot_demand_battery(front_data: list[dict], T: int, DEMAND: np.ndarray,
                         bat_soc_ini: float, bat_soc_min_kwh: float,
                         bat_soc_max_kwh: float, bat_cap: float,
                         eff_c: float, eff_d: float,
                         algo_name: str, out_path: Path) -> None:
    """Genera el plot de cobertura de demanda y SOC para tres soluciones representativas."""
    if not front_data:
        return

    front_arr = np.array([s["objectives"] for s in front_data])
    idx_sorted     = np.argsort(front_arr[:, 0])
    idx_low_cost   = idx_sorted[0]
    idx_low_deficit= idx_sorted[-1]
    idx_mid        = idx_sorted[len(idx_sorted) // 2]

    labels = ["Min coste", "Punto medio", "Min deficit"]
    idxs   = [idx_low_cost, idx_mid, idx_low_deficit]
    colors = ["steelblue", "darkorange", "seagreen"]
    horas  = np.arange(T)

    fig, axes = plt.subplots(len(idxs), 3, figsize=(20, 4.5 * len(idxs)), sharex=True)
    for row, (idx, label, color) in enumerate(zip(idxs, labels, colors)):
        vars_ = np.array(front_data[idx]["variables"])
        x1 = vars_[:T]; x2 = vars_[T:2*T]; b = vars_[2*T:]

        soc = np.empty(T)
        soc_curr = bat_soc_ini
        for t in range(T):
            soc[t] = soc_curr
            soc_curr += b[t] * eff_c if b[t] >= 0 else b[t] / eff_d

        battery_discharge = np.maximum(0, -b)
        battery_charge    = np.maximum(0,  b)
        supply  = x1 + x2 - b
        deficit = np.maximum(0, DEMAND - supply)

        ax = axes[row, 0]
        ax.plot(horas, DEMAND,  color="black",     lw=2.0, label="Demanda")
        ax.plot(horas, x1 + x2, color="goldenrod", lw=1.5, label="Solar+eolico")
        ax.plot(horas, supply,  color="purple",    lw=2.0, label="Suministro neto")
        ax.fill_between(horas, supply, DEMAND, where=supply < DEMAND,
                        color="red", alpha=0.2, label="Deficit")
        ax.set_title(f"{label}: demanda y cobertura"); ax.set_ylabel("kWh/h")
        ax.legend(fontsize=8)

        ax = axes[row, 1]
        ax.bar(horas, x1,                                   color="gold",      alpha=0.85, label="Solar",            width=1.0)
        ax.bar(horas, x2,              bottom=x1,           color="skyblue",   alpha=0.85, label="Eolico",           width=1.0)
        ax.bar(horas, battery_discharge, bottom=x1 + x2,   color="limegreen", alpha=0.85, label="Bateria descarga", width=1.0)
        ax.bar(horas, -battery_charge,                      color="tomato",    alpha=0.6,  label="Bateria carga",    width=1.0)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{label}: desglose horario"); ax.set_ylabel("kWh/h")
        ax.legend(fontsize=8, ncol=2)

        ax = axes[row, 2]
        ax.plot(horas, soc, color=color, lw=2.2, label="SOC")
        ax.axhline(bat_soc_min_kwh, color="gray", ls=":", lw=1.0, label="SOC min/max")
        ax.axhline(bat_soc_max_kwh, color="gray", ls=":", lw=1.0)
        ax.set_ylim(0, bat_cap * 1.05)
        ax.set_title(f"{label}: estado de carga"); ax.set_ylabel("kWh")
        ax.legend(fontsize=8)

    for ax in axes[-1, :]:
        ax.set_xlabel("Hora")
    plt.suptitle(f"Cobertura de la demanda y uso de la bateria — {algo_name}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()


# ===========================================================================
# TAREA 1 — Setup: carga de datos y definicion del problema (secciones 1-5)
# ===========================================================================
@task(name="Setup datos optimizacion", log_prints=True, retries=1)
def tarea_setup_datos(
    data_dir_raw: str,
    data_dir_processed: str,
    data_dir_results: str,
    output_dir: str,
    start_hour: int = START_HOUR,
    pred_year: int = PRED_YEAR,
) -> dict[str, Any]:
    """
    Carga y alinea los datos de la microred.
    Equivale a las secciones 1-5 del notebook (sin la busqueda Optuna).
    Devuelve un diccionario serializable con todos los arrays y parametros.
    """
    # Seccion 2 — carga de datos
    dem = pd.read_csv(os.path.join(data_dir_raw, "demanda_restaurante.csv"))
    d_full = dem.iloc[:, 0].to_numpy(dtype=float)

    pg = pd.read_csv(os.path.join(data_dir_raw, "Precios", "precio2025-peninsula.csv"), sep=";")
    a_grid_full = pg["value"].to_numpy(dtype=float) / 1000.0

    ps_df = pd.read_csv(os.path.join(data_dir_processed, "Precios", "precio_solar_mwh.csv"),  sep=";")
    pe_df = pd.read_csv(os.path.join(data_dir_processed, "Precios", "precio_eolico_mwh.csv"), sep=";")
    a_solar_full = ps_df["precio_eur_mwh"].to_numpy(dtype=float) / 1000.0
    a_wind_full  = pe_df["precio_eur_mwh"].to_numpy(dtype=float) / 1000.0

    sol_df = pd.read_csv(os.path.join(data_dir_results, "Predicciones_Solar.csv"))
    win_df = pd.read_csv(os.path.join(data_dir_results, "Predicciones_Eolico.csv"))
    sol_df["Date"] = pd.to_datetime(sol_df["Date"])
    win_df["Date"] = pd.to_datetime(win_df["Date"])
    Psolar_full = sol_df.loc[sol_df["Date"].dt.year == pred_year, "SystemProduction_AS"].to_numpy(dtype=float)[:8760]
    Pwind_full  = win_df.loc[win_df["Date"].dt.year == pred_year, "Power_AE"].to_numpy(dtype=float)[:8760]

    N = min(len(d_full), len(a_grid_full), len(a_solar_full), len(a_wind_full),
            len(Psolar_full), len(Pwind_full))
    sl = slice(start_hour, start_hour + T_HOURS)
    assert start_hour + T_HOURS <= N, f"La ventana excede los datos disponibles (N={N})."

    DEMAND  = d_full[:N][sl] * 2.5
    P_SOLAR = Psolar_full[:N][sl]
    P_WIND  = Pwind_full[:N][sl]
    A_SOLAR = a_solar_full[:N][sl]
    A_WIND  = a_wind_full[:N][sl]
    A_GRID  = a_grid_full[:N][sl]

    T = len(DEMAND)
    print(f"Datos alineados. Horas en la ventana: T = {T}")
    print(f"  Demanda [kWh/h]: media {DEMAND.mean():.1f} | max {DEMAND.max():.1f}")
    print(f"  Cap. solar     : media {P_SOLAR.mean():.1f} | max {P_SOLAR.max():.1f}")
    print(f"  Cap. eolica    : media {P_WIND.mean():.1f}  | max {P_WIND.max():.1f}")

    # Seccion 4 — cotas y punto de referencia HV
    BATTERY_CAPACITY    = BATTERY_HOURS * DEMAND.mean()
    BATTERY_SOC_MIN_KWH = BATTERY_SOC_MIN * BATTERY_CAPACITY
    BATTERY_SOC_MAX_KWH = BATTERY_SOC_MAX * BATTERY_CAPACITY
    BATTERY_SOC_INI_KWH = BATTERY_SOC_INI * BATTERY_CAPACITY

    F1_MIN, F1_MAX, F2_MIN, F2_MAX = _objective_bounds(DEMAND, P_SOLAR, P_WIND, A_SOLAR, A_WIND, A_GRID)
    area_espacio    = (F1_MAX - F1_MIN) * (F2_MAX - F2_MIN)
    REFERENCE_POINT = (np.array([F1_MAX, F2_MAX]) * (1 + MARGIN)).tolist()

    print(f"f1 (coste)       : [{F1_MIN:.2f}, {F1_MAX:.2f}] €")
    print(f"f2 (energia red) : [{F2_MIN:.2f}, {F2_MAX:.2f}] kWh")
    print(f"Punto de referencia HV: {REFERENCE_POINT}")
    print(f"Bateria: {BATTERY_CAPACITY:.2f} kWh | SOC [{BATTERY_SOC_MIN_KWH:.2f}, {BATTERY_SOC_MAX_KWH:.2f}]")

    # Seccion 2b — plot exploratorio
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    plots_dir = out_path / "plots"
    plots_dir.mkdir(exist_ok=True)

    prod_media   = (P_SOLAR + P_WIND).mean()
    dem_media    = DEMAND.mean()
    ratio_dem    = dem_media / prod_media
    horas_superavit = ((P_SOLAR + P_WIND) >= DEMAND).mean() * 100
    horas_deficit   = 100 - horas_superavit
    ratio_grid_solar = (A_GRID / np.where(A_SOLAR > 0, A_SOLAR, np.nan)).mean()
    ratio_grid_wind  = (A_GRID / np.where(A_WIND  > 0, A_WIND,  np.nan)).mean()

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    horas = np.arange(T)
    ax0 = fig.add_subplot(gs[0, :])
    ax0.fill_between(horas, P_SOLAR + P_WIND, DEMAND,
                     where=(P_SOLAR+P_WIND) >= DEMAND, alpha=0.3, color="green", label="Superavit")
    ax0.fill_between(horas, P_SOLAR + P_WIND, DEMAND,
                     where=(P_SOLAR+P_WIND) < DEMAND,  alpha=0.3, color="red",   label="Deficit")
    ax0.plot(horas, P_SOLAR + P_WIND, label="Renovables (solar+eolico)", lw=1.2)
    ax0.plot(horas, DEMAND, label="Demanda", lw=1.2, color="k")
    ax0.axhline(dem_media, ls="--", color="gray", lw=0.8,
                label=f"Demanda media ({dem_media:.1f} kWh/h)")
    ax0.set_title(
        f"Produccion renovable vs demanda  |  ratio dem/prod = {ratio_dem:.1%}"
        f"  |  superavit: {horas_superavit:.0f}%  |  deficit: {horas_deficit:.0f}%"
    )
    ax0.set_xlabel("Hora"); ax0.set_ylabel("kWh/h"); ax0.legend(fontsize=8)

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.hist(A_SOLAR * 1000, bins=30, alpha=0.6, label="Solar (€/MWh)")
    ax1.hist(A_WIND  * 1000, bins=30, alpha=0.6, label="Eolico (€/MWh)")
    ax1.hist(A_GRID  * 1000, bins=30, alpha=0.6, label="Red (€/MWh)")
    ax1.set_title(f"Distribucion precios  |  red/solar={ratio_grid_solar:.2f}x  |  red/eolico={ratio_grid_wind:.2f}x")
    ax1.set_xlabel("€/MWh"); ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.bar(["Superavit", "Deficit"], [horas_superavit, horas_deficit],
            color=["green", "red"], alpha=0.7)
    ax2.set_ylabel("% horas"); ax2.set_title("Distribucion superavit / deficit")
    ax2.set_ylim(0, 100)
    for bar, val in zip(ax2.patches, [horas_superavit, horas_deficit]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
    plt.suptitle("Analisis exploratorio de los datos", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(plots_dir / "exploratorio_datos.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot guardado: {plots_dir / 'exploratorio_datos.png'}")

    return {
        "DEMAND":  DEMAND.tolist(),
        "P_SOLAR": P_SOLAR.tolist(),
        "P_WIND":  P_WIND.tolist(),
        "A_SOLAR": A_SOLAR.tolist(),
        "A_WIND":  A_WIND.tolist(),
        "A_GRID":  A_GRID.tolist(),
        "T":                T,
        "BATTERY_CAPACITY": BATTERY_CAPACITY,
        "BATTERY_SOC_INI_KWH": BATTERY_SOC_INI_KWH,
        "BATTERY_SOC_MIN_KWH": BATTERY_SOC_MIN_KWH,
        "BATTERY_SOC_MAX_KWH": BATTERY_SOC_MAX_KWH,
        "BATTERY_EFF_C": BATTERY_EFF_C,
        "BATTERY_EFF_D": BATTERY_EFF_D,
        "REFERENCE_POINT": REFERENCE_POINT,
        "area_espacio":   area_espacio,
        "F1_MIN": F1_MIN, "F1_MAX": F1_MAX,
        "F2_MIN": F2_MIN, "F2_MAX": F2_MAX,
        "output_dir": output_dir,
    }


# ===========================================================================
# TAREA 2 — Ejecucion NSGA-II final (seccion 6a del notebook)
# ===========================================================================
@task(name="Ejecutar NSGA-II final", log_prints=True, retries=0)
def tarea_run_nsgaii(
    datos: dict[str, Any],
    optuna_db: str,
    population_size: int = POPULATION_SIZE,
    generations: int = GENERATIONS_FINAL,
    base_seed: int = BASE_SEED,
) -> dict[str, Any]:
    """
    Carga los mejores hiperparametros de NSGA-II desde optuna_microred.db y
    ejecuta el algoritmo con presupuesto completo (GENERATIONS_FINAL).
    Seccion 6a del notebook.
    """
    algo_name = "NSGAII"
    storage_url = f"sqlite:///{optuna_db}"

    study = optuna.load_study(study_name=algo_name, storage=storage_url)
    best_params = study.best_params
    print(f"[{algo_name}] best_params desde DB: {best_params}")

    DEMAND  = np.array(datos["DEMAND"])
    P_SOLAR = np.array(datos["P_SOLAR"])
    P_WIND  = np.array(datos["P_WIND"])
    A_SOLAR = np.array(datos["A_SOLAR"])
    A_WIND  = np.array(datos["A_WIND"])
    A_GRID  = np.array(datos["A_GRID"])

    problem = Microgrid(
        DEMAND, P_SOLAR, P_WIND, A_SOLAR, A_WIND, A_GRID,
        bat_capacity  = datos["BATTERY_CAPACITY"],
        bat_soc_ini   = datos["BATTERY_SOC_INI_KWH"],
        bat_soc_min   = datos["BATTERY_SOC_MIN_KWH"],
        bat_soc_max   = datos["BATTERY_SOC_MAX_KWH"],
        bat_eff_c     = datos["BATTERY_EFF_C"],
        bat_eff_d     = datos["BATTERY_EFF_D"],
    )

    resultado = _run_algorithm(
        algo_name=algo_name, problem=problem,
        population_size=population_size,
        eta_c=best_params["eta_c"], eta_m=best_params["eta_m"],
        crossover_prob=best_params["crossover_prob"],
        seed=base_seed, generations=generations,
    )

    front_arr = np.array([s["objectives"] for s in resultado["front"]])
    hv = HyperVolume(datos["REFERENCE_POINT"]).compute(front_arr.tolist()) if len(front_arr) > 0 else 0.0

    print(f"[{algo_name}] Soluciones: {resultado['n_solutions']}  "
          f"HV: {hv:.6f}  Tiempo: {resultado['elapsed_s']:.1f} s")
    print(f"  Coste: [{front_arr[:,0].min():.2f}, {front_arr[:,0].max():.2f}] €")
    print(f"  Red:   [{front_arr[:,1].min():.2f}, {front_arr[:,1].max():.2f}] kWh")

    # Plot frente de Pareto
    plots_dir = Path(datos["output_dir"]) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(front_arr[:, 0], front_arr[:, 1], c="C0", edgecolors="k", s=30)
    ax.set_title(f"Frente de Pareto — {algo_name}")
    ax.set_xlabel("Coste (€)"); ax.set_ylabel("Energia de red (kWh)")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(str(plots_dir / "pareto_nsgaii.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Plot demanda, cobertura y SOC para 3 soluciones representativas
    _plot_demand_battery(
        front_data=resultado["front"], T=datos["T"], DEMAND=DEMAND,
        bat_soc_ini=datos["BATTERY_SOC_INI_KWH"],
        bat_soc_min_kwh=datos["BATTERY_SOC_MIN_KWH"],
        bat_soc_max_kwh=datos["BATTERY_SOC_MAX_KWH"],
        bat_cap=datos["BATTERY_CAPACITY"],
        eff_c=datos["BATTERY_EFF_C"], eff_d=datos["BATTERY_EFF_D"],
        algo_name=algo_name, out_path=plots_dir / "bateria_nsgaii.png",
    )
    print(f"Plots guardados en: {plots_dir}")

    return {
        "algo_name":   algo_name,
        "front":       resultado["front"],
        "hv":          hv,
        "n_solutions": resultado["n_solutions"],
        "elapsed_s":   resultado["elapsed_s"],
        "best_params": best_params,
    }


# ===========================================================================
# TAREA 3 — Ejecucion SPEA2 final (seccion 6b del notebook)
# ===========================================================================
@task(name="Ejecutar SPEA2 final", log_prints=True, retries=0)
def tarea_run_spea2(
    datos: dict[str, Any],
    optuna_db: str,
    population_size: int = POPULATION_SIZE,
    generations: int = GENERATIONS_FINAL,
    base_seed: int = BASE_SEED,
) -> dict[str, Any]:
    """
    Carga los mejores hiperparametros de SPEA2 desde optuna_microred.db y
    ejecuta el algoritmo con presupuesto completo (GENERATIONS_FINAL).
    Seccion 6b del notebook.
    """
    algo_name = "SPEA2"
    storage_url = f"sqlite:///{optuna_db}"

    study = optuna.load_study(study_name=algo_name, storage=storage_url)
    best_params = study.best_params
    print(f"[{algo_name}] best_params desde DB: {best_params}")

    DEMAND  = np.array(datos["DEMAND"])
    P_SOLAR = np.array(datos["P_SOLAR"])
    P_WIND  = np.array(datos["P_WIND"])
    A_SOLAR = np.array(datos["A_SOLAR"])
    A_WIND  = np.array(datos["A_WIND"])
    A_GRID  = np.array(datos["A_GRID"])

    problem = Microgrid(
        DEMAND, P_SOLAR, P_WIND, A_SOLAR, A_WIND, A_GRID,
        bat_capacity  = datos["BATTERY_CAPACITY"],
        bat_soc_ini   = datos["BATTERY_SOC_INI_KWH"],
        bat_soc_min   = datos["BATTERY_SOC_MIN_KWH"],
        bat_soc_max   = datos["BATTERY_SOC_MAX_KWH"],
        bat_eff_c     = datos["BATTERY_EFF_C"],
        bat_eff_d     = datos["BATTERY_EFF_D"],
    )

    resultado = _run_algorithm(
        algo_name=algo_name, problem=problem,
        population_size=population_size,
        eta_c=best_params["eta_c"], eta_m=best_params["eta_m"],
        crossover_prob=best_params["crossover_prob"],
        seed=base_seed, generations=generations,
    )

    front_arr = np.array([s["objectives"] for s in resultado["front"]])
    hv = HyperVolume(datos["REFERENCE_POINT"]).compute(front_arr.tolist()) if len(front_arr) > 0 else 0.0

    print(f"[{algo_name}] Soluciones: {resultado['n_solutions']}  "
          f"HV: {hv:.6f}  Tiempo: {resultado['elapsed_s']:.1f} s")
    print(f"  Coste: [{front_arr[:,0].min():.2f}, {front_arr[:,0].max():.2f}] €")
    print(f"  Red:   [{front_arr[:,1].min():.2f}, {front_arr[:,1].max():.2f}] kWh")

    plots_dir = Path(datos["output_dir"]) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(front_arr[:, 0], front_arr[:, 1], c="C1", edgecolors="k", s=30)
    ax.set_title(f"Frente de Pareto — {algo_name}")
    ax.set_xlabel("Coste (€)"); ax.set_ylabel("Energia de red (kWh)")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(str(plots_dir / "pareto_spea2.png"), dpi=150, bbox_inches="tight")
    plt.close()

    _plot_demand_battery(
        front_data=resultado["front"], T=datos["T"], DEMAND=DEMAND,
        bat_soc_ini=datos["BATTERY_SOC_INI_KWH"],
        bat_soc_min_kwh=datos["BATTERY_SOC_MIN_KWH"],
        bat_soc_max_kwh=datos["BATTERY_SOC_MAX_KWH"],
        bat_cap=datos["BATTERY_CAPACITY"],
        eff_c=datos["BATTERY_EFF_C"], eff_d=datos["BATTERY_EFF_D"],
        algo_name=algo_name, out_path=plots_dir / "bateria_spea2.png",
    )
    print(f"Plots guardados en: {plots_dir}")

    return {
        "algo_name":   algo_name,
        "front":       resultado["front"],
        "hv":          hv,
        "n_solutions": resultado["n_solutions"],
        "elapsed_s":   resultado["elapsed_s"],
        "best_params": best_params,
    }


# ===========================================================================
# TAREA 4 — Analisis, metricas y guardado (secciones 7-12 del notebook)
# ===========================================================================
@task(name="Analizar resultados y guardar", log_prints=True, retries=1)
def tarea_analizar_resultados(
    datos: dict[str, Any],
    res_nsgaii: dict[str, Any],
    res_spea2:  dict[str, Any],
    optuna_db: str,
) -> str:
    """
    Calcula metricas de calidad del frente (HV, GD, IGD, epsilon, spread),
    genera plots comparativos y guarda resultados en output_dir.
    Secciones 7-12 del notebook.
    """
    output_dir = Path(datos["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    REFERENCE_POINT = datos["REFERENCE_POINT"]
    area_espacio    = datos["area_espacio"]
    T               = datos["T"]

    pareto_fronts: dict[str, list] = {
        "NSGAII": res_nsgaii["front"],
        "SPEA2":  res_spea2["front"],
    }

    # ---- Seccion 8/9: leer trials desde DB y construir DataFrame ----
    storage_url = f"sqlite:///{optuna_db}"
    results = []
    for algo in ["NSGAII", "SPEA2"]:
        study = optuna.load_study(study_name=algo, storage=storage_url)
        for t in study.trials:
            results.append({
                "algorithm":       algo,
                "population_size": POPULATION_SIZE,
                "eta_c":           t.params.get("eta_c"),
                "eta_m":           t.params.get("eta_m"),
                "crossover_prob":  t.params.get("crossover_prob"),
                "hv_median":       t.value,
                "hv_mean":         t.user_attrs.get("hv_mean"),
                "hv_std":          t.user_attrs.get("hv_std"),
                "hv_min":          t.user_attrs.get("hv_min"),
                "hv_max":          t.user_attrs.get("hv_max"),
                "elapsed_mean_s":  t.user_attrs.get("elapsed_mean_s"),
            })

    df = pd.DataFrame(results).sort_values(
        ["algorithm", "hv_median"], ascending=[True, False]
    ).reset_index(drop=True)
    df["hv_norm"]     = df["hv_median"] / area_espacio
    df["hv_norm_std"] = df["hv_std"]    / area_espacio

    # ---- Seccion 7: guardar CSV y best_params ----
    ruta_csv = output_dir / "resultados_optuna_microred.csv"
    df.to_csv(str(ruta_csv), index=False)
    print(f"Guardado: {ruta_csv}")

    best_params_dict = {
        algo: res["best_params"]
        for algo, res in [("NSGAII", res_nsgaii), ("SPEA2", res_spea2)]
    }
    ruta_json = output_dir / "best_params_microred.json"
    with open(str(ruta_json), "w") as f:
        json.dump(best_params_dict, f, indent=2)
    print(f"Guardado: {ruta_json}")

    # ---- Seccion 10: sensibilidad HV vs hiperparametros ----
    hp_params = ["eta_c", "eta_m", "crossover_prob"]
    hp_labels = ["eta_c (SBX)", "eta_m (PM)", "Prob. cruce"]
    algorithms = list(df["algorithm"].unique())
    colors_map = {"NSGAII": "#1f77b4", "SPEA2": "#ff7f0e"}

    fig, axes = plt.subplots(len(algorithms), len(hp_params),
                             figsize=(15, 4.5 * len(algorithms)), sharey="row")
    if len(algorithms) == 1:
        axes = axes.reshape(1, -1)
    for row, algo in enumerate(algorithms):
        da = df[df["algorithm"] == algo]
        for col, (p, lbl) in enumerate(zip(hp_params, hp_labels)):
            ax = axes[row, col]
            ax.scatter(da[p], da["hv_norm"], color=colors_map.get(algo, "gray"), alpha=0.7, s=35)
            ax.set_xlabel(lbl); ax.set_title(algo)
            if col == 0:
                ax.set_ylabel("HV normalizado")
            ax.grid(alpha=0.3)
    plt.suptitle("Influencia de cada hiperparametro sobre el HV normalizado",
                 y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig(str(plots_dir / "sensibilidad_hv.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # HV vs tiempo
    fig, ax = plt.subplots(figsize=(10, 5))
    for algo in algorithms:
        da = df[df["algorithm"] == algo].sort_values("elapsed_mean_s")
        ax.scatter(da["elapsed_mean_s"], da["hv_norm"],
                   label=algo, color=colors_map.get(algo, "gray"), alpha=0.7, s=35)
    ax.set_xlabel("Tiempo medio por run (s)"); ax.set_ylabel("HV normalizado")
    ax.set_title("Calidad del frente vs coste computacional")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(plots_dir / "hv_vs_tiempo.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Seccion 11: frentes de Pareto superpuestos ----
    fig, ax = plt.subplots(figsize=(9, 6))
    markers = {"NSGAII": "o", "SPEA2": "s"}
    for algo_name, front_data in pareto_fronts.items():
        f1 = [s["objectives"][0] for s in front_data]
        f2 = [s["objectives"][1] for s in front_data]
        order = sorted(range(len(f1)), key=lambda i: f1[i])
        ax.plot([f1[i] for i in order], [f2[i] for i in order],
                marker=markers.get(algo_name, "o"), linestyle="-",
                color=colors_map.get(algo_name, "gray"), label=algo_name, alpha=0.8)
    ax.set_xlabel("f1 — Coste total (€)", fontsize=14)
    ax.set_ylabel("f2 — Energia de red (kWh)", fontsize=14)
    ax.set_title(f"Frentes de Pareto (ventana de {T} h)", fontsize=14)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(plots_dir / "pareto_comparativo.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Seccion 12: metricas de calidad ----
    # Frente optimo sintetico combinando ambos algoritmos
    todas_objs = []
    for fd in pareto_fronts.values():
        todas_objs.extend([s["objectives"] for s in fd])
    # Aproximacion al frente optimo: interpolacion suavizada sobre el no-dominado combinado
    real_arr = np.array(todas_objs)
    # Filtrar no dominados manualmente (jmetal espera objetos FloatSolution)
    real_sorted = real_arr[np.argsort(real_arr[:, 0])]
    K, MARGEN_Q = 50, 0.02
    f1_grid = np.linspace(real_sorted[:, 0].min(), real_sorted[:, 0].max(), K)
    f2_grid = np.interp(f1_grid, real_sorted[:, 0], real_sorted[:, 1])
    reference_front_array = np.column_stack([f1_grid, f2_grid]) * (1 - MARGEN_Q)

    metricas: dict[str, dict] = {}
    for algo_name, front_data in pareto_fronts.items():
        fa = np.array([s["objectives"] for s in front_data])
        hv_ind  = HyperVolume(reference_point=REFERENCE_POINT)
        gd_ind  = GenerationalDistance(reference_front=reference_front_array)
        igd_ind = InvertedGenerationalDistance(reference_front=reference_front_array)
        eps_ind = EpsilonIndicator(reference_front=reference_front_array)
        metricas[algo_name] = {
            "HV_raw":          float(hv_ind.compute(fa)),
            "HV_normalizado":  float(hv_ind.compute(fa)) / area_espacio,
            "GD":              float(gd_ind.compute(fa)),
            "IGD":             float(igd_ind.compute(fa)),
            "epsilon":         float(eps_ind.compute(fa)),
            "Spread":          float(_spread(fa, reference_front_array)),
            "n_soluciones":    len(front_data),
            "elapsed_s":       (res_nsgaii if algo_name == "NSGAII" else res_spea2)["elapsed_s"],
        }
        print(f"[{algo_name}] HV_norm={metricas[algo_name]['HV_normalizado']:.6f}  "
              f"GD={metricas[algo_name]['GD']:.4f}  "
              f"IGD={metricas[algo_name]['IGD']:.4f}  "
              f"n={metricas[algo_name]['n_soluciones']}")

    ruta_metricas = output_dir / "metricas_calidad.json"
    with open(str(ruta_metricas), "w") as f:
        json.dump(metricas, f, indent=2)
    print(f"Guardado: {ruta_metricas}")
    print(f"\nTodos los resultados en: {output_dir}")
    return str(output_dir)
