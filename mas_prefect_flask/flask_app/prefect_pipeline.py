from prefect import flow
from prefect import task

from sma_evaluator import run_evaluation

from pathlib import Path
import pandas as pd
import time


# ==================================================
# TASK 1
# ==================================================

@task(name="Load Data")
def load_data():

    t0 = time.perf_counter()

    result = {
        "solar": True,
        "wind": True,
        "load": True,
        "price": True
    }

    elapsed = round(time.perf_counter() - t0, 4)

    return result, elapsed


# ==================================================
# TASK 2
# ==================================================

@task(name="Load Q-Tables")
def load_qtables(mode):

    t0 = time.perf_counter()

    result = {
        "mode": mode
    }

    elapsed = round(time.perf_counter() - t0, 4)

    return result, elapsed


# ==================================================
# TASK 3
# ==================================================

@task(name="Simulate Episode")
def simulate_episode(mode):

    t0 = time.perf_counter()

    result = run_evaluation(mode)

    elapsed = round(time.perf_counter() - t0, 4)

    return result, elapsed


# ==================================================
# TASK 4
# ==================================================

@task(name="Generate Metrics")
def generate_metrics(result):

    t0 = time.perf_counter()

    metrics = {

        "grid_energy": result["grid_energy"],
        "battery_soc": result["battery_soc"],
        "renewable_energy": result["renewable_energy"],
        "renewable_share": result["renewable_share"],

        "consumer_cost": result["consumer_cost"],
        "solar_revenue": result["solar_revenue"],
        "wind_revenue": result["wind_revenue"],

        "solar_strategy": result["solar_strategy"],
        "wind_strategy": result["wind_strategy"],

        "mode": result["mode"]
    }

    elapsed = round(time.perf_counter() - t0, 4)

    return metrics, elapsed


# ==================================================
# TASK 5
# ==================================================

@task(name="Save Results")
def save_results(metrics):

    t0 = time.perf_counter()

    output_dir = Path("/project/results")

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        output_dir / "latest_results.csv",
        index=False
    )

    elapsed = round(time.perf_counter() - t0, 4)

    return True, elapsed


# ==================================================
# FLOW
# ==================================================

@flow(
    name="Evaluate Microgrid"
)
def evaluate_flow(
    mode="competitive"
):

    flow_start = time.perf_counter()

    _, t_load_data = load_data()

    _, t_load_qtables = load_qtables(mode)

    simulation_result, t_simulate = simulate_episode(mode)

    metrics, t_metrics = generate_metrics(
        simulation_result
    )

    _, t_save = save_results(
        metrics
    )

    total_elapsed = round(
        time.perf_counter() - flow_start, 4
    )

    # Tiempos de cada tarea del flow
    task_times = {
        "Load Data": t_load_data,
        "Load Q-Tables": t_load_qtables,
        "Simulate Episode": t_simulate,
        "Generate Metrics": t_metrics,
        "Save Results": t_save,
        "Total Flow": total_elapsed
    }

    metrics["task_times"] = task_times

    return metrics