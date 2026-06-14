from prefect import flow
from prefect import task

from sma_evaluator import run_evaluation

from pathlib import Path
import pandas as pd


# ==================================================
# TASK 1
# ==================================================

@task(name="Load Data")
def load_data():

    return {
        "solar": True,
        "wind": True,
        "load": True,
        "price": True
    }


# ==================================================
# TASK 2
# ==================================================

@task(name="Load Q-Tables")
def load_qtables(mode):

    return {
        "mode": mode
    }


# ==================================================
# TASK 3
# ==================================================

@task(name="Simulate Episode")
def simulate_episode(mode):

    return run_evaluation(mode)


# ==================================================
# TASK 4
# ==================================================

@task(name="Generate Metrics")
def generate_metrics(result):

    return {

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


# ==================================================
# TASK 5
# ==================================================

@task(name="Save Results")
def save_results(metrics):

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

    return True


# ==================================================
# FLOW
# ==================================================

@flow(
    name="Evaluate Microgrid"
)
def evaluate_flow(
    mode="competitive"
):

    load_data()

    load_qtables(mode)

    simulation_result = simulate_episode(mode)

    metrics = generate_metrics(
        simulation_result
    )

    save_results(
        metrics
    )

    return metrics