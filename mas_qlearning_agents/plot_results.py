import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# CARGAR DATOS
# ==========================================

RESULTS_CSV = (
    Path(__file__).parent
    / "results"
    / "mas_qlearning_results.csv"
)

df = pd.read_csv(RESULTS_CSV)

plots_dir = (
    Path(__file__).parent
    / "results"
    / "plots"
)

plots_dir.mkdir(
    exist_ok=True
)

# ==========================================
# 1. ENERGÍA ENTREGADA
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    df["timestep"],
    df["solar_delivered_kw"],
    label="Solar"
)

plt.plot(
    df["timestep"],
    df["wind_delivered_kw"],
    label="Wind"
)

plt.plot(
    df["timestep"],
    df["grid_purchased_kw"],
    label="Grid"
)

plt.xlabel("Timestep")
plt.ylabel("kWh")
plt.title("Energy Supply")

plt.legend()

plt.tight_layout()

plt.savefig(
    plots_dir / "energy_supply.png"
)

plt.close()

# ==========================================
# 2. SHORTFALL
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    df["timestep"],
    df["solar_shortfall_kw"],
    label="Solar Shortfall"
)

plt.plot(
    df["timestep"],
    df["wind_shortfall_kw"],
    label="Wind Shortfall"
)

plt.xlabel("Timestep")
plt.ylabel("kWh")

plt.title("Shortfall")

plt.legend()

plt.tight_layout()

plt.savefig(
    plots_dir / "shortfall.png"
)

plt.close()

# ==========================================
# 3. COBERTURA RENOVABLE
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    df["timestep"],
    df["renewable_coverage_pct"]
)

plt.xlabel("Timestep")
plt.ylabel("%")

plt.title(
    "Renewable Coverage"
)

plt.tight_layout()

plt.savefig(
    plots_dir / "renewable_coverage.png"
)

plt.close()

# ==========================================
# 4. COSTE ACUMULADO
# ==========================================

df["cumulative_cost"] = (
    df["total_cost_eur"]
    .cumsum()
)

plt.figure(figsize=(10, 5))

plt.plot(
    df["timestep"],
    df["cumulative_cost"]
)

plt.xlabel("Timestep")
plt.ylabel("EUR")

plt.title(
    "Cumulative Cost"
)

plt.tight_layout()

plt.savefig(
    plots_dir / "cumulative_cost.png"
)

plt.close()

# ==========================================
# 5. COSTES POR FUENTE
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    df["timestep"],
    df["solar_cost_eur"],
    label="Solar"
)

plt.plot(
    df["timestep"],
    df["wind_cost_eur"],
    label="Wind"
)

plt.plot(
    df["timestep"],
    df["grid_cost_eur"],
    label="Grid"
)

plt.xlabel("Timestep")
plt.ylabel("EUR")

plt.title(
    "Cost by Source"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    plots_dir / "cost_by_source.png"
)

plt.close()

print(
    f"Plots saved in: {plots_dir}"
)