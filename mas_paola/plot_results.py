import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "mas_paola" / "results" / "mas_summary.csv"

OUTPUT_DIR = BASE_DIR / "mas_paola" / "results" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

df["scenario"] = (
    "AS=" + df["strategy_as"] +
    "\nAE=" + df["strategy_ae"]
)

# ---------------------------------------------------
# COSTE TOTAL
# ---------------------------------------------------

plt.figure(figsize=(10, 5))
plt.bar(df["scenario"], df["total_cost_eur"])
plt.title("Coste total por estrategia")
plt.ylabel("Coste total (€)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "coste_total.png")
plt.close()

# ---------------------------------------------------
# COBERTURA RENOVABLE
# ---------------------------------------------------

plt.figure(figsize=(10, 5))
plt.bar(df["scenario"], df["mean_renewable_coverage"])
plt.title("Cobertura renovable media")
plt.ylabel("Cobertura (%)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cobertura_renovable.png")
plt.close()

# ---------------------------------------------------
# ENERGÍA COMPRADA A RED
# ---------------------------------------------------

plt.figure(figsize=(10, 5))
plt.bar(df["scenario"], df["total_grid_purchased_kwh"])
plt.title("Compra total a red")
plt.ylabel("kWh")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "energia_red.png")
plt.close()

# ---------------------------------------------------
# SHORTFALL
# ---------------------------------------------------

plt.figure(figsize=(10, 5))
plt.bar(df["scenario"], df["total_shortfall_kwh"])
plt.title("Déficit por engaño (shortfall)")
plt.ylabel("kWh")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "shortfall.png")
plt.close()

print("\nGráficas guardadas en:")
print(OUTPUT_DIR)