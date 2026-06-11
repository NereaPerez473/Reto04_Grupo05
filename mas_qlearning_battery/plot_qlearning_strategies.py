import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==================================================
# RUTAS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "results"

PLOTS_DIR = RESULTS_DIR / "plots"

PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==================================================
# CARGAR CSV
# ==================================================

competitive_df = pd.read_csv(
    RESULTS_DIR / "competitive_battery_results.csv"
)


cooperative_df = pd.read_csv(
    RESULTS_DIR / "cooperative_battery_results.csv"
)

# ==================================================
# CARGAR Q-TABLES
# ==================================================

competitive_solar_q = np.load(
    RESULTS_DIR / "competitive_battery_solar_qtable.npy"
)

competitive_wind_q = np.load(
    RESULTS_DIR / "competitive_battery_wind_qtable.npy"
)

cooperative_solar_q = np.load(
    RESULTS_DIR / "cooperative_battery_solar_qtable.npy"
)

cooperative_wind_q = np.load(
    RESULTS_DIR / "cooperative_battery_wind_qtable.npy"
)

# ==================================================
# FIGURA 1
# REWARD SOLAR
# ==================================================

plt.figure(figsize=(10, 6))

plt.plot(
    competitive_df["episode"],
    competitive_df["solar_reward"],
    label="Competitivo"
)

plt.plot(
    cooperative_df["episode"],
    cooperative_df["solar_reward"],
    label="Cooperativo"
)

plt.title("Reward Solar")
plt.xlabel("Episodio")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "01_reward_solar.png"
)

plt.close()

# ==================================================
# FIGURA 2
# REWARD WIND
# ==================================================

plt.figure(figsize=(10, 6))

plt.plot(
    competitive_df["episode"],
    competitive_df["wind_reward"],
    label="Competitivo"
)

plt.plot(
    cooperative_df["episode"],
    cooperative_df["wind_reward"],
    label="Cooperativo"
)

plt.title("Reward Wind")
plt.xlabel("Episodio")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "02_reward_wind.png"
)

plt.close()

# ==================================================
# FIGURA 3
# REWARD TOTAL
# ==================================================

plt.figure(figsize=(10, 6))

plt.plot(
    competitive_df["episode"],
    competitive_df["total_reward"],
    label="Competitivo"
)

plt.plot(
    cooperative_df["episode"],
    cooperative_df["total_reward"],
    label="Cooperativo"
)

plt.title("Reward Total")
plt.xlabel("Episodio")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "03_reward_total.png"
)

plt.close()

# ==================================================
# FIGURA 4
# Q MEDIO SOLAR
# ==================================================

plt.figure(figsize=(10, 6))

plt.plot(
    competitive_df["episode"],
    competitive_df["solar_q_mean"],
    label="Competitivo"
)

plt.plot(
    cooperative_df["episode"],
    cooperative_df["solar_q_mean"],
    label="Cooperativo"
)

plt.title("Q Medio Solar")
plt.xlabel("Episodio")
plt.ylabel("Mean Q")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "04_qmean_solar.png"
)

plt.close()

# ==================================================
# FIGURA 5
# Q MEDIO WIND
# ==================================================

plt.figure(figsize=(10, 6))

plt.plot(
    competitive_df["episode"],
    competitive_df["wind_q_mean"],
    label="Competitivo"
)

plt.plot(
    cooperative_df["episode"],
    cooperative_df["wind_q_mean"],
    label="Cooperativo"
)

plt.title("Q Medio Wind")
plt.xlabel("Episodio")
plt.ylabel("Mean Q")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "05_qmean_wind.png"
)

plt.close()

# ==================================================
# FUNCIÓN HEATMAP
# ==================================================

def save_heatmap(qtable, title, filename):
    # Encontramos el valor máximo considerando todos los SoCs y Acciones
    # qtable shape: (3, 3, 3, 3) -> axis=(2,3) colapsa SoC y Acción
    heatmap = np.max(qtable, axis=(2, 3))

    plt.figure(figsize=(6, 5))
    plt.imshow(heatmap, aspect="auto", origin="lower") # origin="lower" para que (0,0) esté abajo
    plt.colorbar(label="Max Q-Value")
    plt.title(title)
    
    # Ejes discretos
    plt.xticks(ticks=[0, 1, 2], labels=["Bajo", "Medio", "Alto"])
    plt.yticks(ticks=[0, 1, 2], labels=["Baja", "Media", "Alta"])
    plt.xlabel("Precio")
    plt.ylabel("Demanda")
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename)
    plt.close()

# ==================================================
# FIGURA 6
# SOLAR COMPETITIVO
# ==================================================

save_heatmap(
    competitive_solar_q,
    "Solar Competitivo",
    "06_heatmap_solar_competitive.png"
)

# ==================================================
# FIGURA 7
# SOLAR COOPERATIVO
# ==================================================

save_heatmap(
    cooperative_solar_q,
    "Solar Cooperativo",
    "07_heatmap_solar_cooperative.png"
)

# ==================================================
# FIGURA 8
# WIND COMPETITIVO
# ==================================================

save_heatmap(
    competitive_wind_q,
    "Wind Competitivo",
    "08_heatmap_wind_competitive.png"
)

# ==================================================
# FIGURA 9
# WIND COOPERATIVO
# ==================================================

save_heatmap(
    cooperative_wind_q,
    "Wind Cooperativo",
    "09_heatmap_wind_cooperative.png"
)

print("\n" + "=" * 60)
print("GRÁFICAS GENERADAS")
print("=" * 60)

for file in sorted(
    PLOTS_DIR.glob("*.png")
):
    print(file.name)

print(
    f"\nGuardadas en:\n{PLOTS_DIR}"
)