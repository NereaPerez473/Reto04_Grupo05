import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuración estética para TFM
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# ==================================================
# RUTAS
# ==================================================
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# CARGAR DATOS
# ==================================================
try:
    competitive_df = pd.read_csv(RESULTS_DIR / "competitive_battery_results.csv")
    cooperative_df = pd.read_csv(RESULTS_DIR / "cooperative_battery_results.csv")
    negotiation_df = pd.read_csv(RESULTS_DIR / "negotiation_battery_results.csv")
except FileNotFoundError as e:
    print(f"Error cargando CSVs: {e}")
    print("Asegúrate de ejecutar primero los tres scripts de entrenamiento.")
    exit()

try:
    comp_solar_q = np.load(RESULTS_DIR / "competitive_battery_solar_qtable.npy")
    comp_wind_q  = np.load(RESULTS_DIR / "competitive_battery_wind_qtable.npy")
    coop_solar_q = np.load(RESULTS_DIR / "cooperative_battery_solar_qtable.npy")
    coop_wind_q  = np.load(RESULTS_DIR / "cooperative_battery_wind_qtable.npy")
    nego_solar_q = np.load(RESULTS_DIR / "negotiation_battery_solar_qtable.npy")
    nego_wind_q  = np.load(RESULTS_DIR / "negotiation_battery_wind_qtable.npy")
except FileNotFoundError as e:
    print(f"Error cargando NPYs: {e}")
    exit()

# Suavizado de curvas (Moving Average) para visualizar tendencias
WINDOW = 50

# ==================================================
# 1. GRÁFICAS DE CURVAS DE APRENDIZAJE (REWARDS)
# ==================================================
def plot_learning_curve(metric, title, filename, ylabel="Euros (€)"):
    plt.figure(figsize=(10, 6))
    
    plt.plot(competitive_df["episode"], competitive_df[metric].rolling(WINDOW).mean(), label="Competitivo", alpha=0.8)
    plt.plot(cooperative_df["episode"], cooperative_df[metric].rolling(WINDOW).mean(), label="Cooperativo", alpha=0.8)
    plt.plot(negotiation_df["episode"], negotiation_df[metric].rolling(WINDOW).mean(), label="Negociación (Bonus)", alpha=0.8)
    
    plt.title(f"{title} (Media Móvil N={WINDOW})", fontweight='bold')
    plt.xlabel("Episodio")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename)
    plt.close()

plot_learning_curve("solar_reward", "Convergencia de Recompensa: Agente Solar", "01_reward_solar.png")
plot_learning_curve("wind_reward", "Convergencia de Recompensa: Agente Eólico", "02_reward_wind.png")
plot_learning_curve("total_reward", "Recompensa Total del Sistema", "03_reward_total.png")

# ==================================================
# 2. MÉTRICAS FÍSICAS: DEPENDENCIA DE LA RED
# ==================================================
plt.figure(figsize=(10, 6))
plt.plot(competitive_df["episode"], competitive_df["grid_purchased_kwh"].rolling(WINDOW).mean(), label="Competitivo", color="red", alpha=0.7)
plt.plot(cooperative_df["episode"], cooperative_df["grid_purchased_kwh"].rolling(WINDOW).mean(), label="Cooperativo", color="green", alpha=0.7)
plt.plot(negotiation_df["episode"], negotiation_df["grid_purchased_kwh"].rolling(WINDOW).mean(), label="Negociación", color="blue", alpha=0.7)

plt.title(f"Dependencia Energética de la Red Eléctrica (Media Móvil N={WINDOW})", fontweight='bold')
plt.xlabel("Episodio")
plt.ylabel("Energía Comprada a Red (kWh)")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "04_grid_purchased.png")
plt.close()

# ==================================================
# 3. EVOLUCIÓN DE ESTRATEGIAS (Solo Negociación)
# ==================================================
def plot_strategy_evolution(df, agent_prefix, title, filename):
    plt.figure(figsize=(10, 6))
    
    # Calcular % de uso por episodio (total de pasos = n_steps - 1)
    # Asumimos n_steps fijo para normalizar, o directamente graficamos recuentos
    plt.plot(df["episode"], df[f"{agent_prefix}_honest"].rolling(WINDOW).mean(), label="Honesto", color="green")
    plt.plot(df["episode"], df[f"{agent_prefix}_hide"].rolling(WINDOW).mean(), label="Ocultación (Premium)", color="orange")
    plt.plot(df["episode"], df[f"{agent_prefix}_deception"].rolling(WINDOW).mean(), label="Engaño (Dumping)", color="red")
    
    plt.title(f"{title} - Evolución de Tácticas", fontweight='bold')
    plt.xlabel("Episodio")
    plt.ylabel("Frecuencia de Uso (suavizada)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename)
    plt.close()

plot_strategy_evolution(negotiation_df, "solar", "Agente Solar (Modo Negociación)", "05_strategies_solar.png")
plot_strategy_evolution(negotiation_df, "wind", "Agente Eólico (Modo Negociación)", "06_strategies_wind.png")

# ==================================================
# 4. HEATMAPS ADAPTATIVOS
# ==================================================
def save_heatmap(qtable, title, filename):
    # Detección de dimensionalidad
    if qtable.ndim == 4:
        # Estado 3D (Competitivo/Cooperativo)
        # qtable shape: (Demanda, Precio, SoC, Acción) -> colapsar SoC y Acción
        heatmap = np.max(qtable, axis=(2, 3))
    elif qtable.ndim == 5:
        # Estado 4D (Negociación)
        # qtable shape: (Demanda, Precio, Prod_Propia, SoC, Acción) -> colapsar Prod, SoC y Acción
        heatmap = np.max(qtable, axis=(2, 3, 4))
    else:
        return

    plt.figure(figsize=(6, 5))
    sns.heatmap(heatmap, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Max Q-Value'})
    
    plt.title(title, fontweight='bold')
    plt.xticks(ticks=[0.5, 1.5, 2.5], labels=["Bajo", "Medio", "Alto"])
    plt.yticks(ticks=[0.5, 1.5, 2.5], labels=["Baja", "Media", "Alta"])
    plt.xlabel("Precio de Red")
    plt.ylabel("Demanda Restaurante")
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename)
    plt.close()

save_heatmap(comp_solar_q, "Max Q-Value Solar (Competitivo)", "07_heatmap_solar_comp.png")
save_heatmap(coop_solar_q, "Max Q-Value Solar (Cooperativo)", "08_heatmap_solar_coop.png")
save_heatmap(nego_solar_q, "Max Q-Value Solar (Negociación)", "09_heatmap_solar_nego.png")

save_heatmap(comp_wind_q, "Max Q-Value Wind (Competitivo)", "10_heatmap_wind_comp.png")
save_heatmap(coop_wind_q, "Max Q-Value Wind (Cooperativo)", "11_heatmap_wind_coop.png")
save_heatmap(nego_wind_q, "Max Q-Value Wind (Negociación)", "12_heatmap_wind_nego.png")

print("\n" + "=" * 60)
print("GRÁFICAS GENERADAS CON ÉXITO")
print("=" * 60)
for file in sorted(PLOTS_DIR.glob("*.png")):
    print(f" - {file.name}")
print(f"\nUbicación: {PLOTS_DIR}")