"""
evaluate_episode.py
===================
Ejecuta un único episodio de evaluación (sin exploración, epsilon=0) 
usando las Q-Tables ya entrenadas. 

Guarda el estado paso a paso (hora a hora) para permitir un análisis
microscópico de la física de la batería y la demanda.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from simple_battery import SimpleBattery
# (Importa aquí tu agente, por ejemplo el competitivo)
from strategy_qlearning_battery import StrategyQLearning 

# ==================================================
# RUTAS Y DATOS (Ajusta según tu estructura)
# ==================================================
BASE_DIR = Path(__file__).resolve().parent.parent
SOLAR_CSV = BASE_DIR / "data" / "results" / "Predicciones_Solar.csv"
WIND_CSV  = BASE_DIR / "data" / "results" / "Predicciones_Eolico.csv"
LOAD_CSV  = BASE_DIR / "data" / "raw" / "demanda_restaurante.csv"
PRICE_CSV = BASE_DIR / "data" / "raw" / "Precios" / "precio2025-peninsula.csv"

# Cargar Q-Tables ya entrenadas (ejemplo: Competitivo)
# IMPORTANTE: Asegúrate de tener los .npy generados tras los 5000 episodios
QTABLE_SOLAR = BASE_DIR / "mas_qlearning_battery" / "results" / "competitive_battery_solar_qtable.npy"
QTABLE_WIND  = BASE_DIR / "mas_qlearning_battery" / "results" / "competitive_battery_wind_qtable.npy"

#RUTA DE GUARDADO
SAVE_PATH_IMG=BASE_DIR / "mas_qlearning_battery" / "results" / "plots" / "evaluate_episode_plot_comp.png"
SAVE_PATH_CSV= BASE_DIR / "mas_qlearning_battery" / "results" / "evaluate_episode_comp.csv"


# ==================================================
# CARGA DE DATOS HORARIOS
# ==================================================
solar = pd.read_csv(SOLAR_CSV)["SystemProduction_AS"].astype(float).values
wind  = pd.read_csv(WIND_CSV)["Power_AE"].astype(float).values
load  = (pd.read_csv(LOAD_CSV)["Electricity:Facility [kW](Hourly)"].astype(float)*2.5).values
price = pd.read_csv(PRICE_CSV, sep=";")["value"].astype(float).values / 1000.0

n_steps = min(len(solar), len(wind), len(load), len(price))
solar, wind, load, price = solar[:n_steps], wind[:n_steps], load[:n_steps], price[:n_steps]

# ==================================================
# INICIALIZACIÓN PARA EVALUACIÓN
# ==================================================
# Epsilon = 0 (No explora, usa el 100% de lo aprendido)
solar_agent = StrategyQLearning(epsilon=0.0)
wind_agent  = StrategyQLearning(epsilon=0.0)

solar_agent.q_table = np.load(QTABLE_SOLAR)
wind_agent.q_table  = np.load(QTABLE_WIND)

battery = SimpleBattery(capacity_kwh=200.0)

# Listas para guardar el registro hora a hora
history = {
    "hour": [],
    "demand": [],
    "solar_prod": [],
    "wind_prod": [],
    "battery_soc": [],
    "effective_demand": [],
    "solar_strategy": [],
    "wind_strategy": [],
    "grid_purchased": []
}

# ==================================================
# EJECUCIÓN PASO A PASO (1 Episodio Completo)
# ==================================================
for t in range(n_steps - 1):
    d = load[t]
    p = price[t]
    s = solar[t]
    w = wind[t]
    
    # 1. Batería Física
    raw_ren = s + w
    battery_contribution = 0.0
    if raw_ren >= d:
        battery.charge(raw_ren - d)
        eff_d = d
    else:
        battery_contribution = battery.discharge(d - raw_ren)
        eff_d = max(0.0, d - battery_contribution)
        
    # 2. Acciones Aprendidas
    s_state = solar_agent.get_state(d, p, battery.soc)
    w_state = wind_agent.get_state(d, p, battery.soc)
    
    s_action = solar_agent.choose_action(s_state)
    w_action = wind_agent.choose_action(w_state)
    
    # 3. Registrar variables
    history["hour"].append(t)
    history["demand"].append(d)
    history["solar_prod"].append(s)
    history["wind_prod"].append(w)
    history["battery_soc"].append(battery.soc)
    history["effective_demand"].append(eff_d)
    history["solar_strategy"].append(solar_agent.action_to_strategy(s_action))
    history["wind_strategy"].append(wind_agent.action_to_strategy(w_action))
    
    # Grid simplificado para el plot
    history["grid_purchased"].append(max(0.0, eff_d - (s+w)))

df_history = pd.DataFrame(history)

# ==================================================
# GRAFICAR UNA SEMANA INTERESANTE
# ==================================================
# Elijamos una semana de verano (ej. Julio, hora 4000 a 4168)
start_h = 4000
end_h = 4168
df_week = df_history.iloc[start_h:end_h]

fig, ax1 = plt.subplots(figsize=(12, 6))

# Eje Y Izquierdo: Potencia (kW)
ax1.plot(df_week["hour"], df_week["demand"], label="Demanda Real", color="black", linestyle="--")
ax1.plot(df_week["hour"], df_week["effective_demand"], label="Demanda Efectiva (Tras Batería)", color="red")
ax1.fill_between(df_week["hour"], 0, df_week["solar_prod"], alpha=0.3, color="orange", label="Solar")
ax1.fill_between(df_week["hour"], df_week["solar_prod"], df_week["solar_prod"] + df_week["wind_prod"], alpha=0.3, color="blue", label="Eólica")

ax1.set_xlabel("Horas del Año")
ax1.set_ylabel("Potencia (kW)")
ax1.legend(loc="upper left")

# Eje Y Derecho: SoC de la Batería
ax2 = ax1.twinx()
ax2.plot(df_week["hour"], df_week["battery_soc"] * 100, label="SoC Batería (%)", color="green", linewidth=2)
ax2.set_ylabel("Estado de Carga (%)", color="green")
ax2.set_ylim(0, 105)
ax2.legend(loc="upper right")

plt.title("Dinámica Física de la Batería y Producción Renovable (1 Semana)- Competitivo")
plt.tight_layout()
plt.savefig(str(SAVE_PATH_IMG), dpi=300, bbox_inches="tight")

plt.show()

# Guardar el CSV paso a paso para Excel/Tableau si lo necesitas
df_history.to_csv(str(SAVE_PATH_CSV), index=False)