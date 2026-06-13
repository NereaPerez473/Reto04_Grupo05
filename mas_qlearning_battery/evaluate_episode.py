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
from strategy_qlearning_battery import StrategyQLearning 
from strategies import NegotiationStrategies # NUEVO: Necesario para la subasta

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
SAVE_PATH_IMG= BASE_DIR / "mas_qlearning_battery" / "results" / "plots" / "evaluate_episode_plot_comp.png"
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
    
    # ==========================================
    # 1. FASE 1: DESCARGA FÍSICA
    # ==========================================
    battery_contribution = battery.discharge(d)
    eff_d = d - battery_contribution
    
    # ==========================================
    # 2. ESTADOS Y ACCIONES (Automático 3D/4D)
    # ==========================================
    if solar_agent.q_table.ndim == 5:
        # Modelo Negociación
        s_state = solar_agent.get_state(eff_d, p, s, battery.soc)
        w_state = wind_agent.get_state(eff_d, p, w, battery.soc)
    else:
        # Modelo Competitivo / Cooperativo
        s_state = solar_agent.get_state(eff_d, p, battery.soc)
        w_state = wind_agent.get_state(eff_d, p, battery.soc)
    
    s_action = solar_agent.choose_action(s_state)
    w_action = wind_agent.choose_action(w_state)
    
    solar_strategy = solar_agent.action_to_strategy(s_action)
    wind_strategy  = wind_agent.action_to_strategy(w_action)

    # ==========================================
    # 3. PRODUCCIÓN DECLARADA
    # ==========================================
    solar_declared = s if solar_strategy == "honest" else (s * 0.7 if solar_strategy == "hide_information" else s * 1.3)
    wind_declared  = w if wind_strategy == "honest" else (w * 0.7 if wind_strategy == "hide_information" else w * 1.3)

    # ==========================================
    # 4. REPARTO CHEAPEST-FIRST (Merit Order)
    # ==========================================
    solar_proposal = NegotiationStrategies.apply(solar_strategy, s, p)
    wind_proposal  = NegotiationStrategies.apply(wind_strategy, w, p)

    proposals = [
        ("solar", solar_declared, solar_proposal.price_eur_kwh),
        ("wind",  wind_declared,  wind_proposal.price_eur_kwh)
    ]

    viable_proposals = [prop for prop in proposals if prop[2] < p]
    viable_proposals.sort(key=lambda x: x[2])

    solar_allocated = 0.0
    wind_allocated  = 0.0
    remaining_demand = eff_d

    for source, declared_kw, price_kwh in viable_proposals:
        if remaining_demand <= 0:
            break
        purchase = min(remaining_demand, declared_kw)
        if source == "solar":
            solar_allocated = purchase
        else:
            wind_allocated = purchase
        remaining_demand = max(0.0, remaining_demand - purchase)

    # ==========================================
    # 5. LIQUIDACIÓN Y BATERÍA
    # ==========================================
    solar_delivered = min(solar_allocated, s)
    wind_delivered  = min(wind_allocated, w)
    renewable_delivered = solar_delivered + wind_delivered

    grid_purchased = 0.0
    if renewable_delivered < eff_d:
        grid_purchased = eff_d - renewable_delivered

    total_physical_production = s + w
    physical_surplus = max(0.0, total_physical_production - renewable_delivered)
    
    if physical_surplus > 0:
        battery.charge(physical_surplus)

    # ==========================================
    # 6. REGISTRO DE DATOS
    # ==========================================
    history["hour"].append(t)
    history["demand"].append(d)
    history["solar_prod"].append(s)
    history["wind_prod"].append(w)
    history["battery_soc"].append(battery.soc)
    history["effective_demand"].append(eff_d)
    history["solar_strategy"].append(solar_strategy)
    history["wind_strategy"].append(wind_strategy)
    history["grid_purchased"].append(grid_purchased)

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

# Asegurar que el directorio de la imagen exista antes de guardar
SAVE_PATH_IMG.parent.mkdir(parents=True, exist_ok=True)

plt.title("Dinámica Física de la Batería y Producción Renovable (1 Semana) - Competitivo")
plt.tight_layout()
plt.savefig(str(SAVE_PATH_IMG), dpi=300, bbox_inches="tight")

plt.show()

# Guardar el CSV paso a paso para Excel/Tableau si lo necesitas
df_history.to_csv(str(SAVE_PATH_CSV), index=False)