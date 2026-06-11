"""
train_multiagent_qlearning_cooperative_battery.py
==================================================
Entrenamiento cooperativo con batería pasiva.

Reward cooperativo con batería
--------------------------------
    total_clean  = solar_delivered + wind_delivered + battery_contribution
    grid_purchased = max(0, effective_demand - renewable_delivered)
    shared_reward  = total_clean − 2 × grid_purchased − total_shortfall

La batería que descarga para cubrir déficit eleva total_clean y reduce
grid_purchased → mejora el reward compartido para todos. Esto incentiva
a AS y AE a dejar que la batería actúe (no sobredeclarar para robarle
cuota de mercado, porque el shortfall penaliza el sistema completo).
"""

import numpy as np
import pandas as pd
from pathlib import Path

from strategy_qlearning_battery import StrategyQLearning
from simple_battery import SimpleBattery
from strategies import NegotiationStrategies



# ==================================================
# RUTAS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SOLAR_CSV = BASE_DIR / "data" / "results" / "Predicciones_Solar.csv"
WIND_CSV  = BASE_DIR / "data" / "results" / "Predicciones_Eolico.csv"
LOAD_CSV  = (BASE_DIR / "data" / "raw" /
             "demanda_restaurante.csv")
PRICE_CSV = BASE_DIR / "data" / "raw" / "Precios" / "precio2025-peninsula.csv"

OUTPUT_DIR = BASE_DIR / "mas_qlearning_battery" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# DATOS
# ==================================================

solar_df = pd.read_csv(SOLAR_CSV)
wind_df  = pd.read_csv(WIND_CSV)
load_df  = pd.read_csv(LOAD_CSV)
price_df = pd.read_csv(PRICE_CSV, sep=";")

solar = solar_df["SystemProduction_AS"].astype(float).values
wind  = wind_df["Power_AE"].astype(float).values
load  = load_df["Electricity:Facility [kW](Hourly)"].astype(float).values
price = price_df["value"].astype(float).values / 1000.0

n_steps = min(len(solar), len(wind), len(load), len(price))
solar = solar[:n_steps]
wind  = wind[:n_steps]
load  = load[:n_steps]
price = price[:n_steps]

print(f"Timesteps: {n_steps}")

# ==================================================
# AGENTES Y BATERÍA
# ==================================================

solar_agent = StrategyQLearning(alpha=0.1, gamma=0.95, epsilon=1.0)
wind_agent  = StrategyQLearning(alpha=0.1, gamma=0.95, epsilon=1.0)

battery = SimpleBattery(
    capacity_kwh=200.0, initial_soc=0.5,
    charge_eff=0.95, discharge_eff=0.95,
    max_power_kw=50.0, soc_min=0.05
)

N_EPISODES = 5000

# ==================================================
# HISTÓRICOS
# ==================================================

solar_rewards_history  = []
wind_rewards_history   = []
total_rewards_history  = []
solar_q_history        = []
wind_q_history         = []
battery_soc_history    = []
grid_purchased_history = []

# ==================================================
# ENTRENAMIENTO
# ==================================================

for episode in range(N_EPISODES):

    battery.reset()

    solar_total_reward = 0.0
    wind_total_reward  = 0.0
    episode_soc        = []
    episode_grid_kwh   = 0.0

    for t in range(n_steps - 1):

        demand        = load[t]
        current_price = price[t]
        solar_power   = solar[t]
        wind_power    = wind[t]

        # ==========================================
        # FÍSICA DE BATERÍA
        # ==========================================

        raw_renewable        = solar_power + wind_power
        battery_contribution = 0.0

        if raw_renewable >= demand:
            battery.charge(raw_renewable - demand)
            effective_demand = demand
        else:
            battery_contribution = battery.discharge(demand - raw_renewable)
            effective_demand = max(0.0, demand - battery_contribution)

        episode_soc.append(battery.soc)

        # ==========================================
        # ESTADOS
        # ==========================================

        solar_state = solar_agent.get_state(demand, current_price, battery.soc)
        wind_state  = wind_agent.get_state(demand, current_price, battery.soc)

        # ==========================================
        # ACCIONES Y ESTRATEGIAS
        # ==========================================

        solar_action   = solar_agent.choose_action(solar_state)
        wind_action    = wind_agent.choose_action(wind_state)
        solar_strategy = solar_agent.action_to_strategy(solar_action)
        wind_strategy  = wind_agent.action_to_strategy(wind_action)

        # ==========================================
        # PRODUCCIÓN DECLARADA
        # ==========================================

        if solar_strategy == "honest":
            solar_declared = solar_power
        elif solar_strategy == "hide_information":
            solar_declared = solar_power * 0.7
        else:
            solar_declared = solar_power * 1.3

        if wind_strategy == "honest":
            wind_declared = wind_power
        elif wind_strategy == "hide_information":
            wind_declared = wind_power * 0.7
        else:
            wind_declared = wind_power * 1.3

        # ==========================================
        # REPARTO PROPORCIONAL
        # ==========================================

        total_declared = solar_declared + wind_declared

        if total_declared > 0:
            solar_share = solar_declared / total_declared
            wind_share  = wind_declared  / total_declared
        else:
            solar_share = wind_share = 0.0

        solar_allocated = effective_demand * solar_share
        wind_allocated  = effective_demand * wind_share

        # ==========================================
        # ENERGÍA ENTREGADA
        # ==========================================

        solar_delivered = min(solar_allocated, solar_power)
        wind_delivered  = min(wind_allocated,  wind_power)

        # ==========================================
        # INGRESOS PROPIOS
        # ==========================================

        # 1. Obtener las propuestas según la estrategia
        solar_proposal = NegotiationStrategies.apply(
            solar_strategy, solar_power, current_price
        )
        wind_proposal = NegotiationStrategies.apply(
            wind_strategy, wind_power, current_price
        )

        # 2. Calcular los ingresos de cada planta
        solar_revenue = solar_delivered * solar_proposal.price_eur_kwh
        wind_revenue  = wind_delivered  * wind_proposal.price_eur_kwh

        # ==========================================
        # EXCESO DECLARADO Y GRID
        # ==========================================

        # Sustituyes el shortfall por el exceso declarado
        solar_declared_excess = max(0.0, solar_declared - solar_power)
        wind_declared_excess  = max(0.0, wind_declared  - wind_power)
        total_declared_excess = solar_declared_excess + wind_declared_excess

        renewable_delivered = solar_delivered + wind_delivered
        grid_purchased = max(0.0, effective_demand - renewable_delivered)
        episode_grid_kwh += grid_purchased

        # ==========================================
        # REWARD COOPERATIVO UNIFICADO EN €
        # ==========================================

        battery_savings = battery_contribution * current_price

        shared_reward = (
            (solar_revenue + wind_revenue + battery_savings)
            - (2.0 * grid_purchased * current_price)
            - (total_declared_excess * current_price) # <-- NUEVO (puedes multiplicar por 2.0 si quieres que duela más)
        )

        solar_reward = shared_reward
        wind_reward  = shared_reward

        solar_total_reward += solar_reward
        wind_total_reward  += wind_reward

        # ==========================================
        # NEXT STATES Y ACTUALIZACIONES
        # ==========================================

        next_solar_state = solar_agent.get_state(load[t+1], price[t+1], battery.soc)
        next_wind_state  = wind_agent.get_state(load[t+1],  price[t+1], battery.soc)

        solar_agent.update(solar_state, solar_action, solar_reward, next_solar_state)
        wind_agent.update(wind_state,   wind_action,  wind_reward,  next_wind_state)

    solar_rewards_history.append(solar_total_reward)
    wind_rewards_history.append(wind_total_reward)
    total_rewards_history.append(solar_total_reward + wind_total_reward)
    solar_q_history.append(np.mean(solar_agent.q_table))
    wind_q_history.append(np.mean(wind_agent.q_table))
    battery_soc_history.append(float(np.mean(episode_soc)) if episode_soc else 0.5)
    grid_purchased_history.append(episode_grid_kwh)

    solar_agent.epsilon = max(0.01, solar_agent.epsilon * 0.999)
    wind_agent.epsilon  = max(0.01, wind_agent.epsilon  * 0.999)

    print(
        f"Episode {episode+1}/{N_EPISODES}"
        f" | Solar={solar_total_reward:.1f}"
        f" | Wind={wind_total_reward:.1f}"
        f" | SoC={battery_soc_history[-1]:.2f}"
        f" | Grid={episode_grid_kwh:.1f} kWh"
    )

# ==================================================
# GUARDAR
# ==================================================

results_df = pd.DataFrame({
    "episode":            np.arange(1, N_EPISODES + 1),
    "solar_reward":       solar_rewards_history,
    "wind_reward":        wind_rewards_history,
    "total_reward":       total_rewards_history,
    "solar_q_mean":       solar_q_history,
    "wind_q_mean":        wind_q_history,
    "battery_soc_mean":   battery_soc_history,
    "grid_purchased_kwh": grid_purchased_history,
})

path = OUTPUT_DIR / "cooperative_battery_results.csv"
results_df.to_csv(path, index=False)
print(f"\nResultados → {path}")

np.save(OUTPUT_DIR / "cooperative_battery_solar_qtable.npy", solar_agent.q_table)
np.save(OUTPUT_DIR / "cooperative_battery_wind_qtable.npy",  wind_agent.q_table)

print("\nPOLITICA SOLAR\n")
for d in range(3):
    for p in range(3):
        for s in range(3):
            a = np.argmax(solar_agent.q_table[d, p, s])
            print(f"  D={d} P={p} SoC={s} -> {solar_agent.action_to_strategy(a)}")

print("\nPOLITICA WIND\n")
for d in range(3):
    for p in range(3):
        for s in range(3):
            a = np.argmax(wind_agent.q_table[d, p, s])
            print(f"  D={d} P={p} SoC={s} -> {wind_agent.action_to_strategy(a)}")
