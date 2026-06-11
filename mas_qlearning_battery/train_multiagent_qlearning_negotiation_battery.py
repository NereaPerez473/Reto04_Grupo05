"""
train_multiagent_qlearning_negotiation_battery.py
=================================================
Entrenamiento de negociación con batería pasiva.

Usa strategy_qlearning_negotiation_battery.py:
    estado = (demanda, precio, producción_propia, soc_batería) → 81 estados

Reward de negociación:
    ingreso + bonus_cuota − 2 × shortfall
    (igual que la versión sin batería, pero el estado es más informativo)

El bonus de cuota (0.1 × allocated) incentiva a los agentes a declarar más
para capturar mayor parte de la demanda efectiva (tras la batería).
La penalización de shortfall (2×) limita la sobredeclaración.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from strategy_qlearning_negotiation_battery import StrategyQLearning
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
load  = (load_df["Electricity:Facility [kW](Hourly)"].astype(float)*2.5).values
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

N_EPISODES          = 5000
MARKET_BONUS_FACTOR = 0.1

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

solar_honest_history    = []
solar_hide_history      = []
solar_deception_history = []
wind_honest_history     = []
wind_hide_history       = []
wind_deception_history  = []

# ==================================================
# ENTRENAMIENTO
# ==================================================

for episode in range(N_EPISODES):

    battery.reset()

    solar_total_reward = 0.0
    wind_total_reward  = 0.0
    episode_soc        = []
    episode_grid_kwh   = 0.0

    solar_honest = solar_hide = solar_deception = 0
    wind_honest  = wind_hide  = wind_deception  = 0

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
        # ESTADOS (demanda × precio × prod_propia × soc)
        # ==========================================

        solar_state = solar_agent.get_state(
            demand, current_price, solar_power, battery.soc
        )
        wind_state = wind_agent.get_state(
            demand, current_price, wind_power, battery.soc
        )

        # ==========================================
        # ACCIONES
        # ==========================================

        solar_action   = solar_agent.choose_action(solar_state)
        wind_action    = wind_agent.choose_action(wind_state)
        solar_strategy = solar_agent.action_to_strategy(solar_action)
        wind_strategy  = wind_agent.action_to_strategy(wind_action)

        # Contadores de estrategia
        if solar_action == 0:   solar_honest    += 1
        elif solar_action == 1: solar_hide      += 1
        else:                   solar_deception += 1

        if wind_action == 0:    wind_honest     += 1
        elif wind_action == 1:  wind_hide       += 1
        else:                   wind_deception  += 1

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
        # REPARTO
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
        # ENTREGADO
        # ==========================================

        solar_delivered = min(solar_allocated, solar_power)
        wind_delivered  = min(wind_allocated,  wind_power)

        # ==========================================
        # EXCESO DECLARADO (Penalización por mentir)
        # ==========================================

        solar_declared_excess = max(0.0, solar_declared - solar_power)
        wind_declared_excess  = max(0.0, wind_declared  - wind_power)

        renewable_delivered = solar_delivered + wind_delivered
        grid_purchased = max(0.0, effective_demand - renewable_delivered)
        episode_grid_kwh += grid_purchased

        # ==========================================
        # INGRESOS Y BONUS DE CUOTA
        # ==========================================

        # 1. Obtener las propuestas según la estrategia elegida
        solar_proposal = NegotiationStrategies.apply(
            solar_strategy, solar_power, current_price
        )
        wind_proposal = NegotiationStrategies.apply(
            wind_strategy, wind_power, current_price
        )

        # 2. Calcular los ingresos usando el precio propio de la propuesta
        solar_revenue = solar_delivered * solar_proposal.price_eur_kwh
        wind_revenue  = wind_delivered  * wind_proposal.price_eur_kwh

        # 3. Monetizar el bonus de cuota (multiplicando por current_price)
        market_bonus_solar = MARKET_BONUS_FACTOR * solar_allocated * current_price
        market_bonus_wind  = MARKET_BONUS_FACTOR * wind_allocated  * current_price

        # ==========================================
        # REWARD DE NEGOCIACIÓN
        # ==========================================

        # 4. Calcular el reward unificando todo en euros (€)
        # 4. Calcular el reward unificando todo en euros (€)
        # Sustituimos 'shortfall' por 'declared_excess' para castigar solo el engaño
        solar_reward = (
            solar_revenue 
            + market_bonus_solar 
            - (2.0 * solar_declared_excess * current_price)
        )
        
        wind_reward = (
            wind_revenue 
            + market_bonus_wind 
            - (2.0 * wind_declared_excess * current_price)
        )

        solar_total_reward += solar_reward
        wind_total_reward  += wind_reward

        # ==========================================
        # NEXT STATES Y ACTUALIZACIONES
        # ==========================================

        next_solar_state = solar_agent.get_state(
            load[t+1], price[t+1], solar[t+1], battery.soc
        )
        next_wind_state = wind_agent.get_state(
            load[t+1], price[t+1], wind[t+1], battery.soc
        )

        solar_agent.update(solar_state, solar_action, solar_reward, next_solar_state)
        wind_agent.update(wind_state,   wind_action,  wind_reward,  next_wind_state)

    solar_rewards_history.append(solar_total_reward)
    wind_rewards_history.append(wind_total_reward)
    total_rewards_history.append(solar_total_reward + wind_total_reward)
    solar_q_history.append(np.mean(solar_agent.q_table))
    wind_q_history.append(np.mean(wind_agent.q_table))
    battery_soc_history.append(float(np.mean(episode_soc)) if episode_soc else 0.5)
    grid_purchased_history.append(episode_grid_kwh)
    solar_honest_history.append(solar_honest)
    solar_hide_history.append(solar_hide)
    solar_deception_history.append(solar_deception)
    wind_honest_history.append(wind_honest)
    wind_hide_history.append(wind_hide)
    wind_deception_history.append(wind_deception)

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
    "solar_honest":       solar_honest_history,
    "solar_hide":         solar_hide_history,
    "solar_deception":    solar_deception_history,
    "wind_honest":        wind_honest_history,
    "wind_hide":          wind_hide_history,
    "wind_deception":     wind_deception_history,
})

path = OUTPUT_DIR / "negotiation_battery_results.csv"
results_df.to_csv(path, index=False)
print(f"\nResultados → {path}")

np.save(OUTPUT_DIR / "negotiation_battery_solar_qtable.npy", solar_agent.q_table)
np.save(OUTPUT_DIR / "negotiation_battery_wind_qtable.npy",  wind_agent.q_table)

# ==================================================
# POLÍTICAS
# ==================================================

print("\nPOLITICA SOLAR (D × P × Prod × SoC) — muestra no-trivial\n")
for d in range(3):
    for p in range(3):
        for prod in range(3):
            for s in range(3):
                a = np.argmax(solar_agent.q_table[d, p, prod, s])
                print(f"  D={d} P={p} Prod={prod} SoC={s} -> {solar_agent.action_to_strategy(a)}")

print("\nPOLITICA WIND (D × P × Prod × SoC)\n")
for d in range(3):
    for p in range(3):
        for prod in range(3):
            for s in range(3):
                a = np.argmax(wind_agent.q_table[d, p, prod, s])
                print(f"  D={d} P={p} Prod={prod} SoC={s} -> {wind_agent.action_to_strategy(a)}")
