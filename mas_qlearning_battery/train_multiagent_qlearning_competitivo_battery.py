"""
train_multiagent_qlearning_competitivo_battery.py
=================================================
Entrenamiento competitivo con batería pasiva.

Cambios respecto a train_multiagent_qlearning_competitivo.py
-------------------------------------------------------------
1. Batería (200 kWh, 50 kW) actúa como buffer pasivo ANTES de la negociación:
     - Excedente renovable → carga batería
     - Déficit renovable   → descarga batería (hasta soc_min)
2. La negociación AS/AE opera sobre la demanda efectiva tras la batería.
3. El SoC se añade como 3ª dimensión del estado (Q-table 3×3×3×3).
4. El grid_purchased (tras batería) se registra pero no penaliza en
   el reward competitivo individual.

Reward competitivo (sin cambios en lógica):
    AS: ingreso − 2 × shortfall
    AE: ingreso − 2 × shortfall
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
    capacity_kwh  = 200.0,
    initial_soc   = 0.5,
    charge_eff    = 0.95,
    discharge_eff = 0.95,
    max_power_kw  = 50.0,
    soc_min       = 0.05
)

# ==================================================
# PARÁMETROS
# ==================================================

N_EPISODES = 5000

# ==================================================
# HISTÓRICOS
# ==================================================

solar_rewards_history  = []
wind_rewards_history   = []
total_rewards_history  = []
solar_q_history        = []
wind_q_history         = []
battery_soc_history    = []     # SoC medio por episodio
grid_purchased_history = []     # kWh totales de red por episodio

# ==================================================
# ENTRENAMIENTO
# ==================================================

for episode in range(N_EPISODES):

    battery.reset()

    solar_total_reward  = 0.0
    wind_total_reward   = 0.0
    episode_soc         = []
    episode_grid_kwh    = 0.0

    for t in range(n_steps - 1):

        demand        = load[t]
        current_price = price[t]
        solar_power   = solar[t]
        wind_power    = wind[t]

        # ==========================================
        # 1. FASE 1: DESCARGA FÍSICA
        # ==========================================
        # La batería intenta cubrir la demanda bruta primero
        battery_contribution = battery.discharge(demand)
        effective_demand = demand - battery_contribution
        
        # Guardamos el SoC que verán los agentes en el "CFP" virtual
        episode_soc.append(battery.soc)

        # ==========================================
        # 2. ESTADOS (Uso de demanda efectiva)
        # ==========================================
        # Los agentes observan la demanda EFECTIVA, no la bruta
        solar_state = solar_agent.get_state(effective_demand, current_price, battery.soc)
        wind_state  = wind_agent.get_state(effective_demand, current_price, battery.soc)

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
        # REPARTO CHEAPEST-FIRST (Merit Order)
        # ==========================================
        # 1. Obtener las propuestas de precio según la estrategia elegida
        solar_proposal = NegotiationStrategies.apply(solar_strategy, solar_power, current_price)
        wind_proposal  = NegotiationStrategies.apply(wind_strategy, wind_power, current_price)

        # 2. Construir la lista de ofertas para la subasta
        proposals = [
            ("solar", solar_declared, solar_proposal.price_eur_kwh),
            ("wind",  wind_declared,  wind_proposal.price_eur_kwh)
        ]

        # 3. Filtrar fuentes racionales (por debajo de la red) y ordenar por precio ascendente
        viable_proposals = [p for p in proposals if p[2] < current_price]
        viable_proposals.sort(key=lambda x: x[2])

        solar_allocated = 0.0
        wind_allocated  = 0.0
        remaining_demand = effective_demand

        # 4. Despacho secuencial de energía
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
        # ENERGÍA ENTREGADA Y GESTIÓN DE EXCEDENTES
        # ==========================================
        # 1. Comercio: Lo que el consumidor realmente compra y absorbe
        solar_delivered = min(solar_allocated, solar_power)
        wind_delivered  = min(wind_allocated,  wind_power)
        renewable_delivered = solar_delivered + wind_delivered
        
        # 2. Red: Si lo comprado no alcanza, se tira de la red eléctrica
        grid_purchased = 0.0
        if renewable_delivered < effective_demand:
            grid_purchased = effective_demand - renewable_delivered

        # 3. Física: Todo el sol/viento que NO ha entrado al restaurante, va a la batería
        total_physical_production = solar_power + wind_power
        physical_surplus = max(0.0, total_physical_production - renewable_delivered)
        
        if physical_surplus > 0:
            battery.charge(physical_surplus)

        episode_grid_kwh += grid_purchased

        # ==========================================
        # INGRESOS Y REWARD COMPETITIVO
        # ==========================================
        solar_revenue = solar_delivered * solar_proposal.price_eur_kwh
        wind_revenue  = wind_delivered  * wind_proposal.price_eur_kwh

        solar_declared_excess = max(0.0, solar_declared - solar_power)
        wind_declared_excess  = max(0.0, wind_declared  - wind_power)

        # NUEVO: Castigo dinámico por vulnerabilidad de la red
        dynamic_penalty_factor = 3.0 - (2.0 * battery.soc)

        # Aplicamos el factor dinámico en lugar del 2.0 fijo
        solar_reward = solar_revenue - (dynamic_penalty_factor * solar_declared_excess * current_price)
        wind_reward  = wind_revenue  - (dynamic_penalty_factor * wind_declared_excess * current_price)

        solar_total_reward += solar_reward
        wind_total_reward  += wind_reward

        # ==========================================
        # NEXT STATES Y ACTUALIZACIONES
        # ==========================================
        # Corrección MDP: Pre-calcular la demanda efectiva esperada en t+1
        next_d = load[t+1]
        next_batt_contrib = min(next_d, battery.available_discharge_kw())
        next_eff_d = next_d - next_batt_contrib

        next_solar_state = solar_agent.get_state(next_eff_d, price[t+1], battery.soc)
        next_wind_state  = wind_agent.get_state(next_eff_d, price[t+1], battery.soc)

        solar_agent.update(solar_state, solar_action, solar_reward, next_solar_state)
        wind_agent.update(wind_state,   wind_action,  wind_reward,  next_wind_state)

    # Fin de episodio
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
        f" | SoC_mean={battery_soc_history[-1]:.2f}"
        f" | Grid={episode_grid_kwh:.1f} kWh"
    )

# ==================================================
# GUARDAR RESULTADOS
# ==================================================

results_df = pd.DataFrame({
    "episode":           np.arange(1, N_EPISODES + 1),
    "solar_reward":      solar_rewards_history,
    "wind_reward":       wind_rewards_history,
    "total_reward":      total_rewards_history,
    "solar_q_mean":      solar_q_history,
    "wind_q_mean":       wind_q_history,
    "battery_soc_mean":  battery_soc_history,
    "grid_purchased_kwh": grid_purchased_history,
})

path = OUTPUT_DIR / "competitive_battery_results.csv"
results_df.to_csv(path, index=False)
print(f"\nResultados → {path}")

np.save(OUTPUT_DIR / "competitive_battery_solar_qtable.npy", solar_agent.q_table)
np.save(OUTPUT_DIR / "competitive_battery_wind_qtable.npy",  wind_agent.q_table)

# ==================================================
# POLÍTICAS APRENDIDAS
# ==================================================

print("\nPOLITICA SOLAR (Demanda × Precio × SoC)\n")
for d in range(3):
    for p in range(3):
        for s in range(3):
            a = np.argmax(solar_agent.q_table[d, p, s])
            print(f"  D={d} P={p} SoC={s} -> {solar_agent.action_to_strategy(a)}")

print("\nPOLITICA WIND (Demanda × Precio × SoC)\n")
for d in range(3):
    for p in range(3):
        for s in range(3):
            a = np.argmax(wind_agent.q_table[d, p, s])
            print(f"  D={d} P={p} SoC={s} -> {wind_agent.action_to_strategy(a)}")
