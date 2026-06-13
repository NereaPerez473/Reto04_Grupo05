import numpy as np
import pandas as pd
from pathlib import Path

from strategy_qlearning import StrategyQLearning
from strategies import NegotiationStrategies

# ==================================================
# RUTAS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SOLAR_CSV = (
    BASE_DIR
    / "data"
    / "results"
    / "Predicciones_Solar.csv"
)

WIND_CSV = (
    BASE_DIR
    / "data"
    / "results"
    / "Predicciones_Eolico.csv"
)

LOAD_CSV = (
    BASE_DIR
    / "data"
    / "raw"
    / "RefBldgFullServiceRestaurantNew2004_v1.3_7.1_6A_USA_MN_MINNEAPOLIS.csv"
)

PRICE_CSV = (
    BASE_DIR
    / "data"
    / "raw"
    / "Precios"
    / "precio2025-peninsula.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "mas_qlearning"
    / "results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==================================================
# DATOS
# ==================================================

solar_df = pd.read_csv(SOLAR_CSV)

wind_df = pd.read_csv(WIND_CSV)

load_df = pd.read_csv(LOAD_CSV)

price_df = pd.read_csv(
    PRICE_CSV,
    sep=";"
)

solar = (
    solar_df["SystemProduction_AS"]
    .astype(float)
    .values
)

wind = (
    wind_df["Power_AE"]
    .astype(float)
    .values
)

load = (
    load_df["Electricity:Facility [kW](Hourly)"]
    .astype(float)
    .values
)

price = (
    price_df["value"]
    .astype(float)
    .values
    / 1000.0
)

n_steps = min(
    len(solar),
    len(wind),
    len(load),
    len(price)
)

solar = solar[:n_steps]
wind = wind[:n_steps]
load = load[:n_steps]
price = price[:n_steps]

print(f"Timesteps: {n_steps}")

# ==================================================
# AGENTES
# ==================================================

solar_agent = StrategyQLearning(
    alpha=0.1,
    gamma=0.95,
    epsilon=0.30
)

wind_agent = StrategyQLearning(
    alpha=0.1,
    gamma=0.95,
    epsilon=0.30
)

# ==================================================
# ENTRENAMIENTO
# ==================================================

N_EPISODES = 50

solar_rewards_history = []
wind_rewards_history = []
total_rewards_history = []

solar_q_history = []
wind_q_history = []

for episode in range(N_EPISODES):

    solar_total_reward = 0
    wind_total_reward = 0

    for t in range(n_steps - 1):

        demand = load[t]

        current_price = price[t]

        solar_power = solar[t]

        wind_power = wind[t]

        # ==========================================
        # ESTADO
        # ==========================================

        solar_state = solar_agent.get_state(
            demand,
            current_price
        )

        wind_state = wind_agent.get_state(
            demand,
            current_price
        )

        # ==========================================
        # ACCIONES
        # ==========================================

        solar_action = solar_agent.choose_action(
            solar_state
        )

        wind_action = wind_agent.choose_action(
            wind_state
        )

        solar_strategy = (
            solar_agent.action_to_strategy(
                solar_action
            )
        )

        wind_strategy = (
            wind_agent.action_to_strategy(
                wind_action
            )
        )

        # ==========================================
        # ENERGÍA DECLARADA
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

        total_declared = (
            solar_declared
            + wind_declared
        )

        if total_declared > 0:

            solar_share = (
                solar_declared
                / total_declared
            )

            wind_share = (
                wind_declared
                / total_declared
            )

        else:

            solar_share = 0
            wind_share = 0

        solar_allocated = (
            demand
            * solar_share
        )

        wind_allocated = (
            demand
            * wind_share
        )

        # ==========================================
        # ENERGÍA ENTREGADA
        # ==========================================

        solar_delivered = min(
            solar_allocated,
            solar_power
        )

        wind_delivered = min(
            wind_allocated,
            wind_power
        )

        # ==========================================
        # SHORTFALL
        # ==========================================

        solar_shortfall = max(
            0,
            solar_allocated - solar_power
        )

        wind_shortfall = max(
            0,
            wind_allocated - wind_power
        )

        # ==========================================
        # REWARD COMPARTIDO
        # ==========================================

        renewable_used = (
            solar_delivered
            + wind_delivered
        )

        grid_energy = max(
            0,
            demand - renewable_used
        )

        total_shortfall = (
            solar_shortfall
            + wind_shortfall
        )

        shared_reward = (
            renewable_used
            - 2 * grid_energy
            - total_shortfall
        )

        solar_reward = shared_reward

        wind_reward = shared_reward

        solar_total_reward += solar_reward
        wind_total_reward += wind_reward

        # ==========================================
        # NEXT STATE
        # ==========================================

        next_solar_state = (
            solar_agent.get_state(
                load[t + 1],
                price[t + 1]
            )
        )

        next_wind_state = (
            wind_agent.get_state(
                load[t + 1],
                price[t + 1]
            )
        )

        # ==========================================
        # UPDATE
        # ==========================================

        solar_agent.update(
            solar_state,
            solar_action,
            solar_reward,
            next_solar_state
        )

        wind_agent.update(
            wind_state,
            wind_action,
            wind_reward,
            next_wind_state
        )

    solar_rewards_history.append(
        solar_total_reward
    )

    wind_rewards_history.append(
        wind_total_reward
    )

    total_rewards_history.append(
        solar_total_reward + wind_total_reward
    )

    solar_q_history.append(
        np.mean(
            solar_agent.q_table
        )
    )

    wind_q_history.append(
        np.mean(
            wind_agent.q_table
        )
    )

    solar_agent.epsilon = max(
        0.01,
        solar_agent.epsilon * 0.995
    )

    wind_agent.epsilon = max(
        0.01,
        wind_agent.epsilon * 0.995
    )

    print(
        f"Episode {episode+1}/{N_EPISODES}"
        f" | Solar={solar_total_reward:.1f}"
        f" | Wind={wind_total_reward:.1f}"
        f" | Total={solar_total_reward + wind_total_reward:.1f}"
    )

# ==================================================
# GUARDAR RESULTADOS
# ==================================================

results_df = pd.DataFrame({

    "episode": np.arange(
        1,
        N_EPISODES + 1
    ),

    "solar_reward": solar_rewards_history,

    "wind_reward": wind_rewards_history,

    "total_reward": total_rewards_history,

    "solar_q_mean": solar_q_history,

    "wind_q_mean": wind_q_history
})

results_path = (
    OUTPUT_DIR
    / "cooperative_training_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print(f"\nResultados guardados en:\n{results_path}")

# ==================================================
# Q TABLES
# ==================================================

print("\nSOLAR Q-TABLE\n")
print(solar_agent.q_table)

print("\nWIND Q-TABLE\n")
print(wind_agent.q_table)

# ==================================================
# POLÍTICAS
# ==================================================

print("\nPOLITICA SOLAR\n")

for demand in range(3):

    for price in range(3):

        action = np.argmax(
            solar_agent.q_table[
                demand,
                price
            ]
        )

        print(
            f"Demanda={demand}"
            f" Precio={price}"
            f" -> "
            f"{solar_agent.action_to_strategy(action)}"
        )

print("\nPOLITICA WIND\n")

for demand in range(3):

    for price in range(3):

        action = np.argmax(
            wind_agent.q_table[
                demand,
                price
            ]
        )

        print(
            f"Demanda={demand}"
            f" Precio={price}"
            f" -> "
            f"{wind_agent.action_to_strategy(action)}"
        )

# ==================================================
# GUARDAR Q-TABLES
# ==================================================

np.save(
    OUTPUT_DIR / "cooperative_solar_qtable.npy",
    solar_agent.q_table
)

np.save(
    OUTPUT_DIR / "cooperative_wind_qtable.npy",
    wind_agent.q_table
)

print(
    "\nQ-tables guardadas:"
)

print(
    OUTPUT_DIR / "cooperative_solar_qtable.npy"
)

print(
    OUTPUT_DIR / "cooperative_wind_qtable.npy"
)