"""
sma_trainer.py
==============
Generador SSE para entrenamiento en vivo de los tres modos:
  competitive / cooperative / negotiation

Emite un evento por episodio con métricas acumuladas de aprendizaje,
permitiendo visualizar en tiempo real cómo evolucionan los agentes.

Tipos de evento emitidos:
  - "train_init"    : metadatos iniciales (n_episodes, n_steps, mode)
  - "train_episode" : al final de cada episodio con todas las métricas
  - "train_done"    : al terminar, con las Q-tables finales guardadas
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path
from collections import Counter


# ==================================================
# GENERADOR PRINCIPAL
# ==================================================

def run_training_streaming(
    mode: str = "competitive",
    n_episodes: int = 200,
    save_qtables: bool = True
):
    """
    Generador que entrena los agentes y emite un evento SSE por episodio.

    Parámetros
    ----------
    mode        : "competitive" | "cooperative" | "negotiation"
    n_episodes  : número de episodios de entrenamiento
    save_qtables: si True guarda las Q-tables en /project/... al terminar
    """

    BASE_DIR = Path("/project")

    SOLAR_CSV = BASE_DIR / "data" / "results" / "Predicciones_Solar.csv"
    WIND_CSV  = BASE_DIR / "data" / "results" / "Predicciones_Eolico.csv"
    LOAD_CSV  = BASE_DIR / "data" / "raw" / "demanda_restaurante.csv"
    PRICE_CSV = BASE_DIR / "data" / "raw" / "Precios" / "precio2025-peninsula.csv"

    OUTPUT_DIR = BASE_DIR / "mas_qlearning_battery" / "results"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Importaciones dinámicas según el modo ──────────────────────────
    from mas_qlearning_battery.simple_battery import SimpleBattery
    from mas_qlearning_battery.strategies import NegotiationStrategies

    if mode == "negotiation":
        from mas_qlearning_battery.strategy_qlearning_negotiation_battery import (
            StrategyQLearning
        )
    else:
        from mas_qlearning_battery.strategy_qlearning_battery import (
            StrategyQLearning
        )

    # ── Carga de datos ─────────────────────────────────────────────────
    t0 = time.perf_counter()

    solar = pd.read_csv(SOLAR_CSV)["SystemProduction_AS"].astype(float).values
    wind  = pd.read_csv(WIND_CSV)["Power_AE"].astype(float).values
    load  = (pd.read_csv(LOAD_CSV)["Electricity:Facility [kW](Hourly)"]
               .astype(float).values * 2.5)
    price = (pd.read_csv(PRICE_CSV, sep=";")["value"]
               .astype(float).values / 1000.0)

    n_steps = min(len(solar), len(wind), len(load), len(price))
    solar = solar[:n_steps]
    wind  = wind[:n_steps]
    load  = load[:n_steps]
    price = price[:n_steps]

    t_load = round(time.perf_counter() - t0, 3)

    # ── Agentes y batería ──────────────────────────────────────────────
    solar_agent = StrategyQLearning(alpha=0.1, gamma=0.95, epsilon=1.0)
    wind_agent  = StrategyQLearning(alpha=0.1, gamma=0.95, epsilon=1.0)

    battery = SimpleBattery(
        capacity_kwh=200.0, initial_soc=0.5,
        charge_eff=0.95, discharge_eff=0.95,
        max_power_kw=50.0, soc_min=0.05
    )

    MARKET_BONUS_FACTOR = 0.1  # solo usado en modo negotiation

    # ── Evento inicial ─────────────────────────────────────────────────
    yield {
        "type":       "train_init",
        "mode":       mode,
        "n_episodes": n_episodes,
        "n_steps":    n_steps,
        "t_load_s":   t_load
    }

    # ── Histórico de recompensas para ventana móvil ────────────────────
    reward_window = []  # últimos 20 episodios para media móvil
    WINDOW = 20

    train_start = time.perf_counter()

    # ==================================================
    # BUCLE DE ENTRENAMIENTO
    # ==================================================
    for episode in range(1, n_episodes + 1):

        ep_start = time.perf_counter()
        battery.reset()

        solar_total_reward = 0.0
        wind_total_reward  = 0.0
        episode_soc        = []
        episode_grid_kwh   = 0.0

        solar_counter = Counter()
        wind_counter  = Counter()

        for t in range(n_steps - 1):

            demand        = load[t]
            current_price = price[t]
            solar_power   = solar[t]
            wind_power    = wind[t]

            # ── Batería: descarga antes de la negociación ──────────────
            battery_contribution = battery.discharge(demand)
            effective_demand     = demand - battery_contribution
            episode_soc.append(battery.soc)

            # ── Estado ────────────────────────────────────────────────
            if mode == "negotiation":
                solar_state = solar_agent.get_state(
                    effective_demand, current_price, solar_power, battery.soc
                )
                wind_state = wind_agent.get_state(
                    effective_demand, current_price, wind_power, battery.soc
                )
            else:
                solar_state = solar_agent.get_state(
                    effective_demand, current_price, battery.soc
                )
                wind_state = wind_agent.get_state(
                    effective_demand, current_price, battery.soc
                )

            # ── Acciones ──────────────────────────────────────────────
            solar_action   = solar_agent.choose_action(solar_state)
            wind_action    = wind_agent.choose_action(wind_state)
            solar_strategy = solar_agent.action_to_strategy(solar_action)
            wind_strategy  = wind_agent.action_to_strategy(wind_action)

            solar_counter[solar_strategy] += 1
            wind_counter[wind_strategy]   += 1

            # ── Producción declarada ───────────────────────────────────
            FACTORS = {"honest": 1.0, "hide_information": 0.7, "deception": 1.3}
            solar_declared = solar_power * FACTORS[solar_strategy]
            wind_declared  = wind_power  * FACTORS[wind_strategy]

            # ── Propuestas de precio ───────────────────────────────────
            solar_proposal = NegotiationStrategies.apply(
                solar_strategy, solar_power, current_price
            )
            wind_proposal = NegotiationStrategies.apply(
                wind_strategy, wind_power, current_price
            )

            # ── Merit-order dispatch ───────────────────────────────────
            proposals = [
                ("solar", solar_declared, solar_proposal.price_eur_kwh),
                ("wind",  wind_declared,  wind_proposal.price_eur_kwh)
            ]
            viable = sorted(
                [p for p in proposals if p[2] < current_price],
                key=lambda x: x[2]
            )

            solar_allocated = wind_allocated = 0.0
            remaining = effective_demand
            for source, declared_kw, _ in viable:
                if remaining <= 0:
                    break
                purchase = min(remaining, declared_kw)
                if source == "solar":
                    solar_allocated = purchase
                else:
                    wind_allocated = purchase
                remaining = max(0.0, remaining - purchase)

            # ── Energía entregada y excedentes ────────────────────────
            solar_delivered = min(solar_allocated, solar_power)
            wind_delivered  = min(wind_allocated,  wind_power)
            renewable_delivered = solar_delivered + wind_delivered

            grid_purchased = max(0.0, effective_demand - renewable_delivered)

            physical_surplus = max(
                0.0,
                solar_power + wind_power - renewable_delivered
            )
            if physical_surplus > 0:
                battery.charge(physical_surplus)

            episode_grid_kwh += grid_purchased

            # ── Rewards ───────────────────────────────────────────────
            solar_revenue = solar_delivered * solar_proposal.price_eur_kwh
            wind_revenue  = wind_delivered  * wind_proposal.price_eur_kwh

            solar_excess = max(0.0, solar_declared - solar_power)
            wind_excess  = max(0.0, wind_declared  - wind_power)

            dpf = 3.0 - 2.0 * battery.soc  # dynamic penalty factor

            if mode == "competitive":
                solar_reward = solar_revenue - dpf * solar_excess * current_price
                wind_reward  = wind_revenue  - dpf * wind_excess  * current_price

            elif mode == "cooperative":
                battery_savings = battery_contribution * current_price
                total_excess    = solar_excess + wind_excess
                shared = (
                    (solar_revenue + wind_revenue + battery_savings)
                    - 2.0 * grid_purchased * current_price
                    - dpf * total_excess * current_price
                )
                solar_reward = wind_reward = shared

            else:  # negotiation
                mb_solar = MARKET_BONUS_FACTOR * solar_allocated * current_price
                mb_wind  = MARKET_BONUS_FACTOR * wind_allocated  * current_price
                solar_reward = solar_revenue + mb_solar - dpf * solar_excess * current_price
                wind_reward  = wind_revenue  + mb_wind  - dpf * wind_excess  * current_price

            solar_total_reward += solar_reward
            wind_total_reward  += wind_reward

            # ── Next-state y actualización Q ──────────────────────────
            next_d   = load[t + 1]
            next_nbc = min(next_d, battery.available_discharge_kw())
            next_eff = next_d - next_nbc

            if mode == "negotiation":
                next_solar_state = solar_agent.get_state(
                    next_eff, price[t + 1], solar[t + 1], battery.soc
                )
                next_wind_state = wind_agent.get_state(
                    next_eff, price[t + 1], wind[t + 1], battery.soc
                )
            else:
                next_solar_state = solar_agent.get_state(
                    next_eff, price[t + 1], battery.soc
                )
                next_wind_state = wind_agent.get_state(
                    next_eff, price[t + 1], battery.soc
                )

            solar_agent.update(solar_state, solar_action, solar_reward, next_solar_state)
            wind_agent.update(wind_state,   wind_action,  wind_reward,  next_wind_state)

        # ── Epsilon decay ──────────────────────────────────────────────
        solar_agent.epsilon = max(0.01, solar_agent.epsilon * 0.999)
        wind_agent.epsilon  = max(0.01, wind_agent.epsilon  * 0.999)

        # ── Métricas del episodio ──────────────────────────────────────
        total_reward   = solar_total_reward + wind_total_reward
        avg_soc        = float(np.mean(episode_soc)) if episode_soc else 0.5
        ep_elapsed     = round(time.perf_counter() - ep_start, 3)
        total_elapsed  = round(time.perf_counter() - train_start, 1)

        reward_window.append(total_reward)
        if len(reward_window) > WINDOW:
            reward_window.pop(0)
        rolling_avg = round(float(np.mean(reward_window)), 2)

        # Estrategia dominante del episodio
        solar_dom = max(solar_counter, key=solar_counter.get) if solar_counter else "honest"
        wind_dom  = max(wind_counter,  key=wind_counter.get)  if wind_counter  else "honest"

        # ── Evento SSE ────────────────────────────────────────────────
        yield {
            "type":        "train_episode",
            "episode":     episode,
            "n_episodes":  n_episodes,
            "progress":    round(100 * episode / n_episodes, 1),

            # Recompensas
            "solar_reward":  round(solar_total_reward, 2),
            "wind_reward":   round(wind_total_reward,  2),
            "total_reward":  round(total_reward, 2),
            "rolling_avg":   rolling_avg,

            # Exploración
            "epsilon":       round(solar_agent.epsilon, 4),

            # Energía
            "grid_kwh":      round(episode_grid_kwh, 1),
            "avg_soc":       round(avg_soc * 100, 1),
            "q_mean_solar":  round(float(np.mean(solar_agent.q_table)), 4),
            "q_mean_wind":   round(float(np.mean(wind_agent.q_table)),  4),

            # Distribución de estrategias (porcentajes)
            "solar_counts": {
                k: round(100 * v / max(1, sum(solar_counter.values())), 1)
                for k, v in solar_counter.items()
            },
            "wind_counts": {
                k: round(100 * v / max(1, sum(wind_counter.values())), 1)
                for k, v in wind_counter.items()
            },

            # Estrategia dominante
            "solar_dominant": solar_dom,
            "wind_dominant":  wind_dom,

            # Tiempos
            "ep_elapsed_s":    ep_elapsed,
            "total_elapsed_s": total_elapsed,
        }

    # ==================================================
    # GUARDAR Q-TABLES
    # ==================================================
    if save_qtables:
        np.save(
            OUTPUT_DIR / f"{mode}_battery_solar_qtable.npy",
            solar_agent.q_table
        )
        np.save(
            OUTPUT_DIR / f"{mode}_battery_wind_qtable.npy",
            wind_agent.q_table
        )

    yield {
        "type":    "train_done",
        "mode":    mode,
        "saved":   save_qtables,
        "q_solar_mean": round(float(np.mean(solar_agent.q_table)), 4),
        "q_wind_mean":  round(float(np.mean(wind_agent.q_table)),  4),
        "total_elapsed_s": round(time.perf_counter() - train_start, 1)
    }
