import numpy as np
import pandas as pd
import time

from pathlib import Path

from mas_qlearning_battery.simple_battery import SimpleBattery
from mas_qlearning_battery.strategy_qlearning_battery import StrategyQLearning
from mas_qlearning_battery.strategies import NegotiationStrategies

from mas_qlearning_battery.strategy_qlearning_battery import (
    StrategyQLearning as StandardQLearning
)

from mas_qlearning_battery.strategy_qlearning_negotiation_battery import (
    StrategyQLearning as NegotiationQLearning
)


# ==================================================
# EVALUACIÓN ESTÁNDAR (sin streaming)
# ==================================================

def run_evaluation(
    mode="competitive"
):
    """Evaluación completa, devuelve dict con KPIs."""

    # Consume el generador y devuelve sólo el resultado final
    result = None

    for event in run_evaluation_streaming(mode):
        if event.get("type") == "result":
            result = event["data"]

    return result


# ==================================================
# EVALUACIÓN CON STREAMING (generador SSE)
# ==================================================

def run_evaluation_streaming(
    mode="competitive"
):
    """
    Generador que emite un dict por cada paso relevante
    de la simulación para enviarlos como Server-Sent Events.

    Tipos de evento emitidos:
      - "init"       : inicio, con n_steps total
      - "step"       : cada paso t de la simulación
      - "summary"    : resumen parcial cada N pasos
      - "result"     : KPIs finales al terminar
    """

    BASE_DIR = Path("/project")

    # ==================================================
    # DATOS
    # ==================================================

    SOLAR_CSV = (
        BASE_DIR / "data" / "results"
        / "Predicciones_Solar.csv"
    )

    WIND_CSV = (
        BASE_DIR / "data" / "results"
        / "Predicciones_Eolico.csv"
    )

    LOAD_CSV = (
        BASE_DIR / "data" / "raw"
        / "demanda_restaurante.csv"
    )

    PRICE_CSV = (
        BASE_DIR / "data" / "raw" / "Precios"
        / "precio2025-peninsula.csv"
    )

    # ==================================================
    # QTABLES
    # ==================================================

    QTABLE_SOLAR = (
        BASE_DIR / "mas_qlearning_battery" / "results"
        / f"{mode}_battery_solar_qtable.npy"
    )

    QTABLE_WIND = (
        BASE_DIR / "mas_qlearning_battery" / "results"
        / f"{mode}_battery_wind_qtable.npy"
    )

    # ==================================================
    # CARGA
    # ==================================================

    t_load = time.perf_counter()

    solar = (
        pd.read_csv(SOLAR_CSV)["SystemProduction_AS"]
        .astype(float).values
    )

    wind = (
        pd.read_csv(WIND_CSV)["Power_AE"]
        .astype(float).values
    )

    load = (
        pd.read_csv(LOAD_CSV)
        ["Electricity:Facility [kW](Hourly)"]
        .astype(float).values * 2.5
    )

    price = (
        pd.read_csv(PRICE_CSV, sep=";")["value"]
        .astype(float).values / 1000.0
    )

    n_steps = min(
        len(solar), len(wind), len(load), len(price)
    )

    solar  = solar[:n_steps]
    wind   = wind[:n_steps]
    load   = load[:n_steps]
    price  = price[:n_steps]

    t_load_elapsed = round(time.perf_counter() - t_load, 4)

    # ==================================================
    # AGENTES
    # ==================================================

    if mode == "negotiation":
        solar_agent = NegotiationQLearning(epsilon=0.3)
        wind_agent  = NegotiationQLearning(epsilon=0.3)
    else:
        solar_agent = StandardQLearning(epsilon=0.3)
        wind_agent  = StandardQLearning(epsilon=0.3)

    solar_agent.q_table = np.load(QTABLE_SOLAR)
    wind_agent.q_table  = np.load(QTABLE_WIND)

    # ==================================================
    # BATERÍA
    # ==================================================

    battery = SimpleBattery(capacity_kwh=200.0)

    # ==================================================
    # MÉTRICAS ACUMULADAS
    # ==================================================

    total_grid_energy      = 0.0
    total_renewable_energy = 0.0
    battery_soc_values     = []

    from collections import Counter
    solar_counter = Counter()
    wind_counter  = Counter()

    consumer_cost  = 0.0
    solar_revenue  = 0.0
    wind_revenue   = 0.0

    # Emit evento de inicio
    yield {
        "type": "init",
        "n_steps": n_steps,
        "mode": mode,
        "t_load_data": t_load_elapsed
    }

    # Cada cuántos pasos emitir un evento "step"
    # (para no saturar el cliente con 8760 eventos)
    EMIT_EVERY = max(1, n_steps // 200)

    sim_start = time.perf_counter()

    # ==================================================
    # SIMULACIÓN
    # ==================================================

    for t in range(n_steps - 1):

        d = load[t]
        p = price[t]
        s = solar[t]
        w = wind[t]

        battery_contribution = battery.discharge(d)
        effective_demand = d - battery_contribution

        if mode == "negotiation":

            state_solar = solar_agent.get_state(
                effective_demand, p, s, battery.soc
            )

            state_wind = wind_agent.get_state(
                effective_demand, p, w, battery.soc
            )

        else:

            state_solar = solar_agent.get_state(
                effective_demand, p, battery.soc
            )

            state_wind = wind_agent.get_state(
                effective_demand, p, battery.soc
            )

        solar_action = solar_agent.choose_action(state_solar)
        wind_action  = wind_agent.choose_action(state_wind)

        solar_strategy = solar_agent.action_to_strategy(solar_action)
        wind_strategy  = wind_agent.action_to_strategy(wind_action)

        solar_counter[solar_strategy] += 1
        wind_counter[wind_strategy]   += 1

        solar_offer = NegotiationStrategies.apply(
            solar_strategy, s, p
        )

        wind_offer = NegotiationStrategies.apply(
            wind_strategy, w, p
        )

        renewable_energy = (
            solar_offer.real_energy_kw
            + wind_offer.real_energy_kw
        )

        renewable_used = min(renewable_energy, effective_demand)

        grid_energy = max(
            0.0,
            effective_demand - renewable_used
        )

        solar_used = min(
            solar_offer.real_energy_kw, renewable_used
        )

        wind_used = min(
            wind_offer.real_energy_kw,
            max(0.0, renewable_used - solar_used)
        )

        consumer_cost  += solar_used * solar_offer.price_eur_kwh
        consumer_cost  += wind_used  * wind_offer.price_eur_kwh
        consumer_cost  += grid_energy * p

        solar_revenue  += solar_used * solar_offer.price_eur_kwh
        wind_revenue   += wind_used  * wind_offer.price_eur_kwh

        total_grid_energy      += grid_energy
        total_renewable_energy += renewable_used

        surplus = max(0.0, renewable_energy - renewable_used)

        if surplus > 0:
            battery.charge(surplus)

        battery_soc_values.append(battery.soc * 100)

        # Emitir evento de paso periódicamente
        if t % EMIT_EVERY == 0:

            elapsed_sim = round(
                time.perf_counter() - sim_start, 4
            )

            yield {
                "type": "step",
                "t": t,
                "n_steps": n_steps,
                "progress": round(100 * t / (n_steps - 1), 1),

                # Estado actual del agente solar
                "solar_strategy": solar_strategy,
                "solar_energy_kw": round(
                    solar_offer.real_energy_kw, 3
                ),
                "solar_price": round(
                    solar_offer.price_eur_kwh, 4
                ),

                # Estado actual del agente eólico
                "wind_strategy": wind_strategy,
                "wind_energy_kw": round(
                    wind_offer.real_energy_kw, 3
                ),
                "wind_price": round(
                    wind_offer.price_eur_kwh, 4
                ),

                # Métricas acumuladas hasta ahora
                "battery_soc": round(battery.soc * 100, 2),
                "grid_energy_acc": round(total_grid_energy, 2),
                "renewable_acc": round(total_renewable_energy, 2),
                "consumer_cost_acc": round(consumer_cost, 2),

                # Tiempo de simulación transcurrido
                "elapsed_sim_s": elapsed_sim,

                # Demanda y precio actuales
                "demand_kw": round(d, 3),
                "price_eur_kwh": round(p, 4)
            }

    # ==================================================
    # KPI FINALES
    # ==================================================

    t_sim_elapsed = round(time.perf_counter() - sim_start, 4)

    avg_soc = float(np.mean(battery_soc_values))

    renewable_share = (
        100 * total_renewable_energy
        / (total_renewable_energy + total_grid_energy)
    )

    solar_dominant = max(solar_counter, key=solar_counter.get)
    wind_dominant  = max(wind_counter,  key=wind_counter.get)

    result = {
        "grid_energy":       round(total_grid_energy, 2),
        "battery_soc":       round(avg_soc, 2),
        "renewable_energy":  round(total_renewable_energy, 2),
        "renewable_share":   round(renewable_share, 2),
        "consumer_cost":     round(consumer_cost, 2),
        "solar_revenue":     round(solar_revenue, 2),
        "wind_revenue":      round(wind_revenue, 2),
        "solar_strategy":    solar_dominant,
        "wind_strategy":     wind_dominant,
        "mode":              mode,
        "t_simulate":        t_sim_elapsed,
        "t_load_data":       t_load_elapsed,
        # Distribución de estrategias al final
        "solar_counts": dict(solar_counter),
        "wind_counts":  dict(wind_counter)
    }

    yield {
        "type": "result",
        "data": result
    }