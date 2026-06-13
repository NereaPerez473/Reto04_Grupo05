"""
strategy_qlearning_negotiation_battery.py
==========================================
Agente Q-Learning para AS/AE en el modo NEGOCIACIÓN con batería pasiva.

Diferencia respecto a strategy_qlearning_negotiation.py (sin batería)
-----------------------------------------------------------------------
El estado incluye el SoC de la batería como cuarta dimensión, además
de la producción propia que ya tenía la versión sin batería.

Estado: demanda [0..2] × precio [0..2] × producción_propia [0..2]
        × soc_batería [0..2] = 81 estados
Acción: 0=honest · 1=hide_information · 2=deception
Q-table: shape (3, 3, 3, 3, 3)
"""

import numpy as np


class StrategyQLearning:

    ACTIONS = [
        "honest",
        "hide_information",
        "deception"
    ]

    def __init__(
        self,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.3
    ):
        self.alpha   = alpha
        self.gamma   = gamma
        self.epsilon = epsilon

        # demanda(3) x precio(3) x produccion(3) x soc_bateria(3) x acción(3)
        self.q_table = np.zeros((3, 3, 3, 3, 3))

    def discretize_production(self, production: float) -> int:
        if production < 10:
            return 0
        elif production < 30:
            return 1
        else:
            return 2

    def get_state(
        self,
        demand_kw: float,
        price_eur_kwh: float,
        own_power: float,
        battery_soc: float
    ) -> tuple:
        """
        Parámetros
        ----------
        demand_kw     : demanda del consumidor [kW]
        price_eur_kwh : precio de red [€/kWh]
        own_power     : producción real propia (solar o eólica) [kW]
        battery_soc   : SoC actual de la batería [0-1]
        """

        if demand_kw < 30:
            demand_state = 0
        elif demand_kw < 45:
            demand_state = 1
        else:
            demand_state = 2

        if price_eur_kwh < 0.10:
            price_state = 0
        elif price_eur_kwh < 0.20:
            price_state = 1
        else:
            price_state = 2

        production_state = self.discretize_production(own_power)

        if battery_soc < 0.30:
            soc_state = 0
        elif battery_soc < 0.70:
            soc_state = 1
        else:
            soc_state = 2

        return (demand_state, price_state, production_state, soc_state)

    def choose_action(self, state: tuple) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 3)
        return int(np.argmax(self.q_table[state]))

    def action_to_strategy(self, action: int) -> str:
        return self.ACTIONS[action]

    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple
    ) -> None:
        current_q    = self.q_table[state][action]
        max_future_q = np.max(self.q_table[next_state])
        new_q = current_q + self.alpha * (
            reward + self.gamma * max_future_q - current_q
        )
        self.q_table[state][action] = new_q
