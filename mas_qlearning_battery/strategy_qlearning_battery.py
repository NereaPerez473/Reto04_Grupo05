"""
strategy_qlearning_battery.py
==============================
Agente Q-Learning para AS/AE en los modos COMPETITIVO y COOPERATIVO
con batería pasiva.

Diferencia respecto a strategy_qlearning.py (sin batería)
-----------------------------------------------------------
El estado incluye el SoC discretizado de la batería como tercera
dimensión. Esto permite que AS y AE aprendan a adaptar su estrategia
según la reserva energética disponible en el sistema:

    SoC bajo  (0) → AC depende más de renovables y red → oportunidad para
                    precios más altos o estrategias de ocultación.
    SoC medio (1) → situación normal.
    SoC alto  (2) → AC tiene energía almacenada → menor dependencia de AS/AE
                    → los productores deben ofrecer mejores condiciones.

Estado:  demanda [0..2] × precio [0..2] × soc_batería [0..2] = 27 estados
Acción:  0=honest · 1=hide_information · 2=deception
Q-table: shape (3, 3, 3, 3)
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

        # demanda(3) x precio(3) x soc_bateria(3) x acción(3)
        self.q_table = np.zeros((3, 3, 3, 3))

    def get_state(
        self,
        demand_kw: float,
        price_eur_kwh: float,
        battery_soc: float
    ) -> tuple:
        """
        Discretiza el estado del mercado incluyendo el SoC de la batería.

        Parámetros
        ----------
        demand_kw     : demanda del consumidor en este timestep [kW]
        price_eur_kwh : precio de importación de red [€/kWh]
        battery_soc   : SoC actual de la batería [0-1]
        """

        # Demanda
        if demand_kw < 30:
            demand_state = 0
        elif demand_kw < 45:
            demand_state = 1
        else:
            demand_state = 2

        # Precio
        if price_eur_kwh < 0.10:
            price_state = 0
        elif price_eur_kwh < 0.20:
            price_state = 1
        else:
            price_state = 2

        # SoC batería: 3 niveles
        if battery_soc < 0.30:
            soc_state = 0   # batería casi vacía → AC dependiente
        elif battery_soc < 0.70:
            soc_state = 1   # batería media
        else:
            soc_state = 2   # batería llena → AC menos dependiente

        return (demand_state, price_state, soc_state)

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
