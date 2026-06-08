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

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # demanda x precio x solar x wind x acción

        self.q_table = np.zeros(
            (
                3,
                3,
                3,
                3
            )
        )

    def discretize_production(
        self,
        production
    ):

        if production < 10:

            return 0

        elif production < 30:

            return 1

        else:

            return 2

    def get_state(
        self,
        demand_kw,
        price_eur_kwh,
        own_power
    ):

        # --------------------------
        # Demanda
        # --------------------------

        if demand_kw < 30:

            demand_state = 0

        elif demand_kw < 45:

            demand_state = 1

        else:

            demand_state = 2

        # --------------------------
        # Precio
        # --------------------------

        if price_eur_kwh < 0.10:

            price_state = 0

        elif price_eur_kwh < 0.20:

            price_state = 1

        else:

            price_state = 2

        production_state = self.discretize_production(
            own_power
        )
        

        return (
            demand_state,
            price_state,
            production_state
        )

    def choose_action(
        self,
        state
    ):

        if np.random.rand() < self.epsilon:

            return np.random.randint(0, 3)

        return np.argmax(
            self.q_table[state]
        )

    def action_to_strategy(
        self,
        action
    ):

        return self.ACTIONS[action]

    def update(
        self,
        state,
        action,
        reward,
        next_state
    ):

        current_q = self.q_table[state][action]

        max_future_q = np.max(
            self.q_table[next_state]
        )

        new_q = current_q + self.alpha * (
            reward
            + self.gamma * max_future_q
            - current_q
        )

        self.q_table[state][action] = new_q