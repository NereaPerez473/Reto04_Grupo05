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

        # demanda(3) x precio(3) x acción(3)
        self.q_table = np.zeros((3, 3, 3))

    def get_state(
        self,
        demand_kw,
        price_eur_kwh
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

        return (
            demand_state,
            price_state
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