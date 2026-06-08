"""
solar_agent.py
==============
Agente Solar (AS) — servidor FIPA-ACL en el puerto 5001.

Hereda toda la lógica de negociación de ProducerAgent.
Solo configura nombre, puerto y columna del CSV específicos del panel fotovoltaico.

CSV esperado (data/results/Predicciones_Solar.csv):
    Date, SystemProduction_AS
    2017-01-02T00:00:00.000000, 1.348889
    ...

Uso
---
    # Desde código:
    from solar_agent import SolarAgent
    agent = SolarAgent(strategy_name="deception")
    agent.start()  # bloquea — lanzar en thread daemon

    # Desde terminal:
    python solar_agent.py --strategy honest
    python solar_agent.py --strategy deception --bluff-factor 1.4
"""

import argparse
from base_agent import ProducerAgent
from pathlib import Path
import numpy as np

PORT_SOLAR = 5001
# .parent.parent es la carpeta raíz 'Reto04_Grupo05'
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CSV = BASE_DIR/"data"/"results"/"Predicciones_Solar.csv"


class SolarAgent(ProducerAgent):
    """
    Agente Solar (AS).

    Parámetros
    ----------
    csv_path      : str    Ruta al CSV de predicciones solares.
    strategy_name : str    "honest" | "deception" | "hide_information".
    bluff_factor  : float  Solo para strategy="deception". Default 1.3.
    hide_factor   : float  Solo para strategy="hide_information". Default 0.70.
    """

    def __init__(self, csv_path: str = str(DEFAULT_CSV),
                 strategy_name: str = "honest",
                 bluff_factor: float = 1.3,
                 hide_factor: float = 0.70):
        super().__init__(
            name="AgenteSolar",
            port=PORT_SOLAR,
            csv_path=csv_path,
            power_column="SystemProduction_AS",
            strategy_name=strategy_name,
            bluff_factor=bluff_factor,
            hide_factor=hide_factor,
        )

        qtable_path = (
            BASE_DIR
            / "mas_qlearning"
            / "results"
            / "negotiation_solar_qtable.npy"
        )

        self.learner.q_table = np.load(
            qtable_path
        )

        self.learner.epsilon = 0.0

        print(
            f"[AgenteSolar] Q-table cargada desde:\n{qtable_path}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agente Solar FIPA-ACL")
    parser.add_argument("--csv", default=DEFAULT_CSV,
                        help="Ruta al CSV de predicciones solares")
    parser.add_argument("--strategy", default="honest",
                        choices=["honest", "deception", "hide_information"])
    parser.add_argument("--bluff-factor", type=float, default=1.3)
    parser.add_argument("--hide-factor", type=float, default=0.70)
    args = parser.parse_args()

    agent = SolarAgent(
        csv_path=args.csv,
        strategy_name=args.strategy,
        bluff_factor=args.bluff_factor,
        hide_factor=args.hide_factor,
    )
    agent.start()
