"""
wind_agent.py
=============
Agente Eólico (AE) — servidor FIPA-ACL en el puerto 5002.

Hereda toda la lógica de negociación de ProducerAgent.
Solo configura nombre, puerto y columna del CSV específicos del aerogenerador.

CSV esperado (data/results/Predicciones_Eolico.csv):
    Date, Power_AE
    2017-01-02T00:00:00.000000, 15.103733
    ...

Uso
---
    # Desde código:
    from wind_agent import WindAgent
    agent = WindAgent(strategy_name="hide_information")
    agent.start()  # bloquear — lanzar en thread daemon

    # Desde terminal:
    python wind_agent.py --strategy hide_information --hide-factor 0.65
"""

import argparse
from base_agent import ProducerAgent
from pathlib import Path
import numpy as np

PORT_WIND = 5002
# .parent.parent es la carpeta raíz 'Reto04_Grupo05'
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CSV = BASE_DIR/"data"/"results"/"Predicciones_Eolico.csv"


class WindAgent(ProducerAgent):
    """
    Agente Eólico (AE).

    Parámetros
    ----------
    csv_path      : str    Ruta al CSV de predicciones eólicas.
    strategy_name : str    "honest" | "deception" | "hide_information".
    bluff_factor  : float  Solo para strategy="deception". Default 1.3.
    hide_factor   : float  Solo para strategy="hide_information". Default 0.70.
    """

    def __init__(self, csv_path: str = str(DEFAULT_CSV),
                 strategy_name: str = "honest",
                 bluff_factor: float = 1.3,
                 hide_factor: float = 0.70):
        super().__init__(
            name="AgenteEolico",
            port=PORT_WIND,
            csv_path=csv_path,
            power_column="Power_AE",
            strategy_name=strategy_name,
            bluff_factor=bluff_factor,
            hide_factor=hide_factor,
        )

        qtable_path = (
            BASE_DIR
            / "mas_qlearning"
            / "results"
            / "negotiation_wind_qtable.npy"
        )

        self.learner.q_table = np.load(
            qtable_path
        )

        self.learner.epsilon = 0.0

        print(
            f"[AgenteEolico] Q-table cargada desde:\n{qtable_path}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agente Eólico FIPA-ACL")
    parser.add_argument("--csv", default=DEFAULT_CSV,
                        help="Ruta al CSV de predicciones eólicas")
    parser.add_argument("--strategy", default="honest",
                        choices=["honest", "deception", "hide_information"])
    parser.add_argument("--bluff-factor", type=float, default=1.3)
    parser.add_argument("--hide-factor", type=float, default=0.70)
    args = parser.parse_args()

    agent = WindAgent(
        csv_path=args.csv,
        strategy_name=args.strategy,
        bluff_factor=args.bluff_factor,
        hide_factor=args.hide_factor,
    )
    agent.start()
