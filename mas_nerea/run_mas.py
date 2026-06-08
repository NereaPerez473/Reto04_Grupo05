"""
run_mas.py
==========
Punto de entrada de la simulación MAS.

Lanza los servidores de AS y AE en threads daemon, espera a que estén listos
y ejecuta la simulación completa a través del AgenteConsumidor.

Uso básico
----------
    # Todos honestos (baseline)
    python run_mas.py

    # AS engaña, AE honesto
    python run_mas.py --strategy-as deception --strategy-ae honest

    # Ambos ocultan información
    python run_mas.py --strategy-as hide_information --strategy-ae hide_information --n-steps 200

Salida
------
    data/results/mas_results_as_{estrategia}_ae_{estrategia}_n{steps}.csv

Comparación de todas las estrategias
--------------------------------------
    python run_comparison.py
"""

import argparse
import threading
import time
import os
import pandas as pd

from solar_agent import SolarAgent
from wind_agent import WindAgent
from consumer_agent import ConsumerAgent
from pathlib import Path

# .parent.parent es la carpeta raíz 'Reto04_Grupo05'
BASE_DIR = Path(__file__).resolve().parent.parent

# Rutas por defecto
SOLAR_CSV = BASE_DIR/"data"/"results"/"Predicciones_Solar.csv"
WIND_CSV  = BASE_DIR /"data"/"results"/"Predicciones_Eolico.csv"
LOAD_CSV  = BASE_DIR/"data"/"raw"/"RefBldgFullServiceRestaurantNew2004_v1.3_7.1_6A_USA_MN_MINNEAPOLIS.csv"
PRICE_CSV = BASE_DIR/"data"/"raw"/"Precios"/"precio2025-peninsula.csv"
OUTPUT_DIR = BASE_DIR/"mas_nerea"/"results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_simulation(
    strategy_as: str = "honest",
    strategy_ae: str = "honest",
    n_steps: int = 500,
    solar_csv: str = str(SOLAR_CSV),
    wind_csv: str  = str(WIND_CSV),
    load_csv: str  = str(LOAD_CSV),
    price_csv: str = str(PRICE_CSV),
    output_dir: str = str(OUTPUT_DIR),
    save_csv: bool = True,
) -> pd.DataFrame:
    """
    Lanza los 3 agentes y ejecuta la simulación completa.

    Parámetros
    ----------
    strategy_as  : Estrategia del Agente Solar.
    strategy_ae  : Estrategia del Agente Eólico.
    n_steps      : Número de timesteps.
    save_csv     : Si True, guarda el DataFrame en output_dir.

    Returns
    -------
    pd.DataFrame con una fila por timestep.
    """
    print(f"\n{'='*60}")
    print(f"  MAS Microred — AS: {strategy_as} | AE: {strategy_ae} | {n_steps} steps")
    print(f"{'='*60}\n")

    # Instanciar agentes
    solar    = SolarAgent(csv_path=solar_csv, strategy_name=strategy_as)
    wind     = WindAgent(csv_path=wind_csv,   strategy_name=strategy_ae)
    consumer = ConsumerAgent(load_csv_path=load_csv, price_csv_path=price_csv,
                             n_steps=n_steps)

    # Arrancar servidores en threads daemon (se detienen al terminar el main)
    t_solar = threading.Thread(target=solar.start, daemon=True, name="Thread-Solar")
    t_wind  = threading.Thread(target=wind.start,  daemon=True, name="Thread-Wind")
    t_solar.start()
    t_wind.start()

    # Esperar a que los servidores estén escuchando
    time.sleep(1.0)

    # Ejecutar simulación (bloquea hasta que termina)
    df_results = consumer.run()

    # Añadir columnas de metadatos para facilitar la comparación
    df_results["strategy_as"] = strategy_as
    df_results["strategy_ae"] = strategy_ae

    # Guardar resultados
    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"mas_results_as_{strategy_as}_ae_{strategy_ae}_n{n_steps}.csv"
        path = os.path.join(output_dir, filename)
        df_results.to_csv(path, index=False)
        print(f"\n[run_mas] Resultados guardados → {path}")

    # Resumen de agentes productores (ingresos)
    print(f"\n[run_mas] Ingresos productores:")
    print(f"  AgenteSolar  ({strategy_as}): {solar.total_revenue_eur:.4f} EUR")
    print(f"  AgenteEolico ({strategy_ae}): {wind.total_revenue_eur:.4f} EUR")

    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulación MAS microred energética con negociación FIPA-ACL"
    )
    parser.add_argument(
        "--strategy-as", default="honest",
        choices=["honest", "deception", "hide_information"],
        help="Estrategia del Agente Solar (default: honest)"
    )
    parser.add_argument(
        "--strategy-ae", default="honest",
        choices=["honest", "deception", "hide_information"],
        help="Estrategia del Agente Eólico (default: honest)"
    )
    parser.add_argument(
        "--n-steps", type=int, default=500,
        help="Número de timesteps a simular (default: 500)"
    )
    parser.add_argument(
        "--solar-csv", default=SOLAR_CSV
    )
    parser.add_argument(
        "--wind-csv", default=WIND_CSV
    )
    parser.add_argument(
        "--load-csv", default=LOAD_CSV
    )
    parser.add_argument(
        "--price-csv", default=PRICE_CSV
    )
    args = parser.parse_args()

    run_simulation(
        strategy_as=args.strategy_as,
        strategy_ae=args.strategy_ae,
        n_steps=args.n_steps,
        solar_csv=args.solar_csv,
        wind_csv=args.wind_csv,
        load_csv=args.load_csv,
        price_csv=args.price_csv,
    )
