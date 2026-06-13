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
from simple_battery import SimpleBattery
from pathlib import Path

# .parent.parent es la carpeta raíz 'Reto04_Grupo05'
BASE_DIR = Path(__file__).resolve().parent.parent

# Rutas por defecto
SOLAR_CSV = BASE_DIR/"data"/"results"/"Predicciones_Solar.csv"
WIND_CSV  = BASE_DIR /"data"/"results"/"Predicciones_Eolico.csv"
LOAD_CSV  = BASE_DIR/"data"/"raw"/"demanda_restaurante.csv"
PRICE_CSV = BASE_DIR/"data"/"raw"/"Precios"/"precio2025-peninsula.csv"
OUTPUT_DIR = BASE_DIR/"mas_qlearning_battery"/"results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_simulation(
    mode: str = "competitive", # NUEVO
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
    print(f"  MAS Microred — Modo: {mode.upper()} | {n_steps} steps")
    print(f"{'='*60}\n")

    # Configurar e instanciar la batería compartida (física)
    shared_battery = SimpleBattery(
        capacity_kwh=200.0,
        initial_soc=0.5,
        charge_eff=0.95,
        discharge_eff=0.95,
        max_power_kw=50.0,
        soc_min=0.05
    )

    # NUEVO: Construir las rutas dinámicas a las Q-Tables basadas en el modo
    qtable_solar_path = os.path.join(output_dir, f"{mode}_battery_solar_qtable.npy")
    qtable_wind_path  = os.path.join(output_dir, f"{mode}_battery_wind_qtable.npy")

    # NUEVO: Instanciar agentes inyectando las rutas de las Q-Tables
    solar = SolarAgent(csv_path=solar_csv, strategy_name=strategy_as, qtable_path=qtable_solar_path)
    wind  = WindAgent(csv_path=wind_csv,   strategy_name=strategy_ae, qtable_path=qtable_wind_path)
    
    # Inyectar la batería en el consumidor
    consumer = ConsumerAgent(
        load_csv_path=load_csv, 
        price_csv_path=price_csv,
        battery=shared_battery,
        n_steps=n_steps
    )

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
        filename = f"mas_battery_results_as_{strategy_as}_ae_{strategy_ae}_n{n_steps}.csv"
        path = os.path.join(output_dir, filename)
        df_results.to_csv(path, index=False)
        print(f"\n[run_mas] Resultados guardados → {path}")

    # Resumen de agentes
    print(f"\n[run_mas] Ingresos productores:")
    print(f"  AgenteSolar  ({strategy_as}): {solar.total_revenue_eur:.4f} EUR")
    print(f"  AgenteEolico ({strategy_ae}): {wind.total_revenue_eur:.4f} EUR")

    # Detener servidores
    solar.stop()
    wind.stop()

    # Esperar a que terminen los hilos
    t_solar.join(timeout=2)
    t_wind.join(timeout=2)

    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulación MAS microred energética con negociación FIPA-ACL"
    )
    # NUEVO: Selector de experimento
    parser.add_argument(
        "--mode", default="competitive",
        choices=["competitive", "cooperative", "negotiation"],
        help="Modo de las Q-Tables entrenadas a cargar (default: competitive)"
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
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--solar-csv", default=SOLAR_CSV)
    parser.add_argument("--wind-csv", default=WIND_CSV)
    parser.add_argument("--load-csv", default=LOAD_CSV)
    parser.add_argument("--price-csv", default=PRICE_CSV)
    
    args = parser.parse_args()

    run_simulation(
        mode=args.mode, # NUEVO
        strategy_as=args.strategy_as,
        strategy_ae=args.strategy_ae,
        n_steps=args.n_steps,
        solar_csv=args.solar_csv,
        wind_csv=args.wind_csv,
        load_csv=args.load_csv,
        price_csv=args.price_csv,
    )
