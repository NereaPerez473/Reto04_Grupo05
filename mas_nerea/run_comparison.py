"""
run_comparison.py
=================
Ejecuta todas las combinaciones de estrategias relevantes y guarda
un único CSV consolidado para el análisis comparativo.

Combinaciones simuladas
------------------------
    AS=honest,           AE=honest            → baseline
    AS=deception,        AE=honest            → solo solar engaña
    AS=honest,           AE=deception         → solo eólico engaña
    AS=hide_information, AE=honest            → solar oculta
    AS=honest,           AE=hide_information  → eólico oculta
    AS=deception,        AE=deception         → ambos engañan
    AS=hide_information, AE=hide_information  → ambos ocultan

Salida
------
    data/results/mas_comparison_all.csv   (todas las combinaciones)
    data/results/mas_summary.csv          (métricas agregadas por combinación)

Uso
---
    python run_comparison.py
    python run_comparison.py --n-steps 200   # para prueba rápida
"""

import argparse
import pandas as pd
from run_mas import run_simulation
from pathlib import Path

# .parent.parent es la carpeta raíz 'Reto04_Grupo05'
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR /"mas_nerea"/"results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR_ALL=OUTPUT_DIR / "mas_comparison_all.csv"
CSV_DIR_SUMMARY=OUTPUT_DIR / "mas_summary.csv"

COMBINATIONS = [
    ("honest",           "honest"),
    ("deception",        "honest"),
    ("honest",           "deception"),
    ("hide_information", "honest"),
    ("honest",           "hide_information"),
    ("deception",        "deception"),
    ("hide_information", "hide_information"),
]


def run_comparison(n_steps: int = 500) -> pd.DataFrame:
    """
    Ejecuta todas las combinaciones y devuelve el DataFrame consolidado.
    """
    all_results = []
    summary_rows = []

    for strategy_as, strategy_ae in COMBINATIONS:
        df = run_simulation(
            strategy_as=strategy_as,
            strategy_ae=strategy_ae,
            n_steps=n_steps,
            save_csv=False,  # guardamos el consolidado al final
        )

        all_results.append(df)

        # Métricas agregadas para esta combinación
        summary_rows.append({
            "strategy_as":               strategy_as,
            "strategy_ae":               strategy_ae,
            "total_cost_eur":            round(df["total_cost_eur"].sum(), 4),
            "mean_cost_per_step_eur":    round(df["total_cost_eur"].mean(), 6),
            "mean_renewable_coverage":   round(df["renewable_coverage_pct"].mean(), 2),
            "total_grid_purchased_kwh":  round(df["grid_purchased_kw"].sum(), 2),
            "total_solar_shortfall_kwh": round(df["solar_shortfall_kw"].sum(), 4),
            "total_wind_shortfall_kwh":  round(df["wind_shortfall_kw"].sum(), 4),
            "total_shortfall_kwh":       round(
                df["solar_shortfall_kw"].sum() + df["wind_shortfall_kw"].sum(), 4),
            "n_steps":                   n_steps,
        })

    # DataFrame consolidado
    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv(str(CSV_DIR_ALL), index=False)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(str(CSV_DIR_SUMMARY), index=False)

    print("\n" + "="*60)
    print("RESUMEN COMPARACIÓN DE ESTRATEGIAS")
    print("="*60)
    print(df_summary.to_string(index=False))
    print(f"\nArchivos guardados:")
    print(f"  mas_nerea/results/mas_comparison_all.csv")
    print(f"  mas_nerea/results/mas_summary.csv")

    return df_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comparación de todas las combinaciones de estrategias MAS"
    )
    parser.add_argument(
        "--n-steps", type=int, default=500,
        help="Timesteps por combinación (default: 500)"
    )
    args = parser.parse_args()
    run_comparison(n_steps=args.n_steps)
