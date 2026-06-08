"""
generar_precios_agentes.py
==========================================================================
Reto 04 - Microred multiagente (MU - Master IA Aplicada)

Genera el precio de la energia (EUR/MWh) que cobraria cada agente generador
de la microred (Agente Solar y Agente Eolico) a partir de:

  1) precio2025-peninsula.csv  -> precio real PVPC 2.0TD Peninsula (ancla)
  2) export_GeneracionTRealSolarFotovoltaica_*.csv -> generacion solar real
  3) export_GeneracionMedidaEolicaTerrestre_*.csv  -> generacion eolica real

Lee de  ../data/raw/Precios  y escribe en  ../data/processed/Precios

IDEA CENTRAL
------------
El precio de cada fuente se modela INVERSAMENTE a su propia generacion:
si una fuente genera mucho respecto a lo habitual, su energia es abundante
-> barata; si genera poco, es escasa -> cara.

  * Solar  -> mucha generacion a mediodia => barato a mediodia / caro de noche
  * Eolico -> mucha generacion en invierno => barato en invierno / caro en verano

Usamos el precio PVPC real como ANCLA para que el nivel resultante sea
realista (EUR/MWh creibles) y conserve la estructura temporal del mercado.

PASOS DEL ESCALADO
------------------
Para cada fuente con generacion g(t):

  (1) Normalizacion de la generacion a [0, 1] (percentiles 5/95, robusto):
          g_norm = clip( (g - p5) / (p95 - p5), 0, 1 )

  (2) Escasez:  escasez = 1 - g_norm        (0 = abundante, 1 = escaso)

  (3) Banda de precio relativo: interpolamos linealmente entre un MINIMO
      (cuando es abundante) y un MAXIMO (cuando es escaso). Son simples
      porcentajes sobre el precio base, faciles de leer:
          multiplicador = MIN + (MAX - MIN) * escasez
      Ej. solar MIN=0.4, MAX=1.3  ->  "a tope de sol paga el 40% del precio
      base; sin sol, el 130%".

  (4) Precio final de la fuente:
          precio_fuente = precio_base * multiplicador

UNIDADES
--------
Todo se mantiene en MWh, sin conversiones:
- Precio del CSV: viene en EUR/MWh -> se usa tal cual.
- Generacion: viene en MWh -> se usa tal cual.
- Salida: precio de cada fuente en EUR/MWh.
El multiplicador es adimensional, por lo que la unidad de la generacion no
afecta al resultado.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# PARAMETROS CONFIGURABLES
# --------------------------------------------------------------------------
# Banda de precio relativo (multiplicador sobre el precio base):
#   abundante (escasez=0) -> MIN  |  escaso (escasez=1) -> MAX
SOLAR_MIN, SOLAR_MAX = 0.4, 1.3    # solar: oscilacion dia/noche fuerte
EOLICO_MIN, EOLICO_MAX = 0.7, 1.2  # eolico: oscilacion estacional suave

# Percentiles para la normalizacion robusta de la generacion
P_LOW, P_HIGH = 5, 95

# Carpetas y ficheros por defecto
RAW_DIR = "../data/raw/Precios"
OUT_DIR = "../data/processed/Precios"
F_PRECIO = "precio2025-peninsula.csv"
F_SOLAR = "export_GeneracionTRealSolarFotovoltaica_2026-06-08_10_11.csv"
F_EOLICO = "export_GeneracionMedidaEolicaTerrestre_2026-06-08_10_13.csv"

SEP = ";"


# --------------------------------------------------------------------------
# FUNCIONES
# --------------------------------------------------------------------------
def cargar_serie(path: str | Path, nombre_valor: str) -> pd.DataFrame:
    """Lee un CSV de ESIOS (id;name;geoid;geoname;value;datetime).

    Devuelve 'dt_utc' (clave de union robusta ante el cambio de hora, ya que
    mezcla +01:00 y +02:00), 'datetime' original y la columna de valor.
    """
    df = pd.read_csv(path, sep=SEP)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["dt_utc"] = pd.to_datetime(df["datetime"], utc=True)
    return df[["dt_utc", "datetime", "value"]].rename(columns={"value": nombre_valor})


def multiplicador(generacion: pd.Series, m_min: float, m_max: float,
                  p_low: int = P_LOW, p_high: int = P_HIGH) -> pd.Series:
    """Multiplicador sobre el precio base, inverso a la generacion.

    (1) normaliza  (2) escasez  (3) interpola en la banda [m_min, m_max].
    abundante -> m_min (descuento) ; escaso -> m_max (recargo).
    """
    lo = np.percentile(generacion, p_low)
    hi = np.percentile(generacion, p_high)
    g_norm = ((generacion - lo) / (hi - lo)).clip(0.0, 1.0)  # (1)
    escasez = 1.0 - g_norm                                    # (2)
    return m_min + (m_max - m_min) * escasez                  # (3)


def construir_precio_fuente(base: pd.DataFrame, gen_col: str,
                            m_min: float, m_max: float) -> pd.DataFrame:
    """DataFrame final de precios para una fuente (todo en MWh)."""
    mult = multiplicador(base[gen_col], m_min, m_max)
    return pd.DataFrame({
        "datetime": base["datetime"],
        "generacion_mwh": base[gen_col].round(3),
        "precio_base_eur_mwh": base["precio_eur_mwh"].round(3),
        "multiplicador": mult.round(4),
        "precio_eur_mwh": (base["precio_eur_mwh"] * mult).round(3),  # (4)
    })


def resumen(df: pd.DataFrame, etiqueta: str) -> None:
    """Estadisticas de validacion del precio generado."""
    p = df["precio_eur_mwh"]
    print(f"\n--- {etiqueta} ---")
    print(f"  precio medio : {p.mean():.2f} EUR/MWh "
          f"(base: {df['precio_base_eur_mwh'].mean():.2f})")
    print(f"  min / max    : {p.min():.2f} / {p.max():.2f} EUR/MWh")
    print(f"  multiplicador min / max: "
          f"{df['multiplicador'].min():.2f} / {df['multiplicador'].max():.2f}")


def main(raw_dir: str = RAW_DIR, out_dir: str = OUT_DIR) -> None:
    raw = Path(raw_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # (0) Cargar las tres series y unirlas por la marca temporal UTC
    precio = cargar_serie(raw / F_PRECIO, "precio_eur_mwh")
    solar = cargar_serie(raw / F_SOLAR, "gen_solar_mwh")[["dt_utc", "gen_solar_mwh"]]
    eolico = cargar_serie(raw / F_EOLICO, "gen_eolico_mwh")[["dt_utc", "gen_eolico_mwh"]]

    base = (precio
            .merge(solar, on="dt_utc", how="inner")
            .merge(eolico, on="dt_utc", how="inner")
            .sort_values("dt_utc")
            .reset_index(drop=True))

    # Sin conversion de unidades: precio en EUR/MWh y generacion en MWh.

    print(f"Filas alineadas (precio + solar + eolico): {len(base)}")

    df_solar = construir_precio_fuente(base, "gen_solar_mwh", SOLAR_MIN, SOLAR_MAX)
    df_eolico = construir_precio_fuente(base, "gen_eolico_mwh", EOLICO_MIN, EOLICO_MAX)

    f_solar = out / "precio_solar_mwh.csv"
    f_eolico = out / "precio_eolico_mwh.csv"
    df_solar.to_csv(f_solar, sep=SEP, index=False)
    df_eolico.to_csv(f_eolico, sep=SEP, index=False)

    resumen(df_solar, "AGENTE SOLAR")
    resumen(df_eolico, "AGENTE EOLICO")
    print(f"\nGenerados:\n  {f_solar}\n  {f_eolico}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Genera precios EUR/MWh solar y eolico.")
    ap.add_argument("--raw", default=RAW_DIR, help="carpeta de los 3 CSV de entrada")
    ap.add_argument("--out", default=OUT_DIR, help="carpeta de salida")
    a = ap.parse_args()
    main(a.raw, a.out)
