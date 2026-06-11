"""
analisis_datos.py
=================
Script para analizar las estadísticas descriptivas (máximos, mínimos, medias)
de los datasets originales de la microred.
"""

import pandas as pd
from pathlib import Path

# ==================================================
# RUTAS
# ==================================================
BASE_DIR = Path(__file__).resolve().parent.parent

SOLAR_CSV = BASE_DIR / "data" / "results" / "Predicciones_Solar.csv"
WIND_CSV  = BASE_DIR / "data" / "results" / "Predicciones_Eolico.csv"
LOAD_CSV  = BASE_DIR / "data" / "raw" / "demanda_restaurante.csv"

# ==================================================
# CARGA DE DATOS
# ==================================================
print("Cargando datasets...\n")

try:
    solar_df = pd.read_csv(SOLAR_CSV)
    wind_df  = pd.read_csv(WIND_CSV)
    load_df  = pd.read_csv(LOAD_CSV)
except FileNotFoundError as e:
    print(f"ERROR: No se encontró el archivo. Revisa las rutas:\n{e}")
    exit()

# ==================================================
# ANÁLISIS
# ==================================================

print("=" * 50)
print("RADIOGRAFÍA: PRODUCCIÓN SOLAR (kW)")
print("=" * 50)
# Ajusta el nombre de la columna si en tu CSV se llama distinto
if "SystemProduction_AS" in solar_df.columns:
    print(solar_df["SystemProduction_AS"].describe().round(2))
else:
    print(f"Columnas disponibles: {solar_df.columns.tolist()}")

print("\n" + "=" * 50)
print("RADIOGRAFÍA: PRODUCCIÓN EÓLICA (kW)")
print("=" * 50)
if "Power_AE" in wind_df.columns:
    print(wind_df["Power_AE"].describe().round(2))
else:
    print(f"Columnas disponibles: {wind_df.columns.tolist()}")

print("\n" + "=" * 50)
print("RADIOGRAFÍA: DEMANDA DEL RESTAURANTE (kW)")
print("=" * 50)
# Ajusta el nombre de la columna si en tu CSV se llama distinto
col_demanda = "Electricity:Facility [kW](Hourly)"
if col_demanda in load_df.columns:
    print(load_df[col_demanda].describe().round(2))
else:
    print(f"Columnas disponibles: {load_df.columns.tolist()}")

print("\n" + "=" * 50)
print("COMPARATIVA DE MÁXIMOS ABSOLUTOS")
print("=" * 50)
print(f"Pico Máximo Solar:    {solar_df['SystemProduction_AS'].max():.2f} kW")
print(f"Pico Máximo Eólico:   {wind_df['Power_AE'].max():.2f} kW")
print(f"Pico Máximo Demanda:  {load_df[col_demanda].max():.2f} kW")
print("=" * 50)