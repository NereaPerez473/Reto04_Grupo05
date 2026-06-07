import polars as pl
from prefect import task

# ---------------------------------------------------------
# TAREA 1: Procesamiento del Agente Eólico
# ---------------------------------------------------------
@task(name="ETL Agente Eólico", log_prints=True, retries=1)
def tarea_feature_engineering_eolico(ruta_entrada: str, ruta_salida: str) -> str:
    """
    Ingesta, transformación y filtrado de los datos eólicos.
    """
    print(f"Procesando datos eólicos desde: {ruta_entrada}")
    
    # 1. Ingesta perezosa y parseo de fecha
    lf = pl.scan_csv(ruta_entrada).with_columns(
        pl.col("Date").str.to_datetime()
    )
    
    # 2. Feature Engineering: Componentes temporales requeridas por el modelo
    lf_features = lf.with_columns([
        pl.col("Date").dt.hour().alias("hour"),
        pl.col("Date").dt.weekday().alias("dayofweek"),
        pl.col("Date").dt.ordinal_day().alias("dayofyear")
    ])
    
    # 3. Filtrar estrictamente las columnas que el .pkl eólico espera
    columnas_ae = [
        'Date', # Mantenemos la fecha para poder cruzar los datos luego
        'temperature_2m', 'relativehumidity_2m', 'dewpoint_2m', 'windspeed_10m',
        'windspeed_100m', 'winddirection_10m', 'winddirection_100m', 'windgusts_10m',
        'hour', 'dayofweek', 'dayofyear'
    ]
    
    # 4. Materializar, limpiar nulos y guardar
    df_listo = lf_features.select(columnas_ae).drop_nulls().collect()
    df_listo.write_csv(ruta_salida)
    
    print(f"¡Éxito! CSV eólico guardado en: {ruta_salida} con {len(df_listo)} registros.")
    return ruta_salida

# ---------------------------------------------------------
# TAREA 2: Procesamiento del Agente Solar
# ---------------------------------------------------------
@task(name="ETL Agente Solar", log_prints=True, retries=1)
def tarea_feature_engineering_solar(ruta_entrada: str, ruta_salida: str) -> str:
    """
    Ingesta, transformación y filtrado de los datos solares.
    Nota: NO renombramos variables a 'PV' porque el .pkl espera los nombres crudos.
    """
    print(f"Procesando datos solares desde: {ruta_entrada}")
    
    lf = pl.scan_csv(ruta_entrada).with_columns(
        pl.col("Date").str.to_datetime()
    )
    
    lf_features = lf.with_columns([
        pl.col("Date").dt.hour().alias("hour"),
        pl.col("Date").dt.weekday().alias("dayofweek"),
        pl.col("Date").dt.ordinal_day().alias("dayofyear")
    ])
    
    # Filtrar estrictamente las columnas que el .pkl solar espera
    columnas_as = [
        'Date', 
        'WindSpeed', 'Sunshine', 'AirPressure', 'Radiation', 'RelativeAirHumidity',
        'hour', 'dayofweek', 'dayofyear'
    ]
    
    df_listo = lf_features.select(columnas_as).drop_nulls().collect()
    df_listo.write_csv(ruta_salida)
    
    print(f"¡Éxito! CSV solar guardado en: {ruta_salida} con {len(df_listo)} registros.")
    return ruta_salida

# ---------------------------------------------------------
# BLOQUE DE PRUEBA LOCAL
# ---------------------------------------------------------
# Si ejecutas este archivo directamente (python src/etl.py), 
# Prefect ejecutará las tareas sin necesidad de un flujo complejo.
if __name__ == "__main__":
    from prefect import flow
    import os
    from pathlib import Path
    
    # 1. Definimos la ruta raíz del proyecto de forma dinámica
    # __file__ es este script (pipeline/etl.py)
    # .parent es la carpeta 'pipeline'
    # .parent.parent es la carpeta raíz 'Reto04_Grupo05'
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # 2. Construimos las rutas absolutas exactas
    raw_eolico = BASE_DIR / "data" / "raw" / "DatosEolicos.csv"
    raw_solar = BASE_DIR / "data" / "raw" / "DatosSolares.csv"
    
    proc_eolico = BASE_DIR / "data" / "processed" / "Features_Eolico.csv"
    proc_solar = BASE_DIR / "data" / "processed" / "Features_Solar.csv"
    
    # Nos aseguramos de que la carpeta de destino existe
    os.makedirs(BASE_DIR / "data" / "processed", exist_ok=True)
    
    @flow(name="Prueba ETL Local")
    def prueba_etl():
        # Pasamos las rutas convertidas a texto (strings)
        tarea_feature_engineering_eolico(
            ruta_entrada=str(raw_eolico), 
            ruta_salida=str(proc_eolico)
        )
        tarea_feature_engineering_solar(
            ruta_entrada=str(raw_solar), 
            ruta_salida=str(proc_solar)
        )
        
    prueba_etl()