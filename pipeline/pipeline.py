import polars as pl
from prefect import task, flow

# ---------------------------------------------------------
# FASE 1: Ingesta (Lazy) y Enriquecimiento (Separados)
# ---------------------------------------------------------
@task(name="Tarea 1: Ingesta y Filtrado Lazy", log_prints=True, retries=2)
def tarea_1_ingesta(ruta_solar: str, ruta_eolico: str) -> pl.DataFrame:
    """
    Usa LazyFrames para leer, cruzar y filtrar datos antes de subirlos a RAM.
    """
    # scan_csv crea el LazyFrame
    lf_solar = pl.scan_csv(ruta_solar)
    lf_eolico = pl.scan_csv(ruta_eolico)
    
    # Unimos y filtramos valores nulos de forma perezosa
    lf_unido = lf_solar.join(lf_eolico, on="Time").drop_nulls()
    
    # Ejecutamos el plan y materializamos
    df = lf_unido.collect()
    print(f"Registros válidos cargados: {len(df)}")
    return df

@task(name="Tarea 2: Feature Engineering", log_prints=True)
def tarea_2_enriquecimiento(df: pl.DataFrame) -> pl.DataFrame:
    """
    Añade variables útiles para los modelos y los agentes.
    (Ejemplo equivalente al when().then() del profesor)
    """
    df_enriquecido = df.with_columns([
        # Ejemplo: Categorizar si es de día o de noche según la radiación
        pl.when(pl.col("Radiation") > 0)
        .then(pl.lit("Dia"))
        .otherwise(pl.lit("Noche"))
        .alias("periodo_dia"),
        
        # Ejemplo: Racha de viento peligrosa para el AE
        pl.when(pl.col("windgusts_10m") > 15)
        .then(pl.lit("Alerta Viento"))
        .otherwise(pl.lit("Normal"))
        .alias("estado_viento")
    ])
    print(f"Columnas tras enriquecimiento: {df_enriquecido.columns}")
    return df_enriquecido

# ---------------------------------------------------------
# FASE 2: Inferencia y Explicabilidad
# ---------------------------------------------------------
@task(name="Tarea 3: Predicción de Generación")
def tarea_3_prediccion(df: pl.DataFrame) -> dict:
    # Lógica de inferencia (Devuelve predicciones en un diccionario o DataFrame)
    return {"energia_solar": 500, "energia_eolica": 300}

@task(name="Tarea 4: Explicabilidad (xAI)")
def tarea_4_xai(df: pl.DataFrame):
    # Auditoría de los modelos
    pass

# ---------------------------------------------------------
# FASE 3: Bifurcación (Ejecución Independiente)
# ---------------------------------------------------------
@task(name="Tarea 5A: Optimización Centralizada")
def tarea_5a_optimizacion(predicciones: dict):
    # Uso de jMetalPy para frente de Pareto
    return "Frente_Pareto_Calculado"

@task(name="Tarea 5B: Negociación Multiagente")
def tarea_5b_multiagente(predicciones: dict):
    # Comunicación FIPA-ACL
    return "Log_Negociacion"

# ---------------------------------------------------------
# ORQUESTACIÓN PRINCIPAL (@flow)
# ---------------------------------------------------------
@flow(name="Pipeline-Microred-Reto4")
def pipeline_microred():
    # 1. Preparación de datos (Estilo profesor)
    df_bruto = tarea_1_ingesta("data/solar.csv", "data/eolico.csv")
    df_listo = tarea_2_enriquecimiento(df_bruto)
    
    # 2. IA y xAI
    predicciones = tarea_3_prediccion(df_listo)
    tarea_4_xai(df_listo) # Puede correr sin bloquear lo demás
    
    # 3. Resolución del problema (Ramas paralelas)
    tarea_5a_optimizacion(predicciones)
    tarea_5b_multiagente(predicciones)

if __name__ == "__main__":
    pipeline_microred()