import polars as pl
import joblib
import os
from pathlib import Path
from prefect import task, flow

# ---------------------------------------------------------
# TAREA 1: Inferencia del Agente Eólico
# ---------------------------------------------------------
@task(name="Inferencia Agente Eólico", log_prints=True, retries=1)
def tarea_predecir_eolico(ruta_datos: str, ruta_modelo: str, ruta_salida: str) -> str:
    """
    Carga los datos limpios y el modelo, genera la predicción 
    y la guarda en la carpeta results.
    """
    print(f"Cargando modelo eólico desde: {ruta_modelo}")
    modelo = joblib.load(ruta_modelo)
    
    print(f"Leyendo datos procesados desde: {ruta_datos}")
    # Leemos el CSV (aquí podemos usar read_csv directo porque el archivo ya es pequeño/filtrado)
    df_features = pl.read_csv(ruta_datos).with_columns(pl.col("Date").str.to_datetime())
    
    # Scikit-learn necesita una matriz numérica pura, así que quitamos la columna 'Date'
    X_eolico = df_features.drop("Date").to_numpy()
    
    # Generamos la predicción
    predicciones = modelo.predict(X_eolico)
    
    # Creamos un nuevo DataFrame solo con la fecha y la predicción final
    df_resultado = pl.DataFrame({
        "Date": df_features["Date"],
        "Power_AE": predicciones
    })
    
    # Guardamos en disco
    df_resultado.write_csv(ruta_salida)
    print(f"¡Éxito! Predicción eólica guardada en: {ruta_salida}")
    
    return ruta_salida

# ---------------------------------------------------------
# TAREA 2: Inferencia del Agente Solar
# ---------------------------------------------------------
@task(name="Inferencia Agente Solar", log_prints=True, retries=1)
def tarea_predecir_solar(ruta_datos: str, ruta_modelo: str, ruta_salida: str) -> str:
    """
    Carga los datos limpios y el modelo, genera la predicción 
    y la guarda en la carpeta results.
    """
    print(f"Cargando modelo solar desde: {ruta_modelo}")
    modelo = joblib.load(ruta_modelo)
    
    df_features = pl.read_csv(ruta_datos).with_columns(pl.col("Date").str.to_datetime())
    
    X_solar = df_features.drop("Date").to_numpy()
    
    predicciones = modelo.predict(X_solar)
    
    df_resultado = pl.DataFrame({
        "Date": df_features["Date"],
        "SystemProduction_AS": predicciones
    })
    
    df_resultado.write_csv(ruta_salida)
    print(f"¡Éxito! Predicción solar guardada en: {ruta_salida}")
    
    return ruta_salida

# ---------------------------------------------------------
# BLOQUE DE PRUEBA LOCAL
# ---------------------------------------------------------
if __name__ == "__main__":
    
    # 1. Definimos la ruta raíz dinámicamente usando pathlib
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # 2. Rutas de entrada (los archivos que generó etl.py)
    proc_eolico = BASE_DIR / "data" / "processed" / "Features_Eolico.csv"
    proc_solar = BASE_DIR / "data" / "processed" / "Features_Solar.csv"
    
    # 3. Rutas de los modelos
    mod_eolico = BASE_DIR / "models" / "modelo_eolico.pkl"
    mod_solar = BASE_DIR / "models" / "modelo_solar.pkl"
    
    # 4. Rutas de salida
    res_eolico = BASE_DIR / "data" / "results" / "Predicciones_Eolico.csv"
    res_solar = BASE_DIR / "data" / "results" / "Predicciones_Solar.csv"
    
    # Aseguramos que la carpeta data/results existe
    os.makedirs(BASE_DIR / "data" / "results", exist_ok=True)
    
    @flow(name="Prueba Inferencia Local")
    def prueba_inferencia():
        tarea_predecir_eolico(
            ruta_datos=str(proc_eolico),
            ruta_modelo=str(mod_eolico),
            ruta_salida=str(res_eolico)
        )
        
        tarea_predecir_solar(
            ruta_datos=str(proc_solar),
            ruta_modelo=str(mod_solar),
            ruta_salida=str(res_solar)
        )
        
    # Recuerda tener `prefect server start` ejecutándose en otra terminal
    prueba_inferencia()