"""
flow.py - Pipeline principal Prefect para la microred multiagente.

Orden de ejecucion:
  1. ETL          (pipeline/etl.py)
  2. Inferencia   (pipeline/inference.py)
  3. Precios      (optimizacion/generar_precios_agentes.py)
  4. Optimizacion (optimizacion/Optimizacion.ipynb via papermill)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from prefect import flow, task

# Directorio raiz del proyecto dentro del contenedor (o en local si se ejecuta
# directamente). Se sobreescribe con la variable de entorno APP_DIR.
BASE_DIR = Path(os.environ.get("APP_DIR", Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Tarea 3 – Generacion de precios por agente
# ---------------------------------------------------------------------------
@task(name="Generar Precios Agentes", log_prints=True, retries=1)
def tarea_generar_precios(raw_dir: str, out_dir: str) -> str:
    """
    Llama a generar_precios_agentes.main().Si la carpeta de precios crudos no existe, emite un aviso y la omite
    para no bloquear el resto del pipeline.
    """
    raw = Path(raw_dir)
    if not raw.exists():
        print(
            f"[AVISO] Carpeta de datos de precios ESIOS no encontrada: {raw}\n"
            "        Coloca los CSV de precio PVPC y generacion solar/eolica en esa ruta.\n"
            "        Se omite la generacion de precios por agente."
        )
        return out_dir

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(BASE_DIR / "optimizacion"))
    import generar_precios_agentes  # noqa: PLC0415

    generar_precios_agentes.main(raw_dir=raw_dir, out_dir=out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# Tarea 4 – Ejecucion del notebook de optimizacion
# ---------------------------------------------------------------------------
@task(name="Optimizacion Notebook", log_prints=True, retries=0)
def tarea_optimizacion(notebook_path: str, output_path: str) -> str:
    """
    Ejecuta Optimizacion.ipynb con papermill.

    - cwd se fija en el directorio del notebook para que los paths relativos
      del propio notebook (../data/...) resuelvan correctamente.
    - timeout de 3 horas para las tiradas largas de NSGA-II / SPEA2.
    - El notebook escribe en optuna_microred.db (persistido en bind mount).
    """
    import papermill as pm  # noqa: PLC0415

    nb_dir = str(Path(notebook_path).parent)
    print(f"Ejecutando notebook: {notebook_path}")
    pm.execute_notebook(
        input_path=notebook_path,
        output_path=output_path,
        cwd=nb_dir,
        kernel_name="python3",
        execution_timeout=10_800,  # 3 h
    )
    print(f"Notebook completado. Salida: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Flow principal
# ---------------------------------------------------------------------------
@flow(name="Pipeline Microred Multiagente", log_prints=True)
def pipeline_microred() -> None:
    from pipeline.etl import (  # noqa: PLC0415
        tarea_feature_engineering_eolico,
        tarea_feature_engineering_solar,
    )
    from pipeline.inference import (  # noqa: PLC0415
        tarea_predecir_eolico,
        tarea_predecir_solar,
    )

    raw_dir  = BASE_DIR / "data" / "raw"
    proc_dir = BASE_DIR / "data" / "processed"
    res_dir  = BASE_DIR / "data" / "results"
    mod_dir  = BASE_DIR / "models"
    opt_dir  = BASE_DIR / "optimizacion"

    proc_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    # 1 – Feature engineering (paralelo: eolico y solar son independientes)
    path_feat_eolico = tarea_feature_engineering_eolico(
        ruta_entrada=str(raw_dir / "DatosEolicos.csv"),
        ruta_salida=str(proc_dir / "Features_Eolico.csv"),
    )
    path_feat_solar = tarea_feature_engineering_solar(
        ruta_entrada=str(raw_dir / "DatosSolares.csv"),
        ruta_salida=str(proc_dir / "Features_Solar.csv"),
    )

    # 2 – Inferencia (depende de los paths devueltos en el paso 1)
    path_pred_eolico = tarea_predecir_eolico(
        ruta_datos=path_feat_eolico,
        ruta_modelo=str(mod_dir / "modelo_eolico.pkl"),
        ruta_salida=str(res_dir / "Predicciones_Eolico.csv"),
    )
    path_pred_solar = tarea_predecir_solar(
        ruta_datos=path_feat_solar,
        ruta_modelo=str(mod_dir / "modelo_solar.pkl"),
        ruta_salida=str(res_dir / "Predicciones_Solar.csv"),
    )

    # 3 – Precios por agente
    # Nota: los datos de entrada son CSVs de ESIOS en data/raw/Precios/
    #       (independientes de la inferencia; esperamos a que termine para
    #       mantener el orden logico del pipeline).
    _ = path_pred_eolico, path_pred_solar  # establece dependencia Prefect
    tarea_generar_precios(
        raw_dir=str(raw_dir / "Precios"),
        out_dir=str(proc_dir / "Precios"),
    )

    # 4 – Optimizacion multiobjetivo (escribe en optuna_microred.db)
    tarea_optimizacion(
        notebook_path=str(opt_dir / "Optimizacion.ipynb"),
        output_path=str(opt_dir / "Optimizacion_output.ipynb"),
    )


if __name__ == "__main__":
    pipeline_microred()
