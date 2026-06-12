"""
flow.py - Pipeline principal Prefect para la microred multiagente.

Orden de ejecucion:
  1. ETL          (pipeline/etl.py)
  2. Inferencia   (pipeline/inference.py)
  3. Precios      (optimizacion/generar_precios_agentes.py)
  4. Optimizacion — cuatro tasks en lugar del notebook completo:
       4a. tarea_setup_datos      -> carga datos y calcula cotas
       4b. tarea_run_nsgaii       -> ejecuta NSGA-II con best_params del DB
       4c. tarea_run_spea2        -> ejecuta SPEA2 con best_params del DB
       4d. tarea_analizar_resultados -> metricas, plots y guardado
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

    # 4 – Optimizacion multiobjetivo en cuatro tasks
    from optimizacion.optimizacion_tasks import (  # noqa: PLC0415
        tarea_setup_datos,
        tarea_run_nsgaii,
        tarea_run_spea2,
        tarea_analizar_resultados,
    )

    opt_results_dir = str(res_dir / "optimizacion")
    optuna_db       = str(opt_dir / "optuna_microred.db")

    # 4a – Carga de datos y calculo de cotas (depende de precios e inferencia)
    datos_ctx = tarea_setup_datos(
        data_dir_raw=str(raw_dir),
        data_dir_processed=str(proc_dir),
        data_dir_results=str(res_dir),
        output_dir=opt_results_dir,
    )

    # 4b y 4c – Ejecucion de cada algoritmo con best_params guardados en el DB
    res_nsgaii = tarea_run_nsgaii(datos=datos_ctx, optuna_db=optuna_db)
    res_spea2  = tarea_run_spea2(datos=datos_ctx,  optuna_db=optuna_db)

    # 4d – Metricas, plots comparativos y guardado
    tarea_analizar_resultados(
        datos=datos_ctx,
        res_nsgaii=res_nsgaii,
        res_spea2=res_spea2,
        optuna_db=optuna_db,
    )


if __name__ == "__main__":
    pipeline_microred()
