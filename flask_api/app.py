"""
flask_api/app.py
================
API REST para consultar los resultados del pipeline de la microred:
  - Estudios y trials de Optuna (optuna_microred.db)
  - Predicciones de potencia (data/results/)
  - Precios por agente (data/processed/Precios/)

Variables de entorno:
  DATA_DIR   -> ruta a la carpeta data/  (default: /app/data)
  OPTUNA_DB  -> ruta al fichero SQLite   (default: /app/optimizacion/optuna_microred.db)

Endpoints:
  GET /health
  GET /api/optuna/studies
  GET /api/optuna/studies/<study_name>/trials[?estado=COMPLETE]
  GET /api/optuna/studies/<study_name>/best
  GET /api/optuna/studies/<study_name>/params
  GET /api/results/predicciones/<eolico|solar>[?n=N]
  GET /api/results/precios/<eolico|solar>[?n=N]
"""

from __future__ import annotations

import os
from pathlib import Path

import optuna
import pandas as pd
from flask import Flask, abort, jsonify, request

app = Flask(__name__)

DATA_DIR  = Path(os.environ.get("DATA_DIR",  "/app/data"))
OPTUNA_DB = Path(os.environ.get("OPTUNA_DB", "/app/optimizacion/optuna_microred.db"))


def _storage_url() -> str:
    return f"sqlite:///{OPTUNA_DB}"


def _csv(path: Path, sep: str = ",", nrows: int | None = None) -> list[dict]:
    """Lee un CSV y lo devuelve como lista de diccionarios."""
    if not path.exists():
        return []
    df = pd.read_csv(path, sep=sep, nrows=nrows)
    return df.where(df.notna(), None).to_dict(orient="records")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "optuna_db_exists": OPTUNA_DB.exists(),
        "data_dir_exists": DATA_DIR.exists(),
    })


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------
@app.get("/api/optuna/studies")
def get_studies():
    """Lista todos los estudios con resumen de trials."""
    if not OPTUNA_DB.exists():
        return jsonify([])

    summaries = optuna.get_all_study_summaries(storage=_storage_url())
    result = []
    for s in summaries:
        entry = {
            "nombre": s.study_name,
            "n_trials": s.n_trials,
            "fecha_inicio": str(s.datetime_start) if s.datetime_start else None,
        }
        # best_trial no existe en estudios multi-objetivo
        try:
            bt = s.best_trial
            entry["mejor_trial"] = bt.number
            entry["mejor_valor"] = bt.value
        except Exception:
            entry["mejor_trial"] = None
            entry["mejor_valor"] = None
        result.append(entry)
    return jsonify(result)


@app.get("/api/optuna/studies/<study_name>/trials")
def get_trials(study_name: str):
    """
    Todos los trials de un estudio.
    ?estado=COMPLETE  filtra por estado (COMPLETE, RUNNING, FAIL, WAITING).
    """
    if not OPTUNA_DB.exists():
        return jsonify([])

    try:
        study = optuna.load_study(study_name=study_name, storage=_storage_url())
    except Exception:
        abort(404, description=f"Estudio '{study_name}' no encontrado.")

    estado = request.args.get("estado", "").upper() or None
    trials = study.trials

    def _trial_dict(t: optuna.trial.FrozenTrial) -> dict:
        return {
            "numero": t.number,
            "estado": t.state.name,
            "valor": t.value,
            "valores": t.values,
            "params": t.params,
            "fecha_inicio": str(t.datetime_start) if t.datetime_start else None,
            "fecha_fin": str(t.datetime_complete) if t.datetime_complete else None,
            "duracion_s": (
                (t.datetime_complete - t.datetime_start).total_seconds()
                if t.datetime_start and t.datetime_complete
                else None
            ),
        }

    data = [
        _trial_dict(t)
        for t in trials
        if estado is None or t.state.name == estado
    ]
    return jsonify({"estudio": study_name, "n_trials": len(data), "trials": data})


@app.get("/api/optuna/studies/<study_name>/best")
def get_best_trial(study_name: str):
    """Mejor trial de un estudio mono-objetivo."""
    if not OPTUNA_DB.exists():
        abort(404, description="Base de datos Optuna no encontrada.")

    try:
        study = optuna.load_study(study_name=study_name, storage=_storage_url())
        bt = study.best_trial
    except Exception as exc:
        abort(400, description=str(exc))

    return jsonify({
        "estudio": study_name,
        "mejor_trial": bt.number,
        "valor": bt.value,
        "params": bt.params,
    })


@app.get("/api/optuna/studies/<study_name>/params")
def get_best_params(study_name: str):
    """Parametros optimos (best_params) de un estudio mono-objetivo."""
    if not OPTUNA_DB.exists():
        abort(404, description="Base de datos Optuna no encontrada.")

    try:
        study = optuna.load_study(study_name=study_name, storage=_storage_url())
        params = study.best_params
    except Exception as exc:
        abort(400, description=str(exc))

    return jsonify({"estudio": study_name, "best_params": params})


# ---------------------------------------------------------------------------
# Predicciones de potencia
# ---------------------------------------------------------------------------
_PRED_FILES = {
    "eolico": "Predicciones_Eolico.csv",
    "solar":  "Predicciones_Solar.csv",
}


@app.get("/api/results/predicciones/<fuente>")
def get_predicciones(fuente: str):
    """
    Devuelve las predicciones de potencia de la fuente indicada.
    ?n=N  limita el numero de filas (util para previsualizar).
    """
    if fuente not in _PRED_FILES:
        abort(404)
    nrows = request.args.get("n", type=int)
    data  = _csv(DATA_DIR / "results" / _PRED_FILES[fuente], nrows=nrows)
    return jsonify({"fuente": fuente, "registros": len(data), "data": data})


# ---------------------------------------------------------------------------
# Precios por agente
# ---------------------------------------------------------------------------
_PRECIO_FILES = {
    "eolico": "precio_eolico_mwh.csv",
    "solar":  "precio_solar_mwh.csv",
}


@app.get("/api/results/precios/<fuente>")
def get_precios(fuente: str):
    """
    Devuelve los precios EUR/MWh generados por generar_precios_agentes.py.
    ?n=N  limita el numero de filas.
    """
    if fuente not in _PRECIO_FILES:
        abort(404)
    nrows = request.args.get("n", type=int)
    path  = DATA_DIR / "processed" / "Precios" / _PRECIO_FILES[fuente]
    data  = _csv(path, sep=";", nrows=nrows)
    return jsonify({"fuente": fuente, "registros": len(data), "data": data})


# ---------------------------------------------------------------------------
# Entrada
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
