"""
flask_api/app.py
================
REST API for querying microgrid pipeline results:
  - Optuna studies and trials (optuna_microred.db)
  - Power predictions (data/results/) joined with input features (data/processed/)
  - Agent prices (data/processed/Precios/)
  - Quality metrics (data/results/optimizacion/metricas_calidad.json)
  - Pareto front data (data/results/optimizacion/pareto_fronts.json)
  - Execution times history (data/results/optimizacion/execution_times.json)
  - Pipeline state (data/pipeline_state.json)

Environment variables:
  DATA_DIR   -> path to data/ folder  (default: /app/data)
  OPTUNA_DB  -> path to SQLite file   (default: /app/optimizacion/optuna_microred.db)

Endpoints:
  GET /health
  GET /api/optuna/studies
  GET /api/optuna/studies/<study_name>/trials[?state=COMPLETE]
  GET /api/optuna/studies/<study_name>/best
  GET /api/optuna/studies/<study_name>/params
  GET /api/results/predicciones/<wind|solar>[?n=N]
  GET /api/results/precios/<wind|solar>[?n=N]
  GET /api/results/metrics
  GET /api/results/pareto-plot/<algorithm>
  GET /api/results/pareto-data
  GET /api/results/execution-times
  GET /api/status
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import optuna
import pandas as pd
from flask import Flask, abort, jsonify, render_template, request, send_file

app = Flask(__name__)

DATA_DIR   = Path(os.environ.get("DATA_DIR",  "/app/data"))
OPTUNA_DB  = Path(os.environ.get("OPTUNA_DB", "/app/optimizacion/optuna_microred.db"))

OPTIM_DIR        = DATA_DIR / "results" / "optimizacion"
PLOTS_DIR        = OPTIM_DIR / "plots"
METRICS_FILE     = OPTIM_DIR / "metricas_calidad.json"
PARETO_FRONTS_FILE = OPTIM_DIR / "pareto_fronts.json"
EXEC_TIMES_FILE  = OPTIM_DIR / "execution_times.json"
STATE_FILE       = DATA_DIR  / "pipeline_state.json"


def _storage_url() -> str:
    return f"sqlite:///{OPTUNA_DB}"


def _csv(path: Path, sep: str = ",", nrows: int | None = None) -> list[dict]:
    if not path.exists():
        return []
    df = pd.read_csv(path, sep=sep, nrows=nrows)
    return df.where(df.notna(), None).to_dict(orient="records")


def _load_metrics() -> dict:
    if not METRICS_FILE.exists():
        return {}
    with open(METRICS_FILE, encoding="utf-8") as f:
        return json.load(f)


def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_csv_df(path: Path, sep: str = ",") -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, sep=sep)


def _guess_algo_key(study_name: str, metrics: dict) -> str | None:
    name_upper = study_name.upper()
    for key in metrics:
        if key.upper() in name_upper or name_upper in key.upper():
            return key
    return None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return jsonify({
        "status":           "ok",
        "optuna_db_exists": OPTUNA_DB.exists(),
        "data_dir_exists":  DATA_DIR.exists(),
        "metrics_exists":   METRICS_FILE.exists(),
        "plots_dir_exists": PLOTS_DIR.exists(),
    })


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------
@app.get("/api/optuna/studies")
def get_studies():
    if not OPTUNA_DB.exists():
        return jsonify([])
    summaries = optuna.get_all_study_summaries(storage=_storage_url())
    metrics   = _load_metrics()
    result    = []
    for s in summaries:
        algo_key = _guess_algo_key(s.study_name, metrics)
        hv = metrics.get(algo_key, {}).get("HV_normalizado") if algo_key else None
        result.append({
            "name":       s.study_name,
            "n_trials":   s.n_trials,
            "start_date": str(s.datetime_start) if s.datetime_start else None,
            "hypervolume": hv,
        })
    return jsonify(result)


@app.get("/api/optuna/studies/<study_name>/trials")
def get_trials(study_name: str):
    if not OPTUNA_DB.exists():
        return jsonify([])
    try:
        study = optuna.load_study(study_name=study_name, storage=_storage_url())
    except Exception:
        abort(404, description=f"Study '{study_name}' not found.")

    state  = request.args.get("state", "").upper() or None
    trials = study.trials

    def _trial_dict(t: optuna.trial.FrozenTrial) -> dict:
        return {
            "number":     t.number,
            "state":      t.state.name,
            "value":      t.value,
            "values":     t.values,
            "params":     t.params,
            "start_date": str(t.datetime_start)    if t.datetime_start    else None,
            "end_date":   str(t.datetime_complete) if t.datetime_complete else None,
            "duration_s": (
                (t.datetime_complete - t.datetime_start).total_seconds()
                if t.datetime_start and t.datetime_complete else None
            ),
        }

    data = [_trial_dict(t) for t in trials if state is None or t.state.name == state]
    return jsonify({"study": study_name, "n_trials": len(data), "trials": data})


@app.get("/api/optuna/studies/<study_name>/best")
def get_best_trial(study_name: str):
    if not OPTUNA_DB.exists():
        abort(404, description="Optuna database not found.")
    try:
        study = optuna.load_study(study_name=study_name, storage=_storage_url())
        bt    = study.best_trial
    except Exception as exc:
        abort(400, description=str(exc))
    return jsonify({
        "study":      study_name,
        "best_trial": bt.number,
        "value":      bt.value,
        "params":     bt.params,
    })


@app.get("/api/optuna/studies/<study_name>/params")
def get_best_params(study_name: str):
    if not OPTUNA_DB.exists():
        abort(404, description="Optuna database not found.")
    try:
        study  = optuna.load_study(study_name=study_name, storage=_storage_url())
        params = study.best_params
    except Exception as exc:
        abort(400, description=str(exc))
    return jsonify({"study": study_name, "best_params": params})


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
_PRED_FILES = {
    "wind":  "Predicciones_Eolico.csv",
    "solar": "Predicciones_Solar.csv",
}
_FEAT_FILES = {
    "wind":  "Features_Eolico.csv",
    "solar": "Features_Solar.csv",
}


def _merge_features_predictions(
    source: str, nrows: int | None = None
) -> tuple[pd.DataFrame | None, list[str], str | None]:
    feat_path = DATA_DIR / "processed" / _FEAT_FILES[source]
    pred_path = DATA_DIR / "results"   / _PRED_FILES[source]

    df_feat = _load_csv_df(feat_path)
    df_pred = _load_csv_df(pred_path)

    if df_feat is None and df_pred is None:
        return None, [], None
    if df_feat is None:
        df = df_pred
        return df.head(nrows) if nrows else df, [], df.columns[-1]
    if df_pred is None:
        df = df_feat
        return df.head(nrows) if nrows else df, list(df.columns), None

    JOIN_COL = "Date"
    if JOIN_COL in df_feat.columns and JOIN_COL in df_pred.columns:
        df_feat[JOIN_COL] = df_feat[JOIN_COL].astype(str)
        df_pred[JOIN_COL] = df_pred[JOIN_COL].astype(str)
        pred_cols_no_key = [c for c in df_pred.columns if c != JOIN_COL]
        df = df_feat.merge(df_pred[pred_cols_no_key + [JOIN_COL]], on=JOIN_COL, how="left")
    else:
        pred_cols_no_key = list(df_pred.columns)
        df = pd.concat(
            [df_feat.reset_index(drop=True), df_pred.reset_index(drop=True)], axis=1
        )
    feat_cols = list(df_feat.columns)
    pred_col  = pred_cols_no_key[-1] if pred_cols_no_key else None
    if nrows:
        df = df.head(nrows)
    return df, feat_cols, pred_col


@app.get("/api/results/predicciones/<source>")
def get_predicciones(source: str):
    if source not in _PRED_FILES:
        abort(404)
    nrows = request.args.get("n", type=int)
    df, feat_cols, pred_col = _merge_features_predictions(source, nrows)
    if df is None:
        return jsonify({"source": source, "records": 0,
                        "feature_cols": [], "prediction_col": None, "data": []})
    data = df.where(df.notna(), None).to_dict(orient="records")
    return jsonify({
        "source":         source,
        "records":        len(data),
        "feature_cols":   feat_cols,
        "prediction_col": pred_col,
        "data":           data,
    })


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------
_PRECIO_FILES = {
    "wind":  "precio_eolico_mwh.csv",
    "solar": "precio_solar_mwh.csv",
}


@app.get("/api/results/precios/<source>")
def get_precios(source: str):
    if source not in _PRECIO_FILES:
        abort(404)
    nrows = request.args.get("n", type=int)
    path  = DATA_DIR / "processed" / "Precios" / _PRECIO_FILES[source]
    data  = _csv(path, sep=";", nrows=nrows)
    return jsonify({"source": source, "records": len(data), "data": data})


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@app.get("/api/results/metrics")
def get_metrics():
    return jsonify(_load_metrics())


# ---------------------------------------------------------------------------
# Pareto plots (PNG served as files — kept for backward compat)
# ---------------------------------------------------------------------------
_PARETO_PLOT_MAP = {
    "nsgaii":      "pareto_nsgaii.png",
    "nsga2":       "pareto_nsgaii.png",
    "spea2":       "pareto_spea2.png",
    "comparative": "pareto_comparativo.png",
}


@app.get("/api/results/pareto-plot/<algorithm>")
def get_pareto_plot(algorithm: str):
    key  = algorithm.lower().replace("-", "").replace("_", "")
    name = _PARETO_PLOT_MAP.get(key)
    if not name:
        abort(404, description=f"No plot registered for algorithm '{algorithm}'.")
    path = PLOTS_DIR / name
    if not path.exists():
        abort(404, description=f"Plot file '{name}' not found on disk.")
    return send_file(path, mimetype="image/png")


# ---------------------------------------------------------------------------
# Real-time pipeline endpoints
# ---------------------------------------------------------------------------
@app.get("/api/results/pareto-data")
def get_pareto_data():
    """Latest Pareto front points as JSON (for Chart.js scatter plots)."""
    data = _load_json(PARETO_FRONTS_FILE)
    if data is None:
        return jsonify({"fronts": {}, "timestamp": None, "window_start_hour": 0})
    return jsonify(data)


@app.get("/api/results/execution-times")
def get_execution_times():
    """Execution times history — list of {timestamp, window_start_hour, NSGAII, SPEA2}."""
    data = _load_json(EXEC_TIMES_FILE)
    return jsonify(data if data is not None else [])


@app.get("/api/status")
def get_pipeline_status():
    """Current sliding-window state written by the scheduler."""
    data = _load_json(STATE_FILE)
    if data is None:
        return jsonify({"runs_completed": 0, "last_run": None,
                        "last_start_hour": 0, "last_elapsed_s": None})
    return jsonify(data)


# ---------------------------------------------------------------------------
# Real-time dashboard (fully JS-driven via Chart.js)
# ---------------------------------------------------------------------------
@app.get("/")
def dashboard():
    return render_template("dashboard.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
