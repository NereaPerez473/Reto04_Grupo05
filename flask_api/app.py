"""
flask_api/app.py
================
REST API for querying microgrid pipeline results:
  - Optuna studies and trials (optuna_microred.db)
  - Power predictions (data/results/) joined with input features (data/processed/)
  - Agent prices (data/processed/Precios/)
  - Quality metrics (data/results/optimizacion/metricas_calidad.json)
  - Pareto front plots (data/results/optimizacion/plots/)

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
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import optuna
import pandas as pd
from flask import Flask, abort, jsonify, render_template_string, request, send_file

app = Flask(__name__)

DATA_DIR   = Path(os.environ.get("DATA_DIR",  "/app/data"))
OPTUNA_DB  = Path(os.environ.get("OPTUNA_DB", "/app/optimizacion/optuna_microred.db"))
OPTIM_DIR  = DATA_DIR / "results" / "optimizacion"
PLOTS_DIR  = OPTIM_DIR / "plots"
METRICS_FILE = OPTIM_DIR / "metricas_calidad.json"


def _storage_url() -> str:
    return f"sqlite:///{OPTUNA_DB}"


def _csv(path: Path, sep: str = ",", nrows: int | None = None) -> list[dict]:
    """Read a CSV and return it as a list of dicts."""
    if not path.exists():
        return []
    df = pd.read_csv(path, sep=sep, nrows=nrows)
    return df.where(df.notna(), None).to_dict(orient="records")


def _load_metrics() -> dict:
    """Load metricas_calidad.json; return {} if missing."""
    if not METRICS_FILE.exists():
        return {}
    with open(METRICS_FILE, encoding="utf-8") as f:
        return json.load(f)


def _img_b64(path: Path) -> str | None:
    """Read a PNG and return a data-URI string, or None if missing."""
    if not path.exists():
        return None
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
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
    """List all studies with trial summary."""
    if not OPTUNA_DB.exists():
        return jsonify([])

    summaries = optuna.get_all_study_summaries(storage=_storage_url())
    metrics   = _load_metrics()
    result    = []
    for s in summaries:
        # Try to match study name to an algorithm key in metrics
        algo_key = _guess_algo_key(s.study_name, metrics)
        hv = metrics.get(algo_key, {}).get("HV_normalizado") if algo_key else None
        entry = {
            "name":       s.study_name,
            "n_trials":   s.n_trials,
            "start_date": str(s.datetime_start) if s.datetime_start else None,
            "hypervolume": hv,
        }
        result.append(entry)
    return jsonify(result)


def _guess_algo_key(study_name: str, metrics: dict) -> str | None:
    """Try to match study_name to a key in the metrics dict (case-insensitive substring)."""
    name_upper = study_name.upper()
    for key in metrics:
        if key.upper() in name_upper or name_upper in key.upper():
            return key
    return None


@app.get("/api/optuna/studies/<study_name>/trials")
def get_trials(study_name: str):
    """
    All trials for a study.
    ?state=COMPLETE  filters by state (COMPLETE, RUNNING, FAIL, WAITING).
    """
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
                if t.datetime_start and t.datetime_complete
                else None
            ),
        }

    data = [
        _trial_dict(t)
        for t in trials
        if state is None or t.state.name == state
    ]
    return jsonify({"study": study_name, "n_trials": len(data), "trials": data})


@app.get("/api/optuna/studies/<study_name>/best")
def get_best_trial(study_name: str):
    """Best trial of a single-objective study."""
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
    """Optimal parameters (best_params) of a single-objective study."""
    if not OPTUNA_DB.exists():
        abort(404, description="Optuna database not found.")
    try:
        study  = optuna.load_study(study_name=study_name, storage=_storage_url())
        params = study.best_params
    except Exception as exc:
        abort(400, description=str(exc))
    return jsonify({"study": study_name, "best_params": params})


# ---------------------------------------------------------------------------
# Power predictions
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
    """
    Load and join Features_*.csv with Predicciones_*.csv on the Date column
    (or by positional index if Date is absent in one of them).

    Returns:
        df          – merged DataFrame (or None if both files missing)
        feat_cols   – list of feature column names (to style them differently)
        pred_col    – name of the prediction column (last col of predictions file)
    """
    feat_path = DATA_DIR / "processed" / _FEAT_FILES[source]
    pred_path = DATA_DIR / "results"   / _PRED_FILES[source]

    df_feat = _load_csv_df(feat_path)
    df_pred = _load_csv_df(pred_path)

    if df_feat is None and df_pred is None:
        return None, [], None

    # If only one is available, return it as-is
    if df_feat is None:
        df = df_pred
        return df.head(nrows) if nrows else df, [], df.columns[-1]
    if df_pred is None:
        df = df_feat
        return df.head(nrows) if nrows else df, list(df.columns), None

    # Determine join key
    JOIN_COL = "Date"
    if JOIN_COL in df_feat.columns and JOIN_COL in df_pred.columns:
        df_feat[JOIN_COL] = df_feat[JOIN_COL].astype(str)
        df_pred[JOIN_COL] = df_pred[JOIN_COL].astype(str)
        # Prediction columns excluding the key
        pred_cols_no_key = [c for c in df_pred.columns if c != JOIN_COL]
        df = df_feat.merge(df_pred[pred_cols_no_key + [JOIN_COL]], on=JOIN_COL, how="left")
    else:
        # Positional join — reset index and concat side-by-side
        pred_cols_no_key = list(df_pred.columns)
        df = pd.concat(
            [df_feat.reset_index(drop=True), df_pred.reset_index(drop=True)],
            axis=1,
        )

    feat_cols = list(df_feat.columns)
    pred_col  = pred_cols_no_key[-1] if pred_cols_no_key else None

    if nrows:
        df = df.head(nrows)
    return df, feat_cols, pred_col


@app.get("/api/results/predicciones/<source>")
def get_predicciones(source: str):
    """
    Returns input features joined with power predictions for the given source.
    ?n=N  limits the number of rows (useful for preview).
    """
    if source not in _PRED_FILES:
        abort(404)
    nrows = request.args.get("n", type=int)
    df, feat_cols, pred_col = _merge_features_predictions(source, nrows)
    if df is None:
        return jsonify({"source": source, "records": 0, "feature_cols": [], "prediction_col": None, "data": []})
    data = df.where(df.notna(), None).to_dict(orient="records")
    return jsonify({
        "source":         source,
        "records":        len(data),
        "feature_cols":   feat_cols,
        "prediction_col": pred_col,
        "data":           data,
    })


# ---------------------------------------------------------------------------
# Agent prices
# ---------------------------------------------------------------------------
_PRECIO_FILES = {
    "wind":  "precio_eolico_mwh.csv",
    "solar": "precio_solar_mwh.csv",
}


@app.get("/api/results/precios/<source>")
def get_precios(source: str):
    """
    Returns EUR/MWh prices generated by generar_precios_agentes.py.
    ?n=N  limits the number of rows.
    """
    if source not in _PRECIO_FILES:
        abort(404)
    nrows = request.args.get("n", type=int)
    path  = DATA_DIR / "processed" / "Precios" / _PRECIO_FILES[source]
    data  = _csv(path, sep=";", nrows=nrows)
    return jsonify({"source": source, "records": len(data), "data": data})


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------
@app.get("/api/results/metrics")
def get_metrics():
    """Returns the algorithm quality metrics from metricas_calidad.json."""
    return jsonify(_load_metrics())


# ---------------------------------------------------------------------------
# Pareto plot endpoint (serve PNG by algorithm name)
# ---------------------------------------------------------------------------
_PARETO_PLOT_MAP = {
    "nsgaii": "pareto_nsgaii.png",
    "nsga2":  "pareto_nsgaii.png",
    "spea2":  "pareto_spea2.png",
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
# Dashboard HTML
# ---------------------------------------------------------------------------
_DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Microgrid Multi-Agent — Dashboard</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: Arial, sans-serif; background: #f0f2f5; color: #222; }

    /* ---- Header ---- */
    header {
      background: #1e3a5f; color: white; padding: 20px 32px;
      display: flex; align-items: center; justify-content: space-between;
    }
    header h1 { font-size: 22px; }
    header .subtitle { font-size: 13px; margin-top: 4px; opacity: .8; }
    .badge {
      background: #4f9eff; border-radius: 12px;
      font-size: 12px; padding: 4px 12px; font-weight: bold;
    }
    .badge.err { background: #c0392b; }

    /* ---- Layout ---- */
    main { max-width: 1200px; margin: 32px auto; padding: 0 20px; }
    .section { margin-bottom: 48px; }
    h2 {
      font-size: 18px; color: #1e3a5f; margin-bottom: 16px;
      border-left: 4px solid #4f9eff; padding-left: 10px;
    }

    /* ---- Study cards ---- */
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
      gap: 16px;
    }
    .card {
      background: white; border-radius: 10px; padding: 20px;
      box-shadow: 0 1px 6px rgba(0,0,0,0.08);
    }
    .card h3 { font-size: 15px; color: #1e3a5f; margin-bottom: 10px; word-break: break-word; }
    .card .meta { font-size: 13px; color: #555; line-height: 2; }
    .pill {
      display: inline-block; border-radius: 12px; padding: 2px 10px;
      font-size: 12px; font-weight: bold; color: white; background: #4f9eff;
    }
    .hv-value { color: #1a7f3c; font-weight: bold; }

    /* ---- Tables ---- */
    .table-wrap { overflow-x: auto; }
    table {
      width: 100%; border-collapse: collapse; background: white;
      border-radius: 10px; overflow: hidden;
      box-shadow: 0 1px 6px rgba(0,0,0,0.08);
      font-size: 13px;
    }
    thead { background: #1e3a5f; color: white; }
    th, td { padding: 9px 13px; text-align: left; }
    tbody tr:nth-child(even) { background: #f7f9fc; }
    tbody tr:hover { background: #e8f0fb; }

    /* ---- Metrics table highlight ---- */
    .best-val { color: #1a7f3c; font-weight: bold; }

    /* ---- Tabs ---- */
    .tabs { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
    .tab-btn {
      padding: 8px 18px; border: none; border-radius: 6px; cursor: pointer;
      font-size: 14px; background: #dde3ed; color: #333; transition: background .15s;
    }
    .tab-btn.active { background: #1e3a5f; color: white; }
    .tab-pane { display: none; }
    .tab-pane.active { display: block; }

    /* ---- Pareto plots ---- */
    .plot-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(480px, 1fr));
      gap: 20px;
    }
    .plot-card {
      background: white; border-radius: 10px; padding: 16px;
      box-shadow: 0 1px 6px rgba(0,0,0,0.08);
    }
    .plot-card h3 { font-size: 14px; color: #1e3a5f; margin-bottom: 10px; }
    .plot-card img { width: 100%; border-radius: 6px; }
    .plot-missing { color: #999; font-style: italic; font-size: 13px; }

    .empty { color: #999; font-style: italic; font-size: 14px; padding: 10px 0; }
  </style>
</head>
<body>

<header>
  <div>
    <h1>Multi-Objective Optimization for Energy Dispatch</h1>
  </div>
  {% if db_ok %}
    <span class="badge">DB OK</span>
  {% else %}
    <span class="badge err">DB not found</span>
  {% endif %}
</header>

<main>

  <!-- ===================== OPTIMIZATION STUDIES ===================== -->
  <div class="section">
    <h2>Optimization Studies (Optuna)</h2>
    {% if studies %}
      <div class="cards">
        {% for s in studies %}
        <div class="card">
          <h3>{{ s.name }}</h3>
          <div class="meta">
            Trials: <span class="pill">{{ s.n_trials }}</span><br>
            {% if s.hypervolume is not none %}
              Best Normalized Hypervolume:
              <span class="hv-value">{{ "%.6f"|format(s.hypervolume) }}</span><br>
            {% else %}
              Hypervolume: <span style="color:#aaa">—</span><br>
            {% endif %}
            {% if s.start_date %}
              <span style="color:#aaa; font-size:12px">{{ s.start_date[:19] }}</span>
            {% endif %}
          </div>
        </div>
        {% endfor %}
      </div>
    {% else %}
      <p class="empty">No studies registered yet.</p>
    {% endif %}
  </div>

  <!-- ===================== ALGORITHM METRICS ===================== -->
  <div class="section">
    <h2>Algorithm Quality Metrics</h2>
    {% if metrics %}
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Algorithm</th>
              <th>HV (raw)</th>
              <th>HV (normalized)</th>
              <th>GD</th>
              <th>IGD</th>
              <th>Epsilon</th>
              <th>Spread</th>
              <th>Solutions</th>
              <th>Time (s)</th>
            </tr>
          </thead>
          <tbody>
            {% for algo, m in metrics.items() %}
            <tr>
              <td><strong>{{ algo }}</strong></td>
              <td>{{ "%.2f"|format(m.get("HV_raw", 0)) if m.get("HV_raw") is not none else "—" }}</td>
              <td class="{% if m.get('HV_normalizado') == best_hv %}best-val{% endif %}">
                {{ "%.6f"|format(m.get("HV_normalizado", 0)) if m.get("HV_normalizado") is not none else "—" }}
              </td>
              <td class="{% if m.get('GD') == best_gd %}best-val{% endif %}">
                {{ "%.4f"|format(m.get("GD", 0)) if m.get("GD") is not none else "—" }}
              </td>
              <td class="{% if m.get('IGD') == best_igd %}best-val{% endif %}">
                {{ "%.4f"|format(m.get("IGD", 0)) if m.get("IGD") is not none else "—" }}
              </td>
              <td class="{% if m.get('epsilon') == best_epsilon %}best-val{% endif %}">
                {{ "%.2f"|format(m.get("epsilon", 0)) if m.get("epsilon") is not none else "—" }}
              </td>
              <td class="{% if m.get('Spread') == best_spread %}best-val{% endif %}">
                {{ "%.4f"|format(m.get("Spread", 0)) if m.get("Spread") is not none else "—" }}
              </td>
              <td>{{ m.get("n_soluciones", "—") }}</td>
              <td>{{ "%.1f"|format(m.get("elapsed_s", 0)) if m.get("elapsed_s") is not none else "—" }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
        <p style="font-size:12px; color:#888; margin-top:8px;">
          <span style="color:#1a7f3c; font-weight:bold;">Green</span> = best value per metric
          (HV↑ · GD↓ · IGD↓ · Epsilon↓ · Spread↓)
        </p>
      </div>
    {% else %}
      <p class="empty">No metrics file found (metricas_calidad.json).</p>
    {% endif %}
  </div>

  <!-- ===================== PARETO FRONTS ===================== -->
  <div class="section">
    <h2>Pareto Fronts</h2>
    <div class="plot-grid">

      {% for label, img in pareto_plots.items() %}
      <div class="plot-card">
        <h3>{{ label }}</h3>
        {% if img %}
          <img src="{{ img }}" alt="Pareto front — {{ label }}">
        {% else %}
          <p class="plot-missing">Plot not available.</p>
        {% endif %}
      </div>
      {% endfor %}

    </div>
  </div>

  <!-- ===================== POWER PREDICTIONS ===================== -->
  <div class="section">
    <h2>Power Predictions</h2>
    <div class="tabs">
      <button class="tab-btn active" onclick="switchTab(event,'pred','pred-wind')">Wind</button>
      <button class="tab-btn"        onclick="switchTab(event,'pred','pred-solar')">Solar</button>
    </div>

    <div id="pred-wind" class="tab-pane active">
      {% if pred_wind %}
        <div class="table-wrap">{{ pred_wind|safe }}</div>
      {% else %}
        <p class="empty">No wind prediction data available.</p>
      {% endif %}
    </div>

    <div id="pred-solar" class="tab-pane">
      {% if pred_solar %}
        <div class="table-wrap">{{ pred_solar|safe }}</div>
      {% else %}
        <p class="empty">No solar prediction data available.</p>
      {% endif %}
    </div>
  </div>

  <!-- ===================== AGENT PRICES ===================== -->
  <div class="section">
    <h2>Agent Prices (EUR/MWh)</h2>
    <div class="tabs">
      <button class="tab-btn active" onclick="switchTab(event,'prec','prec-wind')">Wind</button>
      <button class="tab-btn"        onclick="switchTab(event,'prec','prec-solar')">Solar</button>
    </div>

    <div id="prec-wind" class="tab-pane active">
      {% if prec_wind %}
        <div class="table-wrap">{{ prec_wind|safe }}</div>
      {% else %}
        <p class="empty">No wind price data available.</p>
      {% endif %}
    </div>

    <div id="prec-solar" class="tab-pane">
      {% if prec_solar %}
        <div class="table-wrap">{{ prec_solar|safe }}</div>
      {% else %}
        <p class="empty">No solar price data available.</p>
      {% endif %}
    </div>
  </div>

</main>

<script>
function switchTab(e, group, targetId) {
  // hide all panes in this section
  const section = e.currentTarget.closest('.section');
  section.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  section.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(targetId).classList.add('active');
  e.currentTarget.classList.add('active');
}
</script>
</body>
</html>
"""


def _df_to_html(df: pd.DataFrame, nrows: int = 50) -> str:
    return (
        df.head(nrows)
        .to_html(index=False, border=0, classes="", na_rep="—")
        .replace('<table ', '<table style="width:100%" ')
    )


def _pred_to_html(
    df: pd.DataFrame,
    feat_cols: list[str],
    pred_col: str | None,
    nrows: int = 50,
) -> str:
    """
    Render a merged features+prediction DataFrame as HTML.
    Feature columns get a light-blue header; the prediction column gets a green header.
    """
    df = df.head(nrows)
    feat_set = set(feat_cols)

    # Build <thead>
    headers = []
    for col in df.columns:
        if col == pred_col:
            style = 'style="background:#1a7f3c; white-space:nowrap;"'
        elif col in feat_set:
            style = 'style="background:#2c5f8a; white-space:nowrap;"'
        else:
            style = 'style="white-space:nowrap;"'
        headers.append(f"<th {style}>{col}</th>")

    rows = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            val_str = "—" if pd.isna(val) else str(val)
            if col == pred_col:
                cells.append(f'<td style="font-weight:bold; color:#1a7f3c;">{val_str}</td>')
            else:
                cells.append(f"<td>{val_str}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    html = (
        '<div style="overflow-x:auto;">'
        '<table style="width:100%">'
        "<thead><tr>" + "".join(headers) + "</tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table>"
        "</div>"
    )
    return html


def _load_csv_df(path: Path, sep: str = ",") -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, sep=sep)


def _best_metric(metrics: dict, key: str, maximize: bool = True):
    """Return the best value across all algorithms for a given metric key."""
    vals = [m.get(key) for m in metrics.values() if m.get(key) is not None]
    if not vals:
        return None
    return max(vals) if maximize else min(vals)


@app.get("/")
def dashboard():
    # --- Optuna studies ---
    studies = []
    db_ok   = OPTUNA_DB.exists()
    metrics = _load_metrics()

    if db_ok:
        try:
            for s in optuna.get_all_study_summaries(storage=_storage_url()):
                algo_key = _guess_algo_key(s.study_name, metrics)
                hv = metrics.get(algo_key, {}).get("HV_normalizado") if algo_key else None
                studies.append({
                    "name":        s.study_name,
                    "n_trials":    s.n_trials,
                    "start_date":  str(s.datetime_start) if s.datetime_start else None,
                    "hypervolume": hv,
                })
        except Exception:
            pass

    # --- Best values per metric (for table highlighting) ---
    best_hv     = _best_metric(metrics, "HV_normalizado", maximize=True)
    best_gd     = _best_metric(metrics, "GD",             maximize=False)
    best_igd    = _best_metric(metrics, "IGD",            maximize=False)
    best_spread = _best_metric(metrics, "Spread",         maximize=False)

    # --- Pareto plots (embedded as base64) ---
    pareto_plots = {
        "NSGA-II":      _img_b64(PLOTS_DIR / "pareto_nsgaii.png"),
        "SPEA2":        _img_b64(PLOTS_DIR / "pareto_spea2.png"),
        "Comparative":  _img_b64(PLOTS_DIR / "pareto_comparativo.png"),
    }

    # --- Predictions merged with input features ---
    df_pw, feat_cols_w, pred_col_w = _merge_features_predictions("wind")
    df_ps, feat_cols_s, pred_col_s = _merge_features_predictions("solar")

    # --- Prices ---
    df_cw = _load_csv_df(DATA_DIR / "processed" / "Precios" / "precio_eolico_mwh.csv", sep=";")
    df_cs = _load_csv_df(DATA_DIR / "processed" / "Precios" / "precio_solar_mwh.csv",  sep=";")

    return render_template_string(
        _DASHBOARD_HTML,
        db_ok        = db_ok,
        studies      = studies,
        metrics      = metrics,
        best_hv      = best_hv,
        best_gd      = best_gd,
        best_igd     = best_igd,
        best_spread  = best_spread,
        pareto_plots = pareto_plots,
        pred_wind    = _pred_to_html(df_pw, feat_cols_w, pred_col_w) if df_pw is not None else None,
        pred_solar   = _pred_to_html(df_ps, feat_cols_s, pred_col_s) if df_ps is not None else None,
        prec_wind    = _df_to_html(df_cw) if df_cw is not None else None,
        prec_solar   = _df_to_html(df_cs) if df_cs is not None else None,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)