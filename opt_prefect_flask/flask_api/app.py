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
from flask import Flask, abort, jsonify, request, send_file

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
_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Microgrid Real-time Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: Arial, sans-serif; background: #f0f2f5; color: #222; }

    header {
      background: #1e3a5f; color: white; padding: 14px 28px;
      display: flex; align-items: center; justify-content: space-between; gap: 16px;
    }
    header h1 { font-size: 18px; white-space: nowrap; }
    header .sub { font-size: 11px; opacity: .7; margin-top: 3px; }

    .status-bar { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    .badge {
      border-radius: 12px; font-size: 11px; padding: 3px 10px;
      font-weight: bold; white-space: nowrap;
    }
    .badge.blue  { background: #4f9eff; color: white; }
    .badge.green { background: #1a7f3c; color: white; }
    .badge.red   { background: #c0392b; color: white; }
    .badge.gray  { background: #555; color: white; }
    #countdown   { font-size: 11px; opacity: .65; color: white; }
    #dot { width:8px; height:8px; border-radius:50%; background:#1a7f3c;
           display:inline-block; animation: pulse 2s infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.25} }

    main { max-width: 1400px; margin: 20px auto; padding: 0 18px; }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-bottom: 18px; }
    .card {
      background: white; border-radius: 10px; padding: 18px;
      box-shadow: 0 1px 6px rgba(0,0,0,.08);
    }
    .card h3 { font-size: 13px; color: #1e3a5f; margin-bottom: 12px; font-weight: bold; }
    canvas { max-height: 260px; width: 100% !important; }
    #prices-canvas { max-height: 200px; }

    h2 {
      font-size: 15px; color: #1e3a5f; margin-bottom: 10px;
      border-left: 4px solid #4f9eff; padding-left: 9px;
    }
    table {
      width: 100%; border-collapse: collapse; font-size: 12px;
      background: white; border-radius: 8px; overflow: hidden;
      box-shadow: 0 1px 6px rgba(0,0,0,.08);
    }
    thead { background: #1e3a5f; color: white; }
    th, td { padding: 7px 11px; text-align: left; }
    tbody tr:nth-child(even) { background: #f7f9fc; }
    .best-val { color: #1a7f3c; font-weight: bold; }
    .empty { color: #999; font-style: italic; font-size: 13px; padding: 8px 0; }
    .note { font-size: 11px; color: #888; margin-top: 6px; }
  </style>
</head>
<body>

<header>
  <div>
    <h1>Microgrid Multi-Agent Optimization</h1>
    <div class="sub">NSGA-II &amp; SPEA2 · Sliding 24-hour window · Optuna best params</div>
  </div>
  <div class="status-bar">
    <span id="dot"></span>
    <span id="badge-window" class="badge blue">Window —</span>
    <span id="badge-runs"   class="badge green">Runs: 0</span>
    <span id="badge-last"   class="badge gray">Last run: —</span>
    <span id="badge-time"   class="badge gray">Duration: —</span>
    <span id="countdown">Refresh in 15s</span>
  </div>
</header>

<main>

  <!-- Row 1: Pareto scatter + Execution times bar -->
  <div class="grid-2">
    <div class="card">
      <h3>Pareto Fronts — Current Window (Economic Cost vs Grid Energy)</h3>
      <canvas id="pareto-canvas"></canvas>
    </div>
    <div class="card">
      <h3>Algorithm Execution Times per Run (seconds)</h3>
      <canvas id="times-canvas"></canvas>
    </div>
  </div>

  <!-- Row 2: Predictions -->
  <div class="grid-2">
    <div class="card">
      <h3>Wind Power Predictions (kWh/h)</h3>
      <canvas id="wind-canvas"></canvas>
    </div>
    <div class="card">
      <h3>Solar Power Predictions (kWh/h)</h3>
      <canvas id="solar-canvas"></canvas>
    </div>
  </div>

  <!-- Row 3: Agent prices (full width) -->
  <div class="card" style="margin-bottom:18px;">
    <h3>Agent Prices (EUR/MWh) — Wind &amp; Solar</h3>
    <canvas id="prices-canvas"></canvas>
  </div>

  <!-- Row 4: Tables -->
  <div class="grid-2" style="align-items:start;">
    <div>
      <h2>Optimization Studies (Optuna)</h2>
      <div id="studies-table"><p class="empty">Loading…</p></div>
    </div>
    <div>
      <h2>Algorithm Quality Metrics</h2>
      <div id="metrics-table"><p class="empty">Loading…</p></div>
    </div>
  </div>

</main>

<script>
// ---------------------------------------------------------------------------
// Chart instances
// ---------------------------------------------------------------------------
let paretoChart, timesChart, windChart, solarChart, pricesChart;
const COLORS = { NSGAII: 'rgba(79,158,255,0.85)', SPEA2: 'rgba(255,127,14,0.85)' };
const REFRESH_MS = 15000;
let countdown = REFRESH_MS / 1000;

// ---------------------------------------------------------------------------
// Fetch helper
// ---------------------------------------------------------------------------
async function get(url) {
  try {
    const r = await fetch(url);
    return r.ok ? r.json() : null;
  } catch { return null; }
}

// ---------------------------------------------------------------------------
// Chart factory helpers
// ---------------------------------------------------------------------------
function scatter(id) {
  return new Chart(document.getElementById(id), {
    type: 'scatter',
    data: { datasets: [] },
    options: {
      animation: false,
      plugins: { legend: { position: 'top', labels: { font: { size: 11 } } } },
      scales: {
        x: { title: { display: true, text: 'Economic Cost (€)', font: { size: 11 } } },
        y: { title: { display: true, text: 'Grid Energy (kWh)', font: { size: 11 } } },
      }
    }
  });
}

function bar(id, yLabel) {
  return new Chart(document.getElementById(id), {
    type: 'bar',
    data: {
      labels: [],
      datasets: [
        { label: 'NSGA-II', data: [], backgroundColor: COLORS.NSGAII },
        { label: 'SPEA2',   data: [], backgroundColor: COLORS.SPEA2  },
      ]
    },
    options: {
      animation: false,
      plugins: { legend: { position: 'top', labels: { font: { size: 11 } } } },
      scales: {
        x: { ticks: { font: { size: 10 } } },
        y: { title: { display: true, text: yLabel, font: { size: 11 } }, beginAtZero: true }
      }
    }
  });
}

function line(id, yLabel, datasets) {
  return new Chart(document.getElementById(id), {
    type: 'line',
    data: { labels: [], datasets },
    options: {
      animation: false,
      plugins: { legend: { position: 'top', labels: { font: { size: 11 } } } },
      elements: { point: { radius: 0 } },
      scales: {
        x: { ticks: { maxTicksLimit: 10, font: { size: 9 } } },
        y: { title: { display: true, text: yLabel, font: { size: 11 } } }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Init charts on load
// ---------------------------------------------------------------------------
window.addEventListener('load', () => {
  paretoChart = scatter('pareto-canvas');

  timesChart = bar('times-canvas', 'Seconds');

  windChart = line('wind-canvas', 'kWh/h', [
    { label: 'Wind prediction', data: [],
      borderColor: '#4f9eff', backgroundColor: 'rgba(79,158,255,.1)',
      fill: true, tension: 0.3, borderWidth: 1.5 }
  ]);

  solarChart = line('solar-canvas', 'kWh/h', [
    { label: 'Solar prediction', data: [],
      borderColor: '#f6c90e', backgroundColor: 'rgba(246,201,14,.1)',
      fill: true, tension: 0.3, borderWidth: 1.5 }
  ]);

  pricesChart = line('prices-canvas', 'EUR/MWh', [
    { label: 'Wind (€/MWh)',  data: [], borderColor: '#4f9eff', tension: 0.3, borderWidth: 1.5 },
    { label: 'Solar (€/MWh)', data: [], borderColor: '#f6c90e', tension: 0.3, borderWidth: 1.5 },
  ]);

  refreshAll();
  setInterval(refreshAll, REFRESH_MS);
  setInterval(() => {
    countdown = Math.max(0, countdown - 1);
    document.getElementById('countdown').textContent = `Refresh in ${countdown}s`;
  }, 1000);
});

// ---------------------------------------------------------------------------
// Main refresh cycle
// ---------------------------------------------------------------------------
async function refreshAll() {
  countdown = REFRESH_MS / 1000;

  const [status, pareto, times, wind, solar, priceWind, priceSolar, studies, metrics] =
    await Promise.all([
      get('/api/status'),
      get('/api/results/pareto-data'),
      get('/api/results/execution-times'),
      get('/api/results/predicciones/wind?n=120'),
      get('/api/results/predicciones/solar?n=120'),
      get('/api/results/precios/wind?n=120'),
      get('/api/results/precios/solar?n=120'),
      get('/api/optuna/studies'),
      get('/api/results/metrics'),
    ]);

  updateStatus(status);
  updatePareto(pareto);
  updateTimes(times);
  updatePrediction(windChart,  wind,  wind?.prediction_col);
  updatePrediction(solarChart, solar, solar?.prediction_col);
  updatePrices(priceWind, priceSolar);
  renderStudies(studies);
  renderMetrics(metrics);
}

// ---------------------------------------------------------------------------
// Status bar
// ---------------------------------------------------------------------------
function updateStatus(s) {
  if (!s) return;
  const sh = s.last_start_hour ?? 0;
  document.getElementById('badge-window').textContent = `Window h${sh}–h${sh + 24}`;
  document.getElementById('badge-runs').textContent   = `Runs: ${s.runs_completed ?? 0}`;
  if (s.last_run) {
    const t = new Date(s.last_run);
    document.getElementById('badge-last').textContent = `Last: ${t.toLocaleTimeString()}`;
  }
  if (s.last_elapsed_s != null) {
    document.getElementById('badge-time').textContent = `Duration: ${s.last_elapsed_s}s`;
  }
}

// ---------------------------------------------------------------------------
// Pareto scatter
// ---------------------------------------------------------------------------
function updatePareto(data) {
  if (!data || !data.fronts) return;
  paretoChart.data.datasets = Object.entries(data.fronts).map(([algo, pts]) => ({
    label: algo,
    data:  pts.map(p => ({ x: +p[0].toFixed(1), y: +p[1].toFixed(1) })),
    backgroundColor: COLORS[algo] ?? '#999',
    pointRadius: 5,
  }));
  paretoChart.update('none');
}

// ---------------------------------------------------------------------------
// Execution times bar
// ---------------------------------------------------------------------------
function updateTimes(data) {
  if (!data || !data.length) return;
  const last = data.slice(-12);
  timesChart.data.labels = last.map((d, i) => {
    const n = data.length - last.length + i + 1;
    return `#${n} h${d.window_start_hour ?? 0}`;
  });
  timesChart.data.datasets[0].data = last.map(d => +(d.NSGAII ?? 0).toFixed(1));
  timesChart.data.datasets[1].data = last.map(d => +(d.SPEA2  ?? 0).toFixed(1));
  timesChart.update('none');
}

// ---------------------------------------------------------------------------
// Prediction line charts
// ---------------------------------------------------------------------------
function updatePrediction(chart, data, predCol) {
  if (!data || !data.data || !data.data.length || !predCol) return;
  const rows   = data.data;
  const labels = rows.map((r, i) => r.Date ? String(r.Date).slice(5, 16) : String(i));
  chart.data.labels = labels;
  chart.data.datasets[0].data = rows.map(r => r[predCol] ?? null);
  chart.update('none');
}

// ---------------------------------------------------------------------------
// Prices line chart
// ---------------------------------------------------------------------------
function updatePrices(wind, solar) {
  if (!wind || !wind.data) return;
  const rows = wind.data;
  const labels = rows.map((_, i) => String(i + 1));
  pricesChart.data.labels = labels;
  pricesChart.data.datasets[0].data = rows.map(r => r['precio_eur_mwh'] ?? null);
  if (solar && solar.data) {
    pricesChart.data.datasets[1].data = solar.data.map(r => r['precio_eur_mwh'] ?? null);
  }
  pricesChart.update('none');
}

// ---------------------------------------------------------------------------
// Studies table
// ---------------------------------------------------------------------------
function renderStudies(studies) {
  const el = document.getElementById('studies-table');
  if (!studies || !studies.length) {
    el.innerHTML = '<p class="empty">No studies registered yet.</p>';
    return;
  }
  const rows = studies.map(s => `
    <tr>
      <td><strong>${s.name}</strong></td>
      <td>${s.n_trials}</td>
      <td>${s.hypervolume != null ? s.hypervolume.toFixed(6) : '—'}</td>
      <td style="color:#aaa;font-size:11px;">${s.start_date ? s.start_date.slice(0,19) : '—'}</td>
    </tr>`).join('');
  el.innerHTML = `
    <table>
      <thead><tr><th>Study</th><th>Trials</th><th>HV (norm.)</th><th>Started</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

// ---------------------------------------------------------------------------
// Metrics table
// ---------------------------------------------------------------------------
function renderMetrics(metrics) {
  const el = document.getElementById('metrics-table');
  if (!metrics || !Object.keys(metrics).length) {
    el.innerHTML = '<p class="empty">No metrics file found (metricas_calidad.json).</p>';
    return;
  }
  const vals = Object.values(metrics);
  const bestHV     = Math.max(...vals.map(m => m.HV_normalizado ?? -Infinity));
  const bestGD     = Math.min(...vals.map(m => m.GD     ?? Infinity));
  const bestIGD    = Math.min(...vals.map(m => m.IGD    ?? Infinity));
  const bestSpread = Math.min(...vals.map(m => m.Spread ?? Infinity));

  const rows = Object.entries(metrics).map(([algo, m]) => `
    <tr>
      <td><strong>${algo}</strong></td>
      <td class="${m.HV_normalizado === bestHV ? 'best-val' : ''}">
        ${m.HV_normalizado != null ? m.HV_normalizado.toFixed(6) : '—'}</td>
      <td class="${m.GD === bestGD ? 'best-val' : ''}">
        ${m.GD != null ? m.GD.toFixed(4) : '—'}</td>
      <td class="${m.IGD === bestIGD ? 'best-val' : ''}">
        ${m.IGD != null ? m.IGD.toFixed(4) : '—'}</td>
      <td class="${m.Spread === bestSpread ? 'best-val' : ''}">
        ${m.Spread != null ? m.Spread.toFixed(4) : '—'}</td>
      <td>${m.n_soluciones ?? '—'}</td>
      <td>${m.elapsed_s != null ? m.elapsed_s.toFixed(1) : '—'}</td>
    </tr>`).join('');

  el.innerHTML = `
    <table>
      <thead>
        <tr><th>Algorithm</th><th>HV↑</th><th>GD↓</th><th>IGD↓</th>
            <th>Spread↓</th><th>Solutions</th><th>Time(s)</th></tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
    <p class="note"><span style="color:#1a7f3c;font-weight:bold;">Green</span>
      = best value per metric (HV↑ · GD↓ · IGD↓ · Spread↓)</p>`;
}
</script>
</body>
</html>"""


@app.get("/")
def dashboard():
    return _DASHBOARD_HTML, 200, {"Content-Type": "text/html; charset=utf-8"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
