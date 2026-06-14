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
from flask import Flask, abort, jsonify, render_template_string, request

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
# Dashboard HTML
# ---------------------------------------------------------------------------
_DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Microred Multiagente — Dashboard</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: Arial, sans-serif; background: #f0f2f5; color: #222; }

    header {
      background: #1e3a5f; color: white; padding: 20px 32px;
      display: flex; align-items: center; gap: 16px;
    }
    header h1 { font-size: 22px; }
    header .badge {
      background: #4f9eff; border-radius: 12px;
      font-size: 12px; padding: 3px 10px;
    }

    main { max-width: 1100px; margin: 32px auto; padding: 0 20px; }

    h2 { font-size: 18px; color: #1e3a5f; margin-bottom: 14px;
         border-left: 4px solid #4f9eff; padding-left: 10px; }

    .section { margin-bottom: 40px; }

    /* Tarjetas de estudios */
    .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; }
    .card {
      background: white; border-radius: 10px; padding: 20px;
      box-shadow: 0 1px 6px rgba(0,0,0,0.08);
    }
    .card h3 { font-size: 15px; color: #1e3a5f; margin-bottom: 10px; word-break: break-word; }
    .card .meta { font-size: 13px; color: #555; line-height: 1.8; }
    .pill {
      display: inline-block; border-radius: 12px; padding: 2px 10px;
      font-size: 12px; font-weight: bold; color: white; background: #4f9eff;
    }

    /* Tablas */
    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; background: white;
            border-radius: 10px; overflow: hidden;
            box-shadow: 0 1px 6px rgba(0,0,0,0.08); }
    thead { background: #1e3a5f; color: white; }
    th, td { padding: 10px 14px; text-align: left; font-size: 13px; }
    tbody tr:nth-child(even) { background: #f7f9fc; }
    tbody tr:hover { background: #e8f0fb; }

    .empty { color: #999; font-style: italic; font-size: 14px; padding: 10px 0; }

    /* Tabs */
    .tabs { display: flex; gap: 8px; margin-bottom: 16px; }
    .tab-btn {
      padding: 8px 18px; border: none; border-radius: 6px; cursor: pointer;
      font-size: 14px; background: #dde3ed; color: #333;
    }
    .tab-btn.active { background: #1e3a5f; color: white; }
    .tab-pane { display: none; }
    .tab-pane.active { display: block; }
  </style>
</head>
<body>

<header>
  <div>
    <h1>Microred Multiagente</h1>
    <div style="font-size:13px; margin-top:4px; opacity:.8;">Pipeline ETL · Inferencia · Precios · Optimización</div>
  </div>
  {% if db_ok %}
    <span class="badge">DB OK</span>
  {% else %}
    <span class="badge" style="background:#c0392b;">DB no encontrada</span>
  {% endif %}
</header>

<main>

  <!-- ESTUDIOS OPTUNA -->
  <div class="section">
    <h2>Estudios de Optimización (Optuna)</h2>
    {% if studies %}
      <div class="cards">
        {% for s in studies %}
        <div class="card">
          <h3>{{ s.nombre }}</h3>
          <div class="meta">
            Trials: <span class="pill">{{ s.n_trials }}</span><br>
            {% if s.mejor_valor is not none %}
              Mejor valor: <strong>{{ "%.6f"|format(s.mejor_valor) }}</strong><br>
              Trial #{{ s.mejor_trial }}
            {% else %}
              Multi-objetivo
            {% endif %}
            {% if s.fecha_inicio %}
              <br><span style="color:#aaa">{{ s.fecha_inicio[:19] }}</span>
            {% endif %}
          </div>
        </div>
        {% endfor %}
      </div>
    {% else %}
      <p class="empty">No hay estudios registrados todavía.</p>
    {% endif %}
  </div>

  <!-- PREDICCIONES -->
  <div class="section">
    <h2>Predicciones de Potencia</h2>
    <div class="tabs">
      <button class="tab-btn active" onclick="switchTab(event,'pred','pred-eolico')">Eólico</button>
      <button class="tab-btn"        onclick="switchTab(event,'pred','pred-solar')">Solar</button>
    </div>

    <div id="pred-eolico" class="tab-pane active">
      {% if pred_eolico %}
        <div class="table-wrap">{{ pred_eolico|safe }}</div>
      {% else %}
        <p class="empty">Sin datos de predicción eólica.</p>
      {% endif %}
    </div>

    <div id="pred-solar" class="tab-pane">
      {% if pred_solar %}
        <div class="table-wrap">{{ pred_solar|safe }}</div>
      {% else %}
        <p class="empty">Sin datos de predicción solar.</p>
      {% endif %}
    </div>
  </div>

  <!-- PRECIOS -->
  <div class="section">
    <h2>Precios por Agente (EUR/MWh)</h2>
    <div class="tabs">
      <button class="tab-btn active" onclick="switchTab(event,'prec','prec-eolico')">Eólico</button>
      <button class="tab-btn"        onclick="switchTab(event,'prec','prec-solar')">Solar</button>
    </div>

    <div id="prec-eolico" class="tab-pane active">
      {% if prec_eolico %}
        <div class="table-wrap">{{ prec_eolico|safe }}</div>
      {% else %}
        <p class="empty">Sin datos de precios eólicos.</p>
      {% endif %}
    </div>

    <div id="prec-solar" class="tab-pane">
      {% if prec_solar %}
        <div class="table-wrap">{{ prec_solar|safe }}</div>
      {% else %}
        <p class="empty">Sin datos de precios solares.</p>
      {% endif %}
    </div>
  </div>

</main>

<script>
function switchTab(e, group, targetId) {
  document.querySelectorAll('#' + group + '-eolico, #' + group + '-solar').forEach(p => p.classList.remove('active'));
  e.currentTarget.closest('.section').querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
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


def _load_csv_df(path: Path, sep: str = ",") -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, sep=sep)


@app.get("/")
def dashboard():
    # Estudios Optuna
    studies = []
    db_ok = OPTUNA_DB.exists()
    if db_ok:
        try:
            for s in optuna.get_all_study_summaries(storage=f"sqlite:///{OPTUNA_DB}"):
                entry = {
                    "nombre": s.study_name,
                    "n_trials": s.n_trials,
                    "fecha_inicio": str(s.datetime_start) if s.datetime_start else None,
                    "mejor_trial": None,
                    "mejor_valor": None,
                }
                try:
                    entry["mejor_trial"] = s.best_trial.number
                    entry["mejor_valor"] = s.best_trial.value
                except Exception:
                    pass
                studies.append(entry)
        except Exception:
            pass

    # Predicciones
    df_pe = _load_csv_df(DATA_DIR / "results" / "Predicciones_Eolico.csv")
    df_ps = _load_csv_df(DATA_DIR / "results" / "Predicciones_Solar.csv")
    # Precios
    df_ce = _load_csv_df(DATA_DIR / "processed" / "Precios" / "precio_eolico_mwh.csv", sep=";")
    df_cs = _load_csv_df(DATA_DIR / "processed" / "Precios" / "precio_solar_mwh.csv",  sep=";")

    return render_template_string(
        _DASHBOARD_HTML,
        db_ok=db_ok,
        studies=studies,
        pred_eolico=_df_to_html(df_pe) if df_pe is not None else None,
        pred_solar=_df_to_html(df_ps)  if df_ps is not None else None,
        prec_eolico=_df_to_html(df_ce) if df_ce is not None else None,
        prec_solar=_df_to_html(df_cs)  if df_cs is not None else None,
    )


# ---------------------------------------------------------------------------
# Entrada
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
