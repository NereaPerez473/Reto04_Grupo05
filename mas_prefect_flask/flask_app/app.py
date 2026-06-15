from flask import (
    Flask,
    render_template,
    send_file,
    Response,
    stream_with_context
)

from pathlib import Path
import json
import time

from prefect_pipeline import evaluate_flow


app = Flask(__name__)

# ==================================================
# RUTAS
# ==================================================

BASE_DIR = Path("/project")

PLOTS_DIR = (
    BASE_DIR
    / "mas_qlearning_battery"
    / "results"
    / "plots"
)

# ==================================================
# ESTADO SIMPLE DEL DASHBOARD
# ==================================================

last_results = {
    "grid_energy": "-",
    "battery_soc": "-",
    "renewable_energy": "-",
    "renewable_share": "-",
    "mode": "competitive",
    "consumer_cost": "-",
    "solar_revenue": "-",
    "wind_revenue": "-",
    "solar_strategy": "-",
    "wind_strategy": "-",
    "task_times": {}
}

# ==================================================
# DASHBOARD
# ==================================================

@app.route("/")
def dashboard():

    return render_template(
        "dashboard.html",

        grid_energy=last_results["grid_energy"],
        battery_soc=last_results["battery_soc"],
        renewable_energy=last_results["renewable_energy"],
        renewable_share=last_results["renewable_share"],

        consumer_cost=last_results["consumer_cost"],
        solar_revenue=last_results["solar_revenue"],
        wind_revenue=last_results["wind_revenue"],

        solar_strategy=last_results["solar_strategy"],
        wind_strategy=last_results["wind_strategy"],

        mode=last_results["mode"],
        task_times=last_results["task_times"]
    )

# ==================================================
# EJECUTAR FLOW PREFECT (resultado final)
# ==================================================

@app.route("/run/<mode>")
def run_evaluation(mode):

    if mode not in [
        "competitive",
        "cooperative",
        "negotiation"
    ]:
        mode = "competitive"

    result = evaluate_flow(mode)

    last_results.update(result)

    timing_bars = build_timing_bars(
        result.get("task_times", {})
    )

    return render_template(
        "results.html",
        timing_bars=timing_bars,
        **result
    )


def build_timing_bars(task_times):
    """
    Convierte el dict task_times en una lista de dicts
    listos para renderizar en el template, con el
    porcentaje ya calculado en Python.
    """

    if not task_times:
        return []

    total = float(task_times.get("Total Flow") or 1)

    colors = [
        "bg-primary",
        "bg-info",
        "bg-danger",
        "bg-warning",
        "bg-success",
    ]

    bars = []

    for idx, (name, elapsed) in enumerate(task_times.items()):

        if name == "Total Flow":
            continue

        pct = round(elapsed / total * 100, 1) if total else 0

        bars.append({
            "name":    name,
            "elapsed": elapsed,
            "pct":     pct,
            "color":   colors[idx % len(colors)],
            "style":   "width: " + str(pct) + "%",
        })

    return bars

# ==================================================
# SSE: STREAMING EN TIEMPO REAL
# ==================================================

@app.route("/stream/<mode>")
def stream_evaluation(mode):
    """
    Endpoint SSE que emite eventos en tiempo real.
    Parámetros query:
      - n_episodes (int, default 1): número de episodios
    """
    from flask import request as flask_request

    if mode not in ["competitive", "cooperative", "negotiation"]:
        mode = "competitive"

    try:
        n_episodes = int(flask_request.args.get("n_episodes", 1))
        n_episodes = max(1, min(n_episodes, 50))
    except (ValueError, TypeError):
        n_episodes = 1

    def generate():

        from sma_evaluator import run_evaluation_streaming

        for episode in range(1, n_episodes + 1):

            # Marcador de inicio de episodio
            yield f"data: {json.dumps({'type': 'episode_start', 'episode': episode, 'n_episodes': n_episodes})}\n\n"

            for event in run_evaluation_streaming(mode):

                # Inyectar número de episodio en cada evento
                event["episode"]    = episode
                event["n_episodes"] = n_episodes
                yield f"data: {json.dumps(event)}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

# ==================================================
# SERVIDOR DE IMÁGENES
# ==================================================

@app.route("/plot/<filename>")
def plot(filename):

    file_path = PLOTS_DIR / filename

    return send_file(
        file_path,
        mimetype="image/png"
    )

# ==================================================
# COMPETITIVE
# ==================================================

@app.route("/competitive")
def competitive():

    return render_template(
        "competitive.html"
    )

# ==================================================
# COOPERATIVE
# ==================================================

@app.route("/cooperative")
def cooperative():

    return render_template(
        "cooperative.html"
    )

# ==================================================
# NEGOTIATION
# ==================================================

@app.route("/negotiation")
def negotiation():

    return render_template(
        "negotiation.html"
    )

# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )