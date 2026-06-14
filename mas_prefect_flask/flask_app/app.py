from flask import (
    Flask,
    render_template,
    send_file
)

from pathlib import Path

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
    "wind_strategy": "-"
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

    mode=last_results["mode"]
)

# ==================================================
# EJECUTAR FLOW PREFECT
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

    return render_template(
        "results.html",
        **result
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
        debug=False
    )