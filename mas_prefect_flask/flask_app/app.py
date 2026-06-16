from flask import (
    Flask,
    render_template,
    Response,
    stream_with_context,
    request,
    redirect,
    url_for,
    jsonify
)

import json
import threading
import os

app = Flask(__name__)

# Prefect UI base URL (override with env var PREFECT_UI_URL if needed)
PREFECT_UI_URL = os.environ.get("PREFECT_UI_URL", "http://localhost:4200")

# ==================================================
# REDIRECCIÓN RAÍZ → TRAINING
# ==================================================

@app.route("/")
def index():
    return redirect(url_for("training"))

@app.route("/training")
def training():
    return render_template("training.html")


# ==================================================
# SSE: ENTRENAMIENTO EN TIEMPO REAL
# ==================================================

@app.route("/train/<mode>")
def stream_training(mode):

    if mode not in ["competitive", "cooperative", "negotiation"]:
        mode = "competitive"

    try:
        n_episodes = int(request.args.get("n_episodes", 200))
        n_episodes = max(1, min(n_episodes, 2000))
    except (ValueError, TypeError):
        n_episodes = 200

    save_qtables = request.args.get("save", "1") != "0"

    # ==========================================
    # LANZAR PREFECT EN PARALELO
    # ==========================================

    def launch_prefect():
        from prefect_pipeline import training_flow

        training_flow(
            mode=mode,
            n_episodes=n_episodes
        )

    threading.Thread(
        target=launch_prefect,
        daemon=True
    ).start()

    # ==========================================
    # SSE NORMAL
    # ==========================================

    def generate():
        from sma_trainer import run_training_streaming

        for event in run_training_streaming(
            mode,
            n_episodes,
            save_qtables
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
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
