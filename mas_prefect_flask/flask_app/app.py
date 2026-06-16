from flask import (
    Flask,
    render_template,
    Response,
    stream_with_context,
    request
)

import json


app = Flask(__name__)

# ==================================================
# PÁGINA PRINCIPAL (única pantalla: negociación)
# ==================================================

@app.route("/")
def negotiation():

    return render_template("negotiation.html")

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

    if mode not in ["competitive", "cooperative", "negotiation"]:
        mode = "negotiation"

    try:
        n_episodes = int(request.args.get("n_episodes", 1))
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
# MAIN
# ==================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )
