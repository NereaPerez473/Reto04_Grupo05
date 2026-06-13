# ============================================================
# Imagen unica usada tanto por pipeline-worker como flask-api.
# Los modelos (.pkl) y los datos (data/) se montan como
# bind mounts en docker-compose.yml, NO se copian aqui.
# ============================================================
FROM python:3.11

WORKDIR /app

# Matplotlib sin display
ENV MPLBACKEND=Agg
# El directorio raiz del proyecto dentro del contenedor
ENV APP_DIR=/app
# Python encuentra los modulos de la raiz y de models/
ENV PYTHONPATH=/app:/app/models:/app/optimizacion

# ---- Dependencias del sistema (necesarias para compilar algunos paquetes) --
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Dependencias Python ---------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Codigo fuente (sin datos ni modelos) ----------------------------------
COPY pipeline/   ./pipeline/
COPY optimizacion/ ./optimizacion/
COPY flask_api/  ./flask_api/

# Directorios que seran llenados por los bind mounts en tiempo de ejecucion
RUN mkdir -p data/raw data/processed data/results models

# Comando por defecto: ejecutar el pipeline.
# docker-compose sobreescribe este CMD para el servicio flask-api.
CMD ["python", "pipeline/flow.py"]
