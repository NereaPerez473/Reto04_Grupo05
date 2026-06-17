# Reto 04 - Gestión Inteligente de Microred Eléctrica

## Contexto y Objetivo
Este proyecto desarrolla un sistema multiagente (MAS) para gestionar una microred eléctrica compuesta por un Agente Solar, un Agente Eólico y un Agente de Consumo. El sistema garantiza el suministro energético y maximiza los beneficios mediante estrategias competitivas y optimización multiobjetivo, integrando además una capa de explicabilidad (xAI).

## Autoras (Grupo 05)
* Alai Urra
* Libe Belasko
* Nerea Perez
* Paola Alvarez

## Tecnologías Utilizadas
* **Orquestación:** Prefect
* **Procesamiento de Datos:** Polars (procesamiento eficiente de datos meteorológicos y de carga)
* **Optimización:** jMetalPy (algoritmos NSGA-II y SPEA2 para optimización del despacho de energía)
* **Sistemas Multiagente:** Comunicación basada en el estándar FIPA-ACL y Q-Learning
* **Explicabilidad:** SHAP, LIME, PDP, ALE, Ceteris Paribus, ...
* **Infraestructura:** Docker (despliegue del pipeline y persistencia)

## Estructura del Repositorio

```
RETO4/
├── data/                        # Datos en bruto y resultados intermedios
│   ├── raw/                     # CSVs meteorológicos, demanda y precios 
│   ├── raw/                     # Datos procesados
│   └── results/                 # Predicciones y resultados de optimización
├── models/                      # Modelos entrenados (.pkl) y adaptador de entorno
├── xAI/                         # Notebooks de explicabilidad global y local
├── optimizacion/                # Problema multiobjetivo, tasks Prefect y DB Optuna
├── mas_qlearning_battery/       # Agentes Q-Learning, estrategias y scripts de entrenamiento
├── mas_prefect_flask/           # Pipeline 1: entrenamiento MAS + UI Flask
│   ├── docker-compose.yml
│   └── flask_app/
│       ├── Dockerfile
│       ├── app.py
│       ├── prefect_pipeline.py
│       └── sma_trainer.py
└── opt_prefect_flask/           # Pipeline 2: ETL → inferencia → optimización + API Flask
    ├── docker-compose.yml
    ├── flask_api/
    │   ├── Dockerfile
    │   └── app.py
    └── pipeline/
        ├── etl.py
        ├── inference.py
        ├── flow.py
        └── scheduler.py
```

## Ejecución

El proyecto tiene **dos pipelines independientes**, cada uno con su propio `docker-compose.yml`. Deben levantarse por separado desde su carpeta correspondiente.

### Requisito previo

```bash
git clone <url-del-repositorio>
cd RETO
```



### Pipeline 1 — Entrenamiento MAS (`mas_prefect_flask/`)
Entrena los agentes Q-Learning y expone una interfaz Flask para lanzar y monitorizar entrenamientos.

**Servicios levantados:**
- `prefect-server` → UI de Prefect en [http://localhost:4200](http://localhost:4200)
- `flask-api` → Interfaz de entrenamiento en [http://localhost:5000](http://localhost:5000)

```bash
cd mas_prefect_flask
docker compose up -d --build
```

Para detener:

```bash
docker compose down
```


### Pipeline 2 — ETL + Inferencia + Optimización (`opt_prefect_flask/`)

Ejecuta el pipeline completo: feature engineering → predicción → generación de precios → optimización multiobjetivo (NSGA-II y SPEA2). Incluye un scheduler con ventana deslizante de 24 h y una API REST para consultar resultados.

**Servicios levantados:**
- `prefect-server` → UI de Prefect en [http://localhost:4200](http://localhost:4200)
- `scheduler` → lanza el pipeline cada 2 minutos avanzando 24 h por ejecución
- `flask-api` → API de resultados en [http://localhost:5000](http://localhost:5000)

```bash
cd opt_prefect_flask
docker compose up -d --build
```

Para lanzar un run puntual sin el scheduler (modo manual):

```bash
docker compose --profile manual run --rm pipeline-worker
```

Para detener:

```bash
docker compose down
```

