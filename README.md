# Reto 04 - Gestión Inteligente de Microred Eléctrica

## Contexto y Objetivo
Este proyecto desarrolla un sistema multiagente (MAS) para gestionar una microred eléctrica compuesta por un Agente Solar, un Agente Eólico y un Agente de Consumo. El sistema garantiza el suministro energético y maximiza los beneficios mediante estrategias competitivas y optimización multiobjetivo, integrando además una capa de explicabilidad (xAI).

## Autoras (Grupo 05)
* Alai Urra
* Libe Velasco
* Nerea Perez
* Paola Alvarez

## Tecnologías Utilizadas
* **Orquestación:** Prefect
* **Procesamiento de Datos:** Polars (procesamiento eficiente de datos meteorológicos y de carga)
* **Optimización:** jMetalPy (algoritmos NSGA-II y SPEA2 para optimización del despacho de energía)
* **Sistemas Multiagente:** Comunicación basada en el estándar FIPA-ACL
* **Infraestructura:** Docker (despliegue del pipeline y persistencia)

## Estructura del Proyecto

Reto04_Grupo05/
│
├── data/
│   ├── raw/                        # CSVs originales sin modificar
│   │   ├── DatosEolicos.csv
│   │   └── DatosSolares.csv
│   └── processed/                  # Series temporales de potencia tras inferencia
│       ├── solar_predictions.csv
│       └── wind_predictions.csv
│
├── models/                         # Modelos entrenados y utilidades de inferencia
│   ├── solar_model/                # Modelo predictor solar (proporcionado)
│   ├── wind_model/                 # Modelo predictor eólico (proporcionado)
│   └── inference.py                # Carga modelos, infiere sobre raw data → processed/
│
├── mas/  
│   ├── agents/                         # Sistema multiagente (FIPA-ACL + negociación)
│   │   ├── fipa_acl.py                 # Serialización/deserialización de mensajes ACL
│   │   ├── base_agent.py               # Clase base con lógica de socket compartida
│   │   ├── solar_agent.py              # Agente Solar: lee processed/, negocia oferta
│   │   ├── wind_agent.py               # Agente Eólico: lee processed/, negocia oferta
│   │   ├── consumer_agent.py           # Agente Consumo: emite demanda energética
│   │   ├── coordinator_agent.py        # Coordinador: gestiona CNP, resuelve conflictos
│   │   ├── microgrid_adapter.py        # Wrapper de pymgrid: expone estado y apply_action()
│   │   ├── microgrid_adapter.py        # Wrapper de pymgrid: expone estado y apply_action()
│   │   └── run_mas.py                  # Entry point: arranca todos los agentes
│   │
│   │
│   └── negotiation/                    # Protocolo y estrategias de negociación
│       ├── contract_net.py             # Implementación del Contract Net Protocol
│       └── strategies.py               # Estrategia honesta vs. estrategia competitiva
│
├── optimization/                   # Optimización multiobjetivo (AE/AS)
│   ├── problem_definition.py       # Definición formal del problema
│   ├── operators.py                # Operadores: selección, cruce, mutación
│   └── fitness.py                  # Función(es) de fitness
│
├── xAI/                            # Explicabilidad de los modelos solar y eólico
│   ├── global_explanations.py      # SHAP summary plots, importancia de features
│   ├── local_explanations.py       # SHAP/LIME para 3 casos locales por agente
│   └── what_if_scenarios.py        # Escenarios hipotéticos (cambios en inputs)
│
├── pipeline/                       # Orquestación con Prefect
│   ├── flows/
│   │   ├── inference_flow.py       # Flow: raw data → inferencia → processed/
│   │   ├── mas_flow.py             # Flow: simulación del sistema multiagente
│   │   └── full_pipeline_flow.py   # Flow maestro que encadena todo
│   └── docker-compose.yml          # Despliegue dockerizado de Prefect + agentes
│
├── notebooks/                      # Solo análisis y visualización, nunca código de producción
│   ├── 01_eda.ipynb                # EDA de DatosEolicos y DatosSolares
│   ├── 02_negotiation_results.ipynb
│   └── 03_strategy_comparison.ipynb
│
├── tests/
│   ├── test_agents.py
│   └── test_negotiation.py
│
├── requirements.txt
├── Dockerfile
└── README.md

## Instrucciones de Ejecución
1. Clonar el repositorio.
2. Construir la imagen Docker: `docker build -t microred-reto4 .`
3. Levantar el contenedor: `docker run -d microred-reto4`
4. Iniciar el servidor local de Prefect: `prefect server start`
5. Ejecutar el flujo principal: `python pipeline/main_flow.py`
