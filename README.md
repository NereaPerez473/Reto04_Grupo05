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
* `/data`: Datos meteorológicos y de potencia simulados.
* `/models`: Modelos predictivos y scripts de xAI (explicaciones locales, globales y escenarios what-if).
* `/optimization`: Definición del problema multiobjetivo, operadores y cálculo de fitness.
* `/agents`: Lógica de negociación y comunicación FIPA-ACL.
* `/pipeline`: Flujos de Prefect que orquestan la ejecución.

## Instrucciones de Ejecución
1. Clonar el repositorio.
2. Construir la imagen Docker: `docker build -t microred-reto4 .`
3. Levantar el contenedor: `docker run -d microred-reto4`
4. Iniciar el servidor local de Prefect: `prefect server start`
5. Ejecutar el flujo principal: `python pipeline/main_flow.py`
