# Predicción de mejoría en depresión con MLflow

Proyecto de práctica con el conjunto `scores.csv`. El objetivo
`mermo_depresion` vale `1` cuando `madrs2 < madrs1`, y `0` en caso contrario.
El flujo prepara los datos, compara cuatro clasificadores y registra en MLflow
el de mayor accuracy bajo el alias `challenger`.

## Estructura

- `src/data.py`: prepara las características, aplica one-hot encoding, imputa
  valores faltantes y escala los datos.
- `src/train.py`: entrena los clasificadores y registra métricas, firma y
  ejemplo de entrada en MLflow.
- `src/register.py`: registra el mejor modelo y asigna el alias configurado.
- `src/predict.py`: carga el alias y realiza una predicción local.
- `src/ejemplo_api.py`: genera un ejemplo de petición y respuesta para la API.
- `notebooks/`: notebooks de exploración y práctica.
- `ejemplos-api/`: ejemplos sanitizados de solicitudes a modelos servidos.
- `evidencias/`: capturas de la interfaz de MLflow.

## Preparación

Usa Python 3.12 o una versión compatible con las dependencias del proyecto.
Instala el entorno y copia la configuración:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

El dataset no se incluye en el repositorio. Coloca una copia autorizada como
`data/scores.csv` o configura `DATA_PATH` en `.env` con su ruta local. El CSV
debe incluir las columnas `madrs1` y `madrs2`, además de las características
usadas por el notebook.

## Entrenar y registrar

Desde PowerShell, en la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe src\train.py
.\.venv\Scripts\python.exe src\register.py
```

MLflow guarda sus metadatos en `mlflow.db` y los artefactos en `mlruns/`.
Esos archivos son locales y están excluidos de Git.

## Servir el modelo

```powershell
.\.venv\Scripts\mlflow.exe --env-file .env models serve `
  -m "models:/modelo-mermo-depresion@challenger" `
  --port 5001 `
  --env-manager local
```

En otra terminal, prueba el endpoint usando `curl.exe` en PowerShell:

```powershell
curl.exe http://127.0.0.1:5001/invocations `
  -H "Content-Type: application/json" `
  -d '{"dataframe_split":{"columns":["days","gender","afftype","melanch","inpatient","marriage","work","madrs1","age_25-29","age_30-34","age_35-39","age_40-44","age_45-49","age_50-54","age_55-59","age_60-64","age_65-69","edu_11-15","edu_16-20","edu_6-10"],"data":[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]}}'
```

La respuesta contiene una predicción: `1` indica mejoría y `0` indica que no
disminuyó la depresión. La fila del ejemplo es sintética; no representa a una
persona ni reemplaza el preprocesamiento indicado por el modelo.

## Datos y archivos locales

No publiques el dataset, archivos `.env`, bases de datos de MLflow, modelos
serializados ni entornos virtuales. Consulta `data/README.md` para las
instrucciones sobre el dataset local.
