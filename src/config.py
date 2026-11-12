"""Nombres del experimento y del modelo, y semilla.

`src/` lee estos valores desde aquí. Salen de `.env`; si falta el
archivo, se usan los de `.env.example`.

El objetivo, la partición y la métrica salen del notebook
`notebooks/scores.ipynb`: clasificar `mermo_depresion` y elegir
el run con mayor accuracy.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Ruta raíz del proyecto
RAIZ = Path(__file__).resolve().parents[1]
load_dotenv(RAIZ / ".env")

# Configuración de MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "experimento-mermo-depresion")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "modelo-mermo-depresion")
MLFLOW_MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "challenger")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# Ruta al dataset de scores
_ruta_datos = os.getenv("DATA_PATH", "data/scores.csv")
DATA_PATH = (
    Path(_ruta_datos)
    if Path(_ruta_datos).is_absolute()
    else RAIZ / _ruta_datos
)

# Variable objetivo y métrica de evaluación
TARGET = "mermo_depresion"
TEST_SIZE = 0.3
METRIC = "accuracy"

# 'number' es un identificador y 'madrs2' define TARGET; se excluye para
# evitar fuga de datos.
COLUMNS_TO_DROP = (
    "number",
    "madrs2",
)


def activar_seguimiento():
    """Apunta MLflow al almacén y al experimento de este proyecto."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)