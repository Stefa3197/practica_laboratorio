"""Entrenamiento de clasificadores y registro de experimentos en MLflow."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --- INSERCIÓN OBLIGATORIA PARA RESOLVER EL MÓDULO 'src' ---
RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
# -----------------------------------------------------------

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from src.config import RANDOM_SEED, activar_seguimiento
from src.data import cargar_y_preparar_datos


def entrenar_modelos():
    """Entrena los modelos, calcula el score de train y test, y registra cada run en MLflow."""
    os.environ.pop("MLFLOW_RUN_ID", None)

    # 1. Activar el seguimiento en MLflow
    activar_seguimiento()

    # 2. Cargar y preparar datos
    X_train, X_test, y_train, y_test, _, nombres_columnas = cargar_y_preparar_datos()
    X_train = pd.DataFrame(X_train, columns=nombres_columnas)
    X_test = pd.DataFrame(X_test, columns=nombres_columnas)

    # 3. Diccionario con los 4 algoritmos del notebook original
    modelos = {
        "regresion_logistica": LogisticRegression(random_state=RANDOM_SEED),
        "svc": SVC(random_state=RANDOM_SEED),
        "arbol_decision": DecisionTreeClassifier(random_state=RANDOM_SEED),
        "random_forest": RandomForestClassifier(random_state=RANDOM_SEED),
    }

    print("Iniciando entrenamiento y registro en MLflow...\n")

    # 4. Iterar sobre cada modelo y registrar el Run en MLflow
    for nombre_modelo, modelo in modelos.items():
        with mlflow.start_run(run_name=nombre_modelo):
            # Entrenar modelo
            modelo.fit(X_train, y_train)

            # Evaluar precisión (Score / Accuracy)
            score_train = modelo.score(X_train, y_train)
            score_test = modelo.score(X_test, y_test)

            # Registrar parámetros en MLflow
            mlflow.log_param("model_type", nombre_modelo)
            mlflow.log_param("random_seed", RANDOM_SEED)

            # Registrar métricas en MLflow
            mlflow.log_metric("accuracy_train", score_train)
            mlflow.log_metric("accuracy", score_test)  # Métrica principal

            input_example = X_train.head(1)
            signature = infer_signature(input_example, modelo.predict(input_example))

            # Loguear el modelo entrenado como artefacto de MLflow
            mlflow.sklearn.log_model(
                modelo,
                name="model",
                signature=signature,
                input_example=input_example,
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )

            print(
                f"[{nombre_modelo}] -> Train Accuracy: {score_train:.4f} | Test Accuracy: {score_test:.4f}"
            )

    print("\n¡Entrenamiento finalizado con éxito!")


if __name__ == "__main__":
    entrenar_modelos()