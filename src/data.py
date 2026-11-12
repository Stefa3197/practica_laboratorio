"""Carga, preparación y escalado de datos para el dataset scores.csv.

Aplica la lógica de prototipado desarrollada en notebooks/scores.ipynb:
- Genera la variable objetivo 'mermo_depresion' (madrs2 < madrs1).
- Descarta filas sin mediciones para calcular la variable objetivo.
- Elimina 'number' y 'madrs2' (evita data leakage).
- Codifica categóricas con pd.get_dummies.
- Imputa valores faltantes con la mediana.
- Realiza train_test_split estratificado y escala con StandardScaler.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import COLUMNS_TO_DROP, DATA_PATH, RANDOM_SEED, TARGET, TEST_SIZE


def cargar_y_preparar_datos():
    """Carga el dataset scores.csv, aplica transformaciones y devuelve los conjuntos
    de entrenamiento, prueba y el escalador.
    """
    # 1. Cargar datos y conservar solo las filas con objetivo observable
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["madrs1", "madrs2"]).copy()
    if df.empty:
        raise ValueError(
            f"El dataset {DATA_PATH} no contiene filas con madrs1 y madrs2 válidos."
        )

    # 2. Crear la variable objetivo (1 si mejoró la depresión, 0 si no)
    df[TARGET] = (df["madrs2"] < df["madrs1"]).astype(int)

    # 3. Separar características (X) y variable objetivo (y)
    X = df.drop(columns=list(COLUMNS_TO_DROP) + [TARGET])
    y = df[TARGET]

    # 4. Codificar variables categóricas dinámicamente y llenar nulos con la mediana
    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(X.median(numeric_only=True))

    # 5. División del dataset (70% Train / 30% Test con estratificación)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # 6. Escalado de características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns