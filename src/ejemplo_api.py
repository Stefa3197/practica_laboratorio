"""Genera un documento con la petición y la respuesta de la API de un modelo.

Cada ejecución usa el nombre y el alias que se le pasan. El archivo queda
en `ejemplos-api/`, con esos dos datos en el nombre, así que `challenger`
y otro alias no se pisan.

    python src/ejemplo_api.py
    python src/ejemplo_api.py --modelo modelo-mermo-depresion --alias challenger
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.models import Model, infer_signature
from mlflow.tracking import MlflowClient

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src import config
from src.data import cargar_y_preparar_datos

SALIDA = RAIZ / "ejemplos-api"
PUERTO = 5001


def _uri(modelo: str, alias: str) -> str:
    return f"models:/{modelo}@{alias}"


def _jsonable(valor):
    if isinstance(valor, dict):
        return {clave: _jsonable(item) for clave, item in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_jsonable(item) for item in valor]
    if hasattr(valor, "item"):
        return valor.item()
    return valor


def _payload(ejemplo) -> dict | None:
    if ejemplo is None:
        return None
    if isinstance(ejemplo, pd.DataFrame):
        return {
            "dataframe_split": {
                "columns": list(ejemplo.columns),
                "data": _jsonable(ejemplo.to_numpy().tolist()),
            }
        }
    return _jsonable(ejemplo)


def _cargar_ejemplo(uri: str) -> pd.DataFrame:
    """Obtiene un ejemplo del modelo o construye uno con el preprocesamiento actual."""
    ejemplo = Model.load(uri).load_input_example(uri)
    if isinstance(ejemplo, pd.DataFrame):
        return ejemplo.head(1)

    X_train, _, _, _, _, nombres_columnas = cargar_y_preparar_datos()
    return pd.DataFrame(X_train[:1], columns=nombres_columnas)


def _esquema(firma, lado: str) -> str:
    tensor = getattr(firma, lado, None) if firma is not None else None
    if tensor is None:
        return "No hay firma disponible."
    return json.dumps(tensor.to_dict(), indent=2, ensure_ascii=False)


def generar(modelo: str, alias: str, puerto: int = PUERTO) -> Path:
    """Escribe el documento de esta pareja modelo-alias y devuelve su ruta."""
    config.activar_seguimiento()
    uri = _uri(modelo, alias)
    cliente = MlflowClient(tracking_uri=config.MLFLOW_TRACKING_URI)
    version = cliente.get_model_version_by_alias(modelo, alias)
    info = mlflow.models.get_model_info(uri)
    ejemplo = _cargar_ejemplo(uri)
    payload = _payload(ejemplo)
    modelo_mlflow = mlflow.pyfunc.load_model(uri)
    prediccion = modelo_mlflow.predict(ejemplo)
    predicciones = prediccion.tolist() if hasattr(prediccion, "tolist") else prediccion
    respuesta = {"predictions": _jsonable(predicciones)}
    firma = info.signature or infer_signature(ejemplo, prediccion)

    cuerpo_peticion = (
        json.dumps(payload, indent=2, ensure_ascii=False) if payload is not None else None
    )
    cuerpo_respuesta = (
        json.dumps(respuesta, indent=2, ensure_ascii=False) if respuesta is not None else None
    )
    curl = ""
    if cuerpo_peticion is not None:
        curl = (
            f"curl http://127.0.0.1:{puerto}/invocations \\\n"
            f"  -H 'Content-Type: application/json' \\\n"
            f"  -d '{json.dumps(payload, ensure_ascii=False)}'"
        )

    lineas = [
        f"# {modelo} @ {alias}",
        "",
        f"- URI: `{uri}`",
        f"- Versión: {version.version}",
        f"- Endpoint: `POST http://127.0.0.1:{puerto}/invocations`",
        "",
        "## Esquema de entrada",
        "",
        "```json",
        _esquema(firma, "inputs"),
        "```",
        "",
    ]
    if info.signature is None:
        lineas.extend(
            [
                "La firma se infirió a partir del ejemplo porque esta versión del modelo no la tiene registrada.",
                "",
            ]
        )
    lineas.extend(
        [
            "## Esquema de salida",
            "",
            "```json",
            _esquema(firma, "outputs"),
            "```",
            "",
            "Los valores de entrada deben estar codificados y escalados con el mismo preprocesamiento del entrenamiento.",
            "La predicción `1` indica que disminuyó la depresión; `0` indica que no disminuyó.",
            "",
            "## Petición",
            "",
        ]
    )
    lineas.extend(
        [
            "```bash",
            curl,
            "```",
            "",
            "```json",
            cuerpo_peticion,
            "```",
        ]
    )
    lineas.extend(["", "## Respuesta", ""])
    lineas.extend(["```json", cuerpo_respuesta, "```"])
    lineas.append("")

    SALIDA.mkdir(parents=True, exist_ok=True)
    destino = SALIDA / f"{modelo}__{alias}.md"
    destino.write_text("\n".join(lineas), encoding="utf-8")
    return destino


def main():
    parser = argparse.ArgumentParser(
        description="Genera el ejemplo de petición y respuesta de un modelo registrado."
    )
    parser.add_argument(
        "--modelo",
        default=config.MLFLOW_MODEL_NAME,
        help="Nombre en el registro. Por defecto, MLFLOW_MODEL_NAME.",
    )
    parser.add_argument(
        "--alias",
        action="append",
        default=None,
        help="Alias a documentar. Se puede repetir. Por defecto, MLFLOW_MODEL_ALIAS.",
    )
    parser.add_argument("--puerto", type=int, default=PUERTO, help="Puerto del curl de ejemplo.")
    args = parser.parse_args()
    alias = args.alias or [config.MLFLOW_MODEL_ALIAS]
    for nombre in alias:
        destino = generar(args.modelo, nombre, args.puerto)
        print(destino)


if __name__ == "__main__":
    main()
