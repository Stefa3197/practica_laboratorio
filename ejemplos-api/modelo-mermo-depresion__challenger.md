# modelo-mermo-depresion @ challenger

- URI: `models:/modelo-mermo-depresion@challenger`
- Versión: 1
- Endpoint: `POST http://127.0.0.1:5001/invocations`

## Esquema de entrada

```json
[
  {
    "type": "double",
    "name": "days",
    "required": true
  },
  {
    "type": "double",
    "name": "gender",
    "required": true
  },
  {
    "type": "double",
    "name": "afftype",
    "required": true
  },
  {
    "type": "double",
    "name": "melanch",
    "required": true
  },
  {
    "type": "double",
    "name": "inpatient",
    "required": true
  },
  {
    "type": "double",
    "name": "marriage",
    "required": true
  },
  {
    "type": "double",
    "name": "work",
    "required": true
  },
  {
    "type": "double",
    "name": "madrs1",
    "required": true
  },
  {
    "type": "double",
    "name": "age_25-29",
    "required": true
  },
  {
    "type": "double",
    "name": "age_30-34",
    "required": true
  },
  {
    "type": "double",
    "name": "age_35-39",
    "required": true
  },
  {
    "type": "double",
    "name": "age_40-44",
    "required": true
  },
  {
    "type": "double",
    "name": "age_45-49",
    "required": true
  },
  {
    "type": "double",
    "name": "age_50-54",
    "required": true
  },
  {
    "type": "double",
    "name": "age_55-59",
    "required": true
  },
  {
    "type": "double",
    "name": "age_60-64",
    "required": true
  },
  {
    "type": "double",
    "name": "age_65-69",
    "required": true
  },
  {
    "type": "double",
    "name": "edu_11-15",
    "required": true
  },
  {
    "type": "double",
    "name": "edu_16-20",
    "required": true
  },
  {
    "type": "double",
    "name": "edu_6-10",
    "required": true
  }
]
```

La firma se infirió a partir del ejemplo porque esta versión del modelo no la tiene registrada.

## Esquema de salida

```json
[
  {
    "type": "tensor",
    "tensor-spec": {
      "dtype": "int64",
      "shape": [
        -1
      ]
    }
  }
]
```

Los valores de entrada deben estar codificados y escalados con el mismo preprocesamiento del entrenamiento.
La predicción `1` indica que disminuyó la depresión; `0` indica que no disminuyó.

## Petición

```bash
curl http://127.0.0.1:5001/invocations \
  -H 'Content-Type: application/json' \
  -d '{"dataframe_split": {"columns": ["days", "gender", "afftype", "melanch", "inpatient", "marriage", "work", "madrs1", "age_25-29", "age_30-34", "age_35-39", "age_40-44", "age_45-49", "age_50-54", "age_55-59", "age_60-64", "age_65-69", "edu_11-15", "edu_16-20", "edu_6-10"], "data": [[0.09090909090909091, -1.1338934190276817, 0.674199862463242, -3.8729833462074166, -2.0816659994661326, -0.8819171036881969, 0.2581988897471611, 0.8056094727078105, -0.3779644730092272, 3.8729833462074166, -0.3779644730092272, -0.3779644730092272, -0.674199862463242, -0.3779644730092272, 0.0, -0.2581988897471611, -0.2581988897471611, -0.5773502691896258, 0.0, 0.674199862463242]]}}'
```

```json
{
  "dataframe_split": {
    "columns": [
      "days",
      "gender",
      "afftype",
      "melanch",
      "inpatient",
      "marriage",
      "work",
      "madrs1",
      "age_25-29",
      "age_30-34",
      "age_35-39",
      "age_40-44",
      "age_45-49",
      "age_50-54",
      "age_55-59",
      "age_60-64",
      "age_65-69",
      "edu_11-15",
      "edu_16-20",
      "edu_6-10"
    ],
    "data": [
      [
        0.09090909090909091,
        -1.1338934190276817,
        0.674199862463242,
        -3.8729833462074166,
        -2.0816659994661326,
        -0.8819171036881969,
        0.2581988897471611,
        0.8056094727078105,
        -0.3779644730092272,
        3.8729833462074166,
        -0.3779644730092272,
        -0.3779644730092272,
        -0.674199862463242,
        -0.3779644730092272,
        0.0,
        -0.2581988897471611,
        -0.2581988897471611,
        -0.5773502691896258,
        0.0,
        0.674199862463242
      ]
    ]
  }
}
```

## Respuesta

```json
{
  "predictions": [
    1
  ]
}
```
