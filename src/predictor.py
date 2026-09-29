"""Predicciones con el modelo guardado (la parte de inferencia que pide el enunciado).

Aquí no se entrena ni se evalúa: se carga el Pipeline de models/ y se predicen reservas
nuevas sin reentrenar nada. La demo predice unas cuantas reservas del test:

    python -m src.predictor
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from . import config


def cargar(ruta=None):
    """Carga el Pipeline entrenado y sus metadatos; devuelve (pipeline, metadatos).

    Si gana la red de Keras, la red va aparte en config.MODELO_KERAS. El umbral se lee
    de los metadatos (también en pipeline.metadatos_) y no de config.UMBRAL, para usar
    el mismo con el que se entrenó el modelo. Si no hay modelo guardado, el error dice
    cómo generarlo.
    """
    ruta = Path(config.MODELO_PKL if ruta is None else ruta)
    ruta_metadatos = ruta.with_name(config.METADATOS.name)
    if not ruta.exists() or not ruta_metadatos.exists():
        raise FileNotFoundError(
            f"No hay ningún modelo entrenado en {ruta.parent}. Ejecuta primero "
            "`python main.py` (o `python main.py --demo`) para generarlo.")

    metadatos = json.loads(ruta_metadatos.read_text(encoding="utf-8"))
    pipeline = joblib.load(ruta)
    # Solo hace algo si el modelo es la red, que se carga de su .keras (ver
    # model_trainer.RedKeras); el resto de modelos van enteros en el .pkl.
    pipeline[-1].cargar_aparte(ruta.with_name(config.MODELO_KERAS.name))
    pipeline.metadatos_ = metadatos
    return pipeline, metadatos


def predecir(pipeline, reservas):
    """Devuelve 0/1 por reserva (1 = cancela) con el umbral de los metadatos.

    Las reservas llegan con las mismas columnas originales que el train: el preprocesado
    va dentro del Pipeline, así que no hay que preparar nada a mano.
    """
    umbral = _metadatos(pipeline)["umbral"]
    cancela = predecir_proba(pipeline, reservas) >= umbral
    return cancela.astype(int).rename("cancela")


def predecir_proba(pipeline, reservas):
    """Devuelve la probabilidad de cancelación de cada reserva.

    Es lo que calcula el modelo; el 0/1 de predecir() sale de compararla con el umbral.
    """
    columnas = _metadatos(pipeline)["columnas"]
    faltan = [c for c in columnas if c not in reservas.columns]
    if faltan:
        raise ValueError(f"A las reservas les faltan columnas del modelo: {faltan}")
    # Selecciono las columnas por nombre y en el orden del train, así las que sobren
    # (is_canceled, las fugas de un CSV sin limpiar...) no llegan al modelo.
    proba = pipeline.predict_proba(reservas[columnas])[:, 1]
    return pd.Series(proba, index=reservas.index, name="prob_cancelacion")


def comparar(pipeline, reservas, reales):
    """Tabla con la probabilidad, la predicción y el valor real de cada reserva.

    La usan la demo de este módulo y el último paso de main.py.
    """
    return pd.concat([predecir_proba(pipeline, reservas).round(3),
                      predecir(pipeline, reservas),
                      pd.Series(reales, index=reservas.index, name="real")], axis=1)


def demo(n=10):
    """Predice n reservas del test con el modelo guardado, junto a su valor real.

    Son del test porque el modelo no las vio al entrenar. Se usa el mismo modo (demo o
    completo) con el que se entrenó el modelo, para que la partición sea la misma.
    """
    from . import data_loader  # solo la demo necesita los datos

    pipeline, metadatos = cargar()
    config.DEMO = metadatos["demo"]
    d = data_loader.preparar()
    muestra = d["X_test"].sample(n, random_state=config.SEMILLA)
    tabla = comparar(pipeline, muestra, d["y_test"].loc[muestra.index])

    print(f"\nModelo: {metadatos['ganador']} · umbral {metadatos['umbral']} · "
          f"entrenado {'en modo demo' if metadatos['demo'] else 'con todos los datos'}")
    print(pd.concat([muestra[["hotel", "lead_time", "deposit_type", "market_segment"]],
                     tabla], axis=1).to_string())
    aciertos = int((tabla["cancela"] == tabla["real"]).sum())
    print(f"\nAciertos: {aciertos} de {n}")
    return tabla


def _metadatos(pipeline) -> dict:
    """Devuelve los metadatos que cargar() guarda en el pipeline."""
    if not hasattr(pipeline, "metadatos_"):
        raise ValueError("Este pipeline no trae metadatos: cárgalo con predictor.cargar(), "
                         "que es el que sabe con qué umbral y qué columnas se entrenó.")
    return pipeline.metadatos_


if __name__ == "__main__":
    # Demo de inferencia: python -m src.predictor
    try:
        demo()
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from None
