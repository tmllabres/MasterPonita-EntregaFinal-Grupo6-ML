"""Predicciones con el modelo ya entrenado: el camino de vuelta.

Esto es la INFERENCIA, y es lo que cierra el «flujo completo desde los datos hasta
la inferencia» que pide el enunciado. Este módulo NO entrena, NO evalúa y NO abre
el CSV de train: carga el artefacto y predice.

La prueba de que el artefacto es usable: alguien que solo tenga models/ y este
fichero puede predecir una reserva nueva sin reentrenar nada.

    python -m src.predictor
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from . import config


def cargar(ruta=None):
    """Carga el Pipeline entrenado (preprocesado + modelo) y sus metadatos.

    Lee config.MODELO_PKL y config.METADATOS, los mismos que escribe
    model_trainer.guardar(). Si el ganador es la red de Keras son DOS ficheros:
    config.MODELO_KERAS con arquitectura y pesos, y el .pkl con el
    ColumnTransformer ya ajustado.

    Devuelve (pipeline, metadatos). El umbral sale de los metadatos, NO de
    config.UMBRAL: el que vale es el que se congeló al entrenar este artefacto. Por eso
    los metadatos viajan también pegados al pipeline (pipeline.metadatos_), y
    predecir() los lee de ahí.
    """
    ruta = Path(config.MODELO_PKL if ruta is None else ruta)
    ruta_metadatos = ruta.with_name(config.METADATOS.name)
    if not ruta.exists() or not ruta_metadatos.exists():
        raise FileNotFoundError(
            f"No hay ningún modelo entrenado en {ruta.parent}. Ejecuta primero "
            "`python main.py` (o `python main.py --demo`) para generarlo.")

    metadatos = json.loads(ruta_metadatos.read_text(encoding="utf-8"))
    pipeline = joblib.load(ruta)
    # Los modelos que caben enteros en el .pkl no hacen nada aquí; la red se recupera
    # de su .keras (ver model_trainer.RedKeras).
    pipeline[-1].cargar_aparte(ruta.with_name(config.MODELO_KERAS.name))
    pipeline.metadatos_ = metadatos
    return pipeline, metadatos


def predecir(pipeline, reservas):
    """Devuelve 0/1 por reserva aplicando el umbral de los metadatos.

    Las reservas entran con las MISMAS columnas crudas que el train: el Pipeline
    se encarga del resto. Si tienes que preparar los datos a mano antes de llamar
    aquí, es que el preprocesado se quedó fuera del artefacto.
    """
    umbral = _metadatos(pipeline)["umbral"]
    cancela = predecir_proba(pipeline, reservas) >= umbral
    return cancela.astype(int).rename("cancela")


def predecir_proba(pipeline, reservas):
    """Devuelve la probabilidad de cancelación, que es lo que el modelo calcula
    de verdad. El 0/1 sale luego de comparar contra el umbral."""
    columnas = _metadatos(pipeline)["columnas"]
    faltan = [c for c in columnas if c not in reservas.columns]
    if faltan:
        raise ValueError(f"A las reservas les faltan columnas del modelo: {faltan}")
    # Solo las columnas del train y en su orden: las que sobren (is_canceled, las de
    # fuga de un CSV crudo...) no llegan al modelo.
    proba = pipeline.predict_proba(reservas[columnas])[:, 1]
    return pd.Series(proba, index=reservas.index, name="prob_cancelacion")


def comparar(pipeline, reservas, reales):
    """Tabla de reservas con la probabilidad, la predicción y lo que pasó de verdad.
    Es lo que enseñan la demo de este módulo y el último paso de main.py."""
    return pd.concat([predecir_proba(pipeline, reservas).round(3),
                      predecir(pipeline, reservas),
                      pd.Series(reales, index=reservas.index, name="real")], axis=1)


def demo(n=10):
    """Coge n reservas del test, les quita la respuesta y las predice con el modelo
    guardado, con la respuesta real al lado.

    Del TEST y no del CSV entero: son reservas que el modelo no vio al entrenar. Con
    el modo (demo o completo) con el que se entrenó el artefacto, para que la partición
    sea la misma que la suya.
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
    """Los metadatos que cargar() dejó pegados al pipeline."""
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
