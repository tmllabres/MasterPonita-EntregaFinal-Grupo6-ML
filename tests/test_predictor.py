"""Tests de predictor: cargar el modelo guardado y predecir sin reentrenar.

Me centro en lo que puede salir mal sin avisar: predecir con el umbral de config en
vez del que se guardó con el modelo, o con las columnas en otro orden.

    python -m pytest tests/test_predictor.py -q
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from src import config, data_loader, model_trainer, predictor


@pytest.fixture(scope="module")
def datos():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "DEMO", True)
        return data_loader.preparar()


@pytest.fixture
def artefacto(datos, tmp_path, monkeypatch):
    """Un árbol entrenado y guardado con model_trainer.guardar() en una carpeta
    temporal, con config apuntando a ella, igual que main.py lo deja en models/."""
    for clave in ("MODELO_PKL", "MODELO_KERAS", "METADATOS"):
        monkeypatch.setattr(config, clave, tmp_path / getattr(config, clave).name)
    pipeline = Pipeline([("prep", clone(datos["preprocesador"])),
                         ("modelo", model_trainer.Arbol())])
    pipeline.fit(datos["X_train"], datos["y_train"])
    model_trainer.guardar(pipeline, "arbol", {"f1": 0.6})
    return pipeline


def test_sin_modelo_entrenado_dice_que_hay_que_ejecutar_main(tmp_path):
    """Sin modelo guardado, el error dice que hay que ejecutar main.py.

    No basta con decir que falta un fichero: el mensaje tiene que decir qué hacer."""
    with pytest.raises(FileNotFoundError, match="python main.py"):
        predictor.cargar(tmp_path / "mejor_modelo.pkl")


def test_el_modelo_recargado_predice_lo_mismo_que_el_entrenado(artefacto, datos):
    """El modelo recargado desde disco, sin reentrenar, da las mismas probabilidades
    que el entrenado sobre reservas con las columnas originales."""
    recargado, metadatos = predictor.cargar()
    assert metadatos["ganador"] == "arbol"
    proba = predictor.predecir_proba(recargado, datos["X_test"])
    np.testing.assert_allclose(proba, artefacto.predict_proba(datos["X_test"])[:, 1])
    assert proba.index.equals(datos["X_test"].index)


def test_predecir_usa_el_umbral_de_los_metadatos_y_no_el_de_config(artefacto, datos,
                                                                  monkeypatch):
    """predecir() usa el umbral guardado en metadatos.json, no el de config.

    Si config.UMBRAL cambiara después de guardar, el modelo guardado no debe cambiar
    sus predicciones."""
    ruta_metadatos = config.METADATOS
    metadatos = json.loads(ruta_metadatos.read_text(encoding="utf-8"))
    metadatos["umbral"] = 0.3
    ruta_metadatos.write_text(json.dumps(metadatos), encoding="utf-8")
    monkeypatch.setattr(config, "UMBRAL", 0.9)

    recargado, _ = predictor.cargar()
    X = datos["X_test"]
    esperado = (artefacto.predict_proba(X)[:, 1] >= 0.3).astype(int)
    np.testing.assert_array_equal(predictor.predecir(recargado, X), esperado)


def test_las_columnas_se_toman_por_nombre_y_se_avisa_si_falta_alguna(artefacto, datos):
    """Las columnas se toman por nombre y, si falta alguna, el error dice cuál.

    Una reserva puede llegar con las columnas en otro orden o con columnas de más (un
    CSV crudo trae is_canceled y las de fuga): se usan las del train, por nombre."""
    recargado, _ = predictor.cargar()
    X = datos["X_test"].head(50)
    desordenadas = X[X.columns[::-1]].assign(is_canceled=1, reservation_status="Canceled")
    np.testing.assert_allclose(predictor.predecir_proba(recargado, desordenadas),
                               predictor.predecir_proba(recargado, X))
    with pytest.raises(ValueError, match="lead_time"):
        predictor.predecir_proba(recargado, X.drop(columns="lead_time"))


def test_un_pipeline_sin_metadatos_no_predice(artefacto, datos):
    """Un Pipeline que no ha pasado por cargar() no predice.

    Sin los metadatos no se sabe con qué umbral se entrenó, y prefiero un error a usar
    el de config sin avisar."""
    with pytest.raises(ValueError, match="cargar"):
        predictor.predecir(artefacto, datos["X_test"].head(5))


def test_la_demo_predice_reservas_del_test_con_la_respuesta_al_lado(artefacto, monkeypatch):
    """La demo (python -m src.predictor) enseña n reservas del test con su probabilidad,
    su 0/1 y lo que pasó de verdad."""
    monkeypatch.setattr(config, "DEMO", config.DEMO)  # la demo lo cambia; así se restaura
    tabla = predictor.demo(n=8)
    assert list(tabla.columns) == ["prob_cancelacion", "cancela", "real"]
    assert len(tabla) == 8
    assert tabla["prob_cancelacion"].between(0, 1).all()
    assert set(tabla["cancela"]) <= {0, 1} and set(tabla["real"]) <= {0, 1}
    assert isinstance(tabla, pd.DataFrame)
