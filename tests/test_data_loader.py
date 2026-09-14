"""Tests de la carga y limpieza de datos.

Lo que se prueba aquí no es «que el código corra», sino las dos cosas que, si se
rompen, no dan error y arruinan el proyecto entero en silencio:

  1. que las columnas de fuga desaparecen —si sobreviven, los cinco modelos sacan
     AUC ≈ 1,000 y la comparativa no distingue nada—,
  2. que X e y siguen alineadas después de borrar filas.

    python -m pytest tests/test_data_loader.py -q
"""
from __future__ import annotations

import pytest

from src import config, data_loader


def test_limpiar_elimina_las_columnas_de_fuga(df_falso):
    """reservation_status y reservation_status_date no pueden sobrevivir a limpiar()."""
    limpio = data_loader.limpiar(df_falso)
    for fuga in config.FUGAS:
        assert fuga not in limpio.columns


def test_limpiar_quita_duplicados_e_imposibles(df_falso):
    """De las 5 filas del fixture deben quedar 2: la sana y una de las duplicadas."""
    limpio = data_loader.limpiar(df_falso)
    assert len(limpio) == 2
    assert (limpio["adr"] >= 0).all()


def test_separar_X_y_no_deja_el_objetivo_dentro_de_X(df_falso):
    """El error clásico: is_canceled se queda en X y el modelo predice con la respuesta."""
    X, y = data_loader.separar_X_y(data_loader.limpiar(df_falso))
    assert config.OBJETIVO not in X.columns
    assert len(X) == len(y)


def test_particionar_conserva_el_reparto_de_clases():
    """stratify=y: la proporción de cancelaciones tiene que ser la misma en train y test."""
    d = data_loader.preparar()
    p_train = d["y_train"].mean()
    p_test = d["y_test"].mean()
    assert abs(p_train - p_test) < 0.01


def test_el_preprocesador_llega_sin_ajustar():
    """Si viniera ya ajustado, se habría entrenado con datos de test: fuga de preprocesado."""
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    d = data_loader.preparar()
    with pytest.raises(NotFittedError):
        check_is_fitted(d["preprocesador"])
