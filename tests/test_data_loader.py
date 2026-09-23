"""Tests de la carga y limpieza de datos.

Lo que se prueba aquí no es «que el código corra», sino las dos cosas que, si se
rompen, no dan error y arruinan el proyecto entero en silencio:

  1. que las columnas de fuga desaparecen —si sobreviven, los cinco modelos sacan
     AUC ≈ 1,000 y la comparativa no distingue nada—,
  2. que X e y siguen alineadas después de borrar filas.

    python -m pytest tests/test_data_loader.py -q
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src import config, data_loader


@pytest.fixture(scope="module")
def datos():
    """preparar() una sola vez para los tests de abajo que necesitan el CSV real."""
    return data_loader.preparar()


@pytest.fixture(scope="module")
def prep_ajustado(datos):
    """Un clon del preprocesador ajustado sobre X_train. El original sigue sin ajustar."""
    return clone(datos["preprocesador"]).fit(datos["X_train"])


def _reservas(**columnas) -> pd.DataFrame:
    """Reservas mínimas para probar limpiar(): lo que no se pase, toma un valor sano."""
    n = len(next(iter(columnas.values())))
    base = {
        "hotel": ["City Hotel"] * n, "is_canceled": [1] * n, "adr": [80.0] * n,
        "adults": [2] * n, "children": [0] * n, "babies": [0] * n,
        "reservation_status": ["Canceled"] * n,
        "reservation_status_date": ["2017-07-01"] * n,
    }
    return pd.DataFrame({**base, **columnas})


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


# ── Lo que los cinco de arriba no distinguen ─────────────────────────────────

def test_limpiar_quita_las_fugas_antes_de_deduplicar():
    """Dos reservas que solo se diferencian en una columna de fuga son la misma para el
    modelo, que nunca ve esa columna. Si se deduplicara antes de quitar las fugas,
    sobrevivirían las dos (en el CSV real: 31.994 duplicados en vez de 33.413)."""
    df = _reservas(reservation_status_date=["2017-07-01", "2017-07-02"])
    assert len(data_loader.limpiar(df)) == 1


def test_limpiar_conserva_a_los_ninos_sin_adultos():
    """«Sin huéspedes» es adultos + niños + bebés == 0, no adults == 0: una reserva de
    0 adultos y 2 niños es rara, pero existe (218 en el CSV limpio)."""
    df = _reservas(adults=[0], children=[2])
    assert len(data_loader.limpiar(df)) == 1


def test_limpiar_no_modifica_su_entrada():
    """df_falso es de sesión: si limpiar() lo modificara, el siguiente test recibiría un
    DataFrame ya limpio y pasaría sin probar nada. Por eso este test usa uno propio: con
    df_falso, un test anterior ya se lo habría dejado sin fugas y no vería nada."""
    df = _reservas(adr=[80.0, -1.0])
    copia = df.copy()
    data_loader.limpiar(df)
    pd.testing.assert_frame_equal(df, copia)


def test_separar_X_y_deja_cada_fila_con_su_respuesta(df_falso):
    """Misma longitud no basta: X e y tienen que compartir índice, y cada y tiene que ser
    la is_canceled de su propia fila."""
    limpio = data_loader.limpiar(df_falso)
    X, y = data_loader.separar_X_y(limpio)
    assert X.index.equals(y.index)
    assert (y.to_numpy() == limpio[config.OBJETIVO].to_numpy()).all()


def test_preparar_no_deja_ninguna_fuga_en_X(datos):
    """Las cuatro de config.FUGAS fuera, también las que se rellenan a la llegada del
    cliente (parking y habitación asignada), que el fixture pequeño no trae."""
    for X in (datos["X_train"], datos["X_test"]):
        assert not set(config.FUGAS) & set(X.columns)
        assert X.shape[1] == 27


def test_particionar_da_los_tamanos_del_readme_y_estratifica_de_verdad(datos):
    """85.811 filas limpias -> 68.648 / 17.163 (el test redondea hacia arriba). Si esto
    cambia, cambian las cifras de los apartados 2, 5 y 8 del README.

    Lo de stratify: con él, las cancelaciones del test salen a menos de una fila de lo
    que toca por prevalencia. Sin él, se desvían unas 60 filas arriba o abajo."""
    assert (len(datos["X_train"]), len(datos["X_test"])) == (68_648, 17_163)
    prevalencia = pd.concat([datos["y_train"], datos["y_test"]]).mean()
    assert abs(datos["y_test"].sum() - len(datos["y_test"]) * prevalencia) <= 1


def test_el_preprocesador_ajustado_agrupa_bien_y_da_97_columnas(prep_ajustado, datos):
    """agent y company son float64: si acabaran en la rama numérica, el agente 240 valdría
    el doble que el 120. Salida: 16 numéricas + 48 categóricas + 3 x 11 agrupadas."""
    ramas = {nombre: list(cols) for nombre, _, cols in prep_ajustado.transformers_}
    assert set(ramas["alta"]) == set(config.ALTA_CARDINALIDAD)
    assert not set(ramas["num"]) & set(config.ALTA_CARDINALIDAD)

    salida = prep_ajustado.transform(datos["X_test"])
    assert salida.shape[1] == 97
    assert not np.isnan(salida).any()


def test_agent_entero_o_decimal_activa_la_misma_columna(prep_ajustado, datos):
    """El CSV trae agent como 9.0 (float64), pero en inferencia puede llegar como 9 (int,
    de un JSON) o "9". Si no se normaliza, la reserva cae en silencio en el cajón de
    infrecuentes y la predicción cambia sin ningún error."""
    agente = datos["X_train"]["agent"].mode()[0]  # el más frecuente: tiene columna propia
    fila = datos["X_test"].iloc[[0]].copy()
    fila["agent"] = float(agente)
    esperado = prep_ajustado.transform(fila)

    for otra_forma in (int(agente), str(int(agente))):
        variante = fila.copy()
        variante["agent"] = pd.Series([otra_forma], index=fila.index, dtype=object)
        np.testing.assert_array_equal(prep_ajustado.transform(variante), esperado)


def test_el_preprocesador_ajustado_se_puede_guardar(prep_ajustado, datos):
    """model_trainer.guardar() persiste el Pipeline entero con joblib, que usa pickle. Una
    lambda dentro del preprocesador lo haría imposible."""
    recargado = pickle.loads(pickle.dumps(prep_ajustado))
    np.testing.assert_array_equal(recargado.transform(datos["X_test"]),
                                  prep_ajustado.transform(datos["X_test"]))
