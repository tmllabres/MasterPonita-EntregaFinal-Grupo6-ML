"""Tests de data_loader: carga, limpieza, partición y preprocesado.

Me centro en dos fallos que no dan ningún error:

  1. que sobreviva una columna de fuga: con las directas cualquier modelo acierta casi
     el 100 %, y con las dos que se rellenan al llegar el F1 del modelo de prueba sube
     de 0,678 a 0,710 sin que nada avise;
  2. que X e y dejen de estar alineadas después de borrar filas.

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
    """Llama a preparar() una sola vez para los tests que usan el CSV real."""
    return data_loader.preparar()


@pytest.fixture(scope="module")
def prep_ajustado(datos):
    """Un clon del preprocesador ajustado sobre X_train. El original sigue sin ajustar."""
    return clone(datos["preprocesador"]).fit(datos["X_train"])


def _reservas(**columnas) -> pd.DataFrame:
    """Reservas mínimas para limpiar(); lo que no se pasa lleva un valor válido."""
    n = len(next(iter(columnas.values())))
    base = {
        "hotel": ["City Hotel"] * n, "is_canceled": [1] * n, "adr": [80.0] * n,
        "adults": [2] * n, "children": [0] * n, "babies": [0] * n,
        "reservation_status": ["Canceled"] * n,
        "reservation_status_date": ["2017-07-01"] * n,
        "required_car_parking_spaces": [0] * n, "assigned_room_type": ["A"] * n,
    }
    return pd.DataFrame({**base, **columnas})


def test_limpiar_elimina_las_columnas_de_fuga(df_falso):
    """limpiar() quita las cuatro columnas de config.FUGAS: las dos directas y las dos
    que se rellenan cuando llega el cliente."""
    limpio = data_loader.limpiar(df_falso)
    for fuga in config.FUGAS:
        assert fuga not in limpio.columns


def test_limpiar_quita_duplicados_e_imposibles(df_falso):
    """limpiar() deja 2 de las 5 filas del fixture: la válida y una de las repetidas."""
    limpio = data_loader.limpiar(df_falso)
    assert len(limpio) == 2
    assert (limpio["adr"] >= 0).all()


def test_separar_X_y_no_deja_el_objetivo_dentro_de_X(df_falso):
    """is_canceled no se queda en X: si no, el modelo predeciría con la respuesta."""
    X, y = data_loader.separar_X_y(data_loader.limpiar(df_falso))
    assert config.OBJETIVO not in X.columns
    assert len(X) == len(y)


def test_particionar_conserva_el_reparto_de_clases():
    """Con stratify=y, la proporción de cancelaciones es la misma en train y en test.

    La tolerancia es de una milésima y no de un punto: sin stratify, con la semilla 42 la
    diferencia ya es de 0,0036, y una tolerancia de 0,01 no la detectaría."""
    d = data_loader.preparar()
    p_train = d["y_train"].mean()
    p_test = d["y_test"].mean()
    assert abs(p_train - p_test) < 0.001


def test_el_preprocesador_llega_sin_ajustar():
    """preparar() devuelve el preprocesador sin ajustar.

    Así se ajusta dentro del Pipeline en cada fold, solo con los datos de entrenamiento;
    si llegara ya ajustado, podría haber aprendido de datos de test (fuga)."""
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    d = data_loader.preparar()
    with pytest.raises(NotFittedError):
        check_is_fitted(d["preprocesador"])


# ── Casos que los cinco tests de arriba no cubren ────────────────────────────

def test_limpiar_quita_las_fugas_antes_de_deduplicar():
    """limpiar() quita las fugas antes de buscar duplicados.

    Dos reservas que solo se diferencian en una columna de fuga son la misma para el
    modelo, que no ve esa columna. En el orden contrario sobrevivirían las dos (en el
    CSV real saldrían 31.994 duplicados en vez de 33.413)."""
    df = _reservas(reservation_status_date=["2017-07-01", "2017-07-02"])
    assert len(data_loader.limpiar(df)) == 1


def test_limpiar_conserva_a_los_ninos_sin_adultos():
    """Una reserva con 0 adultos y 2 niños no se borra.

    «Sin huéspedes» es adults + children + babies == 0, no adults == 0. Una reserva así
    es rara, pero existe (218 en el CSV limpio)."""
    df = _reservas(adults=[0], children=[2])
    assert len(data_loader.limpiar(df)) == 1


def test_limpiar_no_modifica_su_entrada():
    """limpiar() no modifica el DataFrame que recibe.

    df_falso es de sesión: si limpiar() lo modificara, los tests siguientes recibirían
    un DataFrame ya limpio y pasarían sin probar nada. Aquí uso uno propio porque
    df_falso ya podría llegar modificado por un test anterior."""
    df = _reservas(adr=[80.0, -1.0])
    copia = df.copy()
    data_loader.limpiar(df)
    pd.testing.assert_frame_equal(df, copia)


def test_separar_X_y_deja_cada_fila_con_su_respuesta(df_falso):
    """X e y comparten índice y cada y es la is_canceled de su propia fila.

    No basta con que tengan la misma longitud."""
    limpio = data_loader.limpiar(df_falso)
    X, y = data_loader.separar_X_y(limpio)
    assert X.index.equals(y.index)
    assert (y.to_numpy() == limpio[config.OBJETIVO].to_numpy()).all()


def test_preparar_no_deja_ninguna_fuga_en_X(datos):
    """X_train y X_test, lo que reciben los modelos, no tienen ninguna columna de fuga
    y se quedan con las 27 predictoras."""
    for X in (datos["X_train"], datos["X_test"]):
        assert not set(config.FUGAS) & set(X.columns)
        assert X.shape[1] == 27


def test_particionar_da_los_tamanos_del_readme_y_estratifica_de_verdad(datos):
    """La partición da 68.648 / 17.163 filas y estratifica las cancelaciones.

    Salen de las 85.811 filas limpias (el test redondea hacia arriba); si cambian,
    cambian las cifras de los apartados 2, 5 y 8 del README. Con stratify, las
    cancelaciones del test quedan a menos de una fila de lo esperado; sin él, se
    desvían unas 50 filas."""
    assert (len(datos["X_train"]), len(datos["X_test"])) == (68_648, 17_163)
    prevalencia = pd.concat([datos["y_train"], datos["y_test"]]).mean()
    assert abs(datos["y_test"].sum() - len(datos["y_test"]) * prevalencia) <= 1


def test_el_preprocesador_ajustado_agrupa_bien_y_da_97_columnas(prep_ajustado, datos):
    """country, agent y company van a la rama de alta cardinalidad y salen 97 columnas.

    agent y company son float64: si acabaran en la rama numérica, el agente 240 valdría
    el doble que el 120. Salida: 16 numéricas + 48 categóricas + 3 x 11 agrupadas."""
    ramas = {nombre: list(cols) for nombre, _, cols in prep_ajustado.transformers_}
    assert set(ramas["alta"]) == set(config.ALTA_CARDINALIDAD)
    assert not set(ramas["num"]) & set(config.ALTA_CARDINALIDAD)

    salida = prep_ajustado.transform(datos["X_test"])
    assert salida.shape[1] == 97
    assert not np.isnan(salida).any()


def test_agent_entero_o_decimal_activa_la_misma_columna(prep_ajustado, datos):
    """agent como 9.0, como 9 o como "9" activa la misma columna del one-hot.

    El CSV trae agent como float64, pero en inferencia puede llegar como int (de un
    JSON) o como texto. Sin normalizarlo, la reserva acabaría en el cajón de
    infrecuentes y la predicción cambiaría sin ningún error."""
    agente = datos["X_train"]["agent"].mode()[0]  # el más frecuente: tiene columna propia
    fila = datos["X_test"].iloc[[0]].copy()
    fila["agent"] = float(agente)
    esperado = prep_ajustado.transform(fila)

    for otra_forma in (int(agente), str(int(agente))):
        variante = fila.copy()
        variante["agent"] = pd.Series([otra_forma], index=fila.index, dtype=object)
        np.testing.assert_array_equal(prep_ajustado.transform(variante), esperado)


def test_el_preprocesador_ajustado_se_puede_guardar(prep_ajustado, datos):
    """El preprocesador ajustado se puede guardar con pickle y recargado da lo mismo.

    model_trainer.guardar() guarda el Pipeline entero con joblib, que usa pickle, y una
    lambda dentro del preprocesador lo impediría."""
    recargado = pickle.loads(pickle.dumps(prep_ajustado))
    np.testing.assert_array_equal(recargado.transform(datos["X_test"]),
                                  prep_ajustado.transform(datos["X_test"]))


# ── Otros fallos que los tests de arriba no detectarían ──────────────────────

def test_preparar_respeta_el_modo_demo_puesto_despues_del_import(monkeypatch):
    """preparar() lee config.DEMO al ejecutarse y la muestra es siempre la misma.

    main.py pone config.DEMO = True después de importar data_loader; si preparar() lo
    leyera con `from .config import DEMO`, --demo entrenaría con las 85.811 filas sin
    avisar. Con la muestra fija, la demo de la defensa da siempre los mismos números."""
    monkeypatch.setattr(config, "DEMO", True)
    primera, segunda = data_loader.preparar(), data_loader.preparar()
    assert len(primera["X_train"]) + len(primera["X_test"]) == config.DEMO_FILAS
    pd.testing.assert_frame_equal(primera["X_train"], segunda["X_train"])


def test_preparar_deja_cada_reserva_con_su_respuesta(datos):
    """En lo que devuelve preparar(), cada fila de X lleva la y de su propia reserva.

    Es como test_separar_X_y_deja_cada_fila_con_su_respuesta, pero sobre lo que reciben
    los modelos: si y_train se barajara conservando el índice, o se cambiara por y_test,
    ningún otro test lo vería y el modelo aprendería la respuesta de otra reserva."""
    limpio = data_loader.limpiar(data_loader.cargar_crudo())
    for X, y in ((datos["X_train"], datos["y_train"]), (datos["X_test"], datos["y_test"])):
        assert X.index.equals(y.index)
        pd.testing.assert_frame_equal(X, limpio.loc[X.index, X.columns])
        assert (y == limpio.loc[y.index, config.OBJETIVO]).all()


def test_la_particion_es_la_que_describe_el_readme(datos):
    """Con la semilla de config, la partición es la que describen el README y config.

    Con otra semilla los tamaños y el reparto no cambian y los tests de arriba pasarían,
    pero dejarían de ser ciertos dos datos que cito: que la reserva con adr de 5.400 cae
    en el test (y por eso no afecta al escalado) y cuántos valores distintos tienen en
    train las tres columnas de alta cardinalidad."""
    assert (datos["X_test"]["adr"] >= 5000).any()
    assert not (datos["X_train"]["adr"] >= 5000).any()
    distintos = {c: datos["X_train"][c].nunique() for c in config.ALTA_CARDINALIDAD}
    assert distintos == {"country": 168, "agent": 325, "company": 320}


def test_la_rama_numerica_sale_escalada(prep_ajustado, datos):
    """Las 16 numéricas, ajustadas con el train, salen de él con media 0 y desviación 1.

    Sin el StandardScaler el test de las 97 columnas seguiría pasando, pero lead_time
    llegaría hasta 737, arrival_date_year valdría 2015-2017 y la logística dejaría de
    converger."""
    nombres = prep_ajustado.get_feature_names_out()
    de_num = [n.startswith("num__") for n in nombres]
    numericas = prep_ajustado.transform(datos["X_train"])[:, de_num]
    assert numericas.shape[1] == 16
    np.testing.assert_allclose(numericas.mean(axis=0), 0, atol=1e-9)
    np.testing.assert_allclose(numericas.std(axis=0), 1, atol=1e-9)


def test_una_categoria_nunca_vista_no_revienta(prep_ajustado, datos):
    """Una categoría que no estaba en el train sale como ceros, sin lanzar un error.

    Pasa en --demo: distribution_channel "Undefined" sale 3 veces en el train completo y
    1 en el de --demo, que en el fold 0 cae en validación. Con handle_unknown="error" la
    comparación fallaría a medias. X_test no trae categorías nuevas, así que sin este
    test nada lo probaría."""
    fila = datos["X_test"].iloc[[0]].copy()
    fila["meal"] = "ZZ"
    salida = pd.Series(prep_ajustado.transform(fila)[0],
                       index=prep_ajustado.get_feature_names_out())
    assert (salida[salida.index.str.startswith("cat__meal_")] == 0).all()


def test_un_agente_nunca_visto_cae_en_el_cajon(prep_ajustado, datos):
    """Un agente que no estaba en el train activa la columna de infrecuentes.

    Es la misma que activan los agentes poco frecuentes que sí estaban. Con
    handle_unknown="ignore" saldrían once ceros, un patrón que el modelo no ha visto
    nunca."""
    fila = datos["X_test"].iloc[[0]].copy()
    fila["agent"] = 99999.0
    salida = pd.Series(prep_ajustado.transform(fila)[0],
                       index=prep_ajustado.get_feature_names_out())
    assert salida["alta__agent_infrequent_sklearn"] == 1


@pytest.mark.parametrize("sucio,columna", [
    ("NULL", "desconocido"), (" NULL ", "desconocido"), ("null", "desconocido"),
    ("", "desconocido"), ("sin agente", "infrequent_sklearn"), ("inf", "infrequent_sklearn"),
])
def test_un_valor_sucio_no_cambia_la_codificacion_de_su_vecina(prep_ajustado, datos,
                                                                sucio, columna):
    """Un valor mal escrito en agent no cambia cómo se codifica la reserva de al lado.

    En inferencia, cada reserva se tiene que codificar igual llegue sola o en un lote.
    Además, el valor mal escrito va a su sitio: el nulo escrito como texto (con o sin
    espacios) a "desconocido", y lo que no es ni número ni nulo, al cajón."""
    agente = datos["X_train"]["agent"].mode()[0]
    fila = datos["X_test"].iloc[[0]].copy()
    fila["agent"] = float(agente)
    sola = prep_ajustado.transform(fila)

    lote = pd.concat([fila, fila], ignore_index=True)
    lote["agent"] = pd.Series([float(agente), sucio], dtype=object)
    salida = pd.DataFrame(prep_ajustado.transform(lote),
                          columns=prep_ajustado.get_feature_names_out())
    np.testing.assert_array_equal(salida.iloc[[0]].to_numpy(), sola)
    assert salida.loc[1, f"alta__agent_{columna}"] == 1
