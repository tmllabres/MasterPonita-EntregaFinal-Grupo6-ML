"""Tests de evaluator: métricas, figuras e informe.

Me centro en los fallos que no dan error: un AUC calculado con los 0/1 en vez de con
las probabilidades, un PNG en blanco, un JSON que el notebook no puede leer o una curva
ROC con un solo modelo cuando el enunciado pide todos en los mismos ejes. Como un PNG
con algo dibujado no dice qué se ha dibujado, las figuras se capturan antes de
guardarlas y se revisa su contenido.

    python -m pytest tests/test_evaluator.py -q
"""
from __future__ import annotations

import ast
import inspect
import json
import warnings
from functools import partial

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from matplotlib.image import imread
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

import main
from src import config, data_loader, evaluator, model_trainer


@pytest.fixture(scope="module")
def prediccion():
    """y real, unas probabilidades con algo de señal y el 0/1 que sale con el umbral."""
    rng = np.random.default_rng(config.SEMILLA)
    y_true = rng.random(2000) < 0.28
    y_proba = np.clip(0.28 + 0.35 * (y_true - 0.28) + rng.normal(0, 0.2, 2000), 0, 1)
    return y_true.astype(int), (y_proba >= config.UMBRAL).astype(int), y_proba


@pytest.fixture
def figuras(monkeypatch):
    """Las Figure que evaluator manda guardar, para revisar qué se ha dibujado.

    Con el PNG solo no se distingue una matriz traspuesta ni una ROC con una curva."""
    capturadas = []
    guardar = evaluator._guardar

    def espia(fig, ruta):
        capturadas.append(fig)
        guardar(fig, ruta)

    monkeypatch.setattr(evaluator, "_guardar", espia)
    return capturadas


def _png_con_contenido(ruta) -> bool:
    """True si el PNG existe y no está en blanco (lo que deja un plt.show() antes de
    guardar): la desviación de sus píxeles tiene que ser apreciable."""
    return ruta.exists() and imread(ruta)[..., :3].std() > 0.05


# ── metricas() ───────────────────────────────────────────────────────────────

def test_metricas_coinciden_con_scikit_learn_y_salen_como_tipos_nativos(prediccion):
    """Las cinco métricas coinciden con scikit-learn (pos_label=1) y son tipos nativos.

    Tienen que ser tipos que json.dumps pueda escribir, porque el informe y los
    metadatos las guardan tal cual."""
    y_true, y_pred, y_proba = prediccion
    m = evaluator.metricas(y_true, y_pred, y_proba)

    assert m["f1"] == pytest.approx(f1_score(y_true, y_pred))
    assert m["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
    assert m["precision"] == pytest.approx(precision_score(y_true, y_pred))
    assert m["recall"] == pytest.approx(recall_score(y_true, y_pred))
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y_true, y_proba))
    assert m["n"] == len(y_true)

    assert all(type(v) is float for k, v in m.items() if k != "n")
    assert type(m["n"]) is int
    json.dumps(m, allow_nan=False)


def test_el_auc_se_calcula_con_las_probabilidades(prediccion):
    """El AUC se calcula con las probabilidades y no con los 0/1.

    Con los 0/1 la curva ROC tiene un solo punto y el AUC mide ese umbral, no lo bien
    que el modelo ordena. Con estos datos los dos números son distintos."""
    y_true, y_pred, y_proba = prediccion
    m = evaluator.metricas(y_true, y_pred, y_proba)
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y_true, y_proba))
    assert m["roc_auc"] != pytest.approx(roc_auc_score(y_true, y_pred), abs=0.01)


def test_sin_probabilidades_no_hay_auc():
    """Sin y_proba, metricas() no devuelve roc_auc en vez de calcularlo con los 0/1."""
    m = evaluator.metricas([0, 1, 1, 0], [0, 1, 0, 0])
    assert "roc_auc" not in m


def test_un_modelo_que_nunca_predice_cancela_no_avisa_ni_revienta():
    """Un modelo que nunca predice «cancela» da precision, recall y F1 de 0 sin avisos.

    Es el caso del baseline: su precision es 0/0, y con zero_division=0 vale 0 sin
    ningún warning."""
    y_true = np.array([0, 1, 1, 0, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = evaluator.metricas(y_true, np.zeros(5, dtype=int), np.full(5, 0.3))
    assert m["precision"] == m["recall"] == m["f1"] == 0.0
    assert m["roc_auc"] == 0.5


# ── Las figuras ──────────────────────────────────────────────────────────────

def test_ninguna_figura_pasa_por_pyplot():
    """evaluator no importa pyplot ni llama a ningún show(): dibuja sobre Figure.

    plt.show() vacía la figura y deja el PNG en blanco, y pyplot acumula figuras en
    memoria de una llamada a otra. Se analiza el código con ast y no el texto, porque
    los docstrings sí nombran a los dos."""
    arbol = ast.parse(inspect.getsource(evaluator))
    importados = [alias.name for nodo in ast.walk(arbol)
                  if isinstance(nodo, (ast.Import, ast.ImportFrom))
                  for alias in nodo.names] + [nodo.module or "" for nodo in ast.walk(arbol)
                                              if isinstance(nodo, ast.ImportFrom)]
    llamadas = [nodo.func.attr for nodo in ast.walk(arbol)
                if isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Attribute)]
    assert not any("pyplot" in nombre for nombre in importados)
    assert "show" not in llamadas


def test_matriz_de_confusion_guarda_un_png_con_contenido(prediccion, tmp_path):
    """La figura se guarda en la ruta pedida, no está en blanco y la matriz que
    devuelve es la de scikit-learn: filas = real, columnas = predicho."""
    y_true, y_pred, _ = prediccion
    ruta = tmp_path / "confusion.png"
    matriz = evaluator.matriz_confusion(y_true, y_pred, ruta)
    np.testing.assert_array_equal(matriz, confusion_matrix(y_true, y_pred, labels=[0, 1]))
    assert _png_con_contenido(ruta)


def test_la_matriz_dibuja_cada_conteo_en_su_celda_con_el_porcentaje_de_su_fila(
        figuras, tmp_path):
    """Cada celda lleva su conteo y el porcentaje sobre su fila, en formato español.

    Con 2.000 reservas que no cancelan (1.500 bien, 500 mal) y 1.000 que sí (250 mal,
    750 bien), por fila salen 75/25 y 25/75; por columna saldrían 85,7/14,3 y 40/60.
    Las cifras van con punto de miles y coma decimal, como en el README."""
    y_true = np.array([0] * 2000 + [1] * 1000)
    y_pred = np.array([0] * 1500 + [1] * 500 + [0] * 250 + [1] * 750)
    evaluator.matriz_confusion(y_true, y_pred, tmp_path / "confusion.png")

    ax = figuras[-1].axes[0]
    # (x, y) = (columna: predicho, fila: real)
    celdas = {tuple(round(c) for c in t.get_position()): t.get_text() for t in ax.texts}
    assert celdas == {(0, 0): "1.500\n75,0 %", (1, 0): "500\n25,0 %",
                      (0, 1): "250\n25,0 %", (1, 1): "750\n75,0 %"}
    assert [t.get_text() for t in ax.get_xticklabels()] == ["No cancela", "Cancela"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["No cancela", "Cancela"]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Predicho", "Real")


def _tres_modelos(prediccion):
    y_true, _, y_proba = prediccion
    rng = np.random.default_rng(1)
    return {
        "baseline": np.full(len(y_true), 0.28),
        "arbol": y_proba,
        "logistica": np.clip(y_proba + rng.normal(0, 0.2, len(y_true)), 0, 1),
    }


def test_la_curva_roc_lleva_todos_los_modelos_y_su_auc(prediccion, tmp_path):
    """curva_roc() devuelve el AUC de cada modelo, calculado con sus probabilidades.

    Recibe el dict entero, no solo el ganador, y dibuja todas las curvas en la misma
    figura."""
    y_true = prediccion[0]
    modelos_proba = _tres_modelos(prediccion)
    ruta = tmp_path / "roc.png"
    aucs = evaluator.curva_roc(modelos_proba, y_true, ruta)
    assert set(aucs) == set(modelos_proba)
    for nombre, proba in modelos_proba.items():
        assert aucs[nombre] == pytest.approx(roc_auc_score(y_true, proba))
    assert aucs["baseline"] == 0.5
    assert _png_con_contenido(ruta)


def test_la_roc_dibuja_la_curva_de_cada_modelo_y_el_baseline_hace_de_diagonal(
        prediccion, figuras, tmp_path):
    """La ROC dibuja una línea por modelo con su curva y el baseline hace de diagonal.

    Cada línea es la curva de las probabilidades (no la de los 0/1), la leyenda va de
    mayor a menor AUC con el valor y el ganador se dibuja por encima. El baseline ya es
    la diagonal del azar, así que no se dibuja otra."""
    y_true = prediccion[0]
    modelos_proba = _tres_modelos(prediccion)
    aucs = evaluator.curva_roc(modelos_proba, y_true, tmp_path / "roc.png")

    ax = figuras[-1].axes[0]
    lineas = {linea.get_label(): linea for linea in ax.get_lines()}
    leyenda = [t.get_text() for t in ax.get_legend().get_texts()]
    assert len(leyenda) == len(ax.get_lines()) == len(modelos_proba)
    por_auc = sorted(aucs, key=aucs.get, reverse=True)
    for nombre, etiqueta in zip(por_auc, leyenda):
        assert etiqueta.startswith(evaluator._ETIQUETAS[nombre])
        assert etiqueta.endswith(f"AUC {evaluator._es(aucs[nombre], 3)}")
        fpr, tpr, _ = roc_curve(y_true, modelos_proba[nombre])
        np.testing.assert_allclose(lineas[etiqueta].get_xdata(), fpr)
        np.testing.assert_allclose(lineas[etiqueta].get_ydata(), tpr)
    alturas = [lineas[etiqueta].get_zorder() for etiqueta in leyenda]
    assert alturas == sorted(alturas, reverse=True), "el ganador queda debajo"
    baseline = lineas[leyenda[-1]]
    assert list(baseline.get_xdata()) == [0, 1] == list(baseline.get_ydata())
    assert "azar" in leyenda[-1]


def test_sin_baseline_la_diagonal_del_azar_va_aparte(prediccion, figuras, tmp_path):
    """Si el baseline no está entre los modelos, la diagonal del azar se dibuja igual."""
    y_true = prediccion[0]
    modelos_proba = {k: v for k, v in _tres_modelos(prediccion).items() if k != "baseline"}
    evaluator.curva_roc(modelos_proba, y_true, tmp_path / "roc.png")

    ax = figuras[-1].axes[0]
    leyenda = [t.get_text() for t in ax.get_legend().get_texts()]
    assert len(leyenda) == len(modelos_proba) + 1
    diagonal = {linea.get_label(): linea for linea in ax.get_lines()}[leyenda[-1]]
    assert list(diagonal.get_xdata()) == [0, 1] == list(diagonal.get_ydata())


# ── importancias() ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def datos_demo():
    """El CSV real en modo demo, para calcular la importancia con un Pipeline real
    sobre las columnas originales."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "DEMO", True)
        return data_loader.preparar()


@pytest.fixture(scope="module")
def arbol_demo(datos_demo):
    d = datos_demo
    X, y = d["X_train"].iloc[:3000], d["y_train"].iloc[:3000]
    return Pipeline([("prep", clone(d["preprocesador"])),
                     ("modelo", model_trainer.Arbol())]).fit(X, y)


@pytest.mark.parametrize("modelo",
                         [model_trainer.Arbol(), model_trainer.RedKeras(epocas=10)],
                         ids=["con_feature_importances", "red_sin_feature_importances"])
def test_las_importancias_salen_por_variable_original(modelo, datos_demo, tmp_path,
                                                      monkeypatch):
    """Sale una fila por variable original (27, no las 97 del one-hot), ordenadas.

    También con la red, que no tiene feature_importances_ y necesita unas cuantas
    épocas: con dos no predice ningún «cancela», su F1 ya es 0 y barajar no puede
    bajarlo. Los procesos siguen la regla de la validación cruzada: la red en uno solo
    (en paralelo, cada proceso cargaría TensorFlow sin ganar tiempo) y el resto con
    config.N_JOBS. Se comprueba el n_jobs pedido y se ejecuta con uno para ir rápido."""
    monkeypatch.setattr(config, "IMPORTANCIA_REPETICIONES", 2)
    pedidos = []
    real = evaluator.permutation_importance

    def espia(*args, **kwargs):
        pedidos.append(kwargs["n_jobs"])
        return real(*args, **{**kwargs, "n_jobs": 1})

    monkeypatch.setattr(evaluator, "permutation_importance", espia)
    d = datos_demo
    X, y = d["X_train"].iloc[:3000], d["y_train"].iloc[:3000]
    pipeline = Pipeline([("prep", clone(d["preprocesador"])), ("modelo", clone(modelo))])
    pipeline.fit(X, y)

    ruta = tmp_path / "importancias.png"
    tabla = evaluator.importancias(pipeline, d["X_test"], d["y_test"], ruta)
    assert pedidos == [pipeline[-1].procesos()]
    assert set(tabla.index) == set(d["X_test"].columns)
    assert len(tabla) == 27
    assert tabla["media"].is_monotonic_decreasing
    assert {"media", "std"} <= set(tabla.columns)
    assert tabla["media"].iloc[0] > 0
    assert _png_con_contenido(ruta)


def test_las_importancias_miden_la_metrica_principal_sobre_la_X_que_reciben(
        arbol_demo, datos_demo, figuras, tmp_path, monkeypatch):
    """La tabla es permutation_importance con la métrica principal al umbral de config.

    Se calcula sobre la X que recibe y cada media va en la fila de su variable. La
    figura muestra las config.TOP_IMPORTANCIAS primeras, la más importante arriba, con
    su valor y en el color del ganador."""
    monkeypatch.setattr(config, "N_JOBS", 1)
    monkeypatch.setattr(config, "IMPORTANCIA_REPETICIONES", 2)
    monkeypatch.setattr(config, "TOP_IMPORTANCIAS", 5)
    X, y = datos_demo["X_test"], datos_demo["y_test"]
    tabla = evaluator.importancias(arbol_demo, X, y, tmp_path / "importancias.png")

    puntuar = partial(evaluator._puntuacion, metrica=config.METRICA_PRINCIPAL,
                      umbral=config.UMBRAL)
    esperado = permutation_importance(arbol_demo, X, y, scoring=puntuar, n_repeats=2,
                                      random_state=config.SEMILLA)
    pd.testing.assert_series_equal(
        tabla["media"].sort_index(),
        pd.Series(esperado.importances_mean, index=X.columns).sort_index(),
        check_names=False, check_index_type=False)

    ax = figuras[-1].axes[0]
    de_arriba_abajo = [t.get_text() for t in ax.get_yticklabels()][::-1]
    assert de_arriba_abajo == list(tabla.index[:5])
    assert [b.get_width() for b in ax.patches][::-1] == pytest.approx(
        list(tabla["media"].iloc[:5]))
    assert ax.patches[0].get_facecolor() == to_rgba(evaluator._COLORES["arbol"])


def test_la_importancia_usa_el_umbral_de_config_tambien_en_paralelo(
        arbol_demo, datos_demo, tmp_path, monkeypatch):
    """La importancia usa el umbral de config, también con varios procesos.

    Con otro umbral, la importancia se mide a ese umbral, igual que las métricas de
    test. Los procesos hijos importan config de cero: si leyeran el umbral de config
    verían el 0,50 y la tabla cambiaría."""
    monkeypatch.setattr(config, "UMBRAL", 0.35)
    monkeypatch.setattr(config, "IMPORTANCIA_REPETICIONES", 2)
    X, y = datos_demo["X_test"].iloc[:1500], datos_demo["y_test"].iloc[:1500]

    monkeypatch.setattr(config, "N_JOBS", 1)
    en_uno = evaluator.importancias(arbol_demo, X, y, tmp_path / "uno.png")
    monkeypatch.setattr(config, "N_JOBS", 2)
    en_dos = evaluator.importancias(arbol_demo, X, y, tmp_path / "dos.png")
    pd.testing.assert_frame_equal(en_uno.sort_index(), en_dos.sort_index())

    al_050 = partial(evaluator._puntuacion, metrica="f1", umbral=0.5)
    esperado = permutation_importance(arbol_demo, X, y, scoring=al_050, n_repeats=2,
                                      random_state=config.SEMILLA)
    medida_al_050 = pd.Series(esperado.importances_mean, index=X.columns).sort_index()
    assert not np.allclose(en_uno.sort_index()["media"], medida_al_050)


# ── informe() ────────────────────────────────────────────────────────────────

@pytest.fixture
def tabla_de_juguete():
    return pd.DataFrame({"f1": [0.69, 0.66, 0.0], "f1_std": [0.01, 0.01, 0.0],
                         "tiempo_s": [9.1, 104.6, 3.0]},
                        index=pd.Index(["boosting", "red_keras", "baseline"],
                                       name="modelo"))


def test_el_informe_escribe_lo_que_lee_el_notebook(tabla_de_juguete, prediccion, tmp_path):
    """El CSV y el JSON de informe() tienen el formato que lee el notebook.

    La tabla se vuelve a leer con su columna «modelo» (el notebook la pasa a
    to_markdown sin índice) y el JSON es plano porque el notebook lo lee con pd.Series.
    El ganador sale con la misma regla que elegir_mejor."""
    y_true, y_pred, y_proba = prediccion
    m = evaluator.metricas(y_true, y_pred, y_proba)
    ruta_tabla, ruta_json = evaluator.informe(tabla_de_juguete, m, tmp_path)

    assert ruta_tabla.name == config.TABLA_COMPARATIVA.name
    assert ruta_json.name == config.METRICAS_TEST.name
    leida = pd.read_csv(ruta_tabla)
    assert list(leida["modelo"]) == list(tabla_de_juguete.index)
    assert leida["f1"].tolist() == pytest.approx(tabla_de_juguete["f1"].tolist())

    resumen = json.loads(ruta_json.read_text(encoding="utf-8"))
    assert resumen["ganador"] == model_trainer.elegir_mejor(tabla_de_juguete) == "boosting"
    assert resumen["metrica_principal"] == config.METRICA_PRINCIPAL
    assert resumen["umbral"] == config.UMBRAL
    for clave, valor in m.items():
        assert resumen[clave] == pytest.approx(valor)
    assert not any(isinstance(v, (dict, list)) for v in resumen.values())
    pd.Series(resumen).to_frame("valor")


def test_el_informe_escribe_en_la_carpeta_que_se_le_pasa(tabla_de_juguete, tmp_path,
                                                         monkeypatch):
    """Con `ruta`, informe() escribe ahí y no toca config.OUTPUTS.

    Si no, cada ejecución de pytest sobrescribiría outputs/, de donde salen las cifras
    del README y del notebook, con la tabla de juguete."""
    reales = tmp_path / "outputs_reales"
    monkeypatch.setattr(config, "OUTPUTS", reales)
    carpeta = tmp_path / "aqui"
    ruta_tabla, ruta_json = evaluator.informe(tabla_de_juguete, {"f1": 0.5}, carpeta)
    assert ruta_tabla == carpeta / config.TABLA_COMPARATIVA.name and ruta_tabla.exists()
    assert ruta_json == carpeta / config.METRICAS_TEST.name and ruta_json.exists()
    assert not reales.exists()


def test_el_informe_normaliza_tipos_y_elige_el_ganador_con_la_tabla_desordenada(
        tmp_path, monkeypatch):
    """informe() convierte los tipos de numpy y no depende del orden de la tabla.

    n sale como entero (el README cita 17.163, no 17163.0), un float32 no hace fallar a
    json, el indicador demo se escribe tal cual, el ganador es el de la métrica
    principal aunque la tabla venga desordenada, y el CSV no redondea las
    desviaciones."""
    monkeypatch.setattr(config, "DEMO", True)
    tabla = pd.DataFrame({"f1": [0.6612, 0.6856, 0.0], "f1_std": [0.0123, 0.0042, 0.0]},
                         index=pd.Index(["red_keras", "boosting", "baseline"],
                                        name="modelo"))
    metricas = {"f1": np.float32(0.6856), "n": np.int64(17163)}
    _, ruta_json = evaluator.informe(tabla, metricas, tmp_path)

    resumen = json.loads(ruta_json.read_text(encoding="utf-8"))
    assert resumen["ganador"] == "boosting"
    assert resumen["n"] == 17163 and type(resumen["n"]) is int
    assert resumen["f1"] == pytest.approx(0.6856)
    assert resumen["demo"] is True
    leida = pd.read_csv(tmp_path / config.TABLA_COMPARATIVA.name)
    assert leida["f1_std"].tolist() == pytest.approx([0.0123, 0.0042, 0.0])


def test_un_informe_con_nan_no_escribe_nada(tabla_de_juguete, tmp_path):
    """Con un NaN en las métricas, informe() lanza un error antes de escribir nada.

    Si no, el NaN acabaría copiado en el README; así no se escribe ni la tabla nueva ni
    el JSON."""
    with pytest.raises(ValueError):
        evaluator.informe(tabla_de_juguete, {"f1": float("nan")}, tmp_path)
    assert not any(tmp_path.iterdir())


# ── De principio a fin ───────────────────────────────────────────────────────

def test_main_llega_hasta_el_final_y_deja_las_salidas(monkeypatch, tmp_path):
    """main.py en modo demo llega hasta el final y deja todas las salidas.

    Con tres modelos rápidos y las salidas en una carpeta temporal, comprueba los cinco
    ficheros de outputs/, el modelo con sus metadatos y que cada función recibe lo que
    le corresponde: la ROC con todos los modelos, la importancia y la matriz sobre el
    test, y el mismo n en el JSON y en los metadatos."""
    salidas, modelos = tmp_path / "outputs", tmp_path / "models"
    for clave, ruta in {
        "OUTPUTS": salidas,
        "FIG_CONFUSION": salidas / config.FIG_CONFUSION.name,
        "FIG_ROC": salidas / config.FIG_ROC.name,
        "FIG_IMPORTANCIAS": salidas / config.FIG_IMPORTANCIAS.name,
        "TABLA_COMPARATIVA": salidas / config.TABLA_COMPARATIVA.name,
        "METRICAS_TEST": salidas / config.METRICAS_TEST.name,
        "MODELO_PKL": modelos / config.MODELO_PKL.name,
        "MODELO_KERAS": modelos / config.MODELO_KERAS.name,
        "METADATOS": modelos / config.METADATOS.name,
    }.items():
        monkeypatch.setattr(config, clave, ruta)
    monkeypatch.setattr(config, "DEMO", False)  # main lo pone a True; así se restaura
    monkeypatch.setattr(config, "MODELOS_ACTIVOS", ["baseline", "logistica", "arbol"])
    monkeypatch.setattr(config, "IMPORTANCIA_REPETICIONES", 2)

    llamadas = {}

    def espiar(nombre):
        original = getattr(evaluator, nombre)

        def espia(*args, **kwargs):
            llamadas[nombre] = (args, original(*args, **kwargs))
            return llamadas[nombre][1]
        return espia

    for nombre in ("metricas", "matriz_confusion", "curva_roc", "importancias"):
        monkeypatch.setattr(evaluator, nombre, espiar(nombre))

    main.main(demo=True)

    for figura in (config.FIG_CONFUSION, config.FIG_ROC, config.FIG_IMPORTANCIAS):
        assert _png_con_contenido(figura)
    assert len(pd.read_csv(config.TABLA_COMPARATIVA)) == 3
    resumen = json.loads(config.METRICAS_TEST.read_text(encoding="utf-8"))
    metadatos = json.loads(config.METADATOS.read_text(encoding="utf-8"))
    assert resumen["ganador"] == metadatos["ganador"]
    assert metadatos["metricas_test"]["f1"] == pytest.approx(resumen["f1"])
    assert config.MODELO_PKL.exists()

    (y_test, _, _), m = llamadas["metricas"]
    (probas, _), aucs = llamadas["curva_roc"]
    assert set(probas) == set(config.MODELOS_ACTIVOS), "la ROC no lleva todos los modelos"
    assert aucs[resumen["ganador"]] == pytest.approx(m["roc_auc"])
    assert llamadas["importancias"][0][1].index.equals(y_test.index), \
        "la importancia no se midió sobre el test"
    np.testing.assert_array_equal(llamadas["matriz_confusion"][1].sum(axis=1),
                                  np.bincount(y_test))
    assert resumen["demo"] is True and metadatos["demo"] is True
    assert resumen["n"] == metadatos["metricas_test"]["n"] == len(y_test)
    assert type(metadatos["metricas_test"]["n"]) is int
