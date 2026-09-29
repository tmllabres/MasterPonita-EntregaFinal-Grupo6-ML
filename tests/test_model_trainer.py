"""Tests de model_trainer: el registro de modelos, la comparación y el guardado.

El registro imita a las librerías de AutoML, así que lo principal es comprobar que las
seis clases son intercambiables: si una no sigue la interfaz común, el bucle que las
compara falla con un mensaje poco claro.

    python -m pytest tests/test_model_trainer.py -q
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src import config, data_loader, model_trainer


def test_todos_los_modelos_activos_estan_en_el_registro():
    """Todos los modelos de config.MODELOS_ACTIVOS están en el REGISTRO."""
    for nombre in config.MODELOS_ACTIVOS:
        assert nombre in model_trainer.REGISTRO


@pytest.mark.parametrize("nombre", config.MODELOS_ACTIVOS)
def test_cada_modelo_cumple_la_interfaz(nombre):
    """Cada modelo se llama como su clave y tiene los métodos de la interfaz común."""
    modelo = model_trainer.REGISTRO[nombre]()
    assert modelo.nombre == nombre
    for metodo in ("construir", "fit", "predict", "predict_proba", "espacio_busqueda"):
        assert callable(getattr(modelo, metodo))


@pytest.mark.parametrize("nombre", config.MODELOS_ACTIVOS)
def test_clone_funciona(nombre):
    """clone() funciona con cada modelo del registro.

    La validación cruzada usa clone() para dar a cada fold un modelo sin entrenar, y
    falla si __init__ hace algo más que guardar hiperparámetros, por ejemplo construir
    la red de Keras en __init__ en vez de en fit."""
    modelo = model_trainer.REGISTRO[nombre]()
    assert clone(modelo) is not modelo


def test_el_espacio_de_busqueda_usa_la_sintaxis_de_pipeline():
    """Las claves de cada rejilla usan paso__hiperparametro y existen en el Pipeline.

    Sin el prefijo, GridSearchCV no encuentra el hiperparámetro y falla al ejecutarse,
    no al escribir la rejilla. set_params valida cada clave igual que la búsqueda."""
    for nombre in config.MODELOS_ACTIVOS:
        clase = model_trainer.REGISTRO[nombre]
        rejilla = clase().espacio_busqueda()
        for clave in rejilla:
            assert "__" in clave, f"{nombre}: la clave «{clave}» no lleva prefijo de paso"
        pipeline = Pipeline([("prep", "passthrough"), ("modelo", clase())])
        pipeline.set_params(**{clave: valores[0] for clave, valores in rejilla.items()})


# ── Cada modelo por separado, con datos de juguete ───────────────────────────

@pytest.fixture(scope="module")
def juguete():
    """400 filas numéricas inventadas, con el reparto de clases del problema.

    Para probar la interfaz no hace falta el CSV, y así la red entrena en segundos."""
    X, y = make_classification(n_samples=400, n_features=6, weights=[0.72],
                               random_state=config.SEMILLA)
    return X, y


@pytest.mark.parametrize("nombre", config.MODELOS_ACTIVOS)
def test_predict_proba_da_dos_columnas_y_la_1_es_cancela(nombre, juguete):
    """predict_proba da dos columnas, la 1 es «cancela» y predict coincide con ella.

    main.py usa predict_proba(...)[:, 1]: con un vector plano el índice fallaría, y con
    las columnas al revés no fallaría nada pero el AUC saldría por debajo de 0,5. Por
    eso la columna 1 tiene que ordenar las reservas muy por encima del azar, y predict
    (de donde salen F1, precision y recall en la validación cruzada) debe coincidir."""
    X, y = juguete
    modelo = model_trainer.REGISTRO[nombre]()
    assert modelo.fit(X, y) is modelo

    proba = modelo.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1, rtol=1e-6)
    assert list(modelo.classes_) == [0, 1]

    sin_empate = proba[:, 1] != 0.5
    np.testing.assert_array_equal(modelo.predict(X)[sin_empate],
                                  (proba[:, 1] >= 0.5).astype(int)[sin_empate])
    if nombre != "baseline":
        assert roc_auc_score(y, proba[:, 1]) > 0.7


@pytest.mark.parametrize("nombre", [n for n in config.MODELOS_ACTIVOS if n != "red_keras"])
def test_dos_modelos_nuevos_dan_lo_mismo(nombre, juguete):
    """Dos modelos nuevos con los mismos datos dan las mismas probabilidades.

    La semilla de config llega a cada estimador; sin ella, la tabla y las cifras del
    README cambiarían entre ejecuciones. Uso atol y no igualdad exacta porque el bosque
    suma en paralelo y el orden cambia el 1e-16. La red tiene su propio test abajo."""
    X, y = juguete
    a = model_trainer.REGISTRO[nombre]().fit(X, y).predict_proba(X)
    b = model_trainer.REGISTRO[nombre]().fit(X, y).predict_proba(X)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)


def test_la_red_se_reconstruye_en_cada_fit(juguete):
    """Dos fit seguidos de la misma red, y un clon, dan exactamente lo mismo.

    La red se construye en fit y no en __init__. Si se construyera una sola vez, el
    segundo fit seguiría entrenando la del primero (saldría otra red, en apariencia
    mejor) y en la validación cruzada cada fold empezaría con lo aprendido en los
    anteriores. Con la red nueva en cada fit y la semilla fijada, las tres coinciden."""
    X, y = juguete
    red = model_trainer.RedKeras(epocas=3)
    primera = red.fit(X, y).predict_proba(X)
    segunda = red.fit(X, y).predict_proba(X)
    clonada = clone(red).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(primera, segunda)
    np.testing.assert_array_equal(primera, clonada)


def test_la_red_lee_el_modo_demo_al_entrenar(juguete, monkeypatch):
    """La red lee config.DEMO en fit y entrena con validation_split.

    main.py pone config.DEMO = True después de importar los módulos; si la red no lo
    leyera en fit, --demo entrenaría todas las épocas y dejaría de ser la versión
    corta. Sin validation_split no habría val_loss y EarlyStopping no pararía nunca."""
    X, y = juguete
    monkeypatch.setattr(config, "DEMO", True)
    red = model_trainer.RedKeras(epocas=50, paciencia=50).fit(X, y)
    assert len(red.model_.history.history["loss"]) == config.DEMO_EPOCAS
    assert "val_loss" in red.model_.history.history


def test_la_red_se_queda_con_los_mejores_pesos(juguete):
    """Con restore_best_weights, la red se queda con la mejor época, no con la última.

    La mejor época se mide en validación, que es el último 10 % del train (lo que
    aparta validation_split)."""
    X, y = juguete
    red = model_trainer.RedKeras(epocas=60, paciencia=2, learning_rate=0.05).fit(X, y)
    historia = red.model_.history.history["val_loss"]
    assert len(historia) < 60, "no ha llegado a pararse: el test no probaría nada"
    n_val = int(len(X) * 0.1)
    perdida = red.model_.evaluate(X[-n_val:].astype("float32"),
                                  y[-n_val:].astype("float32"), batch_size=256, verbose=0)
    assert perdida == pytest.approx(min(historia), rel=1e-4)


# ── El comparador, sobre el CSV real ─────────────────────────────────────────

@pytest.fixture(scope="module")
def comparacion():
    """Ejecuta entrenar_y_comparar() una vez, en modo demo y con tres modelos rápidos.

    El protocolo es el mismo para los seis, y la red tiene sus propios tests arriba."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "DEMO", True)
        mp.setattr(config, "MODELOS_ACTIVOS", ["baseline", "logistica", "arbol"])
        d = data_loader.preparar()
        tabla, modelos = model_trainer.entrenar_y_comparar(d["X_train"], d["y_train"],
                                                           d["preprocesador"])
    return d, tabla, modelos


def test_la_tabla_trae_una_fila_por_modelo_ordenada(comparacion):
    """La tabla tiene una fila por modelo activo, con la media y la desviación de cada
    métrica, y está ordenada por la principal: la primera fila es la del ganador."""
    _, tabla, modelos = comparacion
    assert set(tabla.index) == set(modelos) == {"baseline", "logistica", "arbol"}
    assert tabla[config.METRICA_PRINCIPAL].is_monotonic_decreasing
    for metrica in [config.METRICA_PRINCIPAL, *config.METRICAS_SECUNDARIAS]:
        assert {metrica, f"{metrica}_std"} <= set(tabla.columns)
    assert "tiempo_s" in tabla.columns


def test_el_baseline_es_la_linea_del_suelo(comparacion):
    """El baseline da F1 0, AUC 0,5 y una accuracy igual al porcentaje de «no cancela».

    Es la referencia: dice siempre «no cancela», y los demás modelos tienen que quedar
    por encima en la métrica principal."""
    d, tabla, _ = comparacion
    suelo = tabla.loc["baseline"]
    assert suelo["f1"] == 0 and suelo["roc_auc"] == 0.5
    assert suelo["accuracy"] == pytest.approx(1 - d["y_train"].mean(), abs=1e-3)
    assert (tabla.drop(index="baseline")[config.METRICA_PRINCIPAL] > 0.3).all()


def test_cada_modelo_sale_ajustado_con_su_propio_preprocesador(comparacion):
    """Cada Pipeline sale ajustado con todo el train y con su propio preprocesador.

    Hacen falta ajustados para la ROC. Si compartieran el preprocesador, el refit de
    uno reajustaría el de los demás; y el de preparar() tiene que seguir sin ajustar."""
    d, _, modelos = comparacion
    preprocesadores = [p.named_steps["prep"] for p in modelos.values()]
    assert len({id(p) for p in preprocesadores}) == len(modelos)
    assert all(p is not d["preprocesador"] for p in preprocesadores)
    for pipeline in modelos.values():
        check_is_fitted(pipeline)
        assert pipeline.predict_proba(d["X_test"]).shape == (len(d["X_test"]), 2)
    assert not hasattr(d["preprocesador"], "transformers_")


def test_mismo_protocolo_para_todos(comparacion, monkeypatch):
    """Todos los modelos se validan con los mismos folds y las mismas reglas.

    Mismos folds para todos (los de --demo, estratificados y con la misma semilla), la
    red en un solo proceso, un fold que falla para la ejecución en vez de dar NaN, y la
    tabla lleva la media y la desviación de esos folds. cross_validate se cambia por un
    espía que no entrena: guarda con qué se le llamó y da un valor distinto por fold."""
    d, _, _ = comparacion
    X, y = d["X_train"].iloc[:1500], d["y_train"].iloc[:1500]
    monkeypatch.setattr(config, "DEMO", True)
    monkeypatch.setattr(config, "MODELOS_ACTIVOS", ["baseline", "arbol", "red_keras"])
    # Sin búsqueda, para que los tres pasen por cross_validate; los folds del buscador
    # los comprueba test_la_busqueda_se_activa_desde_config.
    monkeypatch.setattr(config, "BUSQUEDA", None)
    por_fold = np.array([0.2, 0.4, 0.9])
    llamadas = {}

    def espia(estimador, X_, y_, **kw):
        llamadas[estimador[-1].nombre] = kw
        return {f"test_{m}": por_fold for m in kw["scoring"]}

    monkeypatch.setattr(model_trainer, "cross_validate", espia)
    tabla, _ = model_trainer.entrenar_y_comparar(X, y, d["preprocesador"])

    principal = config.METRICA_PRINCIPAL
    particiones = []
    for nombre, kw in llamadas.items():
        cv = kw["cv"]
        assert cv.get_n_splits() == config.DEMO_FOLDS, "--demo no reduce los folds"
        folds = [tuple(te) for _, te in cv.split(X, y)]
        particiones.append(folds)
        for te in folds:  # estratificado: los positivos de cada fold, a ±2 del reparto
            assert abs(y.iloc[list(te)].sum() - len(te) * y.mean()) <= 2
        assert kw["error_score"] == "raise"
        assert tabla.loc[nombre, principal] == pytest.approx(por_fold.mean())
        assert tabla.loc[nombre, f"{principal}_std"] == pytest.approx(por_fold.std())
    assert set(llamadas) == {"baseline", "arbol", "red_keras"}
    assert all(p == particiones[0] for p in particiones), "cada modelo con sus propios folds"
    assert llamadas["red_keras"]["n_jobs"] == 1


def test_elegir_mejor_lee_la_metrica_de_config(monkeypatch):
    """elegir_mejor() elige por la columna de config.METRICA_PRINCIPAL.

    No usa un "f1" escrito a mano: si cambia la métrica, cambia el ganador sin tocar
    el código."""
    tabla = pd.DataFrame({"f1": [0.60, 0.55], "roc_auc": [0.85, 0.90]},
                         index=pd.Index(["a", "b"], name="modelo"))
    assert model_trainer.elegir_mejor(tabla) == "a"
    monkeypatch.setattr(config, "METRICA_PRINCIPAL", "roc_auc")
    assert model_trainer.elegir_mejor(tabla) == "b"


@pytest.mark.parametrize("busqueda,clase", [("grid", "GridSearchCV"),
                                             ("random", "RandomizedSearchCV")])
def test_la_busqueda_se_activa_desde_config(busqueda, clase, monkeypatch):
    """Con config.BUSQUEDA, el comparador lanza de verdad el buscador correspondiente.

    El buscador elige por la métrica principal (y no por accuracy, que es lo que usaría
    por defecto), devuelve su mejor Pipeline y la tabla recoge los folds de esa
    combinación. Con max_depth 1 contra 8 el ganador está claro. El baseline no tiene
    rejilla y se valida sin buscador."""
    monkeypatch.setattr(config, "DEMO", True)
    monkeypatch.setattr(config, "BUSQUEDA", busqueda)
    monkeypatch.setattr(config, "N_ITER_RANDOM", 30)
    monkeypatch.setattr(config, "MODELOS_ACTIVOS", ["baseline", "arbol"])
    monkeypatch.setattr(model_trainer.Arbol, "espacio_busqueda",
                        lambda self: {"modelo__max_depth": [1, 8]})
    buscadores = []
    real = getattr(model_trainer, clase)

    def espia(*args, **kwargs):
        buscadores.append(real(*args, **kwargs))
        return buscadores[-1]

    monkeypatch.setattr(model_trainer, clase, espia)
    d = data_loader.preparar()
    tabla, modelos = model_trainer.entrenar_y_comparar(d["X_train"], d["y_train"],
                                                       d["preprocesador"])
    assert set(tabla.index) == {"baseline", "arbol"}
    assert len(buscadores) == 1, f"BUSQUEDA={busqueda!r}: se esperaba un buscador, el del árbol"
    b, principal = buscadores[0], config.METRICA_PRINCIPAL
    assert b.refit == principal
    assert modelos["arbol"] is b.best_estimator_
    assert modelos["arbol"][-1].max_depth == 8
    assert tabla.loc["arbol", principal] == pytest.approx(
        b.cv_results_[f"mean_test_{principal}"][b.best_index_])
    check_is_fitted(modelos["arbol"])


# ── Guardar ──────────────────────────────────────────────────────────────────

def test_guardar_deja_el_pipeline_completo_y_sus_metadatos(comparacion, tmp_path,
                                                           monkeypatch):
    """guardar() guarda el Pipeline completo y los metadatos que necesita predictor.

    Recargado, el Pipeline predice lo mismo sobre reservas sin preprocesar. El umbral
    se cambia de 0,50 para que un 0,5 escrito a mano no pase el test."""
    d, _, modelos = comparacion
    monkeypatch.setattr(config, "UMBRAL", 0.35)
    ruta = model_trainer.guardar(modelos["arbol"], "arbol", {"f1": np.float64(0.6)},
                                 tmp_path / "mejor_modelo.pkl")

    metadatos = json.loads((tmp_path / config.METADATOS.name).read_text(encoding="utf-8"))
    assert metadatos["ganador"] == "arbol"
    assert metadatos["umbral"] == 0.35
    assert metadatos["columnas"] == list(d["X_train"].columns)
    assert metadatos["metrica_principal"] == config.METRICA_PRINCIPAL
    assert metadatos["semilla"] == config.SEMILLA
    assert metadatos["ficheros"] == [ruta.name]
    assert "scikit-learn" in metadatos["versiones"]
    assert metadatos["hiperparametros"] == modelos["arbol"][-1].get_params()

    recargado = joblib.load(ruta)
    np.testing.assert_array_equal(recargado.predict_proba(d["X_test"]),
                                  modelos["arbol"].predict_proba(d["X_test"]))


def test_guardar_que_falla_no_deja_un_artefacto_a_medias(comparacion, tmp_path):
    """Si guardar() falla, el modelo que ya estaba guardado queda como estaba.

    Aquí falla al escribir los metadatos (una métrica que no es un número). El caso
    peligroso sería un .pkl nuevo junto a los metadatos del ganador anterior, porque
    predictor lo cargaría sin ningún error."""
    _, _, modelos = comparacion
    ruta = model_trainer.guardar(modelos["arbol"], "arbol", {"f1": 0.6},
                                 tmp_path / "mejor_modelo.pkl")
    with pytest.raises(TypeError):
        model_trainer.guardar(modelos["logistica"], "logistica", {"roc_auc": None}, ruta)

    metadatos = json.loads((tmp_path / config.METADATOS.name).read_text(encoding="utf-8"))
    assert metadatos["ganador"] == "arbol"
    assert isinstance(joblib.load(ruta)[-1], model_trainer.Arbol)


def test_guardar_la_red_va_en_dos_ficheros(comparacion, tmp_path):
    """Si gana la red, va a un .keras aparte y el resto del Pipeline al .pkl.

    La red en memoria no se modifica, el .pkl no la lleva dentro y, sin cargar_aparte,
    el error dice qué falta. Si después gana otro modelo en la misma carpeta, el .keras
    antiguo no se puede quedar al lado."""
    d, _, modelos = comparacion
    X, y = d["X_train"].iloc[:2000], d["y_train"].iloc[:2000]
    pipeline = Pipeline([("prep", clone(d["preprocesador"])),
                         ("modelo", model_trainer.RedKeras(epocas=2))]).fit(X, y)
    ruta = model_trainer.guardar(pipeline, "red_keras", {"f1": 0.5},
                                 tmp_path / "mejor_modelo.pkl")
    ruta_keras = tmp_path / config.MODELO_KERAS.name
    assert ruta_keras.exists()
    assert pipeline[-1].model_ is not None
    metadatos = json.loads((tmp_path / config.METADATOS.name).read_text(encoding="utf-8"))
    assert metadatos["ficheros"] == [ruta.name, ruta_keras.name]

    recargado = joblib.load(ruta)
    assert recargado[-1].model_ is None, "la red viaja dentro del .pkl, no solo en el .keras"
    with pytest.raises(RuntimeError, match="cargar_aparte"):
        recargado.predict_proba(d["X_test"])
    recargado[-1].cargar_aparte(ruta_keras)
    np.testing.assert_allclose(recargado.predict_proba(d["X_test"]),
                               pipeline.predict_proba(d["X_test"]), rtol=1e-6)

    model_trainer.guardar(modelos["arbol"], "arbol", {"f1": 0.6}, ruta)
    assert not ruta_keras.exists()
