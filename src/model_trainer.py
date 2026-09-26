"""Entrenamiento y comparación de los modelos.

Aquí está la idea que pide el enunciado: imitar por dentro a una librería de AutoML.
Cada modelo es una CLASE con la misma interfaz, viven todas en un REGISTRO, y un único
bucle las recorre sin un solo `if`. Añadir un modelo más = escribir su clase y una
línea en el registro; el bucle no cambia ni una letra.
"""
from __future__ import annotations

import copy
import json
import os
import platform
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import config


def _keras():
    """Importa Keras solo cuando hace falta.

    TensorFlow tarda unos 10 s en importarse. Si se importara arriba, cualquier cosa que
    toque este módulo —los tests de contrato, o predictor con un ganador que no es la
    red— pagaría esa espera sin llegar a usarlo.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # sin el ruido de arranque de TF
    import keras
    return keras


# ── El contrato ──────────────────────────────────────────────────────────────
class ModeloBase(BaseEstimator, ClassifierMixin):
    """Interfaz común. Todo modelo del registro cumple esto.

    Reglas de scikit-learn que hay que respetar para que clone(), GridSearchCV y
    cross_val_score funcionen:
      · __init__ SOLO guarda hiperparámetros, uno por argumento, sin validar ni
        construir nada. Lo que se aprende no se toca aquí.
      · lo aprendido se guarda en atributos con GUION BAJO FINAL: self.model_
      · fit devuelve self
    """

    nombre = "base"

    @classmethod
    def procesos(cls) -> int:
        """Procesos con los que se reparten los folds de ESTE modelo en la validación
        cruzada. Se lee de config al llamarlo, no al importar. La red lo sobrescribe."""
        return config.N_JOBS

    def construir(self):
        """Devuelve el estimador o Pipeline sin entrenar. Lo implementa cada hijo."""
        raise NotImplementedError

    def espacio_busqueda(self) -> dict:
        """Rejilla de hiperparámetros con la sintaxis paso__hiperparametro.

        Ej.: {"modelo__max_depth": [4, 6, 8]}. Los dobles guiones bajos son el
        camino hasta el tornillo: cada tramo es la etiqueta que le pusiste al paso.
        """
        return {}

    def fit(self, X, y):
        """Construye un estimador NUEVO, lo entrena y devuelve self.

        Nuevo en cada fit, nunca reutilizado: así dos fit seguidos no comparten nada,
        que es lo que clone() da por hecho cuando reparte un modelo virgen a cada fold.
        """
        self.model_ = self.construir()
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        """Siempre una matriz (n, 2): columna 0 «no cancela», columna 1 «cancela»."""
        return self.model_.predict_proba(X)

    # ── Lo que no va dentro del .pkl ──
    def guardar_aparte(self, ruta):
        """Guarda en `ruta` lo que no debe ir dentro del .pkl, y devuelve el modelo tal
        como tiene que entrar en él. Casi todos caben enteros: se devuelven a sí mismos
        y no escriben nada. La excepción es la red: ver RedKeras."""
        return self

    def cargar_aparte(self, ruta) -> None:
        """El camino de vuelta de guardar_aparte(). Lo llama predictor.cargar()."""


# ── La linea del suelo ───────────────────────────────────────────────────────
class Baseline(ModeloBase):
    """DummyClassifier: la referencia contra la que se miden los cinco de verdad.

    No aprende nada, y ese es justo el punto: con strategy="most_frequent" dice
    siempre "no cancela" y ya acierta el 72,36 % (el 62,96 % sobre el CSV crudo, antes
    de quitar los duplicados). Cualquier modelo que no supere claramente esta fila no
    está aportando nada, y sin la fila no hay forma de saberlo.

    Ojo con la métrica: en accuracy saca 0,72, pero en F1 de la clase positiva saca
    0,00, porque nunca predice un 1. Las dos cosas dicen lo mismo desde dos sitios.
    """
    nombre = "baseline"

    def construir(self):
        return DummyClassifier(strategy="most_frequent")


# ── Los cinco obligatorios ───────────────────────────────────────────────────
class Logistica(ModeloBase):
    """Regresión logística. Necesita escalado. C es la inversa de la regularización.

    El escalado ya lo hace el preprocesador, así que aquí no se repite. max_iter va con
    margen: si lbfgs se queda sin iteraciones no falla, solo avisa, y el modelo se queda
    a medio entrenar.
    """
    nombre = "logistica"

    def __init__(self, C=1.0, max_iter=1000):
        self.C = C
        self.max_iter = max_iter

    def construir(self):
        return LogisticRegression(C=self.C, max_iter=self.max_iter)

    def espacio_busqueda(self) -> dict:
        return {"modelo__C": [0.01, 0.1, 1.0, 10.0]}


class Arbol(ModeloBase):
    """Árbol de decisión. No necesita escalado. Vigila max_depth y min_samples_leaf.

    Sin esos dos límites el árbol crece hasta dejar una reserva por hoja: acierta el
    train entero de memoria y en validación se hunde.
    """
    nombre = "arbol"

    def __init__(self, max_depth=10, min_samples_leaf=50):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def construir(self):
        return DecisionTreeClassifier(max_depth=self.max_depth,
                                      min_samples_leaf=self.min_samples_leaf,
                                      random_state=config.SEMILLA)

    def espacio_busqueda(self) -> dict:
        return {"modelo__max_depth": [6, 8, 10, 14, None],
                "modelo__min_samples_leaf": [10, 50, 100]}


class Bosque(ModeloBase):
    """Random Forest. Bagging: cada árbol ve ~63,2 % de filas distintas y un sorteo
    de columnas (max_features). Trae feature_importances_ ya calculado.

    min_samples_leaf=5 y no 1 (el valor de scikit-learn), elegido de antemano por
    tamaño: con hojas de una sola reserva, los 300 árboles pesan 665 MB sin comprimir y
    108 MB con la compresión de guardar(), frente a 142 y 35 MB con 5. Tiene un coste,
    medido después y declarado: en validación cruzada el F1 baja de 0,681 a 0,654 con el
    mismo AUC (0,895), porque las probabilidades salen más prudentes y cruzan menos veces
    el umbral de 0,5. El ganador no cambia.
    """
    nombre = "bosque"

    def __init__(self, n_estimators=300, min_samples_leaf=5, max_features="sqrt"):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

    def construir(self):
        return RandomForestClassifier(n_estimators=self.n_estimators,
                                      min_samples_leaf=self.min_samples_leaf,
                                      max_features=self.max_features,
                                      n_jobs=config.N_JOBS,
                                      random_state=config.SEMILLA)

    def espacio_busqueda(self) -> dict:
        return {"modelo__n_estimators": [200, 300, 500],
                "modelo__min_samples_leaf": [1, 5, 20],
                "modelo__max_features": ["sqrt", 0.3]}


class Boosting(ModeloBase):
    """Gradient Boosting. learning_rate y n_estimators son un solo mando: si bajas
    uno, sube el otro. Sus árboles son de REGRESIÓN: cada hoja guarda una corrección.

    XGBoost, de los tres que acepta el enunciado. subsample y colsample_bytree sortean
    filas y columnas en cada árbol, como el bosque, para que no persigan todos el
    mismo ruido.
    """
    nombre = "boosting"

    def __init__(self, n_estimators=400, learning_rate=0.05, max_depth=6,
                 subsample=0.8, colsample_bytree=0.8):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

    def construir(self):
        return XGBClassifier(n_estimators=self.n_estimators,
                             learning_rate=self.learning_rate,
                             max_depth=self.max_depth,
                             subsample=self.subsample,
                             colsample_bytree=self.colsample_bytree,
                             eval_metric="logloss",
                             random_state=config.SEMILLA)

    def espacio_busqueda(self) -> dict:
        return {"modelo__n_estimators": [200, 400, 800],
                "modelo__learning_rate": [0.1, 0.05, 0.02],
                "modelo__max_depth": [4, 6, 8]}


class RedKeras(ModeloBase):
    """MLP con Keras envuelto en la interfaz de scikit-learn.

    Aquí está la trampa del proyecto: la red se construye en FIT, nunca en __init__.
    Si se construye en __init__, clone() reparte el mismo objeto de Keras a los 5
    folds, y como el fit de Keras no reinicia los pesos, el fold 2 arranca habiendo
    visto ya sus datos de validación. El F1 sale inflado y no salta ningún error.

    EarlyStopping aparta el último 10 % del train de cada fit para decidir cuándo parar,
    y se queda con los pesos de la mejor época, no con los de la última.
    """
    nombre = "red_keras"

    @classmethod
    def procesos(cls) -> int:
        """Siempre uno, y no por rendimiento sino por corrección: los procesos hijos de la
        validación cruzada importan config de cero y no ven el config.DEMO = True que
        main.py pone en tiempo de ejecución. En paralelo, cada fold de --demo entrenaría
        las épocas completas, y la fila de la tabla describiría una red distinta de la
        que se guarda, sin ningún error."""
        return 1

    def __init__(self, capas=(64, 32), dropout=0.2, learning_rate=1e-3, epocas=30,
                 batch_size=256, paciencia=3):
        self.capas = capas
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epocas = epocas
        self.batch_size = batch_size
        self.paciencia = paciencia

    def construir(self):
        keras = _keras()
        ocultas = []
        for neuronas in self.capas:
            ocultas += [keras.layers.Dense(neuronas, activation="relu"),
                        keras.layers.Dropout(self.dropout)]
        # Sin capa Input: el número de columnas lo pone el primer fit, así construir()
        # no necesita saberlo y mantiene la misma firma que los demás.
        red = keras.Sequential([*ocultas, keras.layers.Dense(1, activation="sigmoid")])
        red.compile(optimizer=keras.optimizers.Adam(self.learning_rate),
                    loss="binary_crossentropy")
        return red

    def espacio_busqueda(self) -> dict:
        return {"modelo__capas": [(32,), (64, 32), (128, 64)],
                "modelo__dropout": [0.1, 0.2, 0.4]}

    def fit(self, X, y):
        keras = _keras()
        # La semilla ANTES de construir: fija los pesos iniciales, el dropout y el
        # barajado de cada época. Con ella, dos fit sobre los mismos datos dan la misma
        # red; si el segundo saliera mejor, es que no se está reconstruyendo.
        keras.utils.set_random_seed(config.SEMILLA)
        self.model_ = self.construir()

        # config.DEMO se lee aquí, en tiempo de ejecución: ver data_loader.preparar().
        epocas = min(self.epocas, config.DEMO_EPOCAS) if config.DEMO else self.epocas
        parar = keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.paciencia,
                                              restore_best_weights=True)
        self.model_.fit(_denso(X), np.asarray(y, dtype="float32"), epochs=epocas,
                        batch_size=self.batch_size, validation_split=0.1,
                        callbacks=[parar], verbose=0)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        if self.model_ is None:
            raise RuntimeError("La red va en su propio fichero .keras: después de "
                               "joblib.load hay que llamar a cargar_aparte(ruta del .keras).")
        # La red se llama directamente y no con .predict(): predict() monta su propia
        # función de TensorFlow para cada red, y con muchas redes nuevas (una por fold y
        # por combinación de la búsqueda) TensorFlow acaba avisando de que la recompila.
        # Llamarla directamente da el mismo resultado sin esos avisos.
        salida = self.model_(_denso(X), training=False)
        cancela = _keras().ops.convert_to_numpy(salida).ravel()
        return np.column_stack([1 - cancela, cancela])

    def predict(self, X):
        # 0,5 como el predict de scikit-learn. main.py no lo usa: umbraliza la
        # probabilidad contra config.UMBRAL.
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def guardar_aparte(self, ruta):
        """La red va a su propio fichero .keras y el .pkl se lleva una copia sin ella.

        Keras 3 ya deja meter una red en un pickle, pero el formato que garantiza entre
        versiones es el suyo, .keras. El modelo del Pipeline no se toca: la copia es
        superficial y solo a ella se le quita la red.
        """
        self.model_.save(ruta)
        sin_red = copy.copy(self)
        sin_red.model_ = None
        return sin_red

    def cargar_aparte(self, ruta) -> None:
        self.model_ = _keras().models.load_model(ruta)


def _denso(X):
    """Keras no acepta matrices dispersas y trabaja en float32: todo a denso float32."""
    X = X.toarray() if hasattr(X, "toarray") else X
    return np.asarray(X, dtype="float32")


# ── El registro ──────────────────────────────────────────────────────────────
REGISTRO = {
    "baseline": Baseline,
    "logistica": Logistica,
    "arbol": Arbol,
    "bosque": Bosque,
    "boosting": Boosting,
    "red_keras": RedKeras,
}


# ── El comparador ────────────────────────────────────────────────────────────
def entrenar_y_comparar(X_train, y_train, preprocesador) -> tuple:
    """Recorre config.MODELOS_ACTIVOS y devuelve (tabla, modelos).

    Protocolo idéntico para los seis, que es lo que hace justa la comparación:
      · el mismo StratifiedKFold(config.CV_FOLDS) con la misma semilla
      · el preprocesado DENTRO del Pipeline, para que se ajuste en cada fold
      · scoring=config.METRICA_PRINCIPAL — si no lo pones, GridSearchCV optimiza
        accuracy sin decírtelo, y luego presentas F1 en el informe
      · media ± desviación de la validación cruzada, nunca una sola partición

    Devuelve DOS cosas:
      · tabla:   DataFrame ordenado por la métrica principal, una fila por modelo, con
                 la media y la desviación (ddof=0) de cada métrica en los folds, y
                 tiempo_s: segundos de reloj de la validación cruzada más el refit. El
                 de la red incluye unos 8 s de importar TensorFlow la primera vez.
      · modelos: dict {nombre: Pipeline ajustado con TODO el train}.

    Por qué también los pipelines, y no solo la tabla: el enunciado pide la curva
    ROC comparativa EN LOS MISMOS EJES. Para dibujar las seis curvas hacen falta seis
    `predict_proba` sobre el test, o sea los seis modelos ajustados. Si aquí solo
    saliera la tabla, main.py se quedaría con el ganador y la figura tendría una sola
    línea. El coste es un refit por modelo sobre el train completo, después de la
    validación cruzada.
    """
    # config.DEMO se lee aquí, en tiempo de ejecución: ver data_loader.preparar().
    folds = config.DEMO_FOLDS if config.DEMO else config.CV_FOLDS
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.SEMILLA)
    scoring = _scoring()
    principal = config.METRICA_PRINCIPAL

    filas, modelos = [], {}
    for nombre in config.MODELOS_ACTIVOS:
        clase = REGISTRO[nombre]
        # Un preprocesador propio para cada modelo: si los seis Pipelines compartieran
        # el mismo objeto, cada refit reajustaría el de los otros cinco.
        pipeline = Pipeline([("prep", clone(preprocesador)), ("modelo", clase())])

        inicio = time.perf_counter()
        resultados, modelos[nombre] = _validar(pipeline, X_train, y_train, cv, scoring,
                                               clase.procesos())
        segundos = time.perf_counter() - inicio

        fila = {"modelo": nombre}
        for metrica, valores in resultados.items():
            fila[metrica] = valores.mean()
            fila[f"{metrica}_std"] = valores.std()
        fila["tiempo_s"] = segundos
        filas.append(fila)
        print(f"      {nombre:<10} {principal} {fila[principal]:.3f} "
              f"± {fila[f'{principal}_std']:.3f}  ({segundos:.0f} s)")

    tabla = (pd.DataFrame(filas).set_index("modelo")
             .sort_values(principal, ascending=False))
    return tabla, modelos


def _scoring() -> dict:
    """Las métricas de la tabla, con su nombre de scikit-learn: la principal y las
    secundarias de config.

    precision va con zero_division=0 a propósito: el baseline nunca predice un 1, así
    que su precision es 0/0, y sin esto scikit-learn avisaría en cada fold de algo que
    ya sabemos.
    """
    scoring = {m: m for m in [config.METRICA_PRINCIPAL, *config.METRICAS_SECUNDARIAS]}
    if "precision" in scoring:
        scoring["precision"] = make_scorer(precision_score, zero_division=0)
    return scoring


def _validar(pipeline, X, y, cv, scoring, procesos):
    """Validación cruzada de un Pipeline y refit con todo el train.

    Devuelve ({métrica: array con un valor por fold}, Pipeline ajustado).

    Con config.BUSQUEDA = None, que es lo que se entrega, los hiperparámetros son los
    de cada clase y se validan tal cual. Con "grid" o "random", el buscador prueba la
    rejilla de espacio_busqueda() sobre los MISMOS folds, se queda con la mejor
    combinación según la métrica principal y la tabla recoge los folds de esa
    combinación. Un modelo sin rejilla, como el baseline, se valida tal cual también
    con la búsqueda activada.

    error_score="raise": si un fold falla, se para con su traza. Por defecto
    scikit-learn lo convertiría en NaN, y el modelo perdería la comparación sin que
    nadie viera por qué.
    """
    espacio = pipeline[-1].espacio_busqueda()
    if config.BUSQUEDA is None or not espacio:
        cv_res = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=procesos,
                                error_score="raise")
        resultados = {m: cv_res[f"test_{m}"] for m in scoring}
        return resultados, pipeline.fit(X, y)

    # Las rejillas son listas, así que "random" no puede sortear más combinaciones de
    # las que hay: con más iteraciones las recorrería todas y avisaría. Se recorta aquí.
    n_iter = min(config.N_ITER_RANDOM, len(ParameterGrid(espacio)))
    buscadores = {"grid": (GridSearchCV, {}),
                  "random": (RandomizedSearchCV, {"n_iter": n_iter,
                                                  "random_state": config.SEMILLA})}
    clase_buscador, extra = buscadores[config.BUSQUEDA]
    buscador = clase_buscador(pipeline, espacio, scoring=scoring,
                              refit=config.METRICA_PRINCIPAL, cv=cv, n_jobs=procesos,
                              error_score="raise", **extra)
    buscador.fit(X, y)
    mejor = buscador.best_index_
    resultados = {m: np.array([buscador.cv_results_[f"split{k}_test_{m}"][mejor]
                               for k in range(cv.get_n_splits())])
                  for m in scoring}
    return resultados, buscador.best_estimator_


def elegir_mejor(tabla):
    """Devuelve el nombre del modelo ganador según config.METRICA_PRINCIPAL.

    La columna se lee de config y no se escribe a mano: si la métrica principal pasara
    a ser roc_auc, el ganador cambiaría sin tocar esta función.
    """
    return str(tabla[config.METRICA_PRINCIPAL].idxmax())


def guardar(pipeline, nombre, metricas, ruta=None):
    """Persiste el Pipeline COMPLETO, no solo el estimador, más sus metadatos.

    Si guardas solo el modelo, el preprocesado se pierde y las predicciones salen
    mal sin dar error. Keras es la excepción: config.MODELO_KERAS aparte (pesos y
    arquitectura) y el resto del Pipeline en config.MODELO_PKL.

    Escribe además config.METADATOS con lo que predictor.py necesita para no tener
    que adivinar nada:
      · nombre del modelo ganador
      · umbral realmente usado al predecir (no el que hoy tenga config.UMBRAL:
        el que se congeló al guardar; si luego mueves config.UMBRAL, las
        predicciones de un modelo ya guardado no pueden cambiar en silencio)
      · las columnas crudas que espera de entrada, en orden
      · métrica principal, semilla y versiones de las librerías

    Con otra `ruta`, el .keras y los metadatos van a su lado con los nombres de
    config. Devuelve la ruta del .pkl.
    """
    ruta = Path(config.MODELO_PKL if ruta is None else ruta)
    ruta_keras = ruta.with_name(config.MODELO_KERAS.name)
    ruta_metadatos = ruta.with_name(config.METADATOS.name)

    # Los metadatos se montan ANTES de tocar el disco: si algo falla aquí (una métrica
    # que no es un número), el artefacto anterior se queda entero. Si se escribiera
    # antes el .pkl, quedaría el modelo nuevo con los metadatos del ganador anterior.
    metadatos = {
        "ganador": nombre,
        "umbral": config.UMBRAL,
        "columnas": list(pipeline.feature_names_in_),
        "metrica_principal": config.METRICA_PRINCIPAL,
        "metricas_test": {k: float(v) for k, v in metricas.items()},
        "semilla": config.SEMILLA,
        "demo": bool(config.DEMO),
        "versiones": _versiones(),
    }

    ruta.parent.mkdir(parents=True, exist_ok=True)
    # Un .keras de un ganador anterior no se puede quedar al lado del .pkl nuevo:
    # predictor lo tomaría por la red de este.
    ruta_keras.unlink(missing_ok=True)
    paso, modelo = pipeline.steps[-1]
    para_pkl = Pipeline([*pipeline.steps[:-1], (paso, modelo.guardar_aparte(ruta_keras))])
    joblib.dump(para_pkl, ruta, compress=3)

    metadatos["ficheros"] = [p.name for p in (ruta, ruta_keras) if p.exists()]
    ruta_metadatos.write_text(json.dumps(metadatos, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"      guardado: {', '.join(metadatos['ficheros'])} + {ruta_metadatos.name}")
    return ruta


def _versiones() -> dict:
    """Las versiones con las que se entrenó el artefacto. Se leen de los metadatos del
    paquete instalado, sin importarlo: así no se paga el import de TensorFlow."""
    versiones = {"python": platform.python_version()}
    for paquete in ("scikit-learn", "pandas", "numpy", "joblib", "xgboost",
                    "tensorflow", "keras"):
        try:
            versiones[paquete] = version(paquete)
        except PackageNotFoundError:
            pass
    return versiones
