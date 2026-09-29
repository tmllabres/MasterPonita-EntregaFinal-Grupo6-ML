"""Entrenamiento y comparación de los modelos.

Me inspiré en las librerías de AutoML que menciona el enunciado: cada modelo es una
clase con la misma interfaz, todas están en un registro y un mismo bucle las entrena
y las compara. Para añadir un modelo basta con escribir su clase y añadirla al
registro.
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

    TensorFlow tarda en importarse, y así los tests o el predictor con un ganador que
    no es la red no tienen que esperar.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # oculta los avisos de arranque
    import keras
    return keras


# ── Interfaz común ───────────────────────────────────────────────────────────
class ModeloBase(BaseEstimator, ClassifierMixin):
    """Interfaz común que cumplen todos los modelos del registro.

    Sigue tres reglas de scikit-learn para que clone(), GridSearchCV y
    cross_validate funcionen:
      · __init__ solo guarda los hiperparámetros, uno por argumento, sin validar ni
        construir nada: clone() hace las copias a partir de esos argumentos.
      · lo que se aprende en fit va en atributos acabados en guion bajo (model_).
      · fit devuelve self.
    """

    nombre = "base"

    @classmethod
    def procesos(cls) -> int:
        """Procesos en paralelo para la validación cruzada de este modelo.

        Se lee de config al llamarlo, no al importar. La red lo sobrescribe.
        """
        return config.N_JOBS

    def construir(self):
        """Devuelve el estimador sin entrenar. Lo implementa cada subclase."""
        raise NotImplementedError

    def espacio_busqueda(self) -> dict:
        """Rejilla de hiperparámetros con la sintaxis paso__hiperparametro.

        Ej.: {"modelo__max_depth": [4, 6, 8]}, donde «modelo» es el nombre del paso
        en el Pipeline y max_depth, el hiperparámetro de ese paso.
        """
        return {}

    def fit(self, X, y):
        """Construye un estimador nuevo, lo entrena y devuelve self.

        Se construye en cada fit para que dos fit seguidos no compartan nada: clone()
        da por hecho que cada fold recibe un modelo sin entrenar.
        """
        self.model_ = self.construir()
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        """Devuelve una matriz (n, 2): columna 0 «no cancela» y columna 1 «cancela»."""
        return self.model_.predict_proba(X)

    # ── Lo que no va dentro del .pkl ──
    def guardar_aparte(self, ruta):
        """Guarda en `ruta` lo que no va en el .pkl y devuelve lo que sí va.

        Por defecto no escribe nada y devuelve el propio modelo. Solo la red lo cambia
        (ver RedKeras.guardar_aparte).
        """
        return self

    def cargar_aparte(self, ruta) -> None:
        """Carga lo que guardó guardar_aparte(). Lo llama predictor.cargar()."""


# ── Modelo base ──────────────────────────────────────────────────────────────
class Baseline(ModeloBase):
    """DummyClassifier: la referencia con la que comparo los cinco modelos.

    Con strategy="most_frequent" dice siempre «no cancela» y ya acierta el 72,36 %
    (el 62,96 % sobre el CSV crudo, antes de quitar los duplicados). Un modelo que no
    lo supere claramente no aporta nada. En accuracy saca 0,72, pero en F1 saca 0,
    porque nunca predice un 1: por eso uso F1 y no accuracy (apartado 5 del README).
    """
    nombre = "baseline"

    def construir(self):
        return DummyClassifier(strategy="most_frequent")


# ── Los cinco modelos del enunciado ──────────────────────────────────────────
class Logistica(ModeloBase):
    """Regresión logística. C es la inversa de la fuerza de la regularización.

    Necesita las variables escaladas, pero eso ya lo hace el preprocesador. max_iter va
    con margen porque si lbfgs no converge no da error, solo un aviso, y el modelo se
    queda a medio entrenar.
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
    """Árbol de decisión. No necesita escalado.

    Limito max_depth y min_samples_leaf porque sin ellos el árbol crece hasta dejar una
    reserva por hoja: se aprende el train de memoria y en validación empeora mucho.
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
    """Random Forest (bagging): cada árbol ve un sorteo de filas y de columnas.

    Por defecto uso min_samples_leaf=5 y no 1 (el de scikit-learn) por tamaño: con 1,
    los 300 árboles ocupan 108 MB con la compresión de guardar(), y con 5, 35 MB. A
    cambio, sin búsqueda el F1 en validación cruzada baja de 0,681 a 0,654 con el mismo
    AUC (0,895): las probabilidades salen más moderadas y pasan menos veces el umbral
    de 0,5. El ganador no cambia.
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
    """Gradient Boosting con XGBoost, una de las tres opciones que acepta el enunciado.

    Cada árbol es de regresión y corrige lo que fallan los anteriores, por eso
    learning_rate y n_estimators van juntos: con un learning_rate más bajo hacen falta
    más árboles. subsample y colsample_bytree sortean filas y columnas en cada árbol,
    como en el bosque, para que no se ajusten todos al mismo ruido.
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
    """MLP con Keras adaptado a la interfaz de scikit-learn.

    La red se construye en fit y no en __init__: si no, la misma red pasaría de un fold
    a otro y, como el fit de Keras no reinicia los pesos, un fold empezaría habiendo
    visto ya sus datos de validación. El F1 saldría inflado sin dar ningún error.

    EarlyStopping usa el último 10 % del train de cada fit para decidir cuándo parar y
    se queda con los pesos de la mejor época, no con los de la última.
    """
    nombre = "red_keras"

    @classmethod
    def procesos(cls) -> int:
        """Un solo proceso para la red: así respeta el modo demo.

        Los procesos en paralelo de la validación cruzada vuelven a importar config y no
        ven el config.DEMO = True que pone main.py al ejecutarse. Con --demo, cada fold
        entrenaría todas las épocas y la fila de la tabla no sería la red que se guarda.
        """
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
        # Sin capa Input: el número de columnas se fija en el primer fit, así
        # construir() no necesita saberlo y tiene la misma firma que en los demás.
        red = keras.Sequential([*ocultas, keras.layers.Dense(1, activation="sigmoid")])
        red.compile(optimizer=keras.optimizers.Adam(self.learning_rate),
                    loss="binary_crossentropy")
        return red

    def espacio_busqueda(self) -> dict:
        # Vacía a propósito: cada entrenamiento de la red son ~20 s en un solo proceso,
        # y una rejilla de 9 combinaciones x 5 folds se iría a un cuarto de hora. Fijo
        # la arquitectura de antemano y se valida tal cual, como el baseline.
        return {}

    def fit(self, X, y):
        keras = _keras()
        # La semilla va antes de construir la red: fija los pesos iniciales, el dropout
        # y el barajado de cada época, así que dos fit con los mismos datos dan la
        # misma red.
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
        # Llamo a la red directamente y no con .predict(): predict() crea una función de
        # TensorFlow para cada red y, con una red nueva en cada fold, TensorFlow acaba
        # avisando de que recompila. El resultado es el mismo, pero sin esos avisos.
        salida = self.model_(_denso(X), training=False)
        cancela = _keras().ops.convert_to_numpy(salida).ravel()
        return np.column_stack([1 - cancela, cancela])

    def predict(self, X):
        # Umbral de 0,5, como el predict de scikit-learn. main.py no lo usa: compara la
        # probabilidad con config.UMBRAL.
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def guardar_aparte(self, ruta):
        """Guarda la red en un .keras y devuelve una copia sin ella para el .pkl.

        Keras 3 permite meter una red en un pickle, pero el formato que garantiza entre
        versiones es el suyo, .keras. La copia es superficial y solo a ella se le quita
        la red, así que el modelo del Pipeline no cambia.
        """
        self.model_.save(ruta)
        sin_red = copy.copy(self)
        sin_red.model_ = None
        return sin_red

    def cargar_aparte(self, ruta) -> None:
        self.model_ = _keras().models.load_model(ruta)


def _denso(X):
    """Pasa X a denso y float32: Keras no acepta matrices dispersas."""
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

    Todos se validan igual para que la comparación sea justa: el mismo StratifiedKFold
    con la misma semilla, el preprocesado dentro del Pipeline (se ajusta en cada fold)
    y el scoring explícito: sin él, GridSearchCV elegiría los hiperparámetros por
    accuracy y no por F1.

      · tabla: DataFrame ordenado por la métrica principal, una fila por modelo, con la
        media y la desviación (ddof=0) de cada métrica en los folds, y tiempo_s, los
        segundos de la validación cruzada más el refit (en la red incluye unos 8 s de
        importar TensorFlow la primera vez).
      · modelos: {nombre: Pipeline reentrenado con todo el train}. Devuelvo los seis y
        no solo el ganador porque la curva ROC los compara todos en la misma figura.
    """
    # config.DEMO se lee aquí, en tiempo de ejecución: ver data_loader.preparar().
    folds = config.DEMO_FOLDS if config.DEMO else config.CV_FOLDS
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.SEMILLA)
    scoring = _scoring()
    principal = config.METRICA_PRINCIPAL

    filas, modelos = [], {}
    for nombre in config.MODELOS_ACTIVOS:
        clase = REGISTRO[nombre]
        # Cada modelo lleva su propio preprocesador: si los seis Pipelines compartieran
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
    """Métricas de la tabla con su nombre en scikit-learn: la principal y las demás.

    precision va con zero_division=0 porque el baseline nunca predice un 1 y su
    precision es 0/0: sin esto, scikit-learn daría un aviso en cada fold.
    """
    scoring = {m: m for m in [config.METRICA_PRINCIPAL, *config.METRICAS_SECUNDARIAS]}
    if "precision" in scoring:
        scoring["precision"] = make_scorer(precision_score, zero_division=0)
    return scoring


def _validar(pipeline, X, y, cv, scoring, procesos):
    """Validación cruzada de un Pipeline y refit con todo el train.

    Devuelve ({métrica: array con un valor por fold}, Pipeline ajustado).

    Con "grid" (la que uso, ver config.BUSQUEDA) o "random", el buscador prueba la
    rejilla de espacio_busqueda() sobre los mismos folds, elige la mejor combinación
    por la métrica principal (refit), la reentrena con todo el train y la tabla recoge
    sus folds. No es validación cruzada anidada, así que el F1 sale algo optimista
    (apartado 10 del README). Con config.BUSQUEDA = None, o si el modelo no tiene
    rejilla (baseline y red), se valida con los hiperparámetros de su clase.

    error_score="raise": si falla un fold, se para con el error. Por defecto
    scikit-learn pondría NaN y el modelo perdería la comparación sin ver por qué.
    """
    espacio = pipeline[-1].espacio_busqueda()
    if config.BUSQUEDA is None or not espacio:
        cv_res = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=procesos,
                                error_score="raise")
        resultados = {m: cv_res[f"test_{m}"] for m in scoring}
        return resultados, pipeline.fit(X, y)

    # Con "random", n_iter no puede pasar del número de combinaciones de la rejilla
    # (scikit-learn las probaría todas y daría un aviso), así que se recorta aquí.
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

    La métrica se lee de config: si la principal pasara a ser roc_auc, no habría que
    tocar esta función.
    """
    return str(tabla[config.METRICA_PRINCIPAL].idxmax())


def guardar(pipeline, nombre, metricas, ruta=None):
    """Guarda el Pipeline completo, no solo el estimador, junto con sus metadatos.

    Guardando solo el modelo se perdería el preprocesado y las predicciones saldrían
    mal sin dar error. Si gana la red, va aparte en config.MODELO_KERAS y el resto del
    Pipeline en config.MODELO_PKL.

    En config.METADATOS queda lo que necesita predictor.py: el ganador, el umbral con
    el que se guardó (así, si luego cambia config.UMBRAL, el modelo guardado sigue
    prediciendo igual), las columnas de entrada en orden, la métrica principal, la
    semilla, los hiperparámetros del ganador y las versiones de las librerías.

    Con otra `ruta`, el .keras y los metadatos van a su lado con los nombres de
    config. Devuelve la ruta del .pkl.
    """
    ruta = Path(config.MODELO_PKL if ruta is None else ruta)
    ruta_keras = ruta.with_name(config.MODELO_KERAS.name)
    ruta_metadatos = ruta.with_name(config.METADATOS.name)

    # Los metadatos se preparan antes de escribir en disco: si algo falla aquí (una
    # métrica que no es un número), el modelo guardado anterior queda intacto y no se
    # mezcla un .pkl nuevo con los metadatos del ganador anterior.
    metadatos = {
        "ganador": nombre,
        "umbral": config.UMBRAL,
        "columnas": list(pipeline.feature_names_in_),
        "metrica_principal": config.METRICA_PRINCIPAL,
        # Los que eligió la búsqueda, o los de la clase si no la hubo.
        "hiperparametros": pipeline.steps[-1][1].get_params(),
        # Tipos de Python y no de numpy, para el json: el n de reservas como entero y
        # el resto como float.
        "metricas_test": {k: int(v) if isinstance(v, (int, np.integer)) else float(v)
                          for k, v in metricas.items()},
        "semilla": config.SEMILLA,
        "demo": bool(config.DEMO),
        "versiones": _versiones(),
    }

    ruta.parent.mkdir(parents=True, exist_ok=True)
    # Se borra el .keras de un ganador anterior: si se quedara junto al .pkl nuevo,
    # predictor lo tomaría por la red de este modelo.
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
    """Versiones de Python y de las librerías con las que se entrenó el modelo.

    Se leen de los metadatos de los paquetes instalados, sin importarlos, para no
    tener que esperar a que se importe TensorFlow.
    """
    versiones = {"python": platform.python_version()}
    for paquete in ("scikit-learn", "pandas", "numpy", "joblib", "xgboost",
                    "tensorflow", "keras"):
        try:
            versiones[paquete] = version(paquete)
        except PackageNotFoundError:
            pass
    return versiones
