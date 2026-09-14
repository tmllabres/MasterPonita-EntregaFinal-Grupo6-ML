"""Parámetros y configuración del proyecto.

Todo lo que sea un número, una ruta o una decisión vive aquí y en ningún otro sitio.
La regla: si vas a cambiarlo en la defensa para enseñar algo, tiene que estar en este fichero.
"""
from pathlib import Path

# ── Rutas ────────────────────────────────────────────────────────────────────
RAIZ = Path(__file__).resolve().parent.parent
DATA_RAW = RAIZ / "data" / "raw" / "dataset_practica_final.csv"
DATA_PROC = RAIZ / "data" / "processed"
MODELS = RAIZ / "models"
OUTPUTS = RAIZ / "outputs"

# Los artefactos, con nombre y apellidos. Sin esto, model_trainer.guardar() y
# predictor.cargar() escriben y leen "ruta=None" y el nombre del fichero acaba
# escrito a mano en dos sitios distintos; el día que uno cambie, el otro falla.
MODELO_PKL = MODELS / "mejor_modelo.pkl"          # el Pipeline completo
MODELO_KERAS = MODELS / "mejor_modelo.keras"      # solo si gana la red
METADATOS = MODELS / "metadatos.json"             # ganador, umbral, columnas, versiones

# Lo que escribe evaluator, con nombre fijo porque el README los enlaza.
FIG_CONFUSION = OUTPUTS / "confusion_matrix.png"
FIG_ROC = OUTPUTS / "roc_curve.png"
FIG_IMPORTANCIAS = OUTPUTS / "feature_importance.png"
TABLA_COMPARATIVA = OUTPUTS / "tabla_comparativa.csv"
METRICAS_TEST = OUTPUTS / "metricas_test.json"

# ── Reproducibilidad ─────────────────────────────────────────────────────────
# Una sola semilla para todo: partición, modelos, validación cruzada.
SEMILLA = 42

# ── El problema ──────────────────────────────────────────────────────────────
OBJETIVO = "is_canceled"

# Columnas que determinan el objetivo al 100 %: son fuga y se eliminan siempre.
# reservation_status: Check-Out -> 0 ; Canceled y No-Show -> 1.
FUGAS = ["reservation_status", "reservation_status_date"]

# ── Alta cardinalidad ────────────────────────────────────────────────────────
# Tres columnas tienen demasiadas categorías para un one-hot directo: country (177
# valores + nulo), agent (333) y company (352). Un one-hot ingenuo de las 29
# predictoras se va a ~890 columnas, casi todas ceros: el árbol y el bosque se
# comen la memoria y la logística sobreajusta sobre categorías con 3 reservas.
#
# Se agrupan en las TOP_N_CATEGORIAS más frecuentes y el resto cae en un cajón
# común, con lo que la matriz se queda en 109 columnas. Se aprende dentro del
# Pipeline, fold a fold: qué categorías son «las más frecuentes» es un número que
# se aprende de los datos, y aprenderlo del test sería fuga.
#
# Ojo: agent y company son float64 en el CSV (IDs numéricos con nulos), así que
# select_dtypes("object") NO las ve. Por eso van listadas a mano.
ALTA_CARDINALIDAD = ["country", "agent", "company"]
TOP_N_CATEGORIAS = 10

# ── Partición ────────────────────────────────────────────────────────────────
TEST_SIZE = 0.20
ESTRATIFICAR = True       # conserva el mismo reparto de clases en las dos mitades

# Las cuentas de filas, ya cerradas. Sobre el CSV crudo son 119.390 -> 95.512 / 23.878,
# pero eso es ANTES de limpiar. Con la limpieza que declara data_loader.limpiar()
# (fugas + duplicados exactos + imposibles) quedan 86.971 filas -> 69.576 / 17.395
# (train_test_split redondea el test HACIA ARRIBA: ceil(86.971 x 0,20) = 17.395),
# y el reparto pasa de 62,96/37,04 a 72,69/27,31.

# ── Duplicados exactos: DECISIÓN TOMADA ──────────────────────────────────────
# Son 32.252 filas, el 27 % del dataset, y el 63,4 % de ellas son cancelaciones.
#
# Se ELIMINAN. El razonamiento, que es de dominio y no estadístico: el CSV no trae
# identificador de reserva, así que dos filas idénticas en las 29 predictoras son
# indistinguibles y no hay forma de demostrar que sean reservas distintas. Dejarlas
# tiene un coste asimétrico: si una copia cae en train y su gemela en test, el modelo
# ya vio ese ejemplo exacto y su nota en test sale inflada SIN que salte ningún error.
# Perder algunas reservas de grupo legítimas es un precio menor que publicar una
# métrica de test que no es honesta.
#
# Consecuencia asumida: la prevalencia baja diez puntos (37,04 % -> 27,31 %), y con
# ella sube el acierto del modelo trivial (62,96 % -> 72,69 %), que es justo el
# argumento del apartado 5 contra usar accuracy. Los números de los apartados 2, 5, 7
# y 8 del README salen TODOS de esta decisión.
#
# Se eliminan DESPUÉS de quitar las fugas (si no, reservation_status_date desempata
# filas que son la misma reserva: serían 31.994 en vez de 32.252) y SIEMPRE ANTES de
# particionar, o la propia partición ya habría repartido las gemelas entre los dos lados.
ELIMINAR_DUPLICADOS = True

# ── Evaluación ───────────────────────────────────────────────────────────────
# Métrica principal declarada ANTES de entrenar. Se justifica en el README.
METRICA_PRINCIPAL = "f1"
METRICAS_SECUNDARIAS = ["accuracy", "precision", "recall", "roc_auc"]

CV_FOLDS = 5              # StratifiedKFold dentro del train
UMBRAL = 0.50             # el de la librería; si lo mueves, dilo y justifícalo

# ── Ajuste de hiperparámetros ────────────────────────────────────────────────
# DESACTIVADO a propósito para la entrega del 15 de septiembre.
#
# No es que no sepamos hacerlo: el comparador está diseñado para soportarlo y cada
# modelo declara su espacio_busqueda(). Es una decisión de calendario. Con 9 días,
# 6 modelos x 5 folds x 30 iteraciones sobre 69.576 filas se va a horas de cómputo,
# y el riesgo de descubrir un fallo a las tres horas de ejecución no compensa.
#
# Sin búsqueda, la ejecución completa dura minutos y el sistema entrega entero, que es
# lo que el enunciado puntúa. Se declara como limitación en el apartado 10 del README.
# Para activarlo, si sobrara tiempo: BUSQUEDA = "random". No hay que tocar nada más.
BUSQUEDA = None           # "grid" | "random" | None
N_ITER_RANDOM = 30        # solo si BUSQUEDA == "random"
N_JOBS = -1               # ojo: n_jobs=1 para la red de Keras, o se sobre-suscribe la CPU

# ── Qué modelos entran en la comparación ─────────────────────────────────────
# El bucle de model_trainer recorre esta lista. Añadir un modelo = añadir una línea.
#
# "baseline" no lo pide el enunciado, pero sin él un F1 de 0,80 no dice si el modelo
# aprendió algo o si el problema era fácil. Es la fila que da sentido a las otras cinco.
MODELOS_ACTIVOS = [
    "baseline",
    "logistica",
    "arbol",
    "bosque",
    "boosting",
    "red_keras",
]

# ── Modo demo ────────────────────────────────────────────────────────────────
# La defensa son 30 minutos. Con esto activado el pipeline entrena sobre una
# muestra y con menos folds, para enseñar el circuito completo sin esperar.
DEMO = False
DEMO_FILAS = 20_000
DEMO_FOLDS = 3
