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

# ── Fugas ────────────────────────────────────────────────────────────────────
# Columnas que no se pueden usar porque, cuando toca predecir, todavía no existen. El
# modelo predice al hacerse la reserva o en las semanas siguientes, antes de que llegue
# el cliente: todo lo que se escribe a su llegada, o una vez se sabe si canceló, es la
# respuesta escrita con otras palabras. Se eliminan siempre, y ANTES de buscar
# duplicados (ver más abajo). 32 columnas − is_canceled − 4 fugas = 27 predictoras.

# Directas: determinan el objetivo al 100 %.
# reservation_status: Check-Out -> 0 ; Canceled y No-Show -> 1.
FUGAS_DIRECTAS = ["reservation_status", "reservation_status_date"]

# Posteriores: se rellenan cuando el cliente llega. Evidencia, sobre el CSV crudo:
#   · required_car_parking_spaces: 7.416 reservas con plaza y NINGUNA cancelada ni
#     No-Show. Si se guardara al reservar, cabrían ~2.670 cancelaciones y ~75 No-Show;
#     las peticiones especiales, que sí se hacen al reservar, cancelan un 21,7 %.
#   · assigned_room_type: la habitación se asigna el día de llegada. Es distinta de la
#     reservada en el 18,8 % de las Check-Out y el 17,2 % de los No-Show, pero solo en
#     el 1,4 % de las canceladas: quien cancela antes no llega a tener habitación.
# Con ellas, un gradient boosting sube el F1 en validación cruzada de 0,678 a 0,710
# sobre las mismas filas: es nota regalada, no la que tendría en producción.
FUGAS_POSTERIORES = ["required_car_parking_spaces", "assigned_room_type"]

FUGAS = FUGAS_DIRECTAS + FUGAS_POSTERIORES

# booking_changes se QUEDA, con una duda declarada: el CSV guarda el número FINAL de
# cambios, y parte de ellos pueden ser posteriores al momento de predecir. No es tan
# clara como las dos de arriba (con cambios cancela un 15,6 % frente a un 30,3 % en
# train: separa, pero no a la perfección), y los cambios hechos hasta el momento de
# predecir sí serían información legítima.

# ── Alta cardinalidad ────────────────────────────────────────────────────────
# Tres columnas tienen demasiadas categorías para un one-hot directo. En X_train
# (68.648 filas): country 168 valores, agent 325 y company 320, las tres con nulos.
# Un one-hot ingenuo de las 27 predictoras se va a 880 columnas (16 numéricas + 864
# de categorías), casi todas ceros: el árbol y el bosque se comen la memoria y la
# logística sobreajusta sobre categorías con 3 reservas.
#
# Se agrupan en las TOP_N_CATEGORIAS más frecuentes y el resto cae en un cajón
# común, con lo que la matriz se queda en 97 columnas (16 numéricas + 48 de las 8
# categóricas + 3 x 11 de estas tres: 10 valores y el cajón). Se aprende dentro del
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
# (fugas + duplicados exactos + imposibles) quedan 85.811 filas -> 68.648 / 17.163
# (train_test_split redondea el test HACIA ARRIBA: ceil(85.811 x 0,20) = 17.163),
# y el reparto pasa de 62,96/37,04 a 72,36/27,64.

# ── Duplicados exactos: DECISIÓN TOMADA ──────────────────────────────────────
# Son 33.413 filas, el 28 % del dataset, y el 61,3 % de ellas son cancelaciones.
#
# Se ELIMINAN. Qué cuenta como duplicado: dos filas idénticas en las 28 columnas que
# quedan tras quitar las fugas, es decir, en las 27 predictoras Y en la respuesta. El
# CSV no trae identificador de reserva, así que esas copias no se pueden distinguir.
# Las que coinciden en las 27 predictoras pero acabaron distinto (289 pares, 578 filas)
# NO se tocan: esas sí son, seguro, reservas distintas.
#
# Dejar las copias tiene un coste asimétrico: si una cae en train y su gemela en test,
# el modelo ya vio ese ejemplo exacto y su nota en test sale inflada SIN que salte
# ningún error. No es una hipótesis: partiendo el CSV sin deduplicar (misma semilla y
# stratify), 7.813 de las 23.878 filas de test (32,7 %) tienen una gemela exacta en
# train, y cancelan un 58,2 % frente al 26,7 % del resto.
#
# El precio, que se asume: las copias se concentran en reservas de grupo. Se va el
# 93,1 % de las Non Refund (14.587 -> 1.011) y el 77,6 % del segmento Groups, y la tasa
# de cancelación del City Hotel baja del 41,73 % al 30,21 %. Algunas serán habitaciones
# legítimas de un mismo grupo; perderlas es un precio menor que publicar una métrica
# de test que no es honesta.
#
# Consecuencia asumida: la prevalencia baja casi diez puntos (37,04 % -> 27,64 %), y
# con ella sube el acierto del modelo trivial (62,96 % -> 72,36 %), que es justo el
# argumento del apartado 5 contra usar accuracy. Los números de los apartados 2, 5, 7
# y 8 del README salen TODOS de esta decisión.
#
# Se eliminan DESPUÉS de quitar las fugas (el modelo no las ve, así que no pueden
# decidir qué es un duplicado: con ellas dentro serían 31.994 en vez de 33.413) y
# SIEMPRE ANTES de particionar, o la propia partición ya habría repartido las gemelas
# entre los dos lados.
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
# 6 modelos x 5 folds x 30 iteraciones sobre 68.648 filas se va a horas de cómputo,
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
