"""Parámetros y configuración del proyecto.

Reúno aquí las rutas, la semilla, la métrica, el umbral, los folds y el modo demo para
poder cambiarlos en un solo sitio. Las rejillas de hiperparámetros están en cada modelo,
en src/model_trainer.py (apartado 11 del README).
"""
from pathlib import Path

# ── Rutas ────────────────────────────────────────────────────────────────────
RAIZ = Path(__file__).resolve().parent.parent
DATA_RAW = RAIZ / "data" / "raw" / "dataset_practica_final.csv"
DATA_PROC = RAIZ / "data" / "processed"
MODELS = RAIZ / "models"
OUTPUTS = RAIZ / "outputs"

# Rutas del modelo guardado. Van aquí para que model_trainer.guardar() y
# predictor.cargar() usen las mismas y el nombre no esté escrito a mano en dos sitios.
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
# Una sola semilla para todo (partición, modelos y validación cruzada), para que al
# repetir la ejecución salgan los mismos resultados.
SEMILLA = 42

# ── El problema ──────────────────────────────────────────────────────────────
OBJETIVO = "is_canceled"

# ── Fugas ────────────────────────────────────────────────────────────────────
# Columnas que todavía no existen cuando se predice (al hacerse la reserva o en las
# semanas siguientes, antes de que llegue el cliente), así que dan la respuesta. Se
# quitan antes de buscar duplicados: 32 columnas − is_canceled − 4 fugas = 27
# predictoras. Ver apartado 2 del README.

# Directas: reservation_status es la respuesta (Check-Out -> 0; Canceled y No-Show -> 1)
# y reservation_status_date es la fecha de ese estado.
FUGAS_DIRECTAS = ["reservation_status", "reservation_status_date"]

# Posteriores: se rellenan el día de llegada. Ninguna de las 7.416 reservas con parking
# está cancelada, y la habitación asignada cambia en el 18,8 % de las Check-Out y el
# 17,2 % de los No-Show, pero solo en el 1,4 % de las canceladas. Con ellas, el F1 del
# modelo de prueba (pruebas_modelos.ipynb) sube de 0,678 a 0,710, pero no es real.
FUGAS_POSTERIORES = ["required_car_parking_spaces", "assigned_room_type"]

FUGAS = FUGAS_DIRECTAS + FUGAS_POSTERIORES

# booking_changes se queda, aunque con dudas: el CSV guarda el número final de cambios y
# alguno puede ser posterior al momento de predecir. No se comporta como el parking (el
# 14,7 % de los No-Show, que nunca llegan, tienen algún cambio) y sin ella el F1 del
# modelo de prueba solo baja de 0,678 a 0,671. Ver apartados 2 y 10 del README.

# ── Alta cardinalidad ────────────────────────────────────────────────────────
# country, agent y company tienen cientos de valores (168, 325 y 320 en X_train) y con
# un one-hot directo la matriz tendría 880 columnas. Me quedo con los TOP_N_CATEGORIAS
# más frecuentes y agrupo el resto en «otros»: quedan 97. Se aprende dentro del
# Pipeline, en cada fold, porque sacar los más frecuentes del test sería fuga.
# agent y company son float64 en el CSV (IDs con nulos) y select_dtypes("object") no
# las detecta, por eso van listadas a mano.
ALTA_CARDINALIDAD = ["country", "agent", "company"]
TOP_N_CATEGORIAS = 10

# ── Partición ────────────────────────────────────────────────────────────────
TEST_SIZE = 0.20
ESTRATIFICAR = True       # conserva el mismo reparto de clases en las dos mitades

# Tras la limpieza de data_loader.limpiar() (fugas, duplicados e imposibles) quedan
# 85.811 filas -> 68.648 de train y 17.163 de test (train_test_split redondea el test
# hacia arriba), y la tasa de cancelación pasa del 37,04 % del CSV crudo al 27,64 %.

# ── Duplicados exactos ───────────────────────────────────────────────────────
# Quito las 33.413 filas repetidas en las 27 predictoras y en la respuesta: el CSV no
# trae identificador de reserva y, si una copia cae en train y otra en test, el modelo
# se examina de algo que ya ha visto (sin quitarlas, el 32,7 % del test tendría una
# gemela en train). Las filas iguales en las predictoras pero con distinta respuesta
# no se tocan, porque esas sí son reservas distintas.
#
# Se quitan después de las fugas (el modelo no ve esas columnas, así que no deben
# decidir qué es un duplicado) y antes de partir, para que la partición no reparta las
# copias entre train y test. El coste que asumo: muchas eran reservas de grupo, casi
# desaparecen las Non Refund y las cancelaciones bajan del 37,04 % al 27,64 %.
ELIMINAR_DUPLICADOS = True

# ── Evaluación ───────────────────────────────────────────────────────────────
# Métrica fijada antes de entrenar, para no elegirla según qué modelo saliera mejor
# (apartado 5 del README).
METRICA_PRINCIPAL = "f1"
METRICAS_SECUNDARIAS = ["accuracy", "precision", "recall", "roc_auc"]

CV_FOLDS = 5              # StratifiedKFold dentro del train
UMBRAL = 0.50             # el de la librería; no lo muevo (apartado 10 del README)

# Importancia por permutación en evaluator.importancias(): más repeticiones dan barras
# de error más fiables, pero tarda más. La figura enseña las TOP_IMPORTANCIAS primeras
# y la tabla que devuelve, todas.
IMPORTANCIA_REPETICIONES = 5
TOP_IMPORTANCIAS = 15

# ── Ajuste de hiperparámetros ────────────────────────────────────────────────
# GridSearchCV con la rejilla de espacio_busqueda() de cada modelo, con los mismos folds
# y la misma métrica que la comparación. "grid" y no "random": las rejillas son pequeñas
# e incluyen los valores por defecto, así que la búsqueda nunca deja un modelo peor que
# sin ella (con 10 al azar, en la demo el árbol empeoraba). La red no entra porque cada
# entrenamiento es muy lento (ver RedKeras.espacio_busqueda()).
#
# La validación cruzada no es anidada: el F1 de la tabla sale algo optimista y la cifra
# final es la del test (apartado 10 del README).
BUSQUEDA = "grid"         # "grid" | "random" | None
N_ITER_RANDOM = 10        # solo si BUSQUEDA == "random"
N_JOBS = -1               # la red usa siempre 1 proceso: ver RedKeras.procesos()

# ── Qué modelos entran en la comparación ─────────────────────────────────────
# El bucle de model_trainer recorre esta lista: añadir un modelo es añadir una línea.
# El baseline no lo pide el enunciado, pero lo uso como referencia para ver si los
# demás aprenden algo (en accuracy ya saca 0,724, pero su F1 es 0).
MODELOS_ACTIVOS = [
    "baseline",
    "logistica",
    "arbol",
    "bosque",
    "boosting",
    "red_keras",
]

# ── Modo demo ────────────────────────────────────────────────────────────────
# Versión corta para la defensa (python main.py --demo): entrena sobre una muestra,
# con menos folds y menos épocas en la red, para enseñar el proceso entero sin esperar.
DEMO = False
DEMO_FILAS = 20_000
DEMO_FOLDS = 3
# 15 y no 5: con 5 la red para a medio aprender (su mejor época es siempre la última)
# y en la tabla de la demo cae del 2.º al 4.º puesto. 15 solo tarda unos segundos más.
DEMO_EPOCAS = 15
