"""Métricas y visualización de resultados.

Es el ÚNICO módulo que abre el conjunto de test, y lo abre una sola vez, al final.
Cada vez que el test influye en una decisión —qué modelo, qué hiperparámetro, qué
umbral— deja de ser una estimación honesta.

Las figuras se dibujan sobre un `Figure` de matplotlib y no con pyplot: sin estado
global no hay figuras que se acumulen en memoria de una llamada a otra, ni un
plt.show() que vacíe la figura y deje el PNG en blanco.
"""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
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

from . import config, model_trainer

# ── Aspecto de las figuras ───────────────────────────────────────────────────
# Tinta, ejes y rejilla discretos: lo único que llama la atención son los datos.
_FONDO = "#fcfcfb"
_TINTA = "#0b0b0b"
_TINTA_2 = "#52514e"
_APAGADO = "#898781"
_REJILLA = "#e1e0d9"
_EJE = "#c3c2b7"

# El color va con el MODELO, no con su puesto: el bosque es aguamarina en todas las
# figuras, gane o quede cuarto. Es una paleta categórica validada para daltonismo en
# este orden; el baseline va en gris porque no es un modelo, es la referencia.
_COLORES = {
    "logistica": "#2a78d6",
    "arbol": "#eb6834",
    "bosque": "#1baf7a",
    "boosting": "#eda100",
    "red_keras": "#e87ba4",
    "baseline": _APAGADO,
}
_ETIQUETAS = {
    "baseline": "Baseline (Dummy)",
    "logistica": "Regresión logística",
    "arbol": "Árbol de decisión",
    "bosque": "Random Forest",
    "boosting": "XGBoost",
    "red_keras": "Red neuronal (Keras)",
}

# Un solo tono, de claro a oscuro: en la matriz el color mide cantidad, no identidad.
_AZULES = LinearSegmentedColormap.from_list(
    "azules", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def metricas(y_true, y_pred, y_proba=None) -> dict:
    """Devuelve la métrica principal y las secundarias en un dict.

    Con pos_label=1 ("cancela"): es la clase que quieres detectar, no la que te
    conviene. Define qué es un TP y hacia dónde miran precision y recall.

    El AUC se calcula con las PROBABILIDADES: con los 0/1 de y_pred la curva ROC tiene
    un solo punto y el AUC mide ese umbral, no lo bien que el modelo ordena. Si no se
    pasan probabilidades, no hay AUC: la clave no aparece, en vez de un número falso.

    zero_division=0: un modelo que no predice ningún 1 (el baseline) tiene precision
    0/0; se declara 0, que es lo que vale para decidir, en vez de avisar.

    Todo sale como tipo nativo de Python (float, y n como int): scikit-learn devuelve
    tipos de numpy, y json.dump revienta con algunos de ellos al escribir el informe.
    `n` es el número de reservas evaluadas, el N que cita el README.
    """
    resultado = {
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }
    if y_proba is not None:
        resultado["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    resultado["n"] = len(y_true)
    return resultado


def matriz_confusion(y_true, y_pred, ruta=None):
    """Figura obligatoria del enunciado. Guarda en config.FIG_CONFUSION.

    Pon los números absolutos y el porcentaje por fila: el de la diagonal de TP es
    el recall y el de TN la specificity, y así se lee sola.

    El color de cada celda sigue a su porcentaje de fila, no al número absoluto: si no,
    la fila de «No cancela», que es casi tres veces más grande, lo taparía todo.
    Devuelve la matriz de conteos (filas = real, columnas = predicho).
    """
    ruta = Path(config.FIG_CONFUSION if ruta is None else ruta)
    matriz = confusion_matrix(y_true, y_pred, labels=[0, 1])
    total_fila = matriz.sum(axis=1, keepdims=True)
    por_fila = np.divide(matriz, total_fila, out=np.zeros(matriz.shape),
                         where=total_fila > 0)

    fig, ax = _lienzo(5.6, 4.9)
    ax.imshow(por_fila, cmap=_AZULES, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{_es(matriz[i, j])}\n{_es(100 * por_fila[i, j], 1)} %",
                    ha="center", va="center", fontsize=13,
                    color=_tinta_sobre(_AZULES(por_fila[i, j])))

    clases = ["No cancela", "Cancela"]
    ax.set_xticks([0, 1], clases)
    ax.set_yticks([0, 1], clases)
    ax.set_xlabel("Predicho", color=_TINTA_2)
    ax.set_ylabel("Real", color=_TINTA_2)
    ax.tick_params(length=0, colors=_TINTA_2, labelsize=11)
    for borde in ax.spines.values():
        borde.set_visible(False)
    _titulo(fig, "Matriz de confusión del ganador en test",
            f"{_es(len(y_true))} reservas · umbral {_es(config.UMBRAL, 2)} · "
            "porcentaje sobre cada fila real")
    _guardar(fig, ruta)
    return matriz


def curva_roc(modelos_proba: dict, y_true, ruta=None):
    """Figura obligatoria del enunciado. Guarda en config.FIG_ROC.

    LAS SEIS CURVAS EN LOS MISMOS EJES, con su AUC en la leyenda y la diagonal
    del azar. Seis gráficos sueltos no demuestran nada.

    Recibe {nombre: probabilidad de cancelar} de TODOS los modelos, no solo del
    ganador. La leyenda va ordenada por AUC; el color, por modelo; y cuanto mejor el
    AUC, más arriba se dibuja la curva, para que el ganador no quede tapado donde las
    curvas se cruzan.

    La diagonal del azar ES la curva del baseline: dice siempre lo mismo, así que no
    ordena nada. Por eso, si el baseline está, su curva hace de diagonal (discontinua
    y en gris) y no se dibuja otra encima, que se verían como una sola línea que no
    coincide con ninguna entrada de la leyenda. Si no está, la diagonal va aparte.
    Devuelve {nombre: AUC}.
    """
    ruta = Path(config.FIG_ROC if ruta is None else ruta)
    aucs = {nombre: float(roc_auc_score(y_true, proba))
            for nombre, proba in modelos_proba.items()}

    fig, ax = _lienzo(6.4, 6.3)
    por_auc = sorted(aucs, key=aucs.get, reverse=True)
    for puesto, nombre in enumerate(por_auc):
        fpr, tpr, _ = roc_curve(y_true, modelos_proba[nombre])
        es_baseline = nombre == "baseline"
        etiqueta = _ETIQUETAS.get(nombre, nombre) + (" = azar" if es_baseline else "")
        ax.plot(fpr, tpr, color=_COLORES.get(nombre, _TINTA_2), lw=1.6,
                ls=(0, (4, 3)) if es_baseline else "-", zorder=2 + len(por_auc) - puesto,
                solid_capstyle="round", solid_joinstyle="round",
                label=f"{etiqueta} · AUC {_es(aucs[nombre], 3)}")
    if "baseline" not in aucs:
        ax.plot([0, 1], [0, 1], color=_APAGADO, lw=1.6, ls=(0, (4, 3)), zorder=1,
                label=f"Azar · AUC {_es(0.5, 3)}")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.005)
    ax.set_aspect("equal")
    ax.set_xlabel("Tasa de falsos positivos (1 − especificidad)", color=_TINTA_2)
    ax.set_ylabel("Tasa de verdaderos positivos (recall)", color=_TINTA_2)
    _ejes_discretos(ax, rejilla="both", numericos=("x", "y"))
    leyenda = ax.legend(loc="lower right", frameon=False, fontsize=9.5,
                        handlelength=2.2, labelcolor=_TINTA)
    leyenda.set_title("Ordenados por AUC", prop={"size": 9})
    leyenda.get_title().set_color(_TINTA_2)
    _titulo(fig, f"Curva ROC en test: {len(aucs)} modelos en los mismos ejes",
            f"{_es(len(y_true))} reservas · cuanto más arriba a la izquierda, mejor ordena")
    _guardar(fig, ruta)
    return aucs


def importancias(pipeline, X, y, ruta=None):
    """Gráfico de importancia de variables del ganador. Guarda en config.FIG_IMPORTANCIAS.

    Ojo: feature_importances_ favorece a las columnas con muchos valores distintos.
    Aquí no es country (entra agrupada en 11 columnas binarias), sino las numéricas
    continuas: sumadas por variable, las del bosque suben adr del puesto 11 al 6 y
    arrival_date_day_of_month del 21 al 11. permutation_importance es más honesta y
    funciona con cualquier modelo. Di cuál usaste.

    Se usa permutation_importance, para los seis por igual: baraja una variable del
    test, vuelve a predecir con el Pipeline entero y mide cuánto cae la métrica
    principal. Tres ventajas sobre feature_importances_ (que vive en .model_ y solo
    tienen el árbol, el bosque y el boosting):
      · sirve igual si gana la logística o la red, que no la tienen;
      · se baraja la variable ORIGINAL, así que las columnas del one-hot se mueven
        juntas y la importancia sale ya por variable, sin sumar trozos a mano;
      · se mide en la métrica con la que se elige el modelo, al mismo umbral que las
        métricas de test (config.UMBRAL, no el 0,5 de predict), y sobre reservas que no
        ha visto.
    Barajar no decide nada, así que mirar el test aquí no lo contamina.

    Cómo se lee: es la importancia para ESTE modelo, no la relación de cada variable
    con la cancelación. Entre variables asociadas, el modelo se apoya en una y la otra
    sale casi a cero: hotel y distribution_channel pesan poco porque agent y
    market_segment ya llevan esa información, no porque no tengan que ver.

    La figura enseña las config.TOP_IMPORTANCIAS primeras, con ± 1 desviación típica
    entre repeticiones como barra de error y en el color del ganador. Devuelve la
    tabla completa, de mayor a menor.
    """
    ruta = Path(config.FIG_IMPORTANCIAS if ruta is None else ruta)
    modelo = pipeline[-1]
    nombre = getattr(modelo, "nombre", type(modelo).__name__)
    # La métrica y el umbral se fijan AQUÍ y viajan como valores: los procesos hijos
    # importan config de cero y no verían lo que se haya cambiado en tiempo de ejecución.
    # Los procesos, con la misma regla que la validación cruzada: la red, en uno solo
    # (en paralelo, cada hijo arrancaría TensorFlow y no se gana tiempo).
    puntuar = partial(_puntuacion, metrica=config.METRICA_PRINCIPAL, umbral=config.UMBRAL)
    procesos = getattr(modelo, "procesos", lambda: config.N_JOBS)()
    resultado = permutation_importance(pipeline, X, y, scoring=puntuar,
                                       n_repeats=config.IMPORTANCIA_REPETICIONES,
                                       random_state=config.SEMILLA, n_jobs=procesos)
    tabla = pd.DataFrame(
        {"media": resultado.importances_mean, "std": resultado.importances_std},
        index=pd.Index(X.columns, name="variable"),
    ).sort_values("media", ascending=False)

    # barh dibuja de abajo arriba: se invierte para que la más importante quede arriba.
    top = tabla.head(config.TOP_IMPORTANCIAS).iloc[::-1]
    fig, ax = _lienzo(7.2, 0.34 * len(top) + 1.9)
    ax.barh(top.index, top["media"], height=0.62, color=_COLORES.get(nombre, _TINTA_2),
            xerr=top["std"], error_kw={"ecolor": _TINTA_2, "elinewidth": 1, "capsize": 0})
    ax.axvline(0, color=_EJE, lw=1)
    ax.set_xlabel(f"Caída del {config.METRICA_PRINCIPAL.upper()} al barajar la variable "
                  "(media ± 1 desviación típica)", color=_TINTA_2)
    _ejes_discretos(ax, rejilla="x", numericos=("x",))
    ax.tick_params(axis="y", length=0, labelcolor=_TINTA, labelsize=10)
    _titulo(fig, f"Qué variables pesan más en el ganador: {_ETIQUETAS.get(nombre, nombre)}",
            f"Permutación en test, al umbral {_es(config.UMBRAL, 2)} · "
            f"{config.IMPORTANCIA_REPETICIONES} repeticiones · "
            f"las {len(top)} primeras de {len(tabla)} variables")
    _guardar(fig, ruta)
    return tabla


def _puntuacion(estimador, X, y, metrica, umbral):
    """La métrica principal de un estimador sobre (X, y), calculada igual que en
    metricas(): la probabilidad contra el umbral congelado, no el 0,5 de predict().

    Va a nivel de módulo, y no como lambda, para que permutation_importance pueda
    mandarla a sus procesos hijos.
    """
    proba = estimador.predict_proba(X)[:, 1]
    return metricas(y, (proba >= umbral).astype(int), proba)[metrica]


def informe(tabla_comparativa, metricas_test: dict, ruta=None):
    """Vuelca los resultados a outputs/, listos para pegarlos en el README.

    Dos ficheros, con los nombres que fija config:
      · config.TABLA_COMPARATIVA (.csv) — la tabla del apartado 7
      · config.METRICAS_TEST (.json)    — las métricas del ganador, del apartado 8
    Los lee después notebooks/finales/comparativa_modelos.ipynb, así que el notebook
    no vuelve a entrenar y no puede contradecir al pipeline.

    `ruta` es la CARPETA donde se escriben los dos (config.OUTPUTS si no se pasa). El
    JSON es plano, una clave por valor, porque el notebook lo lee con pd.Series: el
    ganador (con la misma regla que model_trainer.elegir_mejor), el umbral y las
    métricas. Se escribe sin NaN, o el notebook y el README copiarían un NaN; y se
    monta antes de tocar el disco, para no dejar la tabla nueva junto al JSON viejo.
    Devuelve (ruta de la tabla, ruta del JSON).
    """
    carpeta = Path(config.OUTPUTS if ruta is None else ruta)
    ruta_tabla = carpeta / config.TABLA_COMPARATIVA.name
    ruta_json = carpeta / config.METRICAS_TEST.name

    resumen = {
        "ganador": model_trainer.elegir_mejor(tabla_comparativa),
        "metrica_principal": config.METRICA_PRINCIPAL,
        "umbral": config.UMBRAL,
        "demo": bool(config.DEMO),
        **{clave: _nativo(valor) for clave, valor in metricas_test.items()},
    }
    texto = json.dumps(resumen, indent=2, ensure_ascii=False, allow_nan=False)

    carpeta.mkdir(parents=True, exist_ok=True)
    tabla_comparativa.to_csv(ruta_tabla, float_format="%.4f")
    ruta_json.write_text(texto + "\n", encoding="utf-8")
    print(f"      informe: {ruta_tabla.name} + {ruta_json.name}")
    return ruta_tabla, ruta_json


# ── Piezas comunes ───────────────────────────────────────────────────────────

def _lienzo(ancho, alto):
    """Una figura con un solo eje, sin pyplot (ver el docstring del módulo)."""
    fig = Figure(figsize=(ancho, alto), dpi=150, facecolor=_FONDO,
                 layout="constrained")
    ax = fig.subplots()
    ax.set_facecolor(_FONDO)
    return fig, ax


def _titulo(fig, titulo, subtitulo):
    """Título y, debajo, un subtítulo más pequeño que dice qué se está midiendo.

    Alineados con el borde izquierdo de la FIGURA, no del eje: con etiquetas largas en
    el eje y (previous_bookings_not_canceled) el eje empieza muy a la derecha y el
    subtítulo se saldría. Se les reserva su franja arriba, medida en pulgadas para que
    quede igual en las tres figuras aunque tengan alturas distintas.
    """
    ancho, alto = fig.get_size_inches()
    x = 0.14 / ancho
    fig.text(x, 1 - 0.16 / alto, titulo, ha="left", va="top", fontsize=12.5,
             color=_TINTA, fontweight="semibold")
    fig.text(x, 1 - 0.46 / alto, subtitulo, ha="left", va="top", fontsize=9.5,
             color=_TINTA_2)
    fig.get_layout_engine().set(rect=(0, 0, 1, 1 - 0.74 / alto))


def _ejes_discretos(ax, rejilla, numericos):
    """Rejilla fina y continua detrás de los datos, sin marco arriba ni a la derecha, y
    las cifras de los ejes con coma decimal, como en el README."""
    ax.set_axisbelow(True)
    ax.grid(axis=rejilla, color=_REJILLA, lw=0.8)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(_EJE)
    ax.tick_params(colors=_APAGADO, labelcolor=_TINTA_2, labelsize=9.5)
    for eje in numericos:
        getattr(ax, f"{eje}axis").set_major_formatter(
            FuncFormatter(lambda valor, _: _es(valor, 2)))


def _guardar(fig, ruta):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, facecolor=_FONDO)
    print(f"      figura: {ruta.name}")


def _tinta_sobre(color) -> str:
    """Texto blanco o negro según lo oscuro que sea el fondo de la celda."""
    r, g, b = to_rgb(color)
    luminancia = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminancia < 0.45 else _TINTA


def _es(numero, decimales=0) -> str:
    """17163 -> '17.163' y 0.6856 -> '0,686': las cifras como las escribe el README."""
    texto = f"{numero:,.{decimales}f}"
    return texto.replace(",", "·").replace(".", ",").replace("·", ".")


def _nativo(valor):
    """int o float de Python: los tipos de numpy no siempre pasan por json.dumps."""
    return int(valor) if isinstance(valor, (int, np.integer)) else float(valor)
