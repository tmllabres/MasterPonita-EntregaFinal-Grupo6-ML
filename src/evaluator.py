"""Métricas y figuras del modelo ganador sobre el test.

El test se usa una sola vez, al final, con el modelo ya elegido: si influyera en
alguna decisión (qué modelo, qué hiperparámetro, qué umbral), el resultado dejaría
de ser una estimación fiable.

Las figuras se dibujan sobre un `Figure` de matplotlib y no con pyplot, que tiene
estado global: así no se quedan figuras abiertas entre llamadas ni sale un PNG en
blanco porque un plt.show() haya vaciado la figura.
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
# Texto, ejes y rejilla en tonos suaves, para que lo que destaque sean los datos.
_FONDO = "#fcfcfb"
_TINTA = "#0b0b0b"
_TINTA_2 = "#52514e"
_APAGADO = "#898781"
_REJILLA = "#e1e0d9"
_EJE = "#c3c2b7"

# Cada modelo tiene el mismo color en todas las figuras, quede en el puesto que quede.
# La paleta, en este orden, es apta para daltonismo; el baseline va en gris porque es
# la referencia y no un modelo más.
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

# En la matriz, un solo tono de claro a oscuro, porque ahí el color indica cantidad.
_AZULES = LinearSegmentedColormap.from_list(
    "azules", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def metricas(y_true, y_pred, y_proba=None) -> dict:
    """Devuelve la métrica principal y las secundarias en un dict.

    pos_label=1 porque la clase que quiero detectar es «cancela». El AUC se calcula
    con las probabilidades, porque con los 0/1 la curva ROC tendría un solo punto; sin
    y_proba no se incluye. zero_division=0 da precision 0 al baseline, que nunca
    predice 1, en vez de un aviso. Los valores salen como float/int de Python para que
    json.dump no falle con tipos de numpy; `n` es el número de reservas evaluadas.
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
    """Matriz de confusión del ganador en test. Guarda en config.FIG_CONFUSION.

    Cada celda lleva el número absoluto y el porcentaje sobre su fila real, así que la
    celda de TP da el recall y la de TN la specificity. El color sigue al porcentaje y
    no al número porque, si no, la fila «No cancela», casi tres veces más grande, lo
    taparía todo. Devuelve la matriz de conteos (filas = real, columnas = predicho).
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
    """Curva ROC de todos los modelos en los mismos ejes. Guarda en config.FIG_ROC.

    Recibe {nombre: probabilidad de cancelar} de todos los modelos, no solo del
    ganador. La leyenda va ordenada por AUC y las mejores curvas se dibujan encima,
    para que el ganador no quede tapado donde se cruzan. El baseline da siempre la
    misma probabilidad, así que su curva ya es la diagonal del azar; la diagonal solo
    se dibuja aparte si el baseline no está. Devuelve {nombre: AUC}.
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
    """Importancia por permutación del ganador. Guarda en config.FIG_IMPORTANCIAS.

    Uso la permutación y no feature_importances_ (que solo tienen los modelos de
    árboles y favorece a las variables continuas) porque sirve para cualquier modelo y
    baraja la variable original, así que sale por variable y no por columna del
    one-hot. Mide la caída de la métrica principal con config.UMBRAL, igual que las
    métricas de test; barajar no decide nada, así que usar el test aquí no lo
    contamina. Devuelve la tabla completa, de mayor a menor.
    """
    ruta = Path(config.FIG_IMPORTANCIAS if ruta is None else ruta)
    modelo = pipeline[-1]
    nombre = getattr(modelo, "nombre", type(modelo).__name__)
    # La métrica y el umbral se pasan como valores porque los procesos hijos importan
    # config de cero y no verían lo cambiado en tiempo de ejecución. Los procesos siguen
    # la regla de la validación cruzada: la red, en uno solo (en paralelo, cada proceso
    # cargaría TensorFlow sin ganar tiempo).
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
    """Métrica principal del estimador sobre (X, y), calculada igual que en metricas().

    Compara la probabilidad con `umbral` en vez de usar predict(). Está a nivel de
    módulo, y no como lambda, para que permutation_importance pueda mandarla a sus
    procesos hijos.
    """
    proba = estimador.predict_proba(X)[:, 1]
    return metricas(y, (proba >= umbral).astype(int), proba)[metrica]


def informe(tabla_comparativa, metricas_test: dict, ruta=None):
    """Guarda la tabla comparativa (.csv) y las métricas de test (.json) en outputs/.

    Son la tabla del apartado 7 y las métricas del apartado 8 del README. Los lee
    comparativa_modelos.ipynb, que así no vuelve a entrenar y enseña los mismos
    resultados que main.py. `ruta` es la carpeta de salida (config.OUTPUTS si no se
    pasa). El JSON es plano porque el notebook lo lee con pd.Series, no admite NaN y
    se genera antes de escribir nada, para no dejar la tabla nueva junto a un JSON
    viejo. Devuelve (ruta de la tabla, ruta del JSON).
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
    """Título y, debajo, un subtítulo más pequeño con lo que se mide.

    Van alineados con el borde izquierdo de la figura y no del eje, porque con
    etiquetas largas en el eje y (previous_bookings_not_canceled) el subtítulo se
    saldría. Su franja se mide en pulgadas para que quede igual en las tres figuras.
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
