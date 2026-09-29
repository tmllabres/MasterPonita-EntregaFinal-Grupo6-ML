"""Tests de lo que los módulos esperan unos de otros: las firmas que usa main.py.

main.py llama a los cuatro módulos dando por hechas algunas formas: que `preparar()`
devuelve un dict con cinco claves, que `entrenar_y_comparar()` devuelve dos valores o
que `metricas()` recibe las probabilidades como tercer argumento. Los módulos se
escribieron por partes, cada uno contra los stubs de los demás, y si uno cambia una de
esas formas no se nota hasta ejecutar el proceso entero. Estos tests lo avisan antes.

No ejecutan las funciones: `inspect.signature` lee los nombres y el orden de los
parámetros sin llamarlas, así que ya pasaban cuando los cuerpos eran solo
`raise NotImplementedError`.

    python -m pytest tests/test_contrato.py -q
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from src import data_loader, evaluator, model_trainer, predictor

RAIZ = Path(__file__).resolve().parent.parent


def parametros(funcion) -> tuple[str, ...]:
    """Nombres de los parámetros de una función, en orden, sin `self`."""
    nombres = tuple(inspect.signature(funcion).parameters)
    return nombres[1:] if nombres and nombres[0] == "self" else nombres


# ── Las firmas de los cuatro módulos ─────────────────────────────────────────
# Si una firma cambia a propósito, se actualiza también en esta tabla.

FIRMAS = [
    # data_loader
    (data_loader.preparar,                ()),
    (data_loader.cargar_crudo,            ("ruta",)),
    (data_loader.limpiar,                 ("df",)),
    (data_loader.separar_X_y,             ("df",)),
    (data_loader.particionar,             ("X", "y")),
    (data_loader.construir_preprocesador, ("X_train",)),
    # model_trainer
    (model_trainer.entrenar_y_comparar,   ("X_train", "y_train", "preprocesador")),
    (model_trainer.elegir_mejor,          ("tabla",)),
    (model_trainer.guardar,               ("pipeline", "nombre", "metricas", "ruta")),
    # evaluator
    (evaluator.metricas,                  ("y_true", "y_pred", "y_proba")),
    (evaluator.matriz_confusion,          ("y_true", "y_pred", "ruta")),
    (evaluator.curva_roc,                 ("modelos_proba", "y_true", "ruta")),
    (evaluator.importancias,              ("pipeline", "X", "y", "ruta")),
    (evaluator.informe,                   ("tabla_comparativa", "metricas_test", "ruta")),
    # predictor
    (predictor.cargar,                    ("ruta",)),
    (predictor.predecir,                  ("pipeline", "reservas")),
    (predictor.predecir_proba,            ("pipeline", "reservas")),
]


@pytest.mark.parametrize("funcion,esperados", FIRMAS,
                         ids=[f"{f.__module__.split('.')[-1]}.{f.__name__}" for f, _ in FIRMAS])
def test_la_firma_no_ha_cambiado(funcion, esperados):
    """Cada función mantiene los nombres y el orden de sus parámetros.

    El orden importa porque main.py llama por posición, no por nombre:
    `evaluator.metricas(d["y_test"], y_pred, y_proba)`. Si se intercambian dos
    parámetros del mismo tipo no salta ningún error, solo se calcula otra cosa.
    """
    assert parametros(funcion) == esperados


# ── Lo que main.py espera de los módulos ─────────────────────────────────────

CLAVES_DE_PREPARAR = {"X_train", "X_test", "y_train", "y_test", "preprocesador"}


def fuente_main() -> str:
    return (RAIZ / "main.py").read_text(encoding="utf-8")


def test_main_solo_pide_las_claves_que_preparar_promete():
    """main.py solo pide al dict de `preparar()` claves que preparar() devuelve.

    Las claves no aparecen en ninguna firma: si data_loader devolviera "prep" y main.py
    pidiera "preprocesador", fallaría con un KeyError en el paso 2 de main.py, después
    de cargar y limpiar las 119.390 filas.
    """
    usadas = set(re.findall(r'd\["([^"]+)"\]', fuente_main()))
    assert usadas, "no se han encontrado accesos d[\"...\"] en main.py: ¿se ha renombrado `d`?"
    assert usadas <= CLAVES_DE_PREPARAR, (
        f"main.py pide claves que preparar() no promete: {sorted(usadas - CLAVES_DE_PREPARAR)}"
    )


def test_main_espera_dos_valores_de_entrenar_y_comparar():
    """main.py desempaqueta dos valores de `entrenar_y_comparar`: (tabla, modelos).

    Si solo devolviera la tabla, main.py fallaría al desempaquetar; y sin el dict de
    pipelines, la curva ROC saldría con una sola línea en vez de con los seis modelos
    que pide el enunciado.
    """
    assert re.search(r"\btabla\s*,\s*modelos\s*=\s*model_trainer\.entrenar_y_comparar\(",
                     fuente_main()), \
        "main.py ya no desempaqueta dos valores: el contrato de entrenar_y_comparar ha cambiado"


def test_main_usa_la_columna_positiva_de_predict_proba():
    """main.py se queda con la columna 1 de predict_proba, la de la clase «cancela».

    predict_proba devuelve una matriz (n, 2). Si un modelo del registro devolviera un
    vector plano, el `[:, 1]` fallaría; y si devolviera las columnas al revés, no
    fallaría nada y el AUC saldría por debajo de 0,5.
    """
    assert "predict_proba(d[\"X_test\"])[:, 1]" in fuente_main()


# ── La interfaz común del registro de modelos ────────────────────────────────

METODOS_DEL_CONTRATO = ("construir", "espacio_busqueda", "fit", "predict", "predict_proba")


@pytest.mark.parametrize("metodo", METODOS_DEL_CONTRATO)
def test_modelobase_declara_el_metodo(metodo):
    """ModeloBase declara cada método de la interfaz común.

    Es lo que permite que un solo bucle recorra los seis modelos sin ningún `if`; si
    faltara un método, el comparador fallaría al ejecutarse."""
    assert callable(getattr(model_trainer.ModeloBase, metodo, None))


@pytest.mark.parametrize("nombre", sorted(model_trainer.REGISTRO))
def test_cada_modelo_del_registro_cumple_el_contrato(nombre):
    """Cada clase del registro hereda de ModeloBase y su `nombre` es su clave.

    Si no coincidieran, `modelos[ganador]` buscaría una clave que no existe."""
    clase = model_trainer.REGISTRO[nombre]
    assert issubclass(clase, model_trainer.ModeloBase)
    assert clase.nombre == nombre, f"la clase de «{nombre}» se llama a sí misma «{clase.nombre}»"


def test_el_registro_cubre_los_modelos_activos():
    """Todos los modelos de config.MODELOS_ACTIVOS están en el REGISTRO."""
    from src import config

    faltan = [n for n in config.MODELOS_ACTIVOS if n not in model_trainer.REGISTRO]
    assert not faltan, f"en MODELOS_ACTIVOS pero no en el REGISTRO: {faltan}"
