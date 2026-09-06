"""Contrato entre los módulos: las firmas que main.py da por hechas.

Por qué existe este fichero
---------------------------
`main.py` ya estaba escrito antes de empezar, y llama a los cuatro módulos asumiendo
formas concretas: que `preparar()` devuelve un dict con cinco claves exactas, que
`entrenar_y_comparar()` devuelve DOS cosas, que `metricas()` acepta las probabilidades
como tercer argumento. Nada de eso está escrito en el código: vive en los docstrings.

Y durante el sprint A y B trabajan **en paralelo y a ciegas**, cada uno contra su propio
stub. Si uno renombra `preprocesador` a `prep` o devuelve solo la tabla en vez de la
tupla, no se entera nadie hasta el día que se juntan las dos mitades, y entonces se
rompen `main.py`, `evaluator` y `predictor` a la vez.

Estos tests convierten ese acuerdo tácito en algo que se pone ROJO en el acto.

Por qué pasan HOY, con los módulos aún sin implementar
------------------------------------------------------
Porque comprueban la FIRMA, no el comportamiento: `inspect.signature` lee los nombres y
el orden de los parámetros sin llegar a ejecutar la función, así que un cuerpo que sea
`raise NotImplementedError` da igual. Son la única parte de la suite que puede estar en
verde desde el minuto uno, y por eso son el sitio donde se congela el contrato.

Para cambiar una firma: se avisa al otro, se cambia el test EN EL MISMO COMMIT y se dice
en el mensaje del commit. Nunca en silencio.

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


# ── Las cuatro firmas congeladas ─────────────────────────────────────────────
# Si tocas esta tabla, estás cambiando el contrato. Avisa al otro.

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
    """Los nombres y el orden de los parámetros son parte del contrato, no un detalle.

    El orden importa porque main.py llama por posición, no por nombre:
    `evaluator.metricas(d["y_test"], y_pred, y_proba)`. Intercambiar dos parámetros
    del mismo tipo no lanza ningún error: simplemente calcula otra cosa.
    """
    assert parametros(funcion) == esperados


# ── El contrato visto desde quien lo consume: main.py ────────────────────────

CLAVES_DE_PREPARAR = {"X_train", "X_test", "y_train", "y_test", "preprocesador"}


def fuente_main() -> str:
    return (RAIZ / "main.py").read_text(encoding="utf-8")


def test_main_solo_pide_las_claves_que_preparar_promete():
    """El dict de `preparar()` es el punto donde A y B se tocan de verdad.

    Sus claves no las verifica ningún `def`: si data_loader devuelve "prep" y main.py
    pide "preprocesador", el fallo es un KeyError en el paso [2/5], después de haber
    cargado y limpiado 119.390 filas.
    """
    usadas = set(re.findall(r'd\["([^"]+)"\]', fuente_main()))
    assert usadas, "no se han encontrado accesos d[\"...\"] en main.py: ¿se ha renombrado `d`?"
    assert usadas <= CLAVES_DE_PREPARAR, (
        f"main.py pide claves que preparar() no promete: {sorted(usadas - CLAVES_DE_PREPARAR)}"
    )


def test_main_espera_dos_valores_de_entrenar_y_comparar():
    """`entrenar_y_comparar` DEBE devolver (tabla, modelos).

    Devolver solo la tabla es el error más caro del proyecto: revienta al desempaquetar,
    y aunque no reventara, sin el dict de pipelines la curva ROC obligatoria sale con una
    sola línea en vez de con las seis que pide el enunciado.
    """
    assert re.search(r"\btabla\s*,\s*modelos\s*=\s*model_trainer\.entrenar_y_comparar\(",
                     fuente_main()), \
        "main.py ya no desempaqueta dos valores: el contrato de entrenar_y_comparar ha cambiado"


def test_main_usa_la_columna_positiva_de_predict_proba():
    """`predict_proba` devuelve siempre una matriz (n, 2).

    main.py hace `[:, 1]` para quedarse con la probabilidad de la clase «cancela». Si un
    modelo del registro devuelve un vector plano, el índice falla; y si devuelve las
    columnas al revés, no falla nada y el AUC sale por debajo de 0,5.
    """
    assert "predict_proba(d[\"X_test\"])[:, 1]" in fuente_main()


# ── El contrato del registro de modelos ──────────────────────────────────────

METODOS_DEL_CONTRATO = ("construir", "espacio_busqueda", "fit", "predict", "predict_proba")


@pytest.mark.parametrize("metodo", METODOS_DEL_CONTRATO)
def test_modelobase_declara_el_metodo(metodo):
    """La interfaz común es lo que permite que un solo bucle recorra los seis modelos
    sin un solo `if`. Si desaparece un método, el comparador falla en tiempo de ejecución."""
    assert callable(getattr(model_trainer.ModeloBase, metodo, None))


@pytest.mark.parametrize("nombre", sorted(model_trainer.REGISTRO))
def test_cada_modelo_del_registro_cumple_el_contrato(nombre):
    """Toda clase del registro hereda de ModeloBase y su atributo `nombre` coincide con
    su clave. Si divergen, `modelos[ganador]` busca una clave que no existe."""
    clase = model_trainer.REGISTRO[nombre]
    assert issubclass(clase, model_trainer.ModeloBase)
    assert clase.nombre == nombre, f"la clase de «{nombre}» se llama a sí misma «{clase.nombre}»"


def test_el_registro_cubre_los_modelos_activos():
    """config.MODELOS_ACTIVOS y REGISTRO no pueden divergir sin que alguien se entere."""
    from src import config

    faltan = [n for n in config.MODELOS_ACTIVOS if n not in model_trainer.REGISTRO]
    assert not faltan, f"en MODELOS_ACTIVOS pero no en el REGISTRO: {faltan}"
