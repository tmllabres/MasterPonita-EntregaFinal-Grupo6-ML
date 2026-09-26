"""Orquestador del pipeline completo: de los datos a la inferencia.

    python main.py

Este fichero no calcula nada. Solo llama a los módulos de src/ en orden, y por eso
se lee de un vistazo: la prueba del algodón de una arquitectura modular es que
alguien pueda leer main.py y entender el proceso entero sin abrir nada más.

Es lo que se ejecuta delante del profesor en la defensa.
"""
import argparse

from src import config, data_loader, evaluator, model_trainer


def main(demo: bool = False) -> None:
    config.DEMO = demo or config.DEMO

    # 1. Datos: cargar, limpiar, partir, y preparar el preprocesador sin ajustar.
    print("[1/5] Cargando y preparando datos…")
    d = data_loader.preparar()

    # 2. Entrenar los seis con el mismo protocolo y compararlos por validación
    #    cruzada dentro del train. El test no se toca aquí.
    #    Devuelve la tabla Y los pipelines ajustados: los seis hacen falta para la
    #    curva ROC comparativa del paso 4.
    print("[2/5] Entrenando y comparando modelos…")
    tabla, modelos = model_trainer.entrenar_y_comparar(d["X_train"], d["y_train"],
                                                       d["preprocesador"])
    print(tabla)

    # 3. Elegir el ganador por la métrica principal.
    print("[3/5] Eligiendo el mejor modelo…")
    ganador = model_trainer.elegir_mejor(tabla)
    pipeline = modelos[ganador]

    # 4. Evaluar UNA sola vez sobre el test, y generar las figuras obligatorias.
    print(f"[4/5] Evaluando «{ganador}» sobre el test…")
    y_proba = pipeline.predict_proba(d["X_test"])[:, 1]

    # El 0/1 sale de comparar contra config.UMBRAL, no de pipeline.predict(): así el
    # umbral tiene un dueño único y se puede mover y justificar desde un solo sitio.
    y_pred = (y_proba >= config.UMBRAL).astype(int)

    m = evaluator.metricas(d["y_test"], y_pred, y_proba)
    evaluator.matriz_confusion(d["y_test"], y_pred)

    # Las seis curvas en los mismos ejes (la del baseline cae sobre la diagonal del
    # azar), que es la ROC comparativa que pide el enunciado: una predict_proba por
    # modelo sobre el mismo test.
    evaluator.curva_roc({n: p.predict_proba(d["X_test"])[:, 1] for n, p in modelos.items()},
                        d["y_test"])

    evaluator.importancias(pipeline, d["X_test"], d["y_test"])
    evaluator.informe(tabla, m)
    print(m)

    # 5. Persistir el artefacto para que predictor.py pueda usarlo sin reentrenar.
    print("[5/5] Guardando el modelo…")
    model_trainer.guardar(pipeline, ganador, m)
    print("Listo. Figuras en outputs/, modelo en models/.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pipeline de cancelación de reservas")
    p.add_argument("--demo", action="store_true",
                   help="muestra reducida, menos folds y menos épocas en la red, "
                        "para la defensa")
    main(**vars(p.parse_args()))
