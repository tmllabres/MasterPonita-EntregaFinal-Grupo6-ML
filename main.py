"""Ejecuta el proceso completo, de los datos a la predicción con el modelo guardado.

    python main.py
    python main.py --demo

Este fichero no calcula nada: solo llama a los módulos de src/ en orden, así que
leyéndolo se entiende el proceso entero sin abrir nada más. El modo --demo es la
versión corta para la defensa (apartado 6 del README).
"""
import argparse

from src import config, data_loader, evaluator, model_trainer, predictor


def main(demo: bool = False) -> None:
    config.DEMO = demo or config.DEMO

    # 1. Cargar, limpiar y partir los datos, y preparar el preprocesador sin ajustar
    #    (se ajusta dentro del Pipeline, en cada fold).
    print("[1/6] Cargando y preparando datos…")
    d = data_loader.preparar()

    # 2. Comparar los seis modelos con la misma validación cruzada sobre el train; el
    #    test no se usa aquí. Devuelve también los pipelines entrenados, porque la
    #    curva ROC del paso 4 necesita los seis.
    print("[2/6] Entrenando y comparando modelos…")
    tabla, modelos = model_trainer.entrenar_y_comparar(d["X_train"], d["y_train"],
                                                       d["preprocesador"])
    print(tabla)

    # 3. Elegir el ganador por la métrica principal (F1).
    print("[3/6] Eligiendo el mejor modelo…")
    ganador = model_trainer.elegir_mejor(tabla)
    pipeline = modelos[ganador]

    # 4. Evaluar el ganador una sola vez sobre el test y generar las figuras.
    print(f"[4/6] Evaluando «{ganador}» sobre el test…")
    y_proba = pipeline.predict_proba(d["X_test"])[:, 1]

    # El 0/1 sale de comparar con config.UMBRAL y no de pipeline.predict(), para que
    # el umbral se controle desde un solo sitio.
    y_pred = (y_proba >= config.UMBRAL).astype(int)

    m = evaluator.metricas(d["y_test"], y_pred, y_proba)
    evaluator.matriz_confusion(d["y_test"], y_pred)

    # La ROC comparativa del enunciado: las seis curvas en los mismos ejes, con las
    # probabilidades de cada modelo sobre el mismo test. La del baseline es la diagonal.
    evaluator.curva_roc({n: p.predict_proba(d["X_test"])[:, 1] for n, p in modelos.items()},
                        d["y_test"])

    evaluator.importancias(pipeline, d["X_test"], d["y_test"])
    evaluator.informe(tabla, m)
    print(m)

    # 5. Guardar el modelo para que predictor.py lo use sin volver a entrenar.
    print("[5/6] Guardando el modelo…")
    model_trainer.guardar(pipeline, ganador, m)

    # 6. Volver a cargar el modelo desde models/ y predecir unas reservas del test con
    #    la respuesta real al lado, para comprobar que el modelo guardado funciona.
    print("[6/6] Recargando el modelo guardado y prediciendo…")
    recargado, _ = predictor.cargar()
    print(predictor.comparar(recargado, d["X_test"].head(5), d["y_test"].head(5)))
    print("Listo. Figuras en outputs/, modelo en models/.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pipeline de cancelación de reservas")
    p.add_argument("--demo", action="store_true",
                   help="muestra reducida, menos folds y menos épocas en la red, "
                        "para la defensa")
    main(**vars(p.parse_args()))
