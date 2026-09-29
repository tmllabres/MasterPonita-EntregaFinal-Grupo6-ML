"""Código del proyecto de predicción de cancelaciones de reservas de hotel.

    src/
    ├── config.py         parámetros: rutas, semilla, métrica, umbral...
    ├── data_loader.py    carga, limpieza, partición y preprocesado
    ├── model_trainer.py  los modelos y su comparación
    ├── evaluator.py      métricas y figuras sobre el test
    └── predictor.py      predicciones con el modelo guardado

main.py, en la raíz, llama a estos módulos en orden.
"""
