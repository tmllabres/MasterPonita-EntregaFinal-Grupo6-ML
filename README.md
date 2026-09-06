# Predicción de cancelación de reservas de hotel

Sistema modular que entrena, evalúa y compara cinco modelos de clasificación binaria
sobre un conjunto de 119.390 reservas de hotel, selecciona el mejor según una métrica
justificada, y automatiza el flujo completo desde el CSV crudo hasta la inferencia.

> **Máster en IA, Cloud Computing y DevOps** · Machine Learning y Deep Learning
> Práctica de evaluación final · Entrega: 15 de septiembre de 2026
>
> **Repositorio:** <https://github.com/tmllabres/MasterPonita-ML-EntregaFinal-Grupo6>

<!--
Este README es la documentación completa de la práctica. El guion lo permite:
«¿La documentación es el README.md? Si se incluye toda la información necesaria
definida en el apartado de Entregables obligatorios, sin problema, pero deben estar
todos los puntos recogidos.»

Los diez puntos obligatorios son los diez apartados de abajo. No borres ninguno.
Para la plataforma de PontIA hay que subir un PDF: exporta este fichero.
-->

---

## 1. Autores y roles

| Persona | Correo | De qué se hace cargo |
|---|---|---|
| | | |

<!--
OBLIGATORIO. El guion: «En la documentación habrá que indicar en un apartado los
roles llevados a cabo, quién se hace cargo de qué parte», y «en caso de que no se
haga una distinción específica de los roles […] se asignará la misma nota a ambas».
Si trabajas solo, dilo aquí y explica que lo hablaste con el profesor.
-->

---

## 2. El problema y por qué este dataset

<!--
Qué se predice, para quién, y qué decisión cambia el modelo.
Aquí entra la justificación de negocio: qué le cuesta al hotel cada tipo de error.
-->

- **Objetivo:** predecir si una reserva se cancelará (`is_canceled = 1`) o no (`0`).
- **Tipo de problema:** clasificación binaria supervisada.
- **Filas del CSV crudo:** 119.390 · **Columnas:** 32 · **Predictoras reales:** 29
- **Reparto de clases en crudo:** 62,96 % no cancela / 37,04 % cancela (razón 1,70 : 1)
- **Tras la limpieza:** _(rellenar: filas que quedan y reparto resultante)_

<!--
OJO, y esto hay que resolverlo antes de entrenar: los duplicados EXACTOS del CSV son
32.252 filas, el 27 % del dataset, y el 63,4 % de ellas son cancelaciones. Si se
eliminan, quedan 86.971 filas y el reparto pasa a 72,69 / 27,31. Diez puntos de
prevalencia menos cambian la justificación de la métrica del apartado 5 y el tamaño
del test del apartado 8.
¿Son un error de registro o reservas legítimamente idénticas (mismo hotel, misma
noche, mismo precio, grupos)? Decidid, justificadlo aquí, y que los números de los
apartados 2, 5 y 8 salgan todos de la MISMA decisión.
-->

El diccionario de variables está en [`docs/diccionario_datos.md`](docs/diccionario_datos.md).

**Fuga de datos detectada y eliminada.** `reservation_status` y `reservation_status_date`
determinan el objetivo al 100 % (`Check-Out` → 0; `Canceled` y `No-Show` → 1): son la
etiqueta escrita con otras palabras. Si se dejan, los cinco modelos sacan un AUC ≈ 1,000
y la comparación no distingue nada. De ahí la cuenta: **32 − `is_canceled` − 2 de fuga = 29**
columnas predictoras.

---

## 3. Análisis exploratorio (EDA)

<!--
OBLIGATORIO. Puntúa por lo que CAMBIÓ, no por el número de gráficos.
Notebook: notebooks/finales/eda_final.ipynb
La tabla de abajo es el formato que más rinde: cada hallazgo, con su decisión al lado.
-->

| Hallazgo | Evidencia | Decisión que tomamos |
|---|---|---|
| | | |

---

## 4. Diseño del sistema

<!-- OBLIGATORIO. -->

```
Entrega-Final-ML/
├── main.py                          orquestador: python main.py
├── requirements.txt                 dependencias con versión fijada
├── .python-version                  3.12
├── .gitignore
│
├── src/
│   ├── config.py                    parámetros, rutas, semilla y umbral
│   ├── data_loader.py               cargar, limpiar, partir, preprocesador
│   ├── model_trainer.py             registro de modelos y bucle comparador
│   ├── evaluator.py                 métricas y figuras; único que abre el test
│   └── predictor.py                 inferencia con el modelo entrenado
│
├── notebooks/
│   ├── exploracion/                 la cocina: se prueba y se falla
│   │   ├── eda_inicial.ipynb
│   │   └── pruebas_modelos.ipynb
│   └── finales/                     el escaparate: lo que se defiende
│       ├── eda_final.ipynb
│       └── comparativa_modelos.ipynb
│
├── tests/                           pytest: contrato del registro y de la limpieza
│   ├── conftest.py
│   ├── test_data_loader.py
│   └── test_model_trainer.py
│
├── data/
│   ├── raw/dataset_practica_final.csv   el CSV original, intacto y versionado
│   └── processed/                   intermedios (no se versiona)
│
├── models/                          artefactos entrenados (no se versiona)
│   ├── mejor_modelo.pkl             el Pipeline COMPLETO: preprocesado + modelo
│   ├── mejor_modelo.keras           solo si gana la red
│   └── metadatos.json               ganador, umbral, columnas, semilla, versiones
│
├── outputs/                         generado por evaluator.py, SÍ se versiona
│   ├── confusion_matrix.png
│   ├── roc_curve.png                los cinco modelos en los mismos ejes
│   ├── feature_importance.png
│   ├── tabla_comparativa.csv
│   └── metricas_test.json
│
└── docs/
    ├── diccionario_datos.md         las 32 variables del CSV
    ├── guion_practica.pdf           el enunciado
    └── informe_final.pdf            este README exportado: lo que se sube a PontIA
```

**Decisiones de diseño que hay que poder defender:**

- **Un módulo, una responsabilidad.** Cada uno se describe en una frase sin usar «y».
- **La partición se hace una sola vez**, en `data_loader`. Si cada módulo partiera por
  su cuenta, cada modelo se compararía contra un reparto distinto.
- **Todo lo que aprende un número de los datos va dentro del `Pipeline`** (medianas,
  categorías, medias del escalado), para que se ajuste solo con el train de cada fold.
  Lo que **borra filas** (duplicados, imposibles) va fuera y antes de partir.
- **Registro de modelos con interfaz común**, imitando por dentro a una librería de
  AutoML: añadir un sexto modelo es una clase y una línea, sin tocar el bucle.
- **Semilla única** (`config.SEMILLA = 42`) para partición, modelos y validación cruzada.

---

## 5. Métrica principal y por qué

<!--
OBLIGATORIO justificarlo. Elígela ANTES de entrenar y no la cambies después.
-->

**Métrica principal:** _(F1 de la clase positiva)_ · **Secundarias:** accuracy, precision, recall, ROC-AUC.

Con el reparto del CSV crudo (62,96 / 37,04), un modelo que dijera siempre «no cancela»
ya acierta el **62,96 %** sin haber aprendido nada: por eso accuracy no sirve como
criterio. _(Si la limpieza elimina los duplicados, el porcentaje sube a 72,69 % y el
argumento se refuerza: actualizad la cifra según lo que decidáis en el apartado 2.)_

<!--
Ojo, «F1 equilibra precision y recall» NO es una justificación: di qué error te duele
más en este negocio y por qué el equilibrio es lo razonable aquí.
-->

---

## 6. Cómo ejecutar el proyecto

<!-- OBLIGATORIO: versión de Python, entorno virtual y pasos exactos. -->

**Requisitos:** Python 3.12

```bash
git clone https://github.com/tmllabres/MasterPonita-ML-EntregaFinal-Grupo6.git
cd MasterPonita-ML-EntregaFinal-Grupo6

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

python -m pip install -r requirements.txt
```

<details>
<summary>Alternativa rápida con <code>uv</code> (opcional)</summary>

Si tienes [uv](https://docs.astral.sh/uv/) instalado, lee el mismo `requirements.txt`
y tarda bastante menos —importa, porque TensorFlow son varios cientos de MB:

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

No es necesario: los pasos de arriba con `pip` funcionan igual.
</details>

**Pipeline completo** (carga → limpieza → entrenamiento → comparación → evaluación → artefacto):

```bash
python main.py
```

**Versión corta para la defensa** (muestra reducida y menos folds):

```bash
python main.py --demo
```

**Inferencia con el modelo ya entrenado, sin reentrenar nada:**

```bash
python -m src.predictor
```

> `models/` está en el `.gitignore`, así que en un clon recién hecho **todavía no existe
> ningún artefacto**: hay que lanzar `python main.py` (o `--demo`) al menos una vez antes
> de que la inferencia funcione.

Salidas: figuras y tablas en `outputs/`, modelo en `models/`.

---

## 7. Modelos comparados

Los cinco que exige el enunciado, entrenados con el mismo protocolo: mismo
`StratifiedKFold`, misma semilla, preprocesado dentro del `Pipeline` y
`scoring` fijado a la métrica principal.

| Modelo | F1 (CV) | Accuracy | Precision | Recall | ROC-AUC | Tiempo |
|---|---|---|---|---|---|---|
| Baseline (`DummyClassifier`) | | | | | | |
| Regresión logística | | | | | | |
| Árbol de decisión | | | | | | |
| Random Forest | | | | | | |
| Gradient Boosting | | | | | | |
| Red neuronal (Keras) | | | | | | |

<!--
La fila del baseline no es decorativa: sin ella, un número suelto no dice si tus
modelos aportan algo por encima de lo trivial.
Las métricas de CV son media ± desviación sobre el train. La del test se mide UNA vez.
-->

---

## 8. Resultados y elección final

<!-- OBLIGATORIO. -->

**Modelo elegido:** _(…)_ · **Por qué gana:** _(…)_

Métricas del ganador sobre el conjunto de test _(N reservas nunca vistas: 23.878 si no
se eliminan los duplicados, 17.394 si sí — poned el número real)_:

| | valor |
|---|---|
| F1 | |
| Accuracy | |
| Precision | |
| Recall | |
| ROC-AUC | |

**Figuras obligatorias:**

- Matriz de confusión → `outputs/confusion_matrix.png`
- Curva ROC de los cinco modelos en los mismos ejes → `outputs/roc_curve.png`
- Importancia de variables del ganador → `outputs/feature_importance.png`

---

## 9. Conclusiones

<!-- OBLIGATORIO en el README. Qué se aprendió, qué se puede hacer con esto. -->

---

## 10. Reflexión crítica: limitaciones y mejoras

<!--
OBLIGATORIO, y es donde más gente deja puntos. Reconocer una limitación tú vale
bastante más que si la detecta el profesor.
Candidatas honestas:
  · la validación cruzada no es anidada: los hiperparámetros se eligieron mirando
    los mismos folds con los que se mide
  · la partición es aleatoria, no temporal: en producción predices el futuro
  · el dataset no trae el coste en euros de cada error, así que el umbral se
    elige por F1 y no por dinero
  · el modelo mide correlación, no causa
-->

---

## 11. Bonus técnicos implementados

<!--
NO es obligatorio, pero vale hasta 2 puntos adicionales y permite llegar al 10.
El guion es tajante: «Es necesario que el sistema funcione para poder sumar la
puntuación adicional». Un bonus a medias resta tiempo y no suma nota.
Borra las filas que no hagáis: una tabla con seis «pendiente» es peor que tres filas.
-->

| Bonus | Estado | Dónde está | Qué demuestra |
|---|---|---|---|
| Optimización de hiperparámetros (`RandomizedSearchCV`) | | `src/model_trainer.py` · `config.BUSQUEDA` | |
| Interpretabilidad (`feature_importances_` / SHAP) | | `src/evaluator.py` · `outputs/feature_importance.png` | |
| Balanceo de clases (`class_weight` / SMOTE) | | | |
| API REST con FastAPI (`/train`, `/predict`, `/evaluate`) | | | |
| Registro de experimentos con MLflow | | | |
| Interfaz visual (Streamlit / Gradio) | | | |

---

## Anexos

- [Diccionario de variables](docs/diccionario_datos.md) — las 32 columnas del CSV
- [Guion de la práctica](docs/guion_practica.pdf) — el enunciado del profesor
- Notebooks que se defienden:
  - [`notebooks/finales/eda_final.ipynb`](notebooks/finales/eda_final.ipynb) — el EDA presentable
  - [`notebooks/finales/comparativa_modelos.ipynb`](notebooks/finales/comparativa_modelos.ipynb) — la tabla y las figuras
- Notebooks de trabajo (la cocina): [`notebooks/exploracion/`](notebooks/exploracion/)

**Tests:** `python -m pytest -q` desde la raíz.
