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

- **Objetivo:** predecir si una reserva se cancelará (`is_canceled = 1`) o no (`0`).
- **Tipo de problema:** clasificación binaria supervisada.
- **Filas del CSV crudo:** 119.390 · **Columnas:** 32 · **Predictoras reales:** 27
- **Reparto de clases en crudo:** 62,96 % no cancela / 37,04 % cancela (razón 1,70 : 1)
- **Tras la limpieza:** 85.811 filas · 72,36 % no cancela / 27,64 % cancela (razón 2,62 : 1)

El diccionario de variables está en [`docs/diccionario_datos.md`](docs/diccionario_datos.md).

### Para quién es, y qué decisión cambia

El destinatario es el departamento de *revenue management* del hotel, y la decisión
concreta que cambia es **cuántas habitaciones se vuelven a poner a la venta**.

Una reserva confirmada bloquea inventario. Si esa reserva se va a cancelar y nadie lo sabe
hasta el día de la llegada, la habitación ya no se puede vender a nadie: el inventario de
un hotel es perecedero, y la noche del 14 de agosto no se puede vender el día 15. Con una
probabilidad de cancelación **por reserva** se puede hacer overbooking controlado, decidir
a quién se le pide prepago o depósito, y dimensionar plantilla y compras en los días de más
riesgo.

Sin modelo, la única alternativa es aplicar la tasa media de cancelación a todo el mundo
por igual —que es, literalmente, lo que hace el clasificador trivial contra el que se mide
el sistema en el apartado 5.

### Qué le cuesta al hotel cada tipo de error

Los dos errores duelen, y no de la misma manera:

| Error | Qué pasa | Qué cuesta |
|---|---|---|
| **Falso negativo** — se predice «no cancela» y la reserva se cancela | La habitación se queda vacía sin que nadie lo viera venir | La noche entera, y no se recupera: no hubo margen para revenderla |
| **Falso positivo** — se predice «cancela» y el cliente aparece | El hotel ha revendido una habitación que sí se iba a ocupar y tiene que realojar al huésped | Compensación, traslado a otro establecimiento y una reseña negativa que dura mucho más que la noche |

El dataset **no trae el coste en euros de ninguno de los dos**, así que no se puede fijar
una razón coste-beneficio y optimizar por dinero. De ahí que la métrica del apartado 5 sea
F1 —que obliga a atender a los dos errores— y no una métrica asimétrica elegida a ojo. Es
también la limitación que se reconoce en el apartado 10.

### Fugas de datos detectadas y eliminadas

Una fuga es una columna que el modelo no tendría cuando toca predecir. Aquí se predice
**al hacerse la reserva o en las semanas siguientes, antes de que llegue el cliente**, que
es cuando revenue management decide el overbooking o el depósito. Todo lo que se escribe a
la llegada del cliente, o una vez se sabe si canceló, es la respuesta con otras palabras.

**Directas.** `reservation_status` y `reservation_status_date` determinan el objetivo al
100 % (`Check-Out` → 0; `Canceled` y `No-Show` → 1). Si se dejan, los cinco modelos sacan un
AUC ≈ 1,000 y la comparación no distingue nada.

**Posteriores a la llegada.** Otras dos columnas se rellenan cuando el cliente llega, y
los datos lo delatan (cifras sobre el CSV crudo):

| Columna | Evidencia | Lectura |
|---|---|---|
| `required_car_parking_spaces` | 7.416 reservas con plaza: **0 canceladas y 0 No-Show**. Si se anotara al reservar, cabrían unas 2.670 cancelaciones y 75 No-Show; las peticiones especiales, que sí se hacen al reservar, cancelan un 21,7 % | La plaza se anota cuando el cliente llega en coche: quien no llega, nunca la tiene |
| `assigned_room_type` | Distinta de la reservada en el 18,8 % de las Check-Out y el 17,2 % de los No-Show, pero **solo en el 1,4 % de las canceladas** | La habitación se asigna el día de llegada (por eso los No-Show sí la tienen); quien cancela antes nunca llega a ese día |

Con las dos dentro, un gradient boosting sube el F1 en validación cruzada de **0,678 a
0,710** (y el AUC de 0,897 a 0,914) sobre las mismas filas. Es nota regalada: cuando toca
predecir, nadie tiene todavía plaza anotada ni habitación asignada.

**`booking_changes` se queda, con una duda declarada.** El CSV guarda el número *final* de
cambios, y parte puede ser posterior al momento de predecir. Pero no delata el desenlace
como las otras dos (con cambios cancela un 15,6 % frente a un 30,3 %: separa, pero no a la
perfección), y los cambios hechos hasta el momento de predecir sí son información legítima.
Queda como limitación para el apartado 10.

De ahí la cuenta: **32 − `is_canceled` − 4 de fuga = 27** columnas predictoras. Las cuatro
están en `config.FUGAS`, con su porqué.

### Los 33.413 duplicados exactos: se eliminan

Son el **28,0 %** del dataset, y el **61,3 %** de ellas son cancelaciones, así que no es una
decisión menor: mueve la prevalencia del objetivo casi diez puntos.

**Se eliminan.** Qué cuenta como duplicado: dos filas idénticas en las 28 columnas que
quedan tras quitar las fugas, es decir, en las 27 predictoras **y** en la respuesta. El CSV
no trae identificador de reserva, así que esas copias no se pueden distinguir entre sí. Las
filas que coinciden en las 27 predictoras pero acabaron de forma distinta (289 pares, 578
filas) **se conservan**: esas sí son, con seguridad, reservas distintas.

Dejar las copias tiene un coste asimétrico: si una cae en train y su gemela en test, el
modelo ya ha visto ese ejemplo exacto y su nota de test sale inflada **sin que salte ningún
error**. No es una hipótesis: si se parte el CSV sin deduplicar (misma semilla y
`stratify`), **7.813 de las 23.878 filas de test (32,7 %) tienen una gemela exacta en
train**, y cancelan un 58,2 % frente al 26,7 % del resto.

El precio, que se asume: las copias se concentran en reservas de grupo. Al deduplicar se va
el 93,1 % de las reservas `Non Refund` (de 14.587 a 1.011) y el 77,6 % del segmento
`Groups`, y la tasa de cancelación del City Hotel baja del 41,73 % al 30,21 %. Algunas de
esas filas serán habitaciones legítimas de un mismo grupo; perderlas es un precio menor que
publicar una métrica de test que no es honesta.

El orden de la limpieza importa, y está fijado en `data_loader.limpiar()`:

1. **Primero las fugas.** El modelo nunca las ve, así que no pueden decidir qué es un
   duplicado. Con ellas dentro saldrían 31.994 duplicados en vez de 33.413: 1.419 filas
   idénticas en todo lo que ve el modelo sobrevivirían solo por diferir en una columna de
   fuga.
2. **Después los duplicados**, y siempre **antes de particionar**. Si se particiona primero,
   la propia partición ya ha repartido las gemelas entre los dos lados.
3. **Por último los imposibles:** 1 reserva con `adr` negativo (−6,38) y 165 sin ningún
   huésped (0 adultos, 0 niños y 0 bebés). No son valores raros discutibles: un precio
   negativo no es un precio, y una reserva sin personas no es la reserva de nadie. Su
   etiqueta no describe a ningún cliente —16 de esas 165 constan como canceladas—, así que
   solo aportarían ruido.

Lo que **no** se borra, a propósito, porque es raro pero posible:

- **218 reservas con 0 adultos pero con niños o bebés.** «Sin huéspedes» es la suma de las
  tres columnas, no `adults == 0`.
- **1.619 reservas con `adr = 0`.** Entre ellas están las 621 del segmento `Complementary`
  (invitaciones del hotel) y las 587 de 0 noches; un precio 0 no es imposible.
- **Una reserva con `adr = 5.400`**, cuando el siguiente valor más alto es 510. Es un valor
  extremo, no imposible. Con esta partición cae en el test, así que no afecta al escalado,
  que se aprende solo con el train.

**Consecuencia asumida.** La prevalencia de cancelación baja de 37,04 % a **27,64 %**, y con
ella sube el acierto del modelo trivial de 62,96 % a **72,36 %** —que es justo el argumento
del apartado 5 contra usar accuracy como criterio. Los números de los apartados 5, 7 y 8
salen todos de esta misma decisión: de las 85.811 filas limpias, **68.648 van a train y
17.163 al test**, con un 27,64 % de cancelaciones en cada mitad (`stratify=y`).

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
│   ├── roc_curve.png                los seis modelos en los mismos ejes
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
  AutoML: añadir un modelo más es una clase y una línea, sin tocar el bucle.
- **Semilla única** (`config.SEMILLA = 42`) para partición, modelos y validación cruzada.

---

## 5. Métrica principal y por qué

<!--
OBLIGATORIO justificarlo. Elígela ANTES de entrenar y no la cambies después.
-->

**Métrica principal:** _(F1 de la clase positiva)_ · **Secundarias:** accuracy, precision, recall, ROC-AUC.

Con el reparto del CSV crudo (62,96 / 37,04), un modelo que dijera siempre «no cancela»
ya acierta el **62,96 %** sin haber aprendido nada: por eso accuracy no sirve como
criterio. _(Si la limpieza elimina los duplicados, el porcentaje sube a 72,36 % y el
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

**Versión corta para la defensa** (muestra reducida, menos folds y menos épocas en la red):

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
se eliminan los duplicados, 17.163 si sí — poned el número real)_:

| | valor |
|---|---|
| F1 | |
| Accuracy | |
| Precision | |
| Recall | |
| ROC-AUC | |

**Figuras obligatorias:**

- Matriz de confusión → `outputs/confusion_matrix.png`
- Curva ROC de los seis modelos en los mismos ejes → `outputs/roc_curve.png`
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
