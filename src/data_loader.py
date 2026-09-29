"""Carga, limpieza, partición y preprocesado de los datos.

La limpieza quita filas, así que se hace aquí, antes de partir, y no en el Pipeline:
un transformador de scikit-learn devuelve tantas filas como recibe, y si quitara filas
la `y` dejaría de cuadrar con la `X` sin dar ningún error. Lo que se aprende de los
datos (medianas, escalado, categorías) va en el preprocesador, dentro del Pipeline,
para que se aprenda solo con el train.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config

# Los nulos del CSV vienen como el texto "NULL" (country, agent, company) o "NA"
# (children; ningún país del CSV usa ese código). pandas ya los lee como nulo, pero dejo
# la lista escrita para no depender de su lista por defecto. _columna_a_texto la usa
# también al predecir: un "NULL" que faltara aquí caería en el cajón de infrecuentes y
# no en "desconocido".
NA_VALUES = ["NULL", "null", "NA", "", " "]

# Columnas que definen una "reserva sin huéspedes". Se suman las tres porque una
# reserva con 0 adultos y 2 niños es rara, pero no imposible.
HUESPEDES = ["adults", "children", "babies"]


def _miles(n: int) -> str:
    """119390 -> '119.390'. Solo para que los prints cuadren con el README."""
    return f"{n:,}".replace(",", ".")


def cargar_crudo(ruta=None) -> pd.DataFrame:
    """Lee el CSV original, sin limpiar.

    Los nulos escritos como texto ("NULL", "NA") se leen como NaN gracias a NA_VALUES.
    """
    return pd.read_csv(config.DATA_RAW if ruta is None else ruta, na_values=NA_VALUES)


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Quita las columnas de fuga, los duplicados exactos y los registros imposibles.

    El orden importa: primero las fugas, luego los duplicados y al final los imposibles
    (adr negativo y reservas sin huéspedes). Se llama antes de partir en train y test, e
    imprime lo que quita cada paso (son las cifras del apartado 2 del README).
    """
    filas_iniciales = len(df)

    # 1. Fugas (ver config.FUGAS y el apartado 2 del README). Van primero porque el
    #    modelo no ve estas columnas, así que no deben decidir qué es un duplicado: con
    #    ellas dentro saldrían 31.994 duplicados en vez de 33.413.
    fugas = [c for c in config.FUGAS if c in df.columns]
    df = df.drop(columns=fugas)
    print(f"      [limpieza] fugas: -{len(fugas)} columnas ({', '.join(fugas)})")

    # 2. Duplicados exactos, antes de partir: si una copia cayera en train y otra en
    #    test, el modelo se evaluaría con reservas que ya ha visto.
    if config.ELIMINAR_DUPLICADOS:
        antes = len(df)
        df = df.drop_duplicates()
        print(f"      [limpieza] duplicados exactos: -{_miles(antes - len(df))} filas")

    # 3. Imposibles: un adr negativo no es un precio válido y una reserva sin huéspedes
    #    no corresponde a ningún cliente, así que su etiqueta solo añadiría ruido.
    if "adr" in df.columns:
        adr_negativo = df["adr"] < 0
    else:
        adr_negativo = pd.Series(False, index=df.index)

    presentes = [c for c in HUESPEDES if c in df.columns]
    if presentes:
        sin_huespedes = df[presentes].fillna(0).sum(axis=1) == 0
    else:
        sin_huespedes = pd.Series(False, index=df.index)

    # Al contar las de 0 huéspedes descuento las que ya caen por adr negativo, para que
    # los números impresos sumen el total.
    print(f"      [limpieza] adr negativo: -{_miles(int(adr_negativo.sum()))} filas")
    print("      [limpieza] 0 huéspedes: "
          f"-{_miles(int((sin_huespedes & ~adr_negativo).sum()))} filas")
    df = df[~(adr_negativo | sin_huespedes)]

    print(f"      [limpieza] total: {_miles(filas_iniciales)} -> {_miles(len(df))} filas")
    return df.reset_index(drop=True)


def separar_X_y(df: pd.DataFrame):
    """Devuelve (X, y): X con las 27 columnas predictoras e y = is_canceled.

    Son 27 porque al CSV (32 columnas) se le quitan is_canceled y las 4 de fuga.
    """
    # Con un df de limpiar() las fugas ya no están, pero las quito también aquí por si
    # llegan datos sin limpiar: dejarlas en X sería darle la respuesta al modelo.
    fuera = [c for c in [config.OBJETIVO, *config.FUGAS] if c in df.columns]
    return df.drop(columns=fuera), df[config.OBJETIVO].astype(int)


def particionar(X, y):
    """Parte en train y test con el test_size, la semilla y el stratify de config.

    Se parte una sola vez, aquí, para que todos los modelos se comparen con el mismo
    reparto. Con stratify, train y test tienen el mismo porcentaje de cancelaciones, y
    con la semilla fija el reparto sale igual en cada ejecución.
    """
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.SEMILLA,
        stratify=y if config.ESTRATIFICAR else None,
    )


def _columna_a_texto(col: pd.Series) -> pd.Series:
    """Pasa una columna a texto valor a valor, convirtiendo antes cada número a entero.

    Va valor a valor para que la codificación de una reserva no dependa de las demás del
    lote: si se decidiera por columna, un solo "NULL" de texto al predecir dejaría el
    9.0 de otra fila como "9.0" y acabaría en el cajón de infrecuentes. También se
    quitan los espacios de alrededor (" PRT " pasa a "PRT", como en el train).
    """
    texto = col.astype(object).astype(str).str.strip()
    numeros = pd.to_numeric(col, errors="coerce").astype(float)
    # Solo convierto los números que caben en un entero: un "inf" o un 1e20 no son
    # un ID, así que se quedan como texto y van al cajón en vez de dar error.
    es_numero = numeros.abs() < 2**63
    texto[es_numero] = numeros[es_numero].round().astype("Int64").astype(str)
    # Nulo: NaN o None, o uno de los textos de NA_VALUES ("NULL", "", ...) que llegue al
    # predecir (por ejemplo, desde un JSON) sin pasar por cargar_crudo().
    return texto.mask(col.isna() | texto.isin(NA_VALUES), "desconocido")


def _a_texto(X: pd.DataFrame) -> pd.DataFrame:
    """Pasa a texto country, agent y company, con "desconocido" en los nulos.

    agent y company son float64 (IDs con nulos), así que cada ID pasa antes a entero: si
    no, el agente 9 se aprendería como "9.0" y un agent=9 al predecir no coincidiría.
    El nulo es una categoría más porque tiene significado: sin agente, la reserva es
    directa, y sin empresa, no es un viaje de empresa. Es una función y no una lambda
    porque joblib no puede guardar una lambda dentro del Pipeline.
    """
    return X.apply(_columna_a_texto)


def construir_preprocesador(X_train) -> ColumnTransformer:
    """Devuelve el ColumnTransformer sin ajustar, con tres ramas.

      - numéricas: imputar con la mediana y escalar.
      - categóricas: imputar con "desconocido" y one-hot (handle_unknown="ignore").
      - alta cardinalidad (country, agent, company): top-10 y el resto agrupado.

    Se devuelve sin ajustar porque lo ajusta el Pipeline en cada fit, solo con el train
    de cada fold; así no hay fuga en el preprocesado.
    """
    # Las de alta cardinalidad van primero: agent y company son float64 y, si no, irían
    # a las numéricas y se escalarían como si el agente 240 fuera el doble del 120.
    alta = [c for c in config.ALTA_CARDINALIDAD if c in X_train.columns]
    numericas = [c for c in X_train.select_dtypes(include="number").columns if c not in alta]
    categoricas = [c for c in X_train.select_dtypes(include=["object", "category", "bool"]).columns
                   if c not in alta]

    # Con remainder="drop", una columna de un tipo no previsto (una fecha, por ejemplo)
    # se perdería sin avisar, así que prefiero que dé error aquí.
    sin_rama = [c for c in X_train.columns if c not in {*alta, *numericas, *categoricas}]
    if sin_rama:
        raise ValueError(f"columnas sin rama en el preprocesador: {sin_rama}")

    # Mediana y no media: lead_time y adr tienen la cola larga a la derecha.
    rama_numericas = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ])

    # handle_unknown="ignore": una categoría que el train del fold no ha visto se
    # codifica como ceros en vez de dar error a mitad de la validación cruzada.
    rama_categoricas = Pipeline([
        ("imputar", SimpleImputer(strategy="constant", fill_value="desconocido")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # max_categories = TOP_N + 1 porque el cajón del resto cuenta como una categoría:
    # con 10 quedarían 9 valores y el cajón. Con "infrequent_if_exist", las categorías
    # que no se vieron al entrenar también van al cajón.
    rama_alta_cardinalidad = Pipeline([
        ("a_texto", FunctionTransformer(_a_texto, feature_names_out="one-to-one")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            max_categories=config.TOP_N_CATEGORIAS + 1,
            sparse_output=False,
        )),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", rama_numericas, numericas),
            ("cat", rama_categoricas, categoricas),
            ("alta", rama_alta_cardinalidad, alta),
        ],
        remainder="drop",
    )


def preparar() -> dict:
    """Carga y limpia el CSV, lo parte en train y test y crea el preprocesador.

    Devuelve un dict con X_train, X_test, y_train, y_test y el preprocesador.
    """
    df = limpiar(cargar_crudo())

    # config.DEMO se lee aquí y no con `from .config import DEMO` arriba: main.py lo
    # cambia después de importar este módulo, y una copia del import no vería el cambio.
    if config.DEMO:
        n = min(config.DEMO_FILAS, len(df))
        df = df.sample(n=n, random_state=config.SEMILLA).reset_index(drop=True)
        print(f"      [demo] muestra reducida a {_miles(n)} filas")

    X, y = separar_X_y(df)
    X_train, X_test, y_train, y_test = particionar(X, y)
    print(f"      train {_miles(len(X_train))} / test {_miles(len(X_test))} filas"
          f" | cancelan: {y_train.mean():.2%} en train, {y_test.mean():.2%} en test")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        # Sin ajustar: se ajusta dentro del Pipeline en cada fold.
        "preprocesador": construir_preprocesador(X_train),
    }
