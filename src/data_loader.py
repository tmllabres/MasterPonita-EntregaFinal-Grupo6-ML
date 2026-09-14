"""Carga y transformación de datos.

Este módulo hace tres cosas y ninguna más:
  1. leer el CSV crudo,
  2. limpiar lo que BORRA FILAS (duplicados, imposibles, fugas) — fuera del Pipeline,
  3. partir en train/test y construir el ColumnTransformer SIN ajustar.

Por qué la limpieza vive aquí y no en el Pipeline: un transformador de scikit-learn
devuelve tantas filas como recibe. Si tirara filas, la `y` se quedaría descolocada
respecto a la `X` y nadie te avisaría. Todo lo que borra filas va antes de partir;
todo lo que APRENDE un número de los datos (medianas, categorías, medias del escalado)
va dentro del ColumnTransformer, para que lo aprenda solo con el train.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config

# El texto "NULL" que traen company y agent no es un nulo para pandas: lo lee como
# una cadena más y acabaría siendo una categoría del one-hot con 112.593 reservas.
NA_VALUES = ["NULL", "null", "", " "]

# Las tres columnas que definen "reserva sin huéspedes". Se suman las tres: una
# reserva de 0 adultos pero 2 niños es rara, pero no es imposible.
HUESPEDES = ["adults", "children", "babies"]


def _miles(n: int) -> str:
    """119390 -> '119.390'. Solo para que los prints cuadren con el README."""
    return f"{n:,}".replace(",", ".")


def cargar_crudo(ruta=None) -> pd.DataFrame:
    """Lee el CSV tal cual viene, sin tocar nada.

    Ojo con los nulos: `company` y `agent` traen el texto "NULL", que pandas no
    reconoce como ausente salvo que se lo digas con na_values.
    """
    return pd.read_csv(config.DATA_RAW if ruta is None else ruta, na_values=NA_VALUES)


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Quita fugas, duplicados e imposibles. Devuelve MENOS filas de las que recibe.

    Orden que importa:
      1. eliminar config.FUGAS  (determinan el objetivo al 100 %)
      2. eliminar duplicados exactos
      3. eliminar imposibles: adr negativo, reservas con 0 huéspedes
    Deja registrado cuántas filas caen en cada paso: eso va al README.
    """
    filas_iniciales = len(df)

    # 1. Fugas. Primero, y no por capricho de orden: reservation_status_date es una
    #    fecha casi distinta para cada reserva, así que mientras esté ahí apenas hay
    #    dos filas idénticas y el paso 2 no encontraría nada que borrar.
    fugas = [c for c in config.FUGAS if c in df.columns]
    df = df.drop(columns=fugas)
    print(f"      [limpieza] fugas: -{len(fugas)} columnas ({', '.join(fugas)})")

    # 2. Duplicados exactos. El porqué de borrarlos está razonado en config.
    if config.ELIMINAR_DUPLICADOS:
        antes = len(df)
        df = df.drop_duplicates()
        print(f"      [limpieza] duplicados exactos: -{_miles(antes - len(df))} filas")

    # 3. Imposibles. No son valores raros que haya que discutir: son reservas que no
    #    pueden existir. Un adr negativo no es un precio, y una reserva sin ninguna
    #    persona no es una reserva; ni la una ni la otra se pueden cancelar.
    if "adr" in df.columns:
        adr_negativo = df["adr"] < 0
    else:
        adr_negativo = pd.Series(False, index=df.index)

    presentes = [c for c in HUESPEDES if c in df.columns]
    if presentes:
        sin_huespedes = df[presentes].fillna(0).sum(axis=1) == 0
    else:
        sin_huespedes = pd.Series(False, index=df.index)

    # El conteo de "0 huéspedes" descuenta las que ya caían por adr, o los cuatro
    # números que se imprimen no sumarían el total.
    print(f"      [limpieza] adr negativo: -{_miles(int(adr_negativo.sum()))} filas")
    print("      [limpieza] 0 huéspedes: "
          f"-{_miles(int((sin_huespedes & ~adr_negativo).sum()))} filas")
    df = df[~(adr_negativo | sin_huespedes)]

    print(f"      [limpieza] total: {_miles(filas_iniciales)} -> {_miles(len(df))} filas")
    return df.reset_index(drop=True)


def separar_X_y(df: pd.DataFrame):
    """Devuelve (X, y). X son las 29 columnas predictoras; y es is_canceled.

    La cuenta: 32 columnas del CSV − is_canceled − las 2 de fuga = 29.
    """
    # Las fugas ya no están si el df viene de limpiar(), pero esto no es redundancia
    # inútil: separar_X_y también se usa sobre trozos del CSV crudo (la demo de
    # predictor), y ahí dejarlas dentro de X sería regalarle la respuesta al modelo.
    fuera = [c for c in [config.OBJETIVO, *config.FUGAS] if c in df.columns]
    return df.drop(columns=fuera), df[config.OBJETIVO].astype(int)


def particionar(X, y):
    """train_test_split con test_size, semilla y stratify de config.

    Se llama UNA vez y desde aquí. Si cada módulo partiera por su cuenta, cada
    modelo se compararía contra un reparto distinto y la tabla no valdría nada.
    """
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.SEMILLA,
        stratify=y if config.ESTRATIFICAR else None,
    )


def _a_texto(X: pd.DataFrame) -> pd.DataFrame:
    """Pasa a texto las columnas de alta cardinalidad, con "desconocido" por nulo.

    Hace falta porque las tres no son del mismo tipo: country es texto, pero agent y
    company son float64 (IDs numéricos con nulos). Sin esto, el imputador de constante
    revienta al meter la cadena "desconocido" en una columna numérica.

    Y el nulo se convierte en categoría a propósito, no por comodidad: que no haya
    agente significa que la reserva es directa, y que no haya empresa significa que no
    es un viaje corporativo. Imputar ahí la moda sería inventarse un intermediario.

    Va como función del módulo y no como lambda porque joblib guarda el Pipeline
    entero, y una lambda no se puede serializar.
    """
    return X.astype(object).where(X.notna(), "desconocido").astype(str)


def construir_preprocesador(X_train) -> ColumnTransformer:
    """Devuelve el ColumnTransformer SIN ajustar.

    Sin ajustar a propósito: lo ajusta el Pipeline dentro de cada fit, solo con
    el train de ese fold. Eso es lo que impide la fuga de preprocesado.

    Ramas:
      - numéricas    -> imputar (mediana) + escalar
      - categóricas  -> imputar (constante) + one-hot con handle_unknown="ignore"
      - alta cardinalidad (country, agent, company) -> top-N + un cajón para el resto
    """
    # El reparto se hace en este orden a propósito: alta cardinalidad escoge primero,
    # porque agent y company son float64 y "numéricas" se las llevaría a escalar como
    # si el agente 240 fuera el doble del 120.
    alta = [c for c in config.ALTA_CARDINALIDAD if c in X_train.columns]
    numericas = [c for c in X_train.select_dtypes(include="number").columns if c not in alta]
    categoricas = [c for c in X_train.select_dtypes(include=["object", "category", "bool"]).columns
                   if c not in alta]

    # Mediana y no media: lead_time y adr tienen la cola larga a la derecha.
    rama_numericas = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ])

    # handle_unknown="ignore": si en un fold de validación aparece una categoría que el
    # train de ese fold no vio, sale una fila de ceros en vez de una excepción a mitad
    # de la validación cruzada.
    rama_categoricas = Pipeline([
        ("imputar", SimpleImputer(strategy="constant", fill_value="desconocido")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # max_categories = TOP_N + 1 porque el cajón del resto ocupa una de las plazas: con
    # 10 pelados se quedarían 9 países y el cajón. Las categorías raras y las que no se
    # vieron nunca ("infrequent_if_exist") van a parar a esa misma columna.
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
    """Punto de entrada del módulo: del CSV a todo lo que necesita el resto.

    Devuelve un dict con X_train, X_test, y_train, y_test y el preprocesador.
    """
    df = limpiar(cargar_crudo())

    # config.DEMO se lee AQUÍ, en tiempo de ejecución, y no con un `from .config import
    # DEMO` arriba: main.py lo pone a True DESPUÉS de importar este módulo, así que una
    # copia hecha en el import valdría False para siempre y --demo no haría nada.
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
        # Sin ajustar: lo ajusta cada Pipeline en su fold. Ver el docstring de arriba.
        "preprocesador": construir_preprocesador(X_train),
    }
