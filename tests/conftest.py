"""Configuración compartida de los tests.

pytest importa este fichero antes que cualquier test, y por eso es el sitio donde
se pone la raíz del proyecto en sys.path: sin esto, `from src import ...` falla
cuando ejecutas pytest desde una carpeta que no es la raíz.

    python -m pytest -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))


@pytest.fixture(scope="session")
def df_falso():
    """Un DataFrame diminuto con las columnas del CSV real que toca limpiar() (8 de 32).

    Los tests de limpieza no leen data/raw: con un caso hecho a mano se sabe de
    antemano qué tiene que caer. Lleva todo lo que la limpieza tiene que cazar —un
    duplicado exacto, un adr negativo, una reserva sin huéspedes y las dos columnas
    de fuga— y se comprueba que cae exactamente eso y nada más. Los tests de partición
    y de preprocesado sí leen el CSV real, a través de preparar(), y tardan menos de
    un segundo.

    Es de sesión: lo comparten todos los tests, así que nadie puede modificarlo.
    test_limpiar_no_modifica_su_entrada comprueba que limpiar() no lo hace.
    """
    import pandas as pd

    filas = [
        # hotel, is_canceled, adr, adults, children, babies, reservation_status, fecha
        ("City Hotel",   0,  95.0, 2, 0, 0, "Check-Out", "2017-07-01"),
        ("City Hotel",   0,  95.0, 2, 0, 0, "Check-Out", "2017-07-01"),  # duplicado exacto
        ("Resort Hotel", 1, -50.0, 1, 0, 0, "Canceled",  "2017-07-02"),  # adr negativo
        ("Resort Hotel", 1,  80.0, 0, 0, 0, "No-Show",   "2017-07-03"),  # 0 huéspedes
        ("City Hotel",   1, 120.0, 2, 1, 0, "Canceled",  "2017-07-04"),  # fila sana
    ]
    return pd.DataFrame(filas, columns=[
        "hotel", "is_canceled", "adr", "adults", "children", "babies",
        "reservation_status", "reservation_status_date",
    ])
