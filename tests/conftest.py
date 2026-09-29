"""Configuración compartida de los tests.

pytest importa este fichero antes que los tests, así que aquí pongo la raíz del
proyecto en sys.path: sin esto, `from src import ...` falla si pytest se lanza desde
otra carpeta.

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
    """DataFrame pequeño con las 10 columnas del CSV (de 32) que usa limpiar().

    Lo hice a mano para saber de antemano qué tiene que quitar la limpieza: un
    duplicado exacto, un adr negativo, una reserva sin huéspedes y las cuatro columnas
    de fuga. Los tests de partición y preprocesado usan el CSV real con preparar().

    Es de sesión y lo comparten todos los tests, así que ninguno debe modificarlo;
    test_limpiar_no_modifica_su_entrada comprueba que limpiar() no cambia lo que recibe.
    """
    import pandas as pd

    filas = [
        # hotel, is_canceled, adr, adults, children, babies, reservation_status, fecha,
        # plazas de parking, habitación asignada
        ("City Hotel",   0,  95.0, 2, 0, 0, "Check-Out", "2017-07-01", 1, "A"),
        ("City Hotel",   0,  95.0, 2, 0, 0, "Check-Out", "2017-07-01", 1, "A"),  # duplicado exacto
        ("Resort Hotel", 1, -50.0, 1, 0, 0, "Canceled",  "2017-07-02", 0, "A"),  # adr negativo
        ("Resort Hotel", 1,  80.0, 0, 0, 0, "No-Show",   "2017-07-03", 0, "D"),  # 0 huéspedes
        ("City Hotel",   1, 120.0, 2, 1, 0, "Canceled",  "2017-07-04", 0, "A"),  # fila sana
    ]
    return pd.DataFrame(filas, columns=[
        "hotel", "is_canceled", "adr", "adults", "children", "babies",
        "reservation_status", "reservation_status_date",
        "required_car_parking_spaces", "assigned_room_type",
    ])
