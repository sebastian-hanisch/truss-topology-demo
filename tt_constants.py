"""Physikalische Konstanten und Werkstoffe - dieselbe Modellierung wie in
truss-sizing-demo/truss-shape-demo (Vollkreis-Querschnitt, Euler-Knicken),
hier mit kontinuierlichen Flächen (kein Katalog - das war truss-sizing-demo)."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Material:
    name: str
    e_modul: float  # Elastizitätsmodul [Pa]
    dichte: float  # Dichte [kg/m^3]
    sigma_zul: float  # zulässige Spannung (Zug & Druck, ohne Knicken) [Pa]


MATERIALS: dict[str, Material] = {
    "Baustahl S235": Material("Baustahl S235", 210e9, 7850.0, 160e6),
    "Aluminium EN AW-6060": Material("Aluminium EN AW-6060", 70e9, 2700.0, 110e6),
}
DEFAULT_MATERIAL = "Baustahl S235"

KNICK_BEIWERT_K = 1.0


def flaechenmoment(area: np.ndarray) -> np.ndarray:
    return area ** 2 / (4.0 * np.pi)


A_MIN = 1.0e-4  # 1 cm^2 - Mindestquerschnitt, sobald ein Stab existiert
A_MAX = 2.0e-2  # 200 cm^2

# Ein Stab gilt als "nicht Teil der optimierten Topologie", wenn seine Fläche
# unter diesem Anteil von A_MIN bleibt (LP-Relaxation lässt nicht benötigte
# Stäbe rechnerisch nie exakt bei 0 landen, sondern nur numerisch nahe 0).
TOPOLOGIE_SCHWELLE = 0.05
