"""Inneres Sizing-Problem: für eine GEGEBENE Topologie die leichtesten
Querschnitte finden, die Spannungs- UND Knick-Nebenbedingungen einhalten.
Dieselbe Vollspannungs-Entwurf-Fixpunktiteration wie in truss-sizing-demo/
truss-shape-demo - hier der innere Baustein für die äußere
Topologieoptimierung (jede geprüfte Topologie braucht ihre eigenen,
optimalen Querschnitte, um fair verglichen zu werden)."""

import numpy as np

from tt_constants import Material
from tt_model import TrussStructure, gesamtmasse, loese_tragwerk


def _erforderliche_flaeche_je_stab(struktur: TrussStructure, material: Material, flaechen: np.ndarray) -> np.ndarray:
    ergebnis = loese_tragwerk(struktur, flaechen, material)
    kraefte = np.abs(ergebnis.stabkraefte)
    laengen = struktur.stablaengen()

    a_spannung = kraefte / material.sigma_zul
    druck = ergebnis.stabkraefte < 0
    a_knick = np.zeros_like(flaechen)
    a_knick[druck] = np.sqrt(4.0 * kraefte[druck] * laengen[druck] ** 2 / (np.pi * material.e_modul))
    return np.maximum(a_spannung, a_knick)


def vollspannungs_entwurf(
    struktur: TrussStructure, material: Material, a_min: float, a_max: float,
    max_iter: int = 150, tol: float = 1e-8,
) -> np.ndarray:
    n = len(struktur.staebe)
    if n == 0:
        return np.zeros(0)
    flaechen = np.full(n, a_max)
    for _ in range(max_iter):
        erforderlich = _erforderliche_flaeche_je_stab(struktur, material, flaechen)
        neu = np.clip(erforderlich, a_min, a_max)
        if np.max(np.abs(neu - flaechen)) / a_max < tol:
            return neu
        flaechen = neu
    return flaechen


def masse_mit_sizing(
    struktur: TrussStructure, material: Material, a_min: float, a_max: float, max_iter: int = 150,
) -> tuple[float, np.ndarray]:
    """Löst das innere Sizing-Problem (mit Knicken) und gibt (Masse, Flächen)
    zurück. `max_iter` wird während der Metaheuristik-Suche bewusst niedrig
    gehalten (siehe `tt_solver.py`) - bei manchen (statisch unbestimmten)
    Zwischentopologien oszilliert die Fixpunktiteration statt zu
    konvergieren und schöpft sonst bei JEDER der hunderten Suchschritte das
    volle Iterationslimit aus (dieselbe Erkenntnis wie in truss-shape-demo)."""
    flaechen = vollspannungs_entwurf(struktur, material, a_min, a_max, max_iter=max_iter)
    return gesamtmasse(struktur, flaechen, material), flaechen
