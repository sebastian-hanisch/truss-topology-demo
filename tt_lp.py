"""Die klassische Ground-Structure-LP (Dorn, Gomory & Greenberg 1964,
Michell-Fachwerke): für eine reine Spannungs-Nebenbedingung (KEIN Knicken)
lässt sich die gewichtsminimale Auswahl aus einem dichten Kandidatennetz
als LINEARES Programm exakt lösen - eine Seltenheit in diesem Portfolio,
wo "exakt" sonst fast immer "exponentiell" bedeutet (vgl. arc-routing-demo,
wo dasselbe für das Postbotenproblem gilt).

Idee (statischer Satz der Traglasttheorie): JEDE Stabkraftverteilung, die
das Gleichgewicht erfüllt, ist eine zulässige Lösung - Minimierung des
Volumens über alle gleichgewichtstreuen Kraftverteilungen liefert direkt
die leichteste Struktur, die die Last überhaupt tragen kann. Da das
Volumen (|F_i| * L_i / sigma_zul) linear in den (vorzeichenaufgespaltenen)
Stabkräften ist und das Gleichgewicht selbst linear ist, ist das gesamte
Problem ein lineares Programm - OHNE dass vorher über die Topologie
entschieden werden müsste: Stäbe mit optimaler Kraft 0 sind schlicht nicht
Teil der Lösung.
"""

import numpy as np
from scipy.optimize import linprog

from tt_constants import Material
from tt_model import GroundStructure


class LPUnzulaessig(Exception):
    pass


def _gleichgewichtsmatrix(gs: GroundStructure) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Baut B (2*n_frei x n_kandidaten) mit B @ F = P_frei, sowie die
    Stablängen und den Index-zu-DOF-Lookup der freien Freiheitsgrade."""
    n_knoten = len(gs.knoten)
    frei_maske = np.zeros(2 * n_knoten, dtype=bool)
    for k in gs.knoten:
        if not k.fest_x:
            frei_maske[2 * k.id] = True
        if not k.fest_y:
            frei_maske[2 * k.id + 1] = True
    frei_indizes = np.where(frei_maske)[0]
    dof_zu_zeile = {dof: row for row, dof in enumerate(frei_indizes)}

    n_kandidaten = len(gs.kandidaten)
    b = np.zeros((len(frei_indizes), n_kandidaten))
    laengen = np.zeros(n_kandidaten)
    for col, stab in enumerate(gs.kandidaten):
        k1, k2 = gs.knoten[stab.knoten1], gs.knoten[stab.knoten2]
        laenge = float(np.hypot(k2.x - k1.x, k2.y - k1.y))
        laengen[col] = laenge
        c, s = (k2.x - k1.x) / laenge, (k2.y - k1.y) / laenge
        dof_a_x, dof_a_y = 2 * stab.knoten1, 2 * stab.knoten1 + 1
        dof_b_x, dof_b_y = 2 * stab.knoten2, 2 * stab.knoten2 + 1
        if dof_a_x in dof_zu_zeile:
            b[dof_zu_zeile[dof_a_x], col] += c
        if dof_a_y in dof_zu_zeile:
            b[dof_zu_zeile[dof_a_y], col] += s
        if dof_b_x in dof_zu_zeile:
            b[dof_zu_zeile[dof_b_x], col] += -c
        if dof_b_y in dof_zu_zeile:
            b[dof_zu_zeile[dof_b_y], col] += -s

    p = np.zeros(2 * n_knoten)
    for last in gs.lasten:
        p[2 * last.knoten] += last.fx
        p[2 * last.knoten + 1] += last.fy
    p_frei = p[frei_indizes]
    return b, p_frei, laengen


def loese_ground_structure_lp(gs: GroundStructure, material: Material) -> np.ndarray:
    """Löst die Michell-LP und gibt die (kontinuierlichen) Stabkräfte F_i
    zurück (positiv = Zug). |F_i|/sigma_zul ist die dafür nötige Fläche -
    OHNE Knicken, das ist die bewusste Vereinfachung dieser Relaxation."""
    b, p_frei, laengen = _gleichgewichtsmatrix(gs)
    n = len(gs.kandidaten)

    # Variablen: [F_1^+, ..., F_n^+, F_1^-, ..., F_n^-] >= 0, F_i = F_i^+ - F_i^-
    kosten = np.concatenate([laengen, laengen]) / material.sigma_zul
    a_eq = np.hstack([b, -b])
    bounds = [(0, None)] * (2 * n)

    res = linprog(kosten, A_eq=a_eq, b_eq=p_frei, bounds=bounds, method="highs")
    if not res.success:
        raise LPUnzulaessig(
            "Die Ground-Structure-LP ist für diese Konfiguration unzulässig - "
            "das Kandidatennetz kann die Last nicht tragen (zu wenige Kandidaten-Stäbe)."
        )
    f_plus, f_minus = res.x[:n], res.x[n:]
    return f_plus - f_minus
