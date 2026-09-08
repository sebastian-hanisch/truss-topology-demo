"""Drei Lösungsverfahren für dieselbe Topologieoptimierung: aus einer
dichten Ground Structure (Kandidatennetz) die (Teil-)Menge an Stäben
auswählen, die - nach optimaler Bemessung - das leichteste TRAGFÄHIGE und
STABILE Tragwerk ergibt.
"""

import math
import random
import time
from dataclasses import dataclass, field

import numpy as np

from tt_constants import A_MAX, A_MIN, TOPOLOGIE_SCHWELLE, Material
from tt_lp import LPUnzulaessig, loese_ground_structure_lp
from tt_model import GroundStructure, InstabilesTragwerk, loese_tragwerk
from tt_sizing import masse_mit_sizing

STRAF_MASSE = 1e7


@dataclass
class Topologieergebnis:
    methode: str
    maske: np.ndarray  # bool je Kandidat: Stab Teil der Topologie?
    flaechen: np.ndarray  # Flächen NUR der aktiven Stäbe (gleiche Reihenfolge wie maske==True)
    masse: float
    stabil: bool
    rechenzeit: float
    verlauf: list[float] = field(default_factory=list)
    info: dict = field(default_factory=dict)


_SUCH_MAX_ITER = 25  # schnelle (evtl. nicht ganz konvergierte) innere Lösung während der Suche
_FINAL_MAX_ITER = 150  # präzise innere Lösung für die abschließend berichtete Masse


def _bewerte(gs: GroundStructure, material: Material, maske: np.ndarray, max_iter: int = _SUCH_MAX_ITER) -> tuple[float, np.ndarray, bool]:
    struktur = gs.struktur_aus_maske(maske)
    if len(struktur.staebe) == 0:
        return STRAF_MASSE, np.zeros(0), False
    try:
        loese_tragwerk(struktur, np.full(len(struktur.staebe), 1e-3), material)
    except InstabilesTragwerk:
        return STRAF_MASSE, np.zeros(0), False
    masse, flaechen = masse_mit_sizing(struktur, material, A_MIN, A_MAX, max_iter=max_iter)
    return masse, flaechen, True


def _endgueltige_bewertung(gs: GroundStructure, material: Material, maske: np.ndarray) -> tuple[float, np.ndarray, bool]:
    return _bewerte(gs, material, maske, max_iter=_FINAL_MAX_ITER)


def naives_volles_netz(gs: GroundStructure, material: Material) -> Topologieergebnis:
    start = time.perf_counter()
    maske = np.ones(len(gs.kandidaten), dtype=bool)
    masse, flaechen, stabil = _endgueltige_bewertung(gs, material, maske)
    return Topologieergebnis("Volles Ground-Structure-Netz", maske, flaechen, masse, stabil, time.perf_counter() - start)


def _laengen_aller_kandidaten(gs: GroundStructure) -> np.ndarray:
    voll = gs.volle_struktur()
    return voll.stablaengen()


def _repariere_bis_stabil(gs: GroundStructure, material: Material, maske: np.ndarray) -> np.ndarray:
    """Fügt - kürzeste zuerst - inaktive Kandidaten wieder hinzu, bis die
    Struktur kinematisch stabil ist. Die LP löst nur das Gleichgewicht für
    die EINE gegebene Last (statischer Satz) - das reicht i. A. NICHT für
    generelle Steifigkeit/Stabilität, das ist eine bekannte Einschränkung
    der klassischen Ground-Structure-LP (Dorn/Gomory/Greenberg 1964)."""
    maske = maske.copy()
    laengen = _laengen_aller_kandidaten(gs)
    inaktive = [i for i in range(len(gs.kandidaten)) if not maske[i]]
    inaktive.sort(key=lambda i: laengen[i])
    for idx in inaktive:
        struktur = gs.struktur_aus_maske(maske)
        try:
            if len(struktur.staebe) > 0:
                loese_tragwerk(struktur, np.full(len(struktur.staebe), 1e-3), material)
                return maske
        except InstabilesTragwerk:
            pass
        maske[idx] = True
    return maske


def lp_relaxation_mit_reparatur(gs: GroundStructure, material: Material) -> Topologieergebnis:
    start = time.perf_counter()
    try:
        f = loese_ground_structure_lp(gs, material)
    except LPUnzulaessig:
        maske = np.ones(len(gs.kandidaten), dtype=bool)
        masse, flaechen, stabil = _endgueltige_bewertung(gs, material, maske)
        return Topologieergebnis("LP-Relaxation (Michell)", maske, flaechen, masse, stabil, time.perf_counter() - start)

    areas_lp = np.abs(f) / material.sigma_zul
    maske_roh = areas_lp > TOPOLOGIE_SCHWELLE * A_MIN
    n_lp_aktiv = int(maske_roh.sum())
    lp_ideal_masse = float(np.sum(material.dichte * areas_lp * _laengen_aller_kandidaten(gs)))

    maske = _repariere_bis_stabil(gs, material, maske_roh)
    n_hinzugefuegt = int(maske.sum()) - n_lp_aktiv
    masse, flaechen, stabil = _endgueltige_bewertung(gs, material, maske)
    info = {"n_lp_aktiv": n_lp_aktiv, "n_hinzugefuegt": n_hinzugefuegt, "lp_ideal_masse": lp_ideal_masse}
    return Topologieergebnis("LP-Relaxation (Michell)", maske, flaechen, masse, stabil, time.perf_counter() - start, info=info)


def metaheuristik_lokale_suche(
    gs: GroundStructure, material: Material, start_maske: np.ndarray, seed: int, iterationen: int = 1200,
) -> Topologieergebnis:
    """Simulated Annealing über einzelne Stab-Umschaltungen (an/aus),
    startend bei der reparierten LP-Topologie - sucht nach Verbesserungen,
    die der gierigen (kürzeste-zuerst) Reparatur entgangen sind, z. B.
    eine andere Bracing-Kante, die weniger zusätzliche Masse kostet."""
    start = time.perf_counter()
    rng = random.Random(seed)
    maske = start_maske.copy()
    masse, flaechen, _ = _bewerte(gs, material, maske)
    beste_maske, beste_masse, beste_flaechen = maske.copy(), masse, flaechen
    verlauf = [beste_masse]
    n = len(gs.kandidaten)

    for schritt in range(iterationen):
        temperatur = max(1e-6, 1.0 - schritt / iterationen)
        kandidat = maske.copy()
        i = rng.randrange(n)
        kandidat[i] = not kandidat[i]
        neue_masse, neue_flaechen, stabil = _bewerte(gs, material, kandidat)
        if not stabil:
            continue
        delta = neue_masse - masse
        if delta < 0 or rng.random() < math.exp(-delta / (temperatur * max(beste_masse, 1e-9))):
            maske, masse, flaechen = kandidat, neue_masse, neue_flaechen
            if masse < beste_masse:
                beste_maske, beste_masse, beste_flaechen = maske.copy(), masse, flaechen
        verlauf.append(beste_masse)

    beste_masse, beste_flaechen, stabil = _endgueltige_bewertung(gs, material, beste_maske)
    return Topologieergebnis(
        "Metaheuristik (Simulated Annealing)", beste_maske, beste_flaechen, beste_masse, stabil,
        time.perf_counter() - start, verlauf,
    )


def loese_alle(gs: GroundStructure, material: Material, seed: int) -> dict[str, Topologieergebnis]:
    naiv = naives_volles_netz(gs, material)
    lp = lp_relaxation_mit_reparatur(gs, material)
    meta = metaheuristik_lokale_suche(gs, material, lp.maske, seed)
    return {
        "Volles Ground-Structure-Netz": naiv,
        "LP-Relaxation (Michell)": lp,
        "Metaheuristik (Simulated Annealing)": meta,
    }
