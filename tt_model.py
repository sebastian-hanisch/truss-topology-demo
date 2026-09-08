"""Stabtragwerk mit VARIABLER Topologie: dieselbe Fachwerk-FEM (direkte
Steifigkeitsmethode) wie in truss-sizing-demo/truss-shape-demo, hier über
einer "Ground Structure" - einem dichten Kandidatennetz, aus dem die
Topologieoptimierung eine (meist deutlich dünnere) Teilmenge auswählt.
"""

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from tt_constants import KNICK_BEIWERT_K, Material, flaechenmoment


@dataclass(frozen=True)
class Knoten:
    id: int
    x: float
    y: float
    fest_x: bool = False
    fest_y: bool = False


@dataclass(frozen=True)
class Stab:
    id: int
    knoten1: int
    knoten2: int


@dataclass(frozen=True)
class Last:
    knoten: int
    fx: float
    fy: float


@dataclass(frozen=True)
class TrussStructure:
    name: str
    knoten: list[Knoten]
    staebe: list[Stab]
    lasten: list[Last] = field(default_factory=list)

    def stablaenge(self, stab: Stab) -> float:
        k1, k2 = self.knoten[stab.knoten1], self.knoten[stab.knoten2]
        return float(np.hypot(k2.x - k1.x, k2.y - k1.y))

    def stablaengen(self) -> np.ndarray:
        return np.array([self.stablaenge(s) for s in self.staebe])

    def freiheitsgrade(self) -> np.ndarray:
        frei = []
        for k in self.knoten:
            if not k.fest_x:
                frei.append(2 * k.id)
            if not k.fest_y:
                frei.append(2 * k.id + 1)
        return np.array(frei, dtype=int)

    def lastvektor(self) -> np.ndarray:
        n = len(self.knoten)
        f = np.zeros(2 * n)
        for last in self.lasten:
            f[2 * last.knoten] += last.fx
            f[2 * last.knoten + 1] += last.fy
        return f


@dataclass
class Tragwerksergebnis:
    stabkraefte: np.ndarray
    verschiebungen: np.ndarray
    max_verschiebung: float


class InstabilesTragwerk(Exception):
    pass


def _element_steifigkeit(e_modul: float, area: float, laenge: float, c: float, s: float) -> np.ndarray:
    k = e_modul * area / laenge
    t = np.array(
        [
            [c * c, c * s, -c * c, -c * s],
            [c * s, s * s, -c * s, -s * s],
            [-c * c, -c * s, c * c, c * s],
            [-c * s, -s * s, c * s, s * s],
        ]
    )
    return k * t


def loese_tragwerk(struktur: TrussStructure, flaechen: np.ndarray, material: Material) -> Tragwerksergebnis:
    n_knoten = len(struktur.knoten)
    n_dof = 2 * n_knoten
    k_global = np.zeros((n_dof, n_dof))
    richtungen = []
    for stab, area in zip(struktur.staebe, flaechen):
        k1, k2 = struktur.knoten[stab.knoten1], struktur.knoten[stab.knoten2]
        laenge = struktur.stablaenge(stab)
        c = (k2.x - k1.x) / laenge
        s = (k2.y - k1.y) / laenge
        richtungen.append((c, s, laenge))
        k_e = _element_steifigkeit(material.e_modul, float(area), laenge, c, s)
        dofs = [2 * stab.knoten1, 2 * stab.knoten1 + 1, 2 * stab.knoten2, 2 * stab.knoten2 + 1]
        for i_local, i_global in enumerate(dofs):
            for j_local, j_global in enumerate(dofs):
                k_global[i_global, j_global] += k_e[i_local, j_local]

    frei = struktur.freiheitsgrade()
    f = struktur.lastvektor()
    k_ff = k_global[np.ix_(frei, frei)]
    f_f = f[frei]

    if k_ff.size == 0 or np.linalg.cond(k_ff) > 1e12:
        raise InstabilesTragwerk("Tragwerk ist kinematisch instabil (Mechanismus) bei dieser Topologie.")

    u = np.zeros(n_dof)
    u[frei] = np.linalg.solve(k_ff, f_f)

    stabkraefte = np.zeros(len(struktur.staebe))
    for idx, (stab, area) in enumerate(zip(struktur.staebe, flaechen)):
        c, s, laenge = richtungen[idx]
        dofs = [2 * stab.knoten1, 2 * stab.knoten1 + 1, 2 * stab.knoten2, 2 * stab.knoten2 + 1]
        u_e = u[dofs]
        dehnung = (-c, -s, c, s) @ u_e / laenge
        stabkraefte[idx] = material.e_modul * float(area) * dehnung

    return Tragwerksergebnis(stabkraefte, u, float(np.max(np.abs(u))))


def knick_lasten(struktur: TrussStructure, flaechen: np.ndarray, material: Material) -> np.ndarray:
    laengen = struktur.stablaengen()
    i_flaeche = flaechenmoment(flaechen)
    return np.pi ** 2 * material.e_modul * i_flaeche / (KNICK_BEIWERT_K * laengen) ** 2


def constraint_verletzungen(struktur: TrussStructure, flaechen: np.ndarray, material: Material):
    ergebnis = loese_tragwerk(struktur, flaechen, material)
    spannungen = ergebnis.stabkraefte / flaechen
    spannungs_auslastung = np.abs(spannungen) / material.sigma_zul

    p_kr = knick_lasten(struktur, flaechen, material)
    druck = ergebnis.stabkraefte < 0
    knick_auslastung = np.zeros_like(flaechen)
    knick_auslastung[druck] = np.abs(ergebnis.stabkraefte[druck]) / p_kr[druck]

    tol = 1e-6
    erfuellt = bool(np.all(spannungs_auslastung <= 1 + tol) and np.all(knick_auslastung <= 1 + tol))
    return spannungs_auslastung, knick_auslastung, erfuellt


def gesamtmasse(struktur: TrussStructure, flaechen: np.ndarray, material: Material) -> float:
    return float(np.sum(material.dichte * flaechen * struktur.stablaengen()))


@dataclass(frozen=True)
class GroundStructure:
    """Dichtes Kandidatennetz: alle Knoten + alle statisch sinnvollen
    Kandidaten-Stäbe (Paare innerhalb einer maximalen Länge). Die
    Topologieoptimierung wählt daraus eine (Teil-)Menge aus."""

    name: str
    knoten: list[Knoten]
    lasten: list[Last]
    kandidaten: list[Stab]  # alle Kandidaten-Stäbe der Ground Structure

    def struktur_aus_maske(self, maske: np.ndarray) -> TrussStructure:
        """Baut die TrussStructure aus den durch `maske` (bool je Kandidat)
        ausgewählten Stäben - NICHT ausgewählte Stäbe existieren schlicht nicht."""
        staebe = [self.kandidaten[i] for i in range(len(self.kandidaten)) if maske[i]]
        return TrussStructure(self.name, self.knoten, staebe, self.lasten)

    def volle_struktur(self) -> TrussStructure:
        return TrussStructure(self.name, self.knoten, list(self.kandidaten), self.lasten)


def baue_ground_structure(
    name: str, knoten: list[Knoten], lasten: list[Last], max_laenge: float,
) -> GroundStructure:
    """Verbindet jedes Knotenpaar, dessen Abstand <= max_laenge ist, als
    Kandidaten-Stab - Standardkonstruktion der Ground-Structure-Method
    (dichtes Netz, aus dem die eigentliche Topologie ausgewählt wird)."""
    kandidaten = []
    sid = 0
    for i, j in combinations(range(len(knoten)), 2):
        k1, k2 = knoten[i], knoten[j]
        # Zwei reine Lagerknoten miteinander zu verbinden bringt statisch
        # nichts (beide Enden ohnehin fixiert) - überspringen, um die
        # Ground Structure nicht unnötig aufzublähen.
        if (k1.fest_x or k1.fest_y) and (k2.fest_x or k2.fest_y):
            continue
        laenge = float(np.hypot(k2.x - k1.x, k2.y - k1.y))
        if laenge <= max_laenge + 1e-9:
            kandidaten.append(Stab(sid, i, j))
            sid += 1
    return GroundStructure(name, knoten, lasten, kandidaten)
