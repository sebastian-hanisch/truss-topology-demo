import numpy as np
import pytest

from tt_constants import DEFAULT_MATERIAL, MATERIALS
from tt_networks import ENTWURFSGEBIET_NAMEN, baue_ground_structure_aus_gebiet
from tt_lp import loese_ground_structure_lp

MAT = MATERIALS[DEFAULT_MATERIAL]


@pytest.mark.parametrize("name", ENTWURFSGEBIET_NAMEN)
def test_lp_kraefte_erfuellen_knoten_gleichgewicht(name):
    """Unabhängige Gegenprobe (ohne die interne _gleichgewichtsmatrix()
    wiederzuverwenden): rekonstruiert die Knotenkräfte direkt aus den von
    der LP zurückgegebenen Stabkräften und vergleicht sie an jedem freien
    Freiheitsgrad mit der aufgebrachten äußeren Last."""
    gs = baue_ground_structure_aus_gebiet(name)
    f = loese_ground_structure_lp(gs, MAT)

    kraft_am_knoten = np.zeros((len(gs.knoten), 2))
    for stab, kraft in zip(gs.kandidaten, f):
        k1, k2 = gs.knoten[stab.knoten1], gs.knoten[stab.knoten2]
        laenge = np.hypot(k2.x - k1.x, k2.y - k1.y)
        c, s = (k2.x - k1.x) / laenge, (k2.y - k1.y) / laenge
        kraft_am_knoten[stab.knoten1] += kraft * np.array([c, s])
        kraft_am_knoten[stab.knoten2] += kraft * np.array([-c, -s])

    aeussere_last = np.zeros((len(gs.knoten), 2))
    for last in gs.lasten:
        aeussere_last[last.knoten] += [last.fx, last.fy]

    for k in gs.knoten:
        if not k.fest_x:
            assert kraft_am_knoten[k.id, 0] == pytest.approx(aeussere_last[k.id, 0], abs=1e-3)
        if not k.fest_y:
            assert kraft_am_knoten[k.id, 1] == pytest.approx(aeussere_last[k.id, 1], abs=1e-3)


@pytest.mark.parametrize("name", ENTWURFSGEBIET_NAMEN)
def test_lp_loesung_ist_duenn_besetzt(name):
    """Kernbehauptung der Demo: die LP-Lösung nutzt deutlich weniger Stäbe
    als die Ground Structure Kandidaten anbietet (Michell-Sparsity)."""
    gs = baue_ground_structure_aus_gebiet(name)
    f = loese_ground_structure_lp(gs, MAT)
    n_aktiv = int(np.sum(np.abs(f) > 1e-3))
    assert n_aktiv < len(gs.kandidaten) * 0.6
