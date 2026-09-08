import numpy as np
import pytest

from tt_constants import DEFAULT_MATERIAL, MATERIALS
from tt_model import (
    InstabilesTragwerk,
    Knoten,
    Last,
    Stab,
    TrussStructure,
    baue_ground_structure,
    gesamtmasse,
    loese_tragwerk,
)
from tt_networks import ENTWURFSGEBIET_NAMEN, baue_ground_structure_aus_gebiet

MAT = MATERIALS[DEFAULT_MATERIAL]


def test_dreieck_stabkraefte_stimmen_mit_handrechnung_ueberein():
    knoten = [
        Knoten(0, 0.0, 0.0, fest_x=True, fest_y=True),
        Knoten(1, 4.0, 0.0, fest_y=True),
        Knoten(2, 2.0, 3.0),
    ]
    staebe = [Stab(0, 0, 1), Stab(1, 0, 2), Stab(2, 1, 2)]
    lasten = [Last(2, 0.0, -50_000.0)]
    struktur = TrussStructure("Dreieck", knoten, staebe, lasten)
    erg = loese_tragwerk(struktur, np.full(3, 1e-3), MAT)
    erwartet = np.array([16666.6667, -30046.2606, -30046.2606])
    np.testing.assert_allclose(erg.stabkraefte, erwartet, rtol=1e-4)


def test_instabiles_tragwerk_wird_erkannt():
    knoten = [Knoten(0, 0.0, 0.0, fest_x=True, fest_y=True), Knoten(1, 1.0, 0.0)]
    staebe = [Stab(0, 0, 1)]
    lasten = [Last(1, 0.0, -1000.0)]
    struktur = TrussStructure("Instabil", knoten, staebe, lasten)
    with pytest.raises(InstabilesTragwerk):
        loese_tragwerk(struktur, np.full(1, 1e-3), MAT)


def test_ground_structure_ueberspringt_lager_lager_verbindungen():
    knoten = [
        Knoten(0, 0.0, 0.0, fest_x=True, fest_y=True),
        Knoten(1, 0.0, 3.0, fest_x=True, fest_y=True),
        Knoten(2, 3.0, 0.0),
    ]
    gs = baue_ground_structure("Test", knoten, [Last(2, 0.0, -1000.0)], max_laenge=10.0)
    paare = {(k.knoten1, k.knoten2) for k in gs.kandidaten}
    assert (0, 1) not in paare and (1, 0) not in paare


def test_ground_structure_respektiert_max_laenge():
    knoten = [Knoten(0, 0.0, 0.0), Knoten(1, 1.0, 0.0), Knoten(2, 10.0, 0.0)]
    gs = baue_ground_structure("Test", knoten, [], max_laenge=2.0)
    paare = {(k.knoten1, k.knoten2) for k in gs.kandidaten}
    assert (0, 1) in paare
    assert (0, 2) not in paare and (1, 2) not in paare


@pytest.mark.parametrize("name", ENTWURFSGEBIET_NAMEN)
def test_volle_struktur_ist_stabil(name):
    gs = baue_ground_structure_aus_gebiet(name)
    struktur = gs.volle_struktur()
    erg = loese_tragwerk(struktur, np.full(len(struktur.staebe), 5e-3), MAT)
    assert np.isfinite(erg.max_verschiebung)


@pytest.mark.parametrize("name", ENTWURFSGEBIET_NAMEN)
def test_lastfaktor_skaliert_lasten_linear(name):
    gs1 = baue_ground_structure_aus_gebiet(name, lastfaktor=1.0)
    gs2 = baue_ground_structure_aus_gebiet(name, lastfaktor=2.0)
    for l1, l2 in zip(gs1.lasten, gs2.lasten):
        assert l2.fx == pytest.approx(2 * l1.fx)
        assert l2.fy == pytest.approx(2 * l1.fy)


def test_masse_ist_positiv_und_skaliert_mit_flaeche():
    gs = baue_ground_structure_aus_gebiet("Kompaktes Raster (Schnelltest)")
    struktur = gs.volle_struktur()
    m1 = gesamtmasse(struktur, np.full(len(struktur.staebe), 1e-3), MAT)
    m2 = gesamtmasse(struktur, np.full(len(struktur.staebe), 2e-3), MAT)
    assert m1 > 0
    assert m2 == pytest.approx(2 * m1)
