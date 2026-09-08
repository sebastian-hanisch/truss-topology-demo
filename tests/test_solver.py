import numpy as np
import pytest

from tt_constants import DEFAULT_MATERIAL, MATERIALS
from tt_networks import ENTWURFSGEBIET_NAMEN, baue_ground_structure_aus_gebiet
from tt_solver import loese_alle

MAT = MATERIALS[DEFAULT_MATERIAL]


@pytest.mark.parametrize("name", ENTWURFSGEBIET_NAMEN)
def test_alle_methoden_liefern_stabile_topologien(name):
    gs = baue_ground_structure_aus_gebiet(name)
    for methode, ergebnis in loese_alle(gs, MAT, seed=0).items():
        assert ergebnis.stabil, f"{name}/{methode} ist nicht stabil"


@pytest.mark.parametrize("name", ENTWURFSGEBIET_NAMEN)
def test_naives_netz_nutzt_alle_kandidaten(name):
    gs = baue_ground_structure_aus_gebiet(name)
    ergebnisse = loese_alle(gs, MAT, seed=0)
    naiv = ergebnisse["Volles Ground-Structure-Netz"]
    assert int(naiv.maske.sum()) == len(gs.kandidaten)


@pytest.mark.parametrize("name", ENTWURFSGEBIET_NAMEN)
def test_lp_topologie_ist_duenner_als_volles_netz(name):
    gs = baue_ground_structure_aus_gebiet(name)
    ergebnisse = loese_alle(gs, MAT, seed=0)
    lp = ergebnisse["LP-Relaxation (Michell)"]
    assert int(lp.maske.sum()) < len(gs.kandidaten)


@pytest.mark.parametrize("name", ENTWURFSGEBIET_NAMEN)
def test_lp_und_metaheuristik_sind_leichter_als_naives_netz(name):
    gs = baue_ground_structure_aus_gebiet(name)
    ergebnisse = loese_alle(gs, MAT, seed=0)
    naiv = ergebnisse["Volles Ground-Structure-Netz"]
    for methode in ["LP-Relaxation (Michell)", "Metaheuristik (Simulated Annealing)"]:
        assert ergebnisse[methode].masse < naiv.masse


@pytest.mark.parametrize("name", ENTWURFSGEBIET_NAMEN)
def test_metaheuristik_ist_nie_schlechter_als_ihr_lp_startpunkt(name):
    gs = baue_ground_structure_aus_gebiet(name)
    ergebnisse = loese_alle(gs, MAT, seed=0)
    lp = ergebnisse["LP-Relaxation (Michell)"]
    meta = ergebnisse["Metaheuristik (Simulated Annealing)"]
    assert meta.masse <= lp.masse + 1e-6


def test_lp_reparatur_info_ist_konsistent():
    gs = baue_ground_structure_aus_gebiet("Kragarm (Michell-Benchmark)")
    ergebnisse = loese_alle(gs, MAT, seed=0)
    lp = ergebnisse["LP-Relaxation (Michell)"]
    assert lp.info["n_lp_aktiv"] + lp.info["n_hinzugefuegt"] == int(lp.maske.sum())
    assert lp.info["n_hinzugefuegt"] >= 0
