"""Ein-Klick-Beispielszenarien und Permalink-Logik (dasselbe SETTING_SPECS-
Muster wie in den anderen Demos dieses Workspace, z. B. truss_presets.py)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

from tt_constants import DEFAULT_MATERIAL, MATERIALS
from tt_networks import ENTWURFSGEBIET_NAMEN

MATERIAL_NAMEN = list(MATERIALS.keys())


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _parse_gebiet(v):
    return v if v in ENTWURFSGEBIET_NAMEN else ENTWURFSGEBIET_NAMEN[0]


def _parse_material(v):
    return v if v in MATERIAL_NAMEN else DEFAULT_MATERIAL


SETTING_SPECS = {
    "gebiet_select": SettingSpec("gebiet", _parse_gebiet, ENTWURFSGEBIET_NAMEN[0]),
    "material_select": SettingSpec("material", _parse_material, DEFAULT_MATERIAL),
    "lastfaktor_slider": SettingSpec("last", float, 1.0, 0.3, 2.5),
    "seed_input": SettingSpec("seed", int, 0, 0, 2_000_000_000),
}


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def apply_preset(gebiet, material, lastfaktor, seed):
    st.session_state["gebiet_select"] = gebiet
    st.session_state["material_select"] = material
    st.session_state["lastfaktor_slider"] = lastfaktor
    st.session_state["seed_input"] = seed


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, 2_000_000_000)


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if not isinstance(value, str):
                    if spec.lo is not None:
                        value = max(spec.lo, value)
                    if spec.hi is not None:
                        value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    st.session_state["permalink_loaded"] = True


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def sync_query_params(gebiet, material, lastfaktor, seed):
    try:
        st.query_params["gebiet"] = gebiet
        st.query_params["material"] = material
        st.query_params["last"] = str(lastfaktor)
        st.query_params["seed"] = str(int(seed))
    except Exception:
        pass
