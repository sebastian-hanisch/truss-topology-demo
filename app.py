"""
Topologieoptimierung von Stabtragwerken – interaktive Demo
Sebastian Hanisch - Operations Research und Machine Learning

Dritte und letzte Stufe der Strukturoptimierungs-Hierarchie (Sizing -> Form
-> Topologie), Abschluss der mit truss-sizing-demo begonnenen Linie: hier
ist nicht einmal mehr die Topologie fest vorgegeben. Aus einem dichten
Kandidatennetz ("Ground Structure" - jeder Knoten mit jedem nahen Knoten
testweise verbunden) wird ausgewählt, WELCHE Stäbe überhaupt existieren.
Für jede geprüfte Topologie wird intern die vollständige Sizing-
Optimierung gelöst.

Kernthema dieser Demo: Für die reine Spannungs-Nebenbedingung (ohne
Knicken) lässt sich die gewichtsminimale Topologie exakt als LINEARES
PROGRAMM lösen (Dorn, Gomory & Greenberg 1964, "Michell-Fachwerke") - eine
weitere Seltenheit in diesem Portfolio, in der "exakt" nicht "langsam"
bedeutet (vgl. arc-routing-demo). Der Haken: das Gleichgewicht für EINE
gegebene Last zu erfüllen reicht nicht automatisch für eine kinematisch
STABILE Struktur - reine Gleichgewichtslösungen können Mechanismen sein,
die zusätzliche Aussteifung brauchen, sobald Stabilität gefordert wird.

Code-Struktur: Modell, Sizing-Teilproblem, Ground-Structure-LP,
Topologie-Solver, PDF-Export und Visualisierung liegen in den Modulen
tt_*.py neben dieser Datei.
"""

import numpy as np
import pandas as pd
import streamlit as st

from tt_constants import A_MAX, A_MIN, DEFAULT_MATERIAL, MATERIALS
from tt_model import constraint_verletzungen
from tt_networks import ENTWURFSGEBIET_NAMEN, baue_ground_structure_aus_gebiet
from tt_pdf_export import generate_topology_report_pdf
from tt_presets import (
    MATERIAL_NAMEN,
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from tt_sizing import vollspannungs_entwurf
from tt_solver import loese_alle
from tt_visualization import (
    ground_structure_figure,
    konvergenz_figure,
    topologie_figure,
    vergleich_balken_figure,
)

METHODEN_REIHENFOLGE = ["Volles Ground-Structure-Netz", "LP-Relaxation (Michell)", "Metaheuristik (Simulated Annealing)"]
METHODEN_TAB_LABEL = {
    "Volles Ground-Structure-Netz": "🕸️ Volles Netz",
    "LP-Relaxation (Michell)": "📈 LP-Relaxation",
    "Metaheuristik (Simulated Annealing)": "🧬 Metaheuristik",
}


def _auslastung_fuer(gs, material, ergebnis):
    struktur = gs.struktur_aus_maske(ergebnis.maske)
    if len(struktur.staebe) == 0:
        return np.zeros(0)
    spannungsausl, knickausl, _ = constraint_verletzungen(struktur, ergebnis.flaechen, material)
    return np.maximum(spannungsausl, knickausl)


@st.cache_data(show_spinner=False)
def _compute(gebiet_name, material_name, lastfaktor, seed, cache_key):
    gs = baue_ground_structure_aus_gebiet(gebiet_name, lastfaktor)
    material = MATERIALS[material_name]
    ergebnisse = loese_alle(gs, material, seed)
    return gs, material, ergebnisse


st.set_page_config(page_title="Topologieoptimierung von Stabtragwerken – Sebastian Hanisch", layout="wide")

st.title("🕸️ Topologieoptimierung von Stabtragwerken")
st.markdown(
    """
Interaktive Demo zur **Topologieoptimierung von Fachwerken**: Aus einem dichten Kandidatennetz
("**Ground Structure**" - jeder Knoten testweise mit jedem nahen Knoten verbunden) wird ausgewählt,
**welche Stäbe überhaupt existieren**. Für jede geprüfte Topologie wird intern die vollständige
Querschnittsoptimierung gelöst (dasselbe Sizing-Problem wie in truss-sizing-demo) - gesucht ist die
Auswahl, die danach am leichtesten und dabei **kinematisch stabil** ist. Details im Expander "Wie
funktioniert diese Demo?" unten sowie formal hergeleitet im Expander "📐 Mathematische
Formulierung".
"""
)

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_col1, preset_col2, preset_col3 = st.columns(3)
with preset_col1:
    st.button(
        "🏗️ Kragarm (Michell-Benchmark)", use_container_width=True,
        on_click=apply_preset, args=("Kragarm (Michell-Benchmark)", DEFAULT_MATERIAL, 1.0, 0),
        help="Das Standard-Benchmark der Topologieoptimierungs-Literatur - eingespannte Wand, Last an der freien Ecke.",
    )
with preset_col2:
    st.button(
        "🌉 Einfeldträger (Brücke)", use_container_width=True,
        on_click=apply_preset, args=("Einfeldträger (Brücke)", DEFAULT_MATERIAL, 1.0, 0),
        help="Beidseitig gelagert, Last in Feldmitte - die Optimierung darf frei entscheiden, welches Fachwerkmuster entsteht.",
    )
with preset_col3:
    st.button(
        "🔬 Kompaktes Raster (Schnelltest)", use_container_width=True,
        on_click=apply_preset, args=("Kompaktes Raster (Schnelltest)", DEFAULT_MATERIAL, 1.0, 0),
        help="Kleines Kandidatennetz - schnell nachvollziehbar, hier findet die Metaheuristik die größte Verbesserung.",
    )

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    gebiet_name = st.selectbox("Entwurfsgebiet", options=ENTWURFSGEBIET_NAMEN, key="gebiet_select")
    material_name = st.selectbox("Werkstoff", options=MATERIAL_NAMEN, key="material_select")
    lastfaktor = st.slider(
        "Lastfaktor", *bounds("lastfaktor_slider"), step=0.1, key="lastfaktor_slider",
        help="Skaliert die Last - beeinflusst, wie stark sich Knicken gegenüber der LP-Relaxation (nur Spannung) auswirkt.",
    )

    seed_lo, seed_hi = bounds("seed_input")
    seed = st.number_input("Zufalls-Seed (Metaheuristik)", min_value=seed_lo, max_value=seed_hi, step=1, key="seed_input")
    st.button(
        "🎲 Neuen Zufalls-Seed für die Metaheuristik", use_container_width=True, on_click=randomize_seed,
        help="Simulated Annealing ist stochastisch - ein neuer Seed führt zu einer neuen Suchtrajektorie.",
    )

sync_query_params(gebiet_name, material_name, lastfaktor, int(seed))

cache_key = (gebiet_name, material_name, lastfaktor, int(seed))
with st.spinner("Optimiere Topologie (jede geprüfte Auswahl löst intern das Sizing-Problem)..."):
    gs, material, ergebnisse = _compute(gebiet_name, material_name, lastfaktor, int(seed), cache_key)

zulaessige = {k: v for k, v in ergebnisse.items() if v.stabil}
bester_name = min(zulaessige, key=lambda k: zulaessige[k].masse) if zulaessige else min(ergebnisse, key=lambda k: ergebnisse[k].masse)
bestes = ergebnisse[bester_name]
auslastung_best = _auslastung_fuer(gs, material, bestes)

st.markdown(f"## 🎯 Ergebnis: {bester_name}")

m1, m2, m3 = st.columns(3)
m1.metric("Gesamtmasse", f"{bestes.masse:.0f} kg")
m2.metric("Ausgewählte Stäbe", f"{int(bestes.maske.sum())} / {len(gs.kandidaten)}")
m3.metric("Kinematisch stabil", "Ja" if bestes.stabil else "Nein")

st.plotly_chart(
    topologie_figure(gs, bestes.maske, bestes.flaechen, auslastung_best, f"{gs.name} – {bester_name}"),
    use_container_width=True, key="topologie_main",
)

pdf_bytes = generate_topology_report_pdf(gs, material, bestes, auslastung_best)
st.download_button(
    "📄 Topologiebericht als PDF herunterladen", data=pdf_bytes,
    file_name="topologiebericht.pdf", mime="application/pdf", key="primary_pdf_download",
)

st.caption(
    "Ermittelt mit dem besten von drei eigenen Lösungsverfahren für dieses Szenario - jede geprüfte "
    "Topologie löst intern die vollständige Querschnittsoptimierung. Details unten im vollständigen "
    "Methodenvergleich."
)

st.markdown("---")
st.subheader("📐 Warum reicht Gleichgewicht allein nicht für eine stabile Struktur?")
st.markdown(
    """
Die **LP-Relaxation** löst exakt (und sehr schnell) das Gleichgewicht für die EINE gegebene Last -
der statische Satz der Traglasttheorie garantiert, dass jede gleichgewichtstreue Kraftverteilung
eine gültige (untere Schranke der) Lösung ist. Was er NICHT garantiert: dass die resultierende
Struktur auch **kinematisch stabil** ist - ein minimaler Lastpfad kann exakt diese eine Last tragen
und trotzdem ein Mechanismus sein, der unter jeder anderen Last sofort kollabiert.
"""
)

lp_ergebnis = ergebnisse["LP-Relaxation (Michell)"]
info = lp_ergebnis.info
if info:
    c1, c2, c3 = st.columns(3)
    c1.metric("LP-Lösung (nur Gleichgewicht)", f"{info['n_lp_aktiv']} Stäbe")
    c2.metric("Zusätzlich für Stabilität nötig", f"+{info['n_hinzugefuegt']} Stäbe")
    c3.metric("Masse Idealisierung → real (mit Knicken)", f"{info['lp_ideal_masse']:.0f} → {lp_ergebnis.masse:.0f} kg")
    if info["n_hinzugefuegt"] > 0:
        st.warning(
            f"⚠️ Die reine LP-Lösung für **{gs.name}** ist ein Mechanismus - erst **{info['n_hinzugefuegt']} "
            "zusätzliche Aussteifungsstäbe** machen daraus ein tatsächlich stabiles Tragwerk."
        )
    else:
        st.success(
            f"✅ Bei **{gs.name}** ist die reine LP-Lösung bereits kinematisch stabil - hier war "
            "keine zusätzliche Aussteifung nötig."
        )

st.markdown("---")

with st.expander("🔧 Wie wir das erreichen – vollständiger Methodenvergleich", expanded=False):
    st.plotly_chart(ground_structure_figure(gs, f"{gs.name} – Ground Structure ({len(gs.kandidaten)} Kandidaten-Stäbe)"), use_container_width=True, key="ground_structure")
    st.plotly_chart(vergleich_balken_figure(ergebnisse), use_container_width=True, key="vergleich_balken")
    tabelle = pd.DataFrame(
        [
            {
                "Methode": name, "Masse (kg)": ergebnisse[name].masse,
                "Stäbe": int(ergebnisse[name].maske.sum()), "Stabil": "Ja" if ergebnisse[name].stabil else "Nein",
                "Rechenzeit (s)": ergebnisse[name].rechenzeit,
            }
            for name in METHODEN_REIHENFOLGE
        ]
    )
    st.dataframe(tabelle, use_container_width=True, hide_index=True)

    tab_labels = [METHODEN_TAB_LABEL[m] for m in METHODEN_REIHENFOLGE] + ["📊 Konvergenz"]
    tabs = st.tabs(tab_labels)
    beschreibungen = {
        "Volles Ground-Structure-Netz": "Keine Topologieoptimierung - alle Kandidaten-Stäbe bleiben erhalten und werden bemessen. Immer stabil, aber massiv überdimensioniert.",
        "LP-Relaxation (Michell)": "Löst das Gleichgewicht exakt als lineares Programm (nur Spannung, kein Knicken) - danach wird die Stabilität geprüft und bei Bedarf repariert (kürzeste fehlende Stäbe zuerst ergänzt).",
        "Metaheuristik (Simulated Annealing)": "Sucht - startend bei der reparierten LP-Topologie - über einzelne Stab-Umschaltungen nach Verbesserungen, die der gierigen Reparatur entgangen sind.",
    }
    for tab, name in zip(tabs[:-1], METHODEN_REIHENFOLGE):
        with tab:
            ergebnis = ergebnisse[name]
            st.caption(beschreibungen[name])
            ta1, ta2, ta3 = st.columns(3)
            ta1.metric("Masse", f"{ergebnis.masse:.0f} kg")
            ta2.metric("Stäbe", f"{int(ergebnis.maske.sum())} / {len(gs.kandidaten)}")
            ta3.metric("Rechenzeit", f"{ergebnis.rechenzeit:.2f} s")
            auslastung = _auslastung_fuer(gs, material, ergebnis)
            st.plotly_chart(
                topologie_figure(gs, ergebnis.maske, ergebnis.flaechen, auslastung, name),
                use_container_width=True, key=f"topologie_{name}",
            )

    with tabs[-1]:
        meta_verlauf = ergebnisse["Metaheuristik (Simulated Annealing)"].verlauf
        st.caption("Beste bisher gefundene Masse je Suchschritt (Simulated Annealing über Stab-Umschaltungen).")
        st.plotly_chart(konvergenz_figure(meta_verlauf), use_container_width=True, key="konvergenz")

st.markdown("---")

with st.expander("Wie funktioniert diese Demo?"):
    st.markdown(
        r"""
**Die Problemstellung:** Topologieoptimierung ist die dritte und letzte Stufe der klassischen
Dreier-Hierarchie der Strukturoptimierung (**Sizing -> Form -> Topologie**, siehe truss-sizing-demo
und truss-shape-demo für die ersten beiden Stufen). Hier ist nicht einmal mehr die Topologie fest -
gesucht ist, WELCHE Stäbe aus einem dichten Kandidatennetz überhaupt existieren sollen.

**Die Ground-Structure-Method:** Der Standardansatz (seit den 1960ern): verbinde jeden Knoten mit
jedem anderen Knoten innerhalb einer maximalen Reichweite zu einem dichten Kandidatennetz (der
"Ground Structure") und wähle daraus eine (meist deutlich dünnere) Teilmenge aus. Für den
Kragarm-Benchmark z. B. bleiben von 50 Kandidaten-Stäben am Ende nur 5-25 übrig, je nach Methode.

**Der eigentliche Clou - für reine Spannung ist das exakt lösbar:** Ohne Knick-Nebenbedingung lässt
sich die gewichtsminimale Auswahl aus der Ground Structure als **lineares Programm** formulieren
(Dorn, Gomory & Greenberg 1964) - eine Konsequenz des **statischen Satzes der Traglasttheorie**:
jede Stabkraftverteilung, die das Gleichgewicht erfüllt, ist eine gültige Lösung, und die
gewichtsminimale unter allen gleichgewichtstreuen Verteilungen lässt sich direkt berechnen, OHNE
vorher über die Topologie zu entscheiden - Stäbe mit optimaler Kraft 0 sind schlicht nicht Teil der
Lösung. Das ist (wie bei arc-routing-demo) ein seltener Fall in diesem Portfolio, in dem "exakt"
nicht "exponentiell" bedeutet.

**Die Einschränkung - Gleichgewicht ist nicht dasselbe wie Stabilität:** Die LP-Lösung erfüllt das
Gleichgewicht nur für die EINE gegebene Last. Ein minimaler Lastpfad kann diese Last tragen und
trotzdem ein **Mechanismus** sein - kinematisch instabil, sobald die Last auch nur geringfügig
anders angreift. Diese Demo prüft die LP-Lösung deshalb explizit auf Stabilität (Konditionszahl der
Steifigkeitsmatrix) und ergänzt bei Bedarf die günstigsten fehlenden Stäbe, bis die Struktur
tatsächlich stabil ist - eine bekannte, in der Literatur dokumentierte Einschränkung der klassischen
Ground-Structure-LP.

**Drei Lösungsverfahren im Vergleich:**
- **Volles Ground-Structure-Netz**: keine Topologieoptimierung - alle Kandidaten bleiben erhalten
  und werden bemessen. Immer stabil, aber massiv überdimensioniert (typischerweise 2-3x schwerer).
- **LP-Relaxation** (Michell): löst das Gleichgewicht exakt, repariert dann die Stabilität - liefert
  die dünn besetzte, Michell-Fachwerk-artige Grundtopologie.
- **Metaheuristik** (Simulated Annealing): sucht ausgehend von der reparierten LP-Topologie nach
  Verbesserungen über einzelne Stab-Umschaltungen - findet oft eine günstigere Aussteifung als die
  gierige (kürzeste-zuerst) Reparatur.

**Bezug zu den anderen Stufen:** Sizing (truss-sizing-demo) und Form (truss-shape-demo) sind hier
als innerer Baustein enthalten - jede geprüfte Topologie braucht ihre eigene Sizing-Optimierung, um
fair bewertet zu werden. Damit ist die Sizing -> Form -> Topologie-Linie geschlossen.
"""
    )

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
Gegeben eine Ground Structure mit Knoten $j$, Kandidaten-Stäben $i \in \{1, \ldots, n\}$ (Längen
$L_i$, Richtungscosinus $c_i, s_i$), Lagerung und Lasten $\mathbf{P}$ an den freien Freiheitsgraden.

**LP-Relaxation (statischer Satz, ohne Knicken):** Gesucht sind Stabkräfte $F_i$ (positiv = Zug),
die das Gleichgewicht erfüllen und das Gesamtvolumen minimieren:
"""
    )
    st.latex(r"\min_{F_i} \;\; \sum_{i=1}^n \frac{L_i}{\sigma_{\text{zul}}} \, |F_i| \qquad \text{s.t.} \quad \mathbf{B} \, \mathbf{F} = \mathbf{P}")
    st.markdown(
        r"""
mit der Gleichgewichtsmatrix $\mathbf{B}$ (Zeile je freiem Freiheitsgrad, Spalte je Kandidat-Stab,
Einträge $\pm c_i, \pm s_i$). Die Beträge werden durch Aufspalten $F_i = F_i^+ - F_i^-$,
$F_i^+, F_i^- \geq 0$ linearisiert (Standardtrick, dieselbe Idee wie in den anderen LP-Demos dieses
Portfolios) - danach ist $|F_i| = F_i^+ + F_i^-$ und das Problem ein Standard-LP, lösbar mit SciPy
`linprog` (HiGHS). Optimal wird $A_i = |F_i| / \sigma_{\text{zul}}$; Stäbe mit $A_i \approx 0$ sind
nicht Teil der resultierenden Topologie.

**Warum das genügt, um die TOPOLOGIE zu bestimmen:** Die Ground Structure enthält absichtlich mehr
Kandidaten, als für EINE Lastfall-Lösung nötig sind. Das LP-Optimum ist eine **Basislösung** (Ecke
des Zulässigkeitspolyeders) - bei $m$ Gleichgewichtsbedingungen sind höchstens $m$ Variablen ungleich
0. Die "Topologie" ergibt sich also automatisch als Nebenprodukt der LP-Lösung, ohne dass vorher eine
Auswahl getroffen werden müsste.

**Stabilitäts-Reparatur:** Sei $\mathcal{T}_{\text{LP}}$ die Menge der Stäbe mit $A_i > 0$. Ist die
zugehörige Steifigkeitsmatrix $\mathbf{K}(\mathcal{T}_{\text{LP}})$ (reduziert auf die freien
Freiheitsgrade) singulär oder extrem schlecht konditioniert, ist $\mathcal{T}_{\text{LP}}$ ein
Mechanismus. Reparatur: ergänze - kürzeste zuerst - Kandidaten aus $\mathcal{T}_{\text{Ground}}
\setminus \mathcal{T}_{\text{LP}}$, bis $\mathbf{K}$ wieder gut konditioniert ist.

**Metaheuristik:** Simulated Annealing über den Suchraum $\{0,1\}^n$ (Stab aktiv/inaktiv), startend
bei $\mathcal{T}_{\text{LP}}$ nach Reparatur - jede Nachbar-Topologie (ein Stab umgeschaltet) wird
über das vollständige innere Sizing-Problem MIT Knicken bewertet (`tt_sizing.masse_mit_sizing()`),
instabile Nachbarn werden verworfen.

**Bezug zum Code:** `tt_model.py` implementiert die Ground Structure und die Fachwerk-FEM,
`tt_lp.py` die Michell-LP, `tt_sizing.py` das innere Sizing-Problem (identisch zu
truss-sizing-demo), `tt_solver.py` alle drei äußeren Topologie-Verfahren.

**Quelle:** W.S. Dorn, R.E. Gomory, H.J. Greenberg (1964), *Automatic design of optimal structures*,
Journal de Mécanique 3, 25-52.
"""
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
