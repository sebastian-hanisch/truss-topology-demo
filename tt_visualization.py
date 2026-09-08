"""Plotly-Visualisierungen: Ground-Structure- und Topologieplot,
Konvergenz- und Methodenvergleich."""

import numpy as np
import plotly.graph_objects as go

from tt_model import GroundStructure, TrussStructure

_FARBSKALA = [[0.0, "#2ca02c"], [0.6, "#f2c744"], [1.0, "#d62728"]]


def _interpoliere_farbe(u: float) -> str:
    u = max(0.0, min(1.0, u))
    for (p0, c0), (p1, c1) in zip(_FARBSKALA, _FARBSKALA[1:]):
        if p0 <= u <= p1:
            t = (u - p0) / (p1 - p0) if p1 > p0 else 0.0
            rgb0 = tuple(int(c0[i : i + 2], 16) for i in (1, 3, 5))
            rgb1 = tuple(int(c1[i : i + 2], 16) for i in (1, 3, 5))
            rgb = tuple(round(a + (b - a) * t) for a, b in zip(rgb0, rgb1))
            return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
    return _FARBSKALA[-1][1]


def ground_structure_figure(gs: GroundStructure, titel: str) -> go.Figure:
    fig = go.Figure()
    for stab in gs.kandidaten:
        k1, k2 = gs.knoten[stab.knoten1], gs.knoten[stab.knoten2]
        fig.add_trace(
            go.Scatter(
                x=[k1.x, k2.x], y=[k1.y, k2.y], mode="lines",
                line=dict(width=1.0, color="#cccccc"), hoverinfo="skip", showlegend=False,
            )
        )
    _knoten_marker(fig, gs.knoten)
    _lasten_annotationen(fig, gs.knoten, gs.lasten)
    fig.update_layout(
        title=titel, xaxis=dict(scaleanchor="y", scaleratio=1, showgrid=False, zeroline=False, title="m"),
        yaxis=dict(showgrid=False, zeroline=False, title="m"), height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def topologie_figure(
    gs: GroundStructure, maske: np.ndarray, flaechen: np.ndarray, auslastung: np.ndarray, titel: str,
) -> go.Figure:
    fig = go.Figure()

    for i, stab in enumerate(gs.kandidaten):
        if maske[i]:
            continue
        k1, k2 = gs.knoten[stab.knoten1], gs.knoten[stab.knoten2]
        fig.add_trace(
            go.Scatter(
                x=[k1.x, k2.x], y=[k1.y, k2.y], mode="lines",
                line=dict(width=1.0, color="#dddddd", dash="dot"), hoverinfo="skip", showlegend=False,
            )
        )

    aktive_indizes = np.where(maske)[0]
    if len(flaechen) > 0:
        a_min, a_max = float(flaechen.min()), float(flaechen.max())
        spanne = max(a_max - a_min, 1e-12)
        for pos, i in enumerate(aktive_indizes):
            stab = gs.kandidaten[i]
            k1, k2 = gs.knoten[stab.knoten1], gs.knoten[stab.knoten2]
            breite = 2.5 + 8.0 * (flaechen[pos] - a_min) / spanne
            farbe = _interpoliere_farbe(min(auslastung[pos], 1.0))
            fig.add_trace(
                go.Scatter(
                    x=[k1.x, k2.x], y=[k1.y, k2.y], mode="lines",
                    line=dict(width=breite, color=farbe), hoverinfo="text",
                    text=f"Stab {stab.id}: A={flaechen[pos] * 1e4:.1f} cm², Auslastung {auslastung[pos] * 100:.0f}%",
                    showlegend=False,
                )
            )

    _knoten_marker(fig, gs.knoten)
    _lasten_annotationen(fig, gs.knoten, gs.lasten)
    fig.update_layout(
        title=titel, xaxis=dict(scaleanchor="y", scaleratio=1, showgrid=False, zeroline=False, title="m"),
        yaxis=dict(showgrid=False, zeroline=False, title="m"), height=420,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _knoten_marker(fig: go.Figure, knoten) -> None:
    fest_x = [k.x for k in knoten if k.fest_x or k.fest_y]
    fest_y = [k.y for k in knoten if k.fest_x or k.fest_y]
    fig.add_trace(go.Scatter(x=fest_x, y=fest_y, mode="markers", marker=dict(symbol="triangle-up", size=13, color="#555555"), hoverinfo="skip", showlegend=False))
    frei_x = [k.x for k in knoten if not (k.fest_x or k.fest_y)]
    frei_y = [k.y for k in knoten if not (k.fest_x or k.fest_y)]
    fig.add_trace(go.Scatter(x=frei_x, y=frei_y, mode="markers", marker=dict(size=7, color="#333333"), hoverinfo="skip", showlegend=False))


def _lasten_annotationen(fig: go.Figure, knoten, lasten) -> None:
    xs = [k.x for k in knoten]
    last_skala = 0.15 * (max(xs) - min(xs) + 1e-9)
    max_last = max((abs(l.fx) + abs(l.fy) for l in lasten), default=1.0)
    for last in lasten:
        k = knoten[last.knoten]
        betrag = np.hypot(last.fx, last.fy)
        if betrag == 0:
            continue
        dx, dy = last.fx / betrag * last_skala, last.fy / betrag * last_skala
        fig.add_annotation(
            x=k.x, y=k.y, ax=k.x - dx, ay=k.y - dy, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2 + 2 * betrag / max_last,
            arrowcolor="#1f77b4",
        )


def konvergenz_figure(verlauf: list[float]) -> go.Figure:
    fig = go.Figure(go.Scatter(x=list(range(1, len(verlauf) + 1)), y=verlauf, mode="lines"))
    fig.update_layout(
        title="Konvergenz der Metaheuristik: beste Masse je Schritt", xaxis_title="Schritt",
        yaxis_title="Masse [kg]", height=340, margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def vergleich_balken_figure(ergebnisse: dict) -> go.Figure:
    labels = list(ergebnisse.keys())
    massen = [ergebnisse[l].masse for l in labels]
    fig = go.Figure(go.Bar(x=labels, y=massen, marker_color="#1f77b4", text=[f"{m:.0f} kg" for m in massen], textposition="outside"))
    fig.update_layout(title="Gesamtmasse je Methode", yaxis_title="Masse [kg]", height=360, margin=dict(l=20, r=20, t=50, b=20))
    return fig
