"""Entwurfsgebiete (Design-Domains) für die Topologieoptimierung: ein
Knotenraster mit Lagerung und Last, aus dem eine Ground Structure (dichtes
Kandidatennetz) aufgebaut wird."""

from dataclasses import dataclass

from tt_model import GroundStructure, Knoten, Last, baue_ground_structure


@dataclass(frozen=True)
class Entwurfsgebiet:
    name: str
    knoten: list[Knoten]
    lasten: list[Last]
    max_laenge_faktor: float  # Vielfaches des Rasterabstands als Kandidaten-Reichweite


def _kragarm_michell(spacing: float = 3.0) -> Entwurfsgebiet:
    """Der klassische Michell-Kragarm: linke Kante fest eingespannt, Last an
    der unteren rechten Ecke - das Standard-Benchmark der
    Topologieoptimierungs-Literatur."""
    n_spalten, n_zeilen = 4, 3
    knoten = []
    for col in range(n_spalten):
        for row in range(n_zeilen):
            kid = col * n_zeilen + row
            fest = col == 0
            knoten.append(Knoten(kid, col * spacing, row * spacing, fest_x=fest, fest_y=fest))
    last_knoten = (n_spalten - 1) * n_zeilen + 0  # unten rechts
    lasten = [Last(last_knoten, 0.0, -100_000.0)]
    return Entwurfsgebiet("Kragarm (Michell-Benchmark)", knoten, lasten, max_laenge_faktor=2.5)


def _einfeldtraeger(spacing: float = 3.0) -> Entwurfsgebiet:
    """Einfeldträger (Brücke): unten links/rechts gelagert, Last in
    Feldmitte unten - die Topologieoptimierung darf hier frei entscheiden,
    ob ein Fachwerk à la Pratt/Warren oder etwas anderes entsteht."""
    n_spalten, n_zeilen = 5, 3
    knoten = []
    for col in range(n_spalten):
        for row in range(n_zeilen):
            kid = col * n_zeilen + row
            fest_x = col == 0 and row == 0
            fest_y = row == 0 and col in (0, n_spalten - 1)
            knoten.append(Knoten(kid, col * spacing, row * spacing, fest_x=fest_x, fest_y=fest_y))
    mitte_unten = (n_spalten // 2) * n_zeilen + 0
    lasten = [Last(mitte_unten, 0.0, -150_000.0)]
    return Entwurfsgebiet("Einfeldträger (Brücke)", knoten, lasten, max_laenge_faktor=2.5)


def _kompakt() -> Entwurfsgebiet:
    """Minimales 2x3-Raster - schnell nachvollziehbar, für Handrechnung/Tests."""
    spacing = 3.0
    n_spalten, n_zeilen = 3, 2
    knoten = []
    for col in range(n_spalten):
        for row in range(n_zeilen):
            kid = col * n_zeilen + row
            fest = col == 0
            knoten.append(Knoten(kid, col * spacing, row * spacing, fest_x=fest, fest_y=fest))
    last_knoten = (n_spalten - 1) * n_zeilen + 0
    lasten = [Last(last_knoten, 0.0, -60_000.0)]
    return Entwurfsgebiet("Kompaktes Raster (Schnelltest)", knoten, lasten, max_laenge_faktor=2.5)


ENTWURFSGEBIETE: dict[str, Entwurfsgebiet] = {
    e.name: e for e in [_kragarm_michell(), _einfeldtraeger(), _kompakt()]
}
ENTWURFSGEBIET_NAMEN = list(ENTWURFSGEBIETE.keys())


def baue_ground_structure_aus_gebiet(name: str, lastfaktor: float = 1.0) -> GroundStructure:
    gebiet = ENTWURFSGEBIETE[name]
    spacing = gebiet.knoten[1].y - gebiet.knoten[0].y if len(gebiet.knoten) > 1 else 3.0
    # Rasterabstand robust aus den tatsächlichen Koordinaten bestimmen (kleinster
    # positiver Achsenabstand), statt ihn erneut hart zu codieren.
    xs = sorted({k.x for k in gebiet.knoten})
    dx = min(b - a for a, b in zip(xs, xs[1:])) if len(xs) > 1 else 3.0
    max_laenge = gebiet.max_laenge_faktor * dx
    lasten = [Last(l.knoten, l.fx * lastfaktor, l.fy * lastfaktor) for l in gebiet.lasten]
    return baue_ground_structure(gebiet.name, gebiet.knoten, lasten, max_laenge)
