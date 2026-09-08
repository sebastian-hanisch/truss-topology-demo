# 🕸️ Topologieoptimierung von Stabtragwerken

Interaktive Demo zur Topologieoptimierung von Fachwerken: Aus einem dichten Kandidatennetz ("Ground Structure") wird ausgewählt, welche Stäbe überhaupt existieren.

## Worum geht's?

Dritte und letzte Stufe der klassischen Dreier-Hierarchie der Strukturoptimierung (**Sizing → Form → Topologie**), Abschluss der mit [truss-sizing-demo](https://github.com/sebastian-hanisch/truss-sizing-demo) begonnenen und über [truss-shape-demo](https://github.com/sebastian-hanisch/truss-shape-demo) fortgesetzten Linie. Hier ist nicht einmal mehr die Topologie fest vorgegeben: aus einem dichten Kandidatennetz ("Ground Structure" — jeder Knoten testweise mit jedem nahen Knoten verbunden) wird ausgewählt, welche Stäbe überhaupt existieren sollen. Für jede geprüfte Topologie wird intern die vollständige Sizing-Optimierung gelöst.

Kernthema der Demo: Für die reine Spannungs-Nebenbedingung (ohne Knicken) lässt sich die gewichtsminimale Topologie exakt als **lineares Programm** lösen (Dorn, Gomory & Greenberg 1964, "Michell-Fachwerke") — eine weitere Seltenheit in diesem Portfolio, in der "exakt" nicht "langsam" bedeutet (vgl. [arc-routing-demo](https://github.com/sebastian-hanisch/arc-routing-demo)). Der Haken: das Gleichgewicht für eine gegebene Last zu erfüllen reicht nicht automatisch für eine kinematisch stabile Struktur — reine Gleichgewichtslösungen können Mechanismen sein, die erst durch zusätzliche Aussteifung stabil werden.

## Methodik

- Drei Entwurfsgebiete: der klassische Michell-Kragarm-Benchmark (eingespannte Wand, Last an der freien Ecke), ein Einfeldträger/Brücke (die Optimierung entscheidet selbst, welches Fachwerkmuster entsteht), ein kompaktes Testraster
- Ground-Structure-Method: dichtes Kandidatennetz (alle Knotenpaare innerhalb einer maximalen Reichweite), aus dem die eigentliche Topologie ausgewählt wird
- Drei Lösungsverfahren: das volle (unoptimierte) Kandidatennetz als naive Baseline, die exakte Ground-Structure-LP (SciPy `linprog`/HiGHS) mit anschließender Stabilitätsprüfung und -reparatur, eine Simulated-Annealing-Metaheuristik über einzelne Stab-Umschaltungen
- Empirisch verifiziert (u. a. eine unabhängige Knotengleichgewichts-Gegenprobe der LP-Lösung) vor dem App-Bau — inklusive derselben Fixpunkt-Konvergenzbeschleunigung für das innere Sizing-Problem wie in truss-shape-demo
- PDF-Export, Permalink
- Mathematische Herleitung der Ground-Structure-LP und der Stabilitäts-Reparatur im Expander „Mathematische Formulierung“

## Lokal ausführen

```bash
pip install -r requirements-dev.txt
streamlit run app.py
```

Tests: `pytest tests/ -v`

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von [Sebastian Hanisch](https://sebastianhanisch.net) — Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
