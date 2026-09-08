"""Erzeugt einen Topologieoptimierungs-Bericht als downloadbares PDF
(in-memory). Umlaute sind unproblematisch (Latin-1, von der FPDF-Kernschrift
Helvetica unterstützt) - vermieden werden nur echte Sonderzeichen wie
Halbgeviertstriche (–), die die Kernschrift nicht darstellen kann.
"""

import time

import numpy as np

from tt_constants import Material
from tt_model import GroundStructure
from tt_solver import Topologieergebnis


def generate_topology_report_pdf(
    gs: GroundStructure, material: Material, ergebnis: Topologieergebnis, auslastung: np.ndarray,
) -> bytes:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Topologieoptimierungs-Bericht (Demo)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"Erstellt: {time.strftime('%d.%m.%Y %H:%M')} Uhr", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Zusammenfassung", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Entwurfsgebiet: {gs.name}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, f"Werkstoff: {material.name}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, f"Methode: {ergebnis.methode}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, f"Kandidaten-Staebe: {len(gs.kandidaten)}, davon ausgewaehlt: {int(ergebnis.maske.sum())}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, f"Gesamtmasse: {ergebnis.masse:.1f} kg", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Ausgewaehlte Staebe", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    headers = ["Stab", "Flaeche (cm2)", "Auslastung (%)"]
    widths = [30, 50, 50]
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(235, 235, 235)
    for h, w in zip(headers, widths):
        pdf.cell(w, 7, h, border=1, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.ln(7)
    pdf.set_font("Helvetica", "", 9)
    aktive_indizes = np.where(ergebnis.maske)[0]
    for pos, idx in enumerate(aktive_indizes):
        row = [str(gs.kandidaten[idx].id), f"{ergebnis.flaechen[pos] * 1e4:.2f}", f"{auslastung[pos] * 100:.0f}"]
        for val, w in zip(row, widths):
            pdf.cell(w, 6, val, border=1, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.ln(6)

    return bytes(pdf.output())
