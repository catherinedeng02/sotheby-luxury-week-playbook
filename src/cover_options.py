"""Generate refined cover options (C2 variant) in Morandi color palette.

After narrowing down to C2 (Swiss editorial, flush-left layout), this
script renders three color-palette variants for final selection:
  C2-M1 warm linen
  C2-M2 soft greige
  C2-M3 cool ivory

Each adds the subtitle and time period that the original C2 lacked,
removes the redundant "independent market study" line, and removes the
period after "Playbook".

Run: python3 src/cover_options.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import date

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas

from config import OUTPUT_DIR


# ---------------------------------------------------------------------------
# Static text
# ---------------------------------------------------------------------------

AUTHOR_NAME = "Zhuoyao Deng"
TAGLINE = "Market Intelligence Report"
DATE_LABEL = date.today().strftime("%B %Y")  # "June 2026"

TITLE_LINES = ["The Sotheby's", "Luxury Week", "Playbook"]
SUBTITLE_LINES = [
    "Cross-category performance, premium drivers, and the HK\u2013NY buyer corridor",
    "2021\u20132026",
]


# ---------------------------------------------------------------------------
# Three Morandi palettes
# ---------------------------------------------------------------------------

PALETTES = {
    "M1_warm_linen": {
        "bg":       HexColor("#F2EBDD"),   # warm cream, slight yellow
        "ink":      HexColor("#2A2A2A"),   # near-black, never pure black
        "midtone":  HexColor("#6B6357"),   # warm grey-brown
        "accent":   HexColor("#8A7544"),   # smoky gold
    },
    "M2_soft_greige": {
        "bg":       HexColor("#EAE4D6"),   # greige, more grey
        "ink":      HexColor("#2A2A2A"),
        "midtone":  HexColor("#6A6358"),
        "accent":   HexColor("#7E6A3F"),   # deeper, more brown gold
    },
    "M3_cool_ivory": {
        "bg":       HexColor("#EEEBE3"),   # cool ivory, slight green
        "ink":      HexColor("#262626"),
        "midtone":  HexColor("#666660"),   # cool warm-neutral
        "accent":   HexColor("#8B7B4F"),   # cooler smoky gold
    },
}


# ---------------------------------------------------------------------------
# Single cover renderer, parameterized by palette
# ---------------------------------------------------------------------------

def render_cover(c: canvas.Canvas, palette: dict) -> None:
    """C2 Swiss editorial layout in a Morandi palette.

    Composition rationale:
    - Title is flush-left, very large, three lines — establishes voice.
    - Subtitle in italic two lines, immediately below — gives context.
    - Top-right corner holds tagline + date (small caps sans).
    - Bottom-left author block — minimal, anchors the page.
    - No frames, no rules, no ornaments — let the typography work.
    - All text colors are off-black or muted (no pure black, no pure white).
    """
    W, H = LETTER
    bg = palette["bg"]
    ink = palette["ink"]
    midtone = palette["midtone"]
    accent = palette["accent"]

    # Full-bleed background
    c.setFillColor(bg)
    c.rect(0, 0, W, H, fill=1, stroke=0)

    # Top-right metadata block (small caps sans-serif)
    c.setFillColor(ink)
    c.setFont("Helvetica-Bold", 8.5)
    c.drawRightString(W - 1.0 * inch, H - 1.25 * inch, TAGLINE.upper())
    c.setFillColor(midtone)
    c.setFont("Helvetica", 8.5)
    c.drawRightString(W - 1.0 * inch, H - 1.45 * inch, DATE_LABEL.upper())

    # Title — three flush-left lines, very large, off-black
    c.setFillColor(ink)
    c.setFont("Times-Bold", 52)
    title_y_start = H - 4.6 * inch
    line_height = 0.85 * inch
    for i, line in enumerate(TITLE_LINES):
        c.drawString(1.0 * inch, title_y_start - i * line_height, line)

    # Subtitle — first line is the full descriptor, second is the year range.
    # Font size dropped to 11 so the long first line fits on a single line
    # within the cover's text block (letter width minus margins).
    c.setFillColor(midtone)
    sub_y_start = title_y_start - len(TITLE_LINES) * line_height - 0.18 * inch

    # Both subtitle lines render at the same size for visual consistency.
    c.setFont("Times-Italic", 11)
    c.drawString(1.0 * inch, sub_y_start, SUBTITLE_LINES[0])
    c.drawString(1.0 * inch, sub_y_start - 0.28 * inch, SUBTITLE_LINES[1])

    # Thin accent rule under subtitle (anchors the typography block)
    c.setStrokeColor(accent)
    c.setLineWidth(0.6)
    rule_y = sub_y_start - 0.55 * inch
    c.line(1.0 * inch, rule_y, 2.5 * inch, rule_y)

    # Author block bottom-left
    c.setFillColor(ink)
    c.setFont("Helvetica-Bold", 9.5)
    c.drawString(1.0 * inch, 1.25 * inch, "BY " + AUTHOR_NAME.upper())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Render the locked-in cover: C2 Swiss editorial layout, M3 cool-ivory
    Morandi palette. This is the visual anchor used in 09_report.py."""
    proof_dir = OUTPUT_DIR / "cover_proofs"
    proof_dir.mkdir(parents=True, exist_ok=True)

    palette = PALETTES["M3_cool_ivory"]
    out = proof_dir / "cover_final.pdf"
    c = canvas.Canvas(str(out), pagesize=LETTER)
    render_cover(c, palette)
    c.showPage()
    c.save()
    print(f"\nFinal cover \u2192 {out}")


if __name__ == "__main__":
    main()