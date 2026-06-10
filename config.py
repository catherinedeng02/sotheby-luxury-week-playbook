"""Central configuration for the Sotheby's Luxury Week Playbook project.

Every script in this project imports from here. Keeping paths, palette,
and report metadata in one file means you change a setting once and the
whole pipeline picks it up.
"""
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# PROJECT_ROOT resolves to the folder this file lives in. All other paths
# are computed relative to it, so the project is portable — you can move
# the folder anywhere and nothing breaks.

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
SUMMARIES_DIR = OUTPUT_DIR / "summaries"

REPORT_PATH = OUTPUT_DIR / "luxury_week_playbook.pdf"


# ---------------------------------------------------------------------------
# Report metadata — these print on the cover page
# ---------------------------------------------------------------------------

REPORT_TITLE = "The Sotheby's Luxury Week Playbook"
REPORT_SUBTITLE = (
    "Cross-Category Performance, Premium Drivers, "
    "and the HK\u2013NY Buyer Corridor, 2021\u20132026"
)
AUTHOR_NAME = "Zhuoyao Deng"            # <-- change this before final submission
AUTHOR_TAGLINE = "Independent Market Study"


# ---------------------------------------------------------------------------
# Visual palette — restrained, auction-house-appropriate
# ---------------------------------------------------------------------------
# Gold = positive premium drivers. Claret = negative (discount). Ink = text.
# Stone/midtone/parchment = supporting neutrals. These are used consistently
# across every chart so the report reads as one document, not eight.

PALETTE = {
    "ink":          "#1A1A1A",   # headings, axis text
    "midtone":      "#6B6B6B",   # secondary text, gridlines
    "stone":        "#A8A8A8",   # tertiary, captions
    "parchment":    "#F5F1E8",   # subtle background fills
    "gold":         "#B8860B",   # positive coefficients, highlights
    "soft_gold":    "#D4A847",   # secondary gold
    "claret":       "#722F37",   # negative coefficients
    "soft_claret":  "#A85358",   # secondary claret
}


# ---------------------------------------------------------------------------
# Analysis constants
# ---------------------------------------------------------------------------

RANDOM_SEED = 42        # reproducibility for synthesis & any random sampling

# Categories used for macro-level analysis (Pillar 1)
CATEGORIES = [
    "Handbags",
    "Watches",
    "Wine",
    "Spirits",
    "Modern Collectibles",
]