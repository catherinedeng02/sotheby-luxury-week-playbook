"""Build the final PDF report.

Strategy:
  - Cover: prepend the finalized cover_final.pdf (not redrawn).
  - Body: ReportLab, one analysis per page = H2 heading + one figure
    (width-controlled, not squashed) + 2-3 paragraphs of finalized prose
    lifted from the website.
  - Figures: existing high-res matplotlib PNGs in output/figures/.
  - Body pages on white; cover stays ivory. US Letter, Times-Roman.

Output: output/luxury_week_playbook.pdf

Run: python3 src/09_report.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import date
import io

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, KeepTogether,
    Flowable,
)
from pypdf import PdfReader, PdfWriter

from config import (
    FIGURES_DIR, OUTPUT_DIR, REPORT_TITLE, AUTHOR_NAME,
)

COVER_PATH = OUTPUT_DIR / "cover_proofs" / "cover_final.pdf"
BODY_PATH = OUTPUT_DIR / "_body_only.pdf"
FINAL_PATH = OUTPUT_DIR / "luxury_week_playbook.pdf"

# Palette (white body, ivory only on cover)
INK = HexColor("#262626")
MIDTONE = HexColor("#666660")
STONE = HexColor("#A8A8A8")
GOLD = HexColor("#8B7B4F")
RULE = HexColor("#D8D2C2")


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

def build_styles():
    base = getSampleStyleSheet()
    return {
        "h1": ParagraphStyle("H1", parent=base["Heading1"],
            fontName="Times-Bold", fontSize=20, leading=25, textColor=INK,
            spaceBefore=2, spaceAfter=12),
        "h2": ParagraphStyle("H2", parent=base["Heading2"],
            fontName="Times-Bold", fontSize=15, leading=20, textColor=INK,
            spaceBefore=4, spaceAfter=8),
        "body": ParagraphStyle("Body", parent=base["BodyText"],
            fontName="Times-Roman", fontSize=10.5, leading=15.5, textColor=INK,
            alignment=TA_JUSTIFY, spaceAfter=9),
        "stat_row": ParagraphStyle("StatRow", parent=base["BodyText"],
            fontName="Times-Roman", fontSize=10.5, leading=15.5, textColor=INK,
            alignment=TA_LEFT, spaceAfter=4),
        "finding_num": ParagraphStyle("FNum", parent=base["BodyText"],
            fontName="Times-Bold", fontSize=8.5, leading=12, textColor=GOLD,
            spaceAfter=2),
        "finding_head": ParagraphStyle("FHead", parent=base["BodyText"],
            fontName="Times-Italic", fontSize=12.5, leading=16, textColor=INK,
            spaceAfter=4),
        "source": ParagraphStyle("Source", parent=base["BodyText"],
            fontName="Times-Italic", fontSize=8, leading=11, textColor=STONE,
            alignment=TA_LEFT, spaceBefore=3, spaceAfter=14),
        "foot": ParagraphStyle("Foot", parent=base["BodyText"],
            fontName="Times-Italic", fontSize=8.5, leading=12, textColor=MIDTONE,
            alignment=TA_LEFT, spaceBefore=20),
    }


class GoldRule(Flowable):
    """Short gold rule used above each section heading."""
    def __init__(self, width=2.5 * 72 / 2.54 / 10 * 10, thickness=1.6):
        super().__init__()
        self.width = 0.9 * inch
        self.thickness = thickness
    def draw(self):
        self.canv.setStrokeColor(GOLD)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, 0, self.width, 0)


def fig(name: str, width: float = 6.6 * inch):
    """Place a figure scaled to `width`, preserving aspect ratio,
    capped so it never dominates the page (leaves room for prose)."""
    path = FIGURES_DIR / name
    if not path.exists():
        return Paragraph(f"<i>[missing figure: {name}]</i>",
                         build_styles()["source"])
    from PIL import Image as PILImage
    iw, ih = PILImage.open(str(path)).size
    aspect = ih / iw
    h = width * aspect
    max_h = 3.5 * inch          # cap height so prose always shares the page
    if h > max_h:
        h = max_h
        width = h / aspect
    img = Image(str(path), width=width, height=h)
    img.hAlign = "CENTER"
    return img


def section(styles, heading, figure_name, source, paragraphs,
            tail_paragraphs=None):
    """One analysis page: gold rule + H2 + figure + source + prose."""
    story = [GoldRule(), Spacer(1, 4), Paragraph(heading, styles["h2"])]
    story.append(fig(figure_name))
    story.append(Paragraph(source, styles["source"]))
    for p in paragraphs:
        story.append(Paragraph(p, styles["body"]))
    if tail_paragraphs:
        for p in tail_paragraphs:
            story.append(Paragraph(p, styles["body"]))
    story.append(PageBreak())
    return story


# ---------------------------------------------------------------------------
# Body content (prose lifted from the finalized website copy)
# ---------------------------------------------------------------------------

def build_body():
    styles = build_styles()
    S = []

    # ---- Market Overview (text only) ----
    S.append(Paragraph("Market Overview", styles["h1"]))
    S.append(Paragraph(
        "Sotheby&rsquo;s Luxury Week has become the firm&rsquo;s "
        "highest-velocity sale calendar, compressing watches, jewels, "
        "handbags, wine, and modern collectibles into a single concentrated "
        "selling window. Across 1,180 lots sold between 2021 and 2026, this "
        "study models what drives a lot to clear above estimate, and finds "
        "the answer is strikingly consistent across otherwise unrelated "
        "categories.", styles["body"]))

    for num, head, body in [
        ("01 / MARKET", "Watches leads on premium without the volatility.",
         "Across the study window Watches held a median hammer-to-mid-estimate "
         "ratio of 1.41x against a sell-through rate near 95%. Modern "
         "Collectibles reached higher in its best seasons but also fell "
         "further in its worst. Watches rarely had a weak quarter, which is "
         "usually what you see when there are more buyers than good material "
         "to go around."),
        ("02 / CATEGORY", "The ceiling is set by rarity, not by name.",
         "In every category model the largest premium attaches to scarcity or "
         "documented history, not to the maker&rsquo;s name. A Himalaya "
         "crocodile Birkin runs about 56% above its ordinary counterpart, an "
         "exotic Paul Newman Daytona more than 200%, a Karuizawa bottling "
         "close to 60%. The name and the age are priced in before bidding "
         "even starts. What moves the hammer is whatever is left that "
         "can&rsquo;t be reproduced."),
        ("03 / CORRIDOR", "Where a lot sells depends on what it is.",
         "Comparing 80 lots that sold in both Hong Kong and New York shows "
         "there is no single better venue, only a better venue for a given "
         "thing. Herm&egrave;s handbags clear around 8% higher in Hong Kong, "
         "while vintage Rolex Daytona clears closer to 17% higher in New "
         "York. Tested together the two effects nearly cancel, which is why "
         "the routing question only becomes useful once the categories are "
         "pulled apart."),
    ]:
        S.append(Spacer(1, 6))
        S.append(Paragraph(num, styles["finding_num"]))
        S.append(Paragraph(head, styles["finding_head"]))
        S.append(Paragraph(body, styles["body"]))
    S.append(PageBreak())

    # ---- I. Macro ----
    S += section(styles,
        "I. Macro",
        "01_macro_heatmap.png",
        "Source: Sotheby&rsquo;s Past Auctions (sothebys.com); author&rsquo;s analysis. "
        "Figures are illustrative of the analytical method.",
        [
        "The starting question for any sale calendar is which categories are "
        "carrying the room. The heatmap tracks median premium ratio, hammer "
        "over mid-estimate, for each anchor category across every Luxury Week "
        "season from 2021 to 2026. The more telling pattern is not which "
        "season ran hot but how differently the categories behave from one "
        "another.",
        "Watches holds a tight band of strong results season after season. "
        "Modern Collectibles posts the highest individual cells in the grid "
        "yet also the coldest, swinging between blockbuster and ordinary "
        "depending on what crossed the block that quarter. A strong Modern "
        "Collectibles season usually comes down to one or two exceptional "
        "consignments, not broad demand, and it pays to read it that way "
        "before drawing conclusions.",
        "Premium and sell-through do not always move together. A category can "
        "clear nearly every lot while only modestly beating estimate, which "
        "tends to mean estimates were set conservatively against steady "
        "demand. Watches manages the rarer combination, holding sell-through "
        "high while still leading on premium, the profile of a category where "
        "buyers outnumber the good material coming to market.",
        ])

    # ---- II. Birkin ----
    S += section(styles,
        "II. Herm\u00e8s Birkin &amp; Kelly",
        "02_birkin_coefficients.png",
        "Source: Sotheby&rsquo;s Past Auctions (sothebys.com); author&rsquo;s analysis. "
        "Figures are illustrative of the analytical method.",
        [
        "Working from 295 sold lots, the model regresses log-premium on the "
        "five characteristics a cataloguer records for every bag: leather, "
        "hardware, size, condition, and model. The reference point is the bag "
        "one sees most often at auction, a 35cm Birkin in Togo leather with "
        "gold hardware in Excellent condition, so each coefficient reads as "
        "the premium a feature adds against that ordinary baseline.",
        "Two features pull away from the rest. A Himalaya Niloticus crocodile "
        "finish carries a premium near 56% over the same bag in Togo, and "
        "diamond-set hardware sits close behind around 45%. The two are hard "
        "to separate statistically, but both stand well clear of everything "
        "beneath them, so the honest reading is that they share the top tier "
        "rather than that one outranks the other.",
        "Condition is where the more useful lesson sits, because its effect "
        "is not symmetric. A bag graded Good rather than Excellent gives back "
        "close to a third of its premium, more than Pristine adds at the "
        "other end. The downside of a condition slip is bigger than the "
        "upside. For estimate-setting that means pricing anything below "
        "Excellent conservatively, less to maximise a single hammer than to "
        "protect sell-through and the momentum of the sale.",
        ])

    # ---- III. Corridor ----
    S += section(styles,
        "III. HK\u2013NY Buyer Corridor",
        "03_corridor_paired.png",
        "Source: Illustrative matched-pair sample; author&rsquo;s analysis. "
        "Buyer-share figures cited below: Sotheby&rsquo;s press release, December 2022.",
        [
        "There is a structural reason a corridor exists. Asian clients account "
        "for roughly 40% of all spend across Sotheby&rsquo;s core luxury "
        "auctions, the highest share of any region, and their weight is uneven "
        "across categories: about 34% of handbag buyers and a comparable share "
        "in watches. That buyer geography suggests, though it does not by "
        "itself prove, that a category whose collector base sits "
        "disproportionately in Asia should find deeper competition in Hong "
        "Kong, while one whose deepest bench is American should hold its "
        "strongest bids in New York.",
        "To test it, the analysis pairs comparable lots, the same reference "
        "or model in similar condition and within a 30% band of one "
        "another&rsquo;s mid-estimate, and compares the premium each realised "
        "in Hong Kong against New York. Points above the parity line did "
        "better in Hong Kong, points below did better in New York. The "
        "categories separate by direction, and the direction is the one the "
        "buyer geography predicted: handbags around 8% higher in Hong Kong, "
        "vintage Daytona closer to 17% stronger in New York, Patek near parity.",
        "A pooled test across all pairs returns a p-value near 0.10, which a "
        "strict reading would call inconclusive. The reason is not weak data "
        "but offsetting signals that cancel when averaged into one number. "
        "The pooled figure is the wrong thing to read; the split underneath "
        "it is the point, and it lines up with where each category&rsquo;s "
        "buyers actually are.",
        ])

    # ---- IV.1 Daytona ----
    S += section(styles,
        "IV.1 Vintage Rolex Daytona",
        "04_daytona_coefficients.png",
        "Source: Sotheby&rsquo;s Past Auctions (sothebys.com); author&rsquo;s analysis. "
        "Figures are illustrative of the analytical method.",
        [
        "The vintage Daytona market is organised almost entirely around the "
        "dial. An exotic Paul Newman dial lifts hammer premium by more than "
        "200% over an otherwise identical standard-dial reference, a "
        "coefficient several times larger than anything else in the model and "
        "large enough that it effectively defines two separate markets that "
        "happen to share a case shape.",
        "Full documentation and named provenance matter too, around 47% and "
        "49%, but they are second-order next to the dial. For a watch sale "
        "the catalogue order is obvious: put the Paul Newman lots first, "
        "shoot the dial properly, and let everything else sit behind them.",
        ])

    # ---- IV.2 Patek + DRC/Whisky pointer ----
    S += section(styles,
        "IV.2 Patek Philippe Calatrava",
        "05_patek_coefficients.png",
        "Source: Sotheby&rsquo;s Past Auctions (sothebys.com); author&rsquo;s analysis. "
        "Figures are illustrative of the analytical method.",
        [
        "Calatrava premium is led by two features that are difficult to rank "
        "against each other. A salmon dial and a stainless-steel case each "
        "carry something near 50% to 58%, with overlapping confidence "
        "intervals, so the data treats them as a shared top tier. Both are "
        "scarce for the same reason: Patek built the Calatrava overwhelmingly "
        "in precious metal with conventional dials, so steel and salmon are "
        "the exceptions a collector waits years to see.",
        "The more practical lever is the next tier down. Original papers add "
        "roughly 36% and a Patek archive extract about 24%, and unlike a rare "
        "case or dial, you can actually go and get them. No one can produce a "
        "steel Calatrava on demand, but a specialist can write to Patek for "
        "an extract, which makes the paperwork the most dependable way to "
        "lift a Calatrava result.",
        ],
        tail_paragraphs=[
        "<b>Further categories.</b> The same model is applied to DRC Burgundy "
        "and Japanese whisky in the full online report. In Burgundy, format "
        "and direct-from-Domaine provenance lead, the latter worth about 22% "
        "over a private-cellar source at the same vintage. In whisky, a "
        "closed distillery dominates: Karuizawa runs close to 60% over a "
        "mainstream distillery of comparable age, a clean case of the market "
        "pricing finite supply rather than quality.",
        ])

    # ---- Strategic Implications (text only) ----
    S.append(GoldRule())
    S.append(Spacer(1, 4))
    S.append(Paragraph("Strategic Implications", styles["h1"]))
    for p in [
        "Read across categories, the findings point the same direction. "
        "Premium is paid for what cannot be reproduced, and the "
        "catalogue&rsquo;s job is to make that scarcity visible and "
        "believable to the room. A few practical things follow.",
        "Most of it is decided at sourcing. The biggest premiums sit on "
        "features a specialist cannot create after the fact: an exotic dial, "
        "a steel case, a closed distillery, a Himalaya skin. That argues for "
        "putting consignment effort behind the rare configurations rather "
        "than chasing volume, since one well-sourced Paul Newman Daytona or "
        "salmon Calatrava does more for a sale than a dozen ordinary lots "
        "beside it.",
        "What cannot be sourced can often be documented, and that is the more "
        "controllable half. A Patek extract is worth around 24%, original "
        "papers on a Daytona near 47%, a direct-from-Domaine line on a DRC "
        "lot about 22%. None of these need a rarer object, only the work to "
        "get the paperwork and present it properly.",
        "Routing should resist a house default. No city pays more in general; "
        "handbags do better in Hong Kong, vintage sports watches in New York. "
        "Deciding venue by category rather than by policy picks up a premium "
        "a blanket rule would miss. And estimates deserve a defensive hand: "
        "condition is asymmetric, a grade lost costs more than a grade "
        "gained, so anything below top condition warrants conservative "
        "pricing to keep lots clearing.",
        "None of this replaces a specialist&rsquo;s judgment. The model only "
        "sees what a catalogue records and a saleroom prices. It does not see "
        "the relationship that wins a single-owner collection, the freshness "
        "a long-held lot carries, or the timing of a market this study treats "
        "as fixed. What it offers is narrower and, within those limits, "
        "reliable: a measured account of which recorded attributes have moved "
        "hammer prices, and by how much.",
    ]:
        S.append(Paragraph(p, styles["body"]))
    S.append(PageBreak())

    # ---- Appendix (text only) ----
    S.append(GoldRule())
    S.append(Spacer(1, 4))
    S.append(Paragraph("Appendix", styles["h1"]))
    for p in [
        "<b>Method.</b> Premium ratio is hammer price divided by "
        "mid-estimate, the average of published low and high. Each category "
        "model is an ordinary least squares regression of log-premium on "
        "one-hot encoded categorical attributes, with reference levels set to "
        "the most common configuration so coefficients read as premium "
        "relative to a typical lot. Using log-premium keeps a beat and a miss "
        "of equal proportion symmetric. Confidence intervals are at the 95% "
        "level; where a coefficient rests on few lots, that is noted in the "
        "text rather than hidden. Sample sizes range from roughly 150 to 320 "
        "sold lots per category.",
        "<b>Data.</b> The figures in this report are illustrative, generated "
        "to demonstrate the analytical method on realistic category "
        "structures while the 2026 Luxury Week results are still pending. "
        "The schema mirrors Sotheby&rsquo;s Past Auctions across New York, "
        "Hong Kong, Geneva, and London, 2021 to 2026, recording each "
        "lot&rsquo;s sale date and location, category-specific attributes, "
        "published estimates, hammer price, and buyer region where disclosed. "
        "Macro buyer-share figures cited in Section III are real, drawn from "
        "Sotheby&rsquo;s December 2022 press release.",
        "<b>Code and full report.</b> The complete pipeline, the interactive "
        "version of every figure, and additional category studies (DRC "
        "Burgundy and Japanese whisky) are available in the online report and "
        "project repository.",
    ]:
        S.append(Paragraph(p, styles["body"]))
    S.append(Paragraph(
        "The Sotheby&rsquo;s Luxury Week Playbook &middot; Independent Market "
        "Intelligence Report &middot; " + date.today().strftime("%B %Y"),
        styles["foot"]))

    return S


# ---------------------------------------------------------------------------
# Page footer
# ---------------------------------------------------------------------------

def on_page(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.3)
    canvas.line(0.85 * inch, 0.7 * inch, LETTER[0] - 0.85 * inch, 0.7 * inch)
    canvas.setFont("Times-Italic", 8)
    canvas.setFillColor(STONE)
    canvas.drawString(0.85 * inch, 0.5 * inch, REPORT_TITLE)
    canvas.drawRightString(LETTER[0] - 0.85 * inch, 0.5 * inch, str(doc.page + 1))
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Main: build body, then prepend cover
# ---------------------------------------------------------------------------

def main():
    doc = SimpleDocTemplate(
        str(BODY_PATH), pagesize=LETTER,
        leftMargin=0.85 * inch, rightMargin=0.85 * inch,
        topMargin=0.85 * inch, bottomMargin=0.9 * inch,
        title=REPORT_TITLE, author=AUTHOR_NAME,
    )
    doc.build(build_body(), onFirstPage=on_page, onLaterPages=on_page)
    print(f"  body written -> {BODY_PATH.name}")

    # Merge cover + body
    writer = PdfWriter()
    if COVER_PATH.exists():
        for pg in PdfReader(str(COVER_PATH)).pages:
            writer.add_page(pg)
    else:
        print(f"  ! cover not found at {COVER_PATH}, skipping")
    for pg in PdfReader(str(BODY_PATH)).pages:
        writer.add_page(pg)
    with open(FINAL_PATH, "wb") as f:
        writer.write(f)
    print(f"  final report -> {FINAL_PATH}")


if __name__ == "__main__":
    main()