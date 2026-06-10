# The Sotheby's Luxury Week Playbook

An independent market intelligence report analyzing cross-category premium
drivers and the HK–NY buyer corridor at Sotheby's Luxury Week auctions,
2021–2026.

**[→ Read the interactive report](https://catherinedeng02.github.io/sotheby-luxury-week-playbook/)**

**[→ Download the report](output/luxury_week_playbook.pdf)**

---

## What this is

A data-analysis study modeling what drives a luxury lot to clear above
estimate. Across eight high-collectible categories, it fits ordinary least
squares regressions on log-premium (hammer over mid-estimate) against
lot-level attributes, then translates the coefficients into sourcing,
documentation, routing, and estimate-setting implications.

Three pillars:

- **Macro** — cross-category premium and sell-through across Luxury Week seasons
- **Micro** — Hermès Birkin & Kelly premium-driver regression, extended to
  vintage Rolex Daytona, Patek Philippe Calatrava, DRC Burgundy, and Japanese whisky
- **Corridor** — a matched-pair analysis of venue-conditional premium (HK vs NY)

## Method

- Premium ratio = hammer / mid-estimate, mid = (low + high) / 2
- OLS on log-premium with one-hot categoricals; reference levels set to the
  typical lot, so coefficients read as premium uplift versus that baseline
- 95% confidence intervals reported throughout

## Data note

The lot-level figures are illustrative, generated to demonstrate the method on
realistic category structures while the 2026 Luxury Week results are pending.
Macro buyer-share figures cited in the corridor section are real, drawn from
Sotheby's published press material.

## Repository structure

| Path | Contents |
|------|----------|
| `src/` | Analysis pipeline (data synthesis, regressions, figure generation, PDF build) |
| `docs/` | Interactive report source (MkDocs Material) |
| `data/raw/` | Category datasets |
| `output/` | Generated figures, cover, and the final PDF report |

## Running locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 src/synthesize.py        # generate datasets
python3 src/build_plotly.py      # generate interactive figures
mkdocs serve                     # preview at localhost:8000
```

---

*By Zhuoyao Deng · June 2026*
