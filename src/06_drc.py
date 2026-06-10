"""Section IV.3 — Domaine de la Romanée-Conti (DRC) Burgundy premium drivers.

Reads:   data/raw/wine_drc.csv
Schema:  lot_id, sale_date, sale_location, cuvee, vintage, format, source,
         owc_intact, low_estimate, high_estimate, hammer_price, buyer_region

Outputs:
  output/figures/06_drc_coefficients.png    OLS coefficient bar chart
  output/figures/06_drc_combos.png          top winning combinations
  output/summaries/06_drc.json              headline takeaways + stats

Encoded priors: Romanée-Conti cuvée dominates the lineup; great vintages
(1990, 1999, 2005, 2009, 2015, 2018, 2019) command premium; larger
formats (Magnum, Jeroboam) > 750ml; direct-from-Domaine provenance
adds material premium over private-cellar source.

Note on feature engineering: the raw `vintage` column is an integer year
(1985–2020). For the regression, we convert it to a categorical
`vintage_tier` indicating whether the vintage falls within the Burgundy
collector community's consensus 'great vintages' set. A continuous-year
predictor would assume a linear time trend that doesn't exist in this
market — what matters is the binary 'is this a great vintage' read.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FIGURES_DIR
from src.common import (
    apply_style, load_csv, compute_premium, fit_premium_model,
    plot_coefficients, winning_combo_table, plot_combo_table,
    write_summary, top_driver_sentences, add_caption_and_source,
)


# Burgundy collector consensus 'great vintages' for DRC
GREAT_VINTAGES = {1990, 1999, 2005, 2009, 2015, 2018, 2019}


def main() -> None:
    apply_style()
    df = load_csv("wine_drc")
    df = compute_premium(df)
    n_lots = len(df)
    print(f"Loaded {n_lots} sold DRC lots")

    # ---- Feature engineering: vintage → vintage_tier
    df["vintage_tier"] = df["vintage"].apply(
        lambda v: "Great vintage" if int(v) in GREAT_VINTAGES else "Regular vintage"
    )

    # ---- OLS regression
    # Reference: Richebourg cuvée, Regular vintage, 750ml, Private cellar
    # source, no OWC. This is the median lot configuration.
    features = ["cuvee", "vintage_tier", "format", "source", "owc_intact"]
    reference_levels = {
        "cuvee": "Richebourg",
        "vintage_tier": "Regular vintage",
        "format": "750ml",
        "source": "Private cellar",
        "owc_intact": "No",
    }
    model = fit_premium_model(df, features, reference_levels=reference_levels)
    print(model.summary())

    # ---- Figure 1: coefficient bar chart
    years = sorted(df["sale_date"].dt.year.dropna().unique().astype(int))
    year_range = f"{years[0]}\u2013{years[-1]}" if years else ""

    plot_coefficients(
        model,
        title=(
            f"DRC Burgundy: Hammer Premium Drivers "
            f"(n = {n_lots}, {year_range})"
        ),
        output=FIGURES_DIR / "06_drc_coefficients.png",
    )

    # ---- Figure 2: winning combinations table
    combo_df = winning_combo_table(
        df, feature_cols=["cuvee", "vintage_tier", "format"],
        min_n=3, top_k=8,
    )
    plot_combo_table(
        combo_df,
        title=(
            "DRC Burgundy: Top Configurations by Median Premium "
            "(min n = 3 per combination)"
        ),
        output=FIGURES_DIR / "06_drc_combos.png",
    )

    # ---- Summary takeaways
    sentences = top_driver_sentences(model, n=4, pct_threshold=5.0)

    payload = {
        "title": "DRC Burgundy — Premium Drivers",
        "n_lots": int(n_lots),
        "features": features,
        "reference_levels": {k: str(v) for k, v in reference_levels.items()},
        "great_vintages_set": sorted(GREAT_VINTAGES),
        "r_squared": round(float(model.rsquared), 3),
        "adj_r_squared": round(float(model.rsquared_adj), 3),
        "n_features": int(len(model.params) - 1),
        "takeaway_sentences": sentences,
        "headline_takeaway": sentences[0] if sentences else "",
        "winning_combos": combo_df.to_dict(orient="records"),
        "figures": [
            "06_drc_coefficients.png",
            "06_drc_combos.png",
        ],
    }
    write_summary("06_drc", payload)

    print("\nHeadline drivers:")
    for s in sentences:
        print(f"  \u2022 {s}")


if __name__ == "__main__":
    main()