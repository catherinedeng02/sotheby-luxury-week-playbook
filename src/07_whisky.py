"""Section IV.4 — Japanese whisky premium drivers (Karuizawa, Ichiro's, Yamazaki).

Reads:   data/raw/whisky_japanese.csv
Schema:  lot_id, sale_date, sale_location, distillery, distilled_year,
         bottled_year, cask_type, abv, label_series, low_estimate,
         high_estimate, hammer_price, buyer_region

Outputs:
  output/figures/07_whisky_coefficients.png  OLS coefficient bar chart
  output/figures/07_whisky_combos.png        top winning combinations
  output/summaries/07_whisky.json            headline takeaways + stats

Encoded priors: Karuizawa (distillery closed in 2000) commands massive
premium given finite remaining stock; Ichiro's Hanyu Card Series is the
single most-collected label series in the category; mizunara casks
add Japanese-specific aging premium; pre-1980 distillation captures
the 'golden era' of Japanese whisky before consolidation.

Feature engineering note: from raw `distilled_year` and `bottled_year`
we derive (1) `age_tier` capturing maturation length in bins, and
(2) `distill_era` (pre-1980 / 1980+). These are the variables Japanese
whisky collectors actually use; raw years carry too much sparsity to
regress on directly.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import FIGURES_DIR
from src.common import (
    apply_style, load_csv, compute_premium, fit_premium_model,
    plot_coefficients, winning_combo_table, plot_combo_table,
    write_summary, top_driver_sentences, add_caption_and_source,
)


def main() -> None:
    apply_style()
    df = load_csv("whisky_japanese")
    df = compute_premium(df)
    n_lots = len(df)
    print(f"Loaded {n_lots} sold Japanese whisky lots")

    # ---- Feature engineering
    df["age_years"] = df["bottled_year"] - df["distilled_year"]
    df["age_tier"] = pd.cut(
        df["age_years"],
        bins=[0, 20, 30, 100],
        labels=["<20Y", "20-30Y", "30+Y"],
        include_lowest=True,
    ).astype(str)
    df["distill_era"] = df["distilled_year"].apply(
        lambda y: "Pre-1980 (golden era)" if int(y) < 1980 else "1980+"
    )

    # ---- OLS regression
    # Reference: Yamazaki (the mass-market benchmark), Bourbon cask,
    # Standard label, 20-30Y age, post-1980 distillation. This is the
    # canonical 'typical Japanese whisky auction lot'.
    features = ["distillery", "cask_type", "label_series", "age_tier", "distill_era"]
    reference_levels = {
        "distillery": "Yamazaki",
        "cask_type": "Bourbon",
        "label_series": "Standard",
        "age_tier": "20-30Y",
        "distill_era": "1980+",
    }
    model = fit_premium_model(df, features, reference_levels=reference_levels)
    print(model.summary())

    # ---- Figure 1: coefficient bar chart
    years = sorted(df["sale_date"].dt.year.dropna().unique().astype(int))
    year_range = f"{years[0]}\u2013{years[-1]}" if years else ""

    plot_coefficients(
        model,
        title=(
            f"Japanese Whisky: Hammer Premium Drivers "
            f"(n = {n_lots}, {year_range})"
        ),
        output=FIGURES_DIR / "07_whisky_coefficients.png",
    )

    # ---- Figure 2: winning combinations table
    combo_df = winning_combo_table(
        df, feature_cols=["distillery", "cask_type", "label_series"],
        min_n=3, top_k=8,
    )
    plot_combo_table(
        combo_df,
        title=(
            "Japanese Whisky: Top Configurations by Median Premium "
            "(min n = 3 per combination)"
        ),
        output=FIGURES_DIR / "07_whisky_combos.png",
    )

    # ---- Summary takeaways
    sentences = top_driver_sentences(model, n=4, pct_threshold=10.0)

    payload = {
        "title": "Japanese Whisky — Premium Drivers",
        "n_lots": int(n_lots),
        "features": features,
        "reference_levels": {k: str(v) for k, v in reference_levels.items()},
        "r_squared": round(float(model.rsquared), 3),
        "adj_r_squared": round(float(model.rsquared_adj), 3),
        "n_features": int(len(model.params) - 1),
        "takeaway_sentences": sentences,
        "headline_takeaway": sentences[0] if sentences else "",
        "winning_combos": combo_df.to_dict(orient="records"),
        "figures": [
            "07_whisky_coefficients.png",
            "07_whisky_combos.png",
        ],
    }
    write_summary("07_whisky", payload)

    print("\nHeadline drivers:")
    for s in sentences:
        print(f"  \u2022 {s}")


if __name__ == "__main__":
    main()