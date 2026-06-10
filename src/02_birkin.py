"""Pillar 2 (primary deep-dive) — Hermès Birkin & Kelly premium drivers.

Reads:   data/raw/handbags.csv
Schema:  lot_id, sale_date, sale_location, brand, model, leather, hardware,
         size_cm, year, condition, low_estimate, high_estimate, hammer_price,
         buyer_region

Outputs:
  output/figures/02_birkin_coefficients.png   OLS coefficient bar chart
  output/figures/02_birkin_combos.png         top winning combinations
  output/summaries/02_birkin.json             headline takeaways + stats

Methodology:
  - Premium = hammer / mid-estimate, mid = (low + high) / 2
  - Target = log(premium) so coefficients are symmetric above/below estimate
  - OLS with one-hot encoded categoricals; reference levels chosen to be the
    'typical' lot (Togo leather, GHW hardware, 35cm size, Excellent
    condition, Birkin model) so coefficients read as 'premium uplift vs the
    typical Birkin/Kelly'.
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


def main() -> None:
    apply_style()
    df = load_csv("handbags")
    df = compute_premium(df)
    n_lots = len(df)
    print(f"Loaded {n_lots} sold Birkin/Kelly lots")

    # ---- OLS regression
    # Reference categories chosen to represent the "typical" lot, so
    # coefficients read naturally as "premium uplift vs typical Birkin".
    features = ["leather", "hardware", "size_cm", "condition", "model"]
    reference_levels = {
        "leather": "Togo",         # most common calf leather
        "hardware": "GHW",         # gold hardware = market baseline
        "size_cm": 35,             # iconic Birkin 35
        "condition": "Excellent",  # market median
        "model": "Birkin",         # the namesake model
    }
    model = fit_premium_model(df, features, reference_levels=reference_levels)

    print(model.summary())

    # ---- Figure 1: coefficient bar chart
    # Compute year range from sale_date for the title, so it stays accurate
    # when real data replaces the synthesized data.
    years = sorted(df["sale_date"].dt.year.dropna().unique().astype(int))
    year_range = f"{years[0]}\u2013{years[-1]}" if years else ""

    plot_coefficients(
        model,
        title=(
            f"Hermès Birkin & Kelly: Hammer Premium Drivers "
            f"(n = {n_lots}, {year_range})"
        ),
        output=FIGURES_DIR / "02_birkin_coefficients.png",
    )

    # ---- Figure 2: winning combinations table
    # Look at leather × hardware × size_cm combinations, keep those with
    # at least 3 observations, take top 8 by median premium.
    combo_df = winning_combo_table(
        df, feature_cols=["leather", "hardware", "size_cm"],
        min_n=3, top_k=8,
    )
    plot_combo_table(
        combo_df,
        title=(
            f"Hermès Birkin & Kelly: Top Configurations by Median Premium "
            f"(min n = 3 per combination)"
        ),
        output=FIGURES_DIR / "02_birkin_combos.png",
    )

    # ---- Summary takeaways
    # Pull the top 4 magnitude drivers (positive or negative) as takeaway
    # sentences. Robust to which leather/hardware ends up as the dummy
    # reference level.
    sentences = top_driver_sentences(model, n=4, pct_threshold=10.0)

    payload = {
        "title": "Hermès Birkin & Kelly — Premium Drivers",
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
            "02_birkin_coefficients.png",
            "02_birkin_combos.png",
        ],
    }
    write_summary("02_birkin", payload)

    print("\nHeadline drivers:")
    for s in sentences:
        print(f"  \u2022 {s}")


if __name__ == "__main__":
    main()