"""Section IV.1 — Vintage Rolex Daytona premium drivers.

Reads:   data/raw/watches_daytona.csv
Schema:  lot_id, sale_date, sale_location, reference, dial_type, bezel,
         movement, papers, provenance, low_estimate, high_estimate,
         hammer_price, buyer_region

Outputs:
  output/figures/04_daytona_coefficients.png  OLS coefficient bar chart
  output/figures/04_daytona_combos.png        top winning combinations
  output/summaries/04_daytona.json            headline takeaways + stats

The most heavily traded vintage chronograph reference family. Encoded
priors: exotic 'Paul Newman' dials dominate; full set provenance
(box and papers) and celebrity provenance are massive uplifts;
steel bezel >> gold bezel for vintage 6263/6265 specifically.
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
    df = load_csv("watches_daytona")
    df = compute_premium(df)
    n_lots = len(df)
    print(f"Loaded {n_lots} sold vintage Daytona lots")

    # ---- OLS regression
    # Reference categories chosen to represent the "typical" vintage Daytona:
    # ref 6263 (most common production run), standard dial, acrylic bezel,
    # Valjoux 727 movement, no papers, single-owner provenance.
    features = ["reference", "dial_type", "bezel", "movement", "papers", "provenance"]
    reference_levels = {
        "reference": "6263",
        "dial_type": "Standard",
        "bezel": "Acrylic",
        "movement": "Valjoux 727",
        "papers": "None",
        "provenance": "Single owner",
    }
    model = fit_premium_model(df, features, reference_levels=reference_levels)
    print(model.summary())

    # ---- Figure 1: coefficient bar chart
    years = sorted(df["sale_date"].dt.year.dropna().unique().astype(int))
    year_range = f"{years[0]}\u2013{years[-1]}" if years else ""

    plot_coefficients(
        model,
        title=(
            f"Vintage Rolex Daytona: Hammer Premium Drivers "
            f"(n = {n_lots}, {year_range})"
        ),
        output=FIGURES_DIR / "04_daytona_coefficients.png",
    )

    # ---- Figure 2: winning combinations table
    # Dial × papers × provenance is the canonical "what made this lot
    # exceptional?" tuple for vintage Daytona.
    combo_df = winning_combo_table(
        df, feature_cols=["dial_type", "papers", "provenance"],
        min_n=3, top_k=8,
    )
    plot_combo_table(
        combo_df,
        title=(
            "Vintage Rolex Daytona: Top Configurations by Median Premium "
            "(min n = 3 per combination)"
        ),
        output=FIGURES_DIR / "04_daytona_combos.png",
    )

    # ---- Summary takeaways
    sentences = top_driver_sentences(model, n=4, pct_threshold=10.0)

    payload = {
        "title": "Vintage Rolex Daytona — Premium Drivers",
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
            "04_daytona_coefficients.png",
            "04_daytona_combos.png",
        ],
    }
    write_summary("04_daytona", payload)

    print("\nHeadline drivers:")
    for s in sentences:
        print(f"  \u2022 {s}")


if __name__ == "__main__":
    main()