"""Section IV.2 — Patek Philippe Calatrava premium drivers.

Reads:   data/raw/watches_patek.csv
Schema:  lot_id, sale_date, sale_location, reference, case_material,
         dial_color, extract_archives, original_papers, low_estimate,
         high_estimate, hammer_price, buyer_region

Outputs:
  output/figures/05_patek_coefficients.png   OLS coefficient bar chart
  output/figures/05_patek_combos.png         top winning combinations
  output/summaries/05_patek.json             headline takeaways + stats

Encoded priors: stainless-steel case enormous (Patek historically only
produced precious-metal cases; steel Calatrava is among the rarest
configurations); Extract from the Patek Archives premium; original
papers premium; salmon dial > champagne > silver.

This section aligns directly with the June 10, 2026 Important Watches
sale at Sotheby's NY, which includes 20+ special-caliber Calatrava
including 2526, 2551, 2552, 3415, and 3444.
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
    df = load_csv("watches_patek")
    df = compute_premium(df)
    n_lots = len(df)
    print(f"Loaded {n_lots} sold Patek Calatrava lots")

    # ---- OLS regression
    # Reference categories chosen to represent the "typical" Calatrava:
    # ref 3444 (a more common postwar reference), yellow gold case,
    # champagne dial, no archives extract, no original papers.
    features = ["reference", "case_material", "dial_color",
                "extract_archives", "original_papers"]
    reference_levels = {
        "reference": "3444",
        "case_material": "Yellow gold",
        "dial_color": "Champagne",
        "extract_archives": "No",
        "original_papers": "No",
    }
    model = fit_premium_model(df, features, reference_levels=reference_levels)
    print(model.summary())

    # ---- Figure 1: coefficient bar chart
    years = sorted(df["sale_date"].dt.year.dropna().unique().astype(int))
    year_range = f"{years[0]}\u2013{years[-1]}" if years else ""

    plot_coefficients(
        model,
        title=(
            f"Patek Philippe Calatrava: Hammer Premium Drivers "
            f"(n = {n_lots}, {year_range})"
        ),
        output=FIGURES_DIR / "05_patek_coefficients.png",
    )

    # ---- Figure 2: winning combinations table
    # Case material × dial color × papers is the canonical Patek tuple.
    combo_df = winning_combo_table(
        df, feature_cols=["case_material", "original_papers", "dial_color"],
        min_n=3, top_k=8,
    )
    plot_combo_table(
        combo_df,
        title=(
            "Patek Philippe Calatrava: Top Configurations by Median Premium "
            "(min n = 3 per combination)"
        ),
        output=FIGURES_DIR / "05_patek_combos.png",
    )

    # ---- Summary takeaways
    sentences = top_driver_sentences(model, n=4, pct_threshold=10.0)

    payload = {
        "title": "Patek Philippe Calatrava — Premium Drivers",
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
            "05_patek_coefficients.png",
            "05_patek_combos.png",
        ],
    }
    write_summary("05_patek", payload)

    print("\nHeadline drivers:")
    for s in sentences:
        print(f"  \u2022 {s}")


if __name__ == "__main__":
    main()