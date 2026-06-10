"""Section IV.5 — Game-worn / photo-matched sneakers premium drivers.

Reads:   data/raw/sneakers.csv
Schema:  lot_id, sale_date, sale_location, player, model, season,
         game_worn, photo_matched, signed, condition, low_estimate,
         high_estimate, hammer_price, buyer_region

Outputs:
  output/figures/08_sneakers_coefficients.png  OLS coefficient bar chart
  output/figures/08_sneakers_combos.png        top winning combinations
  output/summaries/08_sneakers.json            headline takeaways + stats

Encoded priors: Michael Jordan dominates among players; photo-matched
authentication (via MeiGray, Resolution Photomatching, etc) materially
exceeds plain game-worn — it converts a 'player-worn' artifact into a
specific historical moment; signed adds further uplift; condition
asymmetric (heavily worn = significant discount).

Feature engineering: derive `era` from `season` to capture MJ Rookie
1984-86, Bulls Dynasty 1987-98, Post-MJ 1999-2015, Modern 2016+. This
mirrors how Modern Collectibles specialists actually segment the market.
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


def _era_from_season(s: int) -> str:
    s = int(s)
    if s <= 1986:
        return "MJ Rookie (1984\u20131986)"
    elif s <= 1998:
        return "Bulls Dynasty (1987\u20131998)"
    elif s <= 2015:
        return "Post-MJ (1999\u20132015)"
    else:
        return "Modern (2016+)"


def main() -> None:
    apply_style()
    df = load_csv("sneakers")
    df = compute_premium(df)
    n_lots = len(df)
    print(f"Loaded {n_lots} sold sneaker lots")

    # ---- Feature engineering
    df["era"] = df["season"].apply(_era_from_season)
    # Convert booleans to readable strings so coefficients label cleanly
    df["game_worn"] = df["game_worn"].map({True: "Yes", False: "No"})
    df["photo_matched"] = df["photo_matched"].map({True: "Yes", False: "No"})
    df["signed"] = df["signed"].map({True: "Yes", False: "No"})

    # ---- OLS regression
    # Reference: Other player, not game-worn, not photo-matched, not signed,
    # Good condition, Post-MJ era. This is the 'typical' lot in the synthetic
    # data — a non-game-worn, post-MJ sneaker in average condition.
    features = ["player", "game_worn", "photo_matched", "signed",
                "condition", "era"]
    reference_levels = {
        "player": "Other",
        "game_worn": "No",
        "photo_matched": "No",
        "signed": "No",
        "condition": "Good",
        "era": "Post-MJ (1999\u20132015)",
    }
    model = fit_premium_model(df, features, reference_levels=reference_levels)
    print(model.summary())

    # ---- Figure 1: coefficient bar chart
    years = sorted(df["sale_date"].dt.year.dropna().unique().astype(int))
    year_range = f"{years[0]}\u2013{years[-1]}" if years else ""

    plot_coefficients(
        model,
        title=(
            f"Game-Worn Sneakers: Hammer Premium Drivers "
            f"(n = {n_lots}, {year_range})"
        ),
        output=FIGURES_DIR / "08_sneakers_coefficients.png",
    )

    # ---- Figure 2: winning combinations table
    # Player × game_worn × photo_matched is the canonical sneaker tuple.
    combo_df = winning_combo_table(
        df, feature_cols=["player", "game_worn", "photo_matched"],
        min_n=3, top_k=8,
    )
    plot_combo_table(
        combo_df,
        title=(
            "Game-Worn Sneakers: Top Configurations by Median Premium "
            "(min n = 3 per combination)"
        ),
        output=FIGURES_DIR / "08_sneakers_combos.png",
    )

    # ---- Summary takeaways
    sentences = top_driver_sentences(model, n=4, pct_threshold=10.0)

    payload = {
        "title": "Game-Worn Sneakers — Premium Drivers",
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
            "08_sneakers_coefficients.png",
            "08_sneakers_combos.png",
        ],
    }
    write_summary("08_sneakers", payload)

    print("\nHeadline drivers:")
    for s in sentences:
        print(f"  \u2022 {s}")


if __name__ == "__main__":
    main()