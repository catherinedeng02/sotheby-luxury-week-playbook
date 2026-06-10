"""Pillar 1 — Macro: cross-category Luxury Week performance, 2021–2026.

Reads:   data/raw/macro.csv
Schema:  lot_id, sale_date, sale_season, sale_year, category,
         low_estimate, high_estimate, hammer_price, sold

Outputs:
  output/figures/01_macro_heatmap.png       category × season median premium
  output/figures/01_macro_sellthrough.png   sell-through over time
  output/summaries/01_macro.json            key statistics
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import FIGURES_DIR, PALETTE
from src.common import apply_style, load_csv, write_summary, add_caption_and_source


def main() -> None:
    apply_style()
    df = load_csv("macro")

    # ---- Compute per-(category, season-year) metrics
    df_sold = df[df["sold"]].copy()
    df_sold["mid_estimate"] = (df_sold["low_estimate"] + df_sold["high_estimate"]) / 2
    df_sold["premium_ratio"] = df_sold["hammer_price"] / df_sold["mid_estimate"]
    df_sold["above_high"] = df_sold["hammer_price"] > df_sold["high_estimate"]
    df_sold["top_decile"] = df_sold["premium_ratio"] > 2.0  # 200%+ of mid

    df["season_year"] = df["sale_year"].astype(str) + " " + df["sale_season"]
    df_sold["season_year"] = df_sold["sale_year"].astype(str) + " " + df_sold["sale_season"]

    # Sell-through by category × season
    sellthrough = (
        df.groupby(["category", "season_year"])
          .agg(n=("lot_id", "count"), sold_n=("sold", "sum"))
    )
    sellthrough["sell_through"] = sellthrough["sold_n"] / sellthrough["n"]

    # Median premium ratio by category × season
    premium = (
        df_sold.groupby(["category", "season_year"])["premium_ratio"]
        .median()
        .unstack()
    )

    # Top-decile share
    top_share = (
        df_sold.groupby(["category", "season_year"])["top_decile"]
        .mean()
        .unstack()
    )

    # Order columns chronologically (Spring before Autumn within each year)
    def _key(s):
        year, season = s.split(" ")
        return (int(year), 0 if season == "Spring" else 1)
    premium = premium[sorted(premium.columns, key=_key)]
    top_share = top_share[sorted(top_share.columns, key=_key)]

   # ---- Figure 1: heatmap of median premium ratio
    # Diverging gold/claret colormap centered on 1.0x (the mid-estimate
    # benchmark): claret = lots underperformed estimates, gold = lots beat
    # estimates. This makes the business read of the figure immediate.
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    diverging_cmap = LinearSegmentedColormap.from_list(
        "claret_gold",
        [PALETTE["claret"], PALETTE["soft_claret"], PALETTE["parchment"],
         PALETTE["soft_gold"], PALETTE["gold"]],
        N=256,
    )
    # Clip the color scale at the 5th/95th percentile of data, then expand
    # symmetrically around the 1.0x benchmark. This keeps a couple of
    # extreme outliers from washing out the mid-range contrast that's
    # actually meaningful for category-by-season storytelling.
    vlo = float(np.nanpercentile(premium.values, 5))
    vhi = float(np.nanpercentile(premium.values, 95))
    half_range = max(1.0 - vlo, vhi - 1.0)
    vmin = 1.0 - half_range
    vmax = 1.0 + half_range
    norm = TwoSlopeNorm(vcenter=1.0, vmin=vmin, vmax=vmax)

    # Compute year range for the title from the actual data
    years = sorted({int(s.split(" ")[0]) for s in premium.columns})
    year_range = f"{years[0]}\u2013{years[-1]}"

    # Wider, slightly shorter figure so the heatmap body has visual weight
    # and isn't crowded by labels on either side.
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    sns.heatmap(
        premium,
        annot=True, fmt=".2f",
        cmap=diverging_cmap, norm=norm,
        # Slim colorbar: shrink=0.65 makes it ~65% of the axis height;
        # aspect=18 makes it thin; pad=0.02 brings it closer to the heatmap.
        cbar_kws={
            "label": "Premium (×)",
            "shrink": 0.65,
            "aspect": 18,
            "pad": 0.02,
        },
        linewidths=0.6, linecolor="white",
        annot_kws={"fontsize": 9, "color": PALETTE["ink"]},
        ax=ax,
    )
    # Style the colorbar label to be more compact and informative
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Premium (×)", fontsize=9, labelpad=6)

    ax.set_title(
        f"Median Premium Ratio by Category and Season ({year_range})",
        loc="left", pad=12,
    )
    ax.set_xlabel("Sale Season", fontsize=10, labelpad=8)
    ax.set_ylabel("Category", fontsize=10, labelpad=8)
    ax.tick_params(axis="both", labelsize=9)
    plt.xticks(rotation=40, ha="right")
    plt.yticks(rotation=0)

    # Reserve bottom margin for caption + source instead of letting
    # tight_layout push the heatmap to the top. Values are figure-relative:
    # bottom=0.22 means heatmap body occupies y=0.22 to ~0.93, leaving the
    # bottom 22% for x-tick labels + caption + source.
    fig.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.28)
    add_caption_and_source(
        fig,
        note=(
            "1.0\u00d7 = hammer equal to mid-estimate. "
            "Color scale clipped at 5th/95th percentile to preserve "
            "mid-range contrast. Unsold lots excluded from premium calculation."
        ),
        note_y=0.08,
        source_y=0.03,
    )
    fig.savefig(FIGURES_DIR / "01_macro_heatmap.png")
    plt.close(fig)

    # ---- Figure 2: sell-through line chart
    st = sellthrough["sell_through"].unstack().T
    st = st.reindex(sorted(st.index, key=_key))

    # Cohesive palette: gold variants + ink + claret for category lines
    line_palette = [
        PALETTE["gold"], PALETTE["claret"], PALETTE["ink"],
        PALETTE["soft_gold"], PALETTE["soft_claret"], PALETTE["midtone"],
        PALETTE["stone"],
    ]

    # Compute year range from the actual data so the title stays accurate
    # if you re-run the analysis with different date filters.
    years = sorted({int(s.split(" ")[0]) for s in st.index})
    year_range = f"{years[0]}\u2013{years[-1]}"

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for i, cat in enumerate(st.columns):
        ax.plot(st.index, st[cat], marker="o", linewidth=1.6,
                color=line_palette[i % len(line_palette)],
                markersize=4, label=cat)
    # Y-axis: format as percentage, leave headroom above 1.0 so the top
    # markers aren't clipped by the spine.
    ax.set_ylim(0.5, 1.05)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%")
    )
    ax.set_ylabel("Sell-through Rate (%)", fontsize=10)
    ax.set_xlabel("Sale Season", fontsize=10)
    ax.set_title(
        f"Sell-Through Rate by Category Across Luxury Week Seasons ({year_range})",
        loc="left",
    )
    ax.tick_params(axis="both", labelsize=9)
    plt.xticks(rotation=40, ha="right")

    # Move legend below the plot, horizontal, no frame — standard research-
    # report convention. bbox_to_anchor positions it just below the x-axis
    # label; ncol spreads categories across one or two rows depending on count.
    n_cats = len(st.columns)
    legend_ncol = min(n_cats, 5)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=legend_ncol,
        fontsize=9,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.6,
    )

    # Reserve bottom margin for legend + caption + source. Legend is
    # anchored to the axes at y=-0.22, so we need ~35% bottom space.
    fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.42)
    add_caption_and_source(
        fig,
        note=(
            "Sell-through rate = lots sold / lots offered, "
            "computed per category per season. "
            "Industry health benchmark: 80\u201390%; white-glove sale: 100%."
        ),
        note_y=0.10,
        source_y=0.05,
    )
    fig.savefig(FIGURES_DIR / "01_macro_sellthrough.png")
    plt.close(fig)

    # ---- Summary stats
    overall_sellthrough = float(df["sold"].mean())
    overall_median_premium = float(df_sold["premium_ratio"].median())

    # Best/worst categories
    cat_premium = df_sold.groupby("category")["premium_ratio"].median().sort_values(ascending=False)
    cat_sellthrough = df.groupby("category")["sold"].mean().sort_values(ascending=False)

    # If the same category leads on both premium and sell-through, mention
    # the runner-up for sell-through so the sentence stays informative.
    if cat_premium.index[0] == cat_sellthrough.index[0] and len(cat_sellthrough) > 1:
        st_leader = cat_sellthrough.index[1]
        st_value = float(cat_sellthrough.iloc[1])
        st_clause = (
            f"{cat_premium.index[0]} also leads on sell-through "
            f"({cat_sellthrough.iloc[0]:.1%}), with "
            f"{st_leader} the next-strongest at {st_value:.1%}"
        )
    else:
        st_leader = cat_sellthrough.index[0]
        st_value = float(cat_sellthrough.iloc[0])
        st_clause = (
            f"{st_leader} led on sell-through ({st_value:.1%})"
        )

    payload = {
        "title": "Macro — Cross-Category Luxury Week Performance",
        "n_lots_total": int(len(df)),
        "n_lots_sold": int(df["sold"].sum()),
        "overall_sell_through": round(overall_sellthrough, 3),
        "overall_median_premium": round(overall_median_premium, 3),
        "best_premium_category": cat_premium.index[0],
        "best_premium_value": round(float(cat_premium.iloc[0]), 3),
        "best_sellthrough_category": cat_sellthrough.index[0],
        "best_sellthrough_value": round(float(cat_sellthrough.iloc[0]), 3),
        "category_median_premium": {k: round(float(v), 3) for k, v in cat_premium.items()},
        "category_sell_through": {k: round(float(v), 3) for k, v in cat_sellthrough.items()},
        "figures": ["01_macro_heatmap.png", "01_macro_sellthrough.png"],
        "takeaway": (
            f"{cat_premium.index[0]} achieved the highest median premium "
            f"({cat_premium.iloc[0]:.2f}x mid-estimate) across the study "
            f"window; {st_clause}."
        ),
    }
    write_summary("01_macro", payload)
    print(payload["takeaway"])


if __name__ == "__main__":
    main()