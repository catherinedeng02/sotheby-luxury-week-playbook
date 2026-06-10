"""Pillar 3 — Buyer Corridor: HK vs NY premium differential for matched lots.

For each comparable pair (same reference / leather, both sold within a
30% mid-estimate band), compute premium ratio in HK vs NY. The strategic
question: which venue should Sotheby's route a given consignment to,
by category?

Reads:   data/raw/corridor.csv
Schema:  pair_id, reference_or_model, category, hk_lot_id, hk_hammer,
         hk_mid_estimate, hk_sale_date, ny_lot_id, ny_hammer,
         ny_mid_estimate, ny_sale_date

Outputs:
  output/figures/03_corridor_paired.png       paired log-log scatter
  output/figures/03_corridor_distribution.png boxplot by category + t-test
  output/summaries/03_corridor.json           routing recommendations
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from config import FIGURES_DIR, PALETTE
from src.common import apply_style, load_csv, write_summary, add_caption_and_source


def main() -> None:
    apply_style()
    df = load_csv("corridor")
    print(f"Loaded {len(df)} HK-NY paired observations")

    # ---- Premium ratios and log-ratio (the test statistic)
    df["hk_premium"] = df["hk_hammer"] / df["hk_mid_estimate"]
    df["ny_premium"] = df["ny_hammer"] / df["ny_mid_estimate"]
    df["log_ratio"] = np.log(df["hk_premium"] / df["ny_premium"])
    # log_ratio > 0  →  HK paid more
    # log_ratio < 0  →  NY paid more
    # log_ratio = 0  →  parity

    # Compute year range from the actual data for the title
    df["hk_sale_date"] = pd.to_datetime(df["hk_sale_date"], errors="coerce")
    df["ny_sale_date"] = pd.to_datetime(df["ny_sale_date"], errors="coerce")
    years = pd.concat([df["hk_sale_date"].dt.year,
                       df["ny_sale_date"].dt.year]).dropna().astype(int)
    year_range = f"{years.min()}\u2013{years.max()}" if len(years) else ""

    # ---- Aggregate by category (the table behind routing recommendations)
    by_cat = df.groupby("category").agg(
        n_pairs=("pair_id", "count"),
        hk_median=("hk_premium", "median"),
        ny_median=("ny_premium", "median"),
        median_log_ratio=("log_ratio", "median"),
    )
    # exp(median_log_ratio) - 1 = HK's median percentage premium over NY
    by_cat["hk_advantage_pct"] = (np.exp(by_cat["median_log_ratio"]) - 1) * 100
    print("\nBy-category corridor stats:")
    print(by_cat.round(3))

    # ---- Paired t-test on log ratio (H0: HK premium = NY premium overall)
    # Using log_ratio (not raw difference) keeps the test symmetric and
    # the t-statistic interpretable as "is the log-ratio distribution
    # centered on 0?" — the cleanest test of corridor neutrality.
    t_stat, p_val = stats.ttest_1samp(df["log_ratio"].dropna(), 0)

    # ----------------------------------------------------------------
    # Figure 1: matched-pair scatter, log-log scale
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    cats = df["category"].unique()
    cat_colors = [PALETTE["gold"], PALETTE["claret"], PALETTE["midtone"]]
    for cat, col in zip(cats, cat_colors):
        sub = df[df["category"] == cat]
        ax.scatter(
            sub["ny_premium"], sub["hk_premium"],
            alpha=0.75, s=46, color=col, edgecolor="white", linewidth=0.7,
            label=cat,
        )

    # 45° parity line: HK premium = NY premium
    pmin = max(0.3, min(df["hk_premium"].min(), df["ny_premium"].min()) * 0.9)
    pmax = min(8.0, max(df["hk_premium"].max(), df["ny_premium"].max()) * 1.1)
    ax.plot([pmin, pmax], [pmin, pmax],
            color=PALETTE["ink"], linewidth=0.9, linestyle="--",
            label="Equal premium (parity)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(pmin, pmax)
    ax.set_ylim(pmin, pmax)
    ax.set_aspect("equal")
    ax.set_xlabel("New York Premium (Hammer / Mid-Estimate, log scale)",
                  fontsize=10, labelpad=8)
    ax.set_ylabel("Hong Kong Premium (Hammer / Mid-Estimate, log scale)",
                  fontsize=10, labelpad=8)
    ax.set_title(
        f"HK\u2013NY Matched-Pair Premium Comparison "
        f"(n = {len(df)}, {year_range})",
        loc="left", pad=12,
    )
    # Legend below the plot, horizontal — same convention as macro chart
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4, fontsize=9, frameon=False,
        handlelength=2.0, columnspacing=1.6,
    )
    fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.30)
    add_caption_and_source(
        fig,
        note=(
            "Each point is a matched pair: the same reference or model "
            "sold in both HK and NY within a 30% mid-estimate band. "
            "Points above the dashed line indicate HK paid more; "
            "points below indicate NY paid more."
        ),
        note_y=0.09,
        source_y=0.04,
    )
    fig.savefig(FIGURES_DIR / "03_corridor_paired.png")
    plt.close(fig)

    # ----------------------------------------------------------------
    # Figure 2: distribution of log ratios by category, boxplot
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    cats_list = list(df["category"].unique())
    data_by_cat = [
        df[df["category"] == c]["log_ratio"].dropna().values
        for c in cats_list
    ]
    bp = ax.boxplot(
        data_by_cat,
        tick_labels=cats_list,
        vert=True, patch_artist=True, widths=0.5,
        medianprops=dict(color=PALETTE["ink"], linewidth=1.5),
        whiskerprops=dict(color=PALETTE["midtone"]),
        capprops=dict(color=PALETTE["midtone"]),
        flierprops=dict(marker="o", markerfacecolor=PALETTE["stone"],
                        markeredgecolor="none", markersize=4, alpha=0.5),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(PALETTE["soft_gold"])
        patch.set_edgecolor(PALETTE["midtone"])

    # Reference line at 0 = no HK/NY differential
    ax.axhline(0, color=PALETTE["ink"], linewidth=0.8, linestyle="--")

    ax.set_ylabel("log(HK Premium / NY Premium)", fontsize=10, labelpad=8)
    ax.set_xlabel("Category", fontsize=10, labelpad=8)
    ax.set_title(
        f"HK\u2013NY Premium Differential by Category "
        f"(Paired t-test: t = {t_stat:.2f}, p = {p_val:.3f})",
        loc="left", pad=12,
    )
    ax.tick_params(axis="both", labelsize=9)

    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.30)
    add_caption_and_source(
        fig,
        note=(
            "Boxes show median and interquartile range of paired log-ratios "
            "(log of HK premium divided by NY premium) for each category. "
            "Values above zero favor Hong Kong as the routing venue; "
            "values below favor New York."
        ),
        note_y=0.10,
        source_y=0.04,
    )
    fig.savefig(FIGURES_DIR / "03_corridor_distribution.png")
    plt.close(fig)

    # ---- Routing recommendations (the deliverable)
    # Threshold of ±5% chosen as the smallest premium differential
    # consistent with the cost of cross-Pacific freight, insurance, and
    # additional specialist time. Below 5%, "near parity" is the
    # appropriate read.
    recommendations = []
    for cat, row in by_cat.iterrows():
        adv = row["hk_advantage_pct"]
        if adv > 5:
            rec = f"Route {cat} to Hong Kong (+{adv:.1f}% median premium vs NY)"
        elif adv < -5:
            rec = f"Route {cat} to New York ({adv:.1f}% HK disadvantage)"
        else:
            rec = f"{cat}: near-parity — route on logistics / consignor preference"
        recommendations.append(rec)

    payload = {
        "title": "HK\u2013NY Buyer Corridor",
        "n_pairs": int(len(df)),
        "overall_paired_t": round(float(t_stat), 3),
        "overall_p_value": round(float(p_val), 4),
        "by_category": by_cat.round(3).reset_index().to_dict(orient="records"),
        "routing_recommendations": recommendations,
        "figures": ["03_corridor_paired.png", "03_corridor_distribution.png"],
        "headline_takeaway": recommendations[0] if recommendations else "",
    }
    write_summary("03_corridor", payload)
    print("\nRouting recommendations:")
    for r in recommendations:
        print(f"  \u2022 {r}")


if __name__ == "__main__":
    main()