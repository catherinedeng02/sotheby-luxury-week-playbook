"""Generate interactive Plotly HTML for the website.

Reads the same CSVs the matplotlib pipeline reads, but emits interactive
Plotly figures with hover tooltips, zoom, and pan. Output HTML files are
self-contained and ready to be embedded in MkDocs markdown pages.

Outputs go to docs/assets/plots/ as:
    01_macro_heatmap.html
    01_macro_sellthrough.html
    02_birkin_coefficients.html
    03_corridor_paired.html
    04_daytona_coefficients.html
    05_patek_coefficients.html
    06_drc_coefficients.html
    07_whisky_coefficients.html
    08_sneakers_coefficients.html
    09_additional_combined.html  (for Section 7 widget — dropdown)

Run: python3 src/build_plotly.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats as scipy_stats

from config import PALETTE, RAW_DIR
from src.common import compute_premium, fit_premium_model, _humanize


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "docs" / "assets" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Shared Plotly layout settings — applied to every figure for consistency
# ---------------------------------------------------------------------------

# Cool ivory site palette
SITE_BG = "#EEEBE3"
INK = "#262626"
MIDTONE = "#666660"
ACCENT_GOLD = "#8B7B4F"
ACCENT_CLARET = "#722F37"
SOFT_GOLD = "#B89668"
RULE = "#D8D2C2"


def base_layout(title: str, height: int = 480) -> dict:
    """Default layout dict applied via .update_layout() on every figure."""
    return dict(
        title=dict(
            text=title,
            font=dict(family="EB Garamond, Times New Roman, serif",
                      size=18, color=INK),
            x=0.0, xanchor="left",
            pad=dict(t=10, b=20),
        ),
        font=dict(family="EB Garamond, Times New Roman, serif",
                  size=12, color=INK),
        paper_bgcolor=SITE_BG,
        plot_bgcolor="white",
        height=height,
        margin=dict(l=70, r=40, t=70, b=70),
        hoverlabel=dict(
            bgcolor="white",
            font=dict(family="EB Garamond, serif", size=12, color=INK),
            bordercolor=RULE,
        ),
        xaxis=dict(
            gridcolor=RULE, gridwidth=0.5,
            linecolor=MIDTONE, ticks="outside",
            tickfont=dict(size=10, color=MIDTONE),
        ),
        yaxis=dict(
            gridcolor=RULE, gridwidth=0.5,
            linecolor=MIDTONE, ticks="outside",
            tickfont=dict(size=10, color=MIDTONE),
        ),
        legend=dict(
            orientation="h", yanchor="top", y=-0.15,
            xanchor="center", x=0.5,
            font=dict(size=11, color=INK),
        ),
    )


def write_html(fig: go.Figure, name: str) -> None:
    """Write a Plotly figure as a self-contained HTML fragment.

    include_plotlyjs='cdn' keeps each file tiny (~5KB) by loading plotly.js
    from CDN at render time. full_html=False makes it embeddable as a
    fragment inside MkDocs pages.
    """
    out = PLOTS_DIR / f"{name}.html"
    fig.write_html(
        str(out),
        include_plotlyjs="cdn",
        full_html=True,
        config=dict(
            displayModeBar=True,
            modeBarButtonsToRemove=["lasso2d", "select2d"],
            displaylogo=False,
            toImageButtonOptions=dict(
                format="png", filename=name, scale=2,
            ),
        ),
    )
    print(f"  wrote {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Helper for coefficient plots (used by Sections II, IV.1-5)
# ---------------------------------------------------------------------------

def coefficient_figure(
    model, title: str,
    height: int = 520,
) -> go.Figure:
    """Build a horizontal coefficient bar chart from an OLSResults object.

    Coefficients sorted by magnitude. Gold = positive premium, claret =
    negative discount. Error bars show 95% CI. Hover shows coefficient,
    standard error, p-value, and the implied % premium.
    """
    params = model.params.drop("const")
    conf = model.conf_int().drop("const")
    se = model.bse.drop("const")
    pvals = model.pvalues.drop("const")

    # Sort by absolute magnitude descending; we want largest on top
    order = params.abs().sort_values(ascending=True).index
    params = params.loc[order]
    conf = conf.loc[order]
    se = se.loc[order]
    pvals = pvals.loc[order]

    labels = [_humanize(i) for i in params.index]
    colors = [ACCENT_GOLD if v > 0 else ACCENT_CLARET for v in params.values]

    # 95% CI half-widths for asymmetric error bars (here symmetric since OLS)
    err_lo = (params.values - conf[0].values)
    err_hi = (conf[1].values - params.values)

    # Implied premium % = exp(beta) - 1
    pct_implied = (np.exp(params.values) - 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=params.values,
        y=labels,
        orientation="h",
        marker=dict(color=colors, line=dict(color=INK, width=0.6)),
        error_x=dict(
            type="data", symmetric=False,
            array=err_hi, arrayminus=err_lo,
            color=INK, thickness=1, width=4,
        ),
        customdata=np.column_stack([
            params.values, se.values, pvals.values, pct_implied,
        ]),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Coefficient: %{customdata[0]:+.3f}<br>"
            "Std. error: %{customdata[1]:.3f}<br>"
            "p-value: %{customdata[2]:.3f}<br>"
            "Implied premium: %{customdata[3]:+.1f}%<br>"
            "<extra></extra>"
        ),
        showlegend=False,
    ))
    fig.add_vline(x=0, line=dict(color=INK, width=1))
    fig.update_layout(**base_layout(title, height=height))
    fig.update_layout(
        xaxis_title="Coefficient on log(Hammer / Mid-Estimate)",
        yaxis_title=None,
        bargap=0.25,
    )
    return fig


# ---------------------------------------------------------------------------
# Section I — Macro
# ---------------------------------------------------------------------------

def build_macro() -> None:
    print("\n[Section I] Macro figures")
    df = pd.read_csv(RAW_DIR / "macro.csv")
    df["sale_date"] = pd.to_datetime(df["sale_date"])

    df_sold = df[df["sold"]].copy()
    df_sold["mid_estimate"] = (df_sold["low_estimate"] + df_sold["high_estimate"]) / 2
    df_sold["premium_ratio"] = df_sold["hammer_price"] / df_sold["mid_estimate"]
    df["season_year"] = df["sale_date"].dt.year.astype(str) + " " + df["sale_date"].dt.month.map(
        lambda m: "Spring" if m <= 6 else "Autumn"
    )
    df_sold["season_year"] = df_sold["sale_date"].dt.year.astype(str) + " " + df_sold["sale_date"].dt.month.map(
        lambda m: "Spring" if m <= 6 else "Autumn"
    )

    # Chronological order of season columns
    def _key(s):
        year, season = s.split(" ")
        return (int(year), 0 if season == "Spring" else 1)

    # ---- Heatmap
    premium = (
        df_sold.groupby(["category", "season_year"])["premium_ratio"]
        .median().unstack()
    )
    premium = premium[sorted(premium.columns, key=_key)]
    years = sorted({int(s.split(" ")[0]) for s in premium.columns})
    year_range = f"{years[0]}\u2013{years[-1]}"

    # Build a diverging colorscale centered at 1.0 (premium parity)
    fig = go.Figure(data=go.Heatmap(
        z=premium.values,
        x=premium.columns,
        y=premium.index,
        colorscale=[
            [0.0, ACCENT_CLARET], [0.4, "#D8C9A8"],
            [0.5, "#F5F2EA"],
            [0.6, "#D8C9A8"], [1.0, ACCENT_GOLD],
        ],
        zmid=1.0,
        text=premium.values,
        texttemplate="%{text:.2f}",
        textfont=dict(family="EB Garamond, serif", size=11, color=INK),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "%{x}<br>"
            "Median premium: %{z:.2f}\u00d7<br>"
            "<extra></extra>"
        ),
        colorbar=dict(
            title=dict(text="Premium (\u00d7)", font=dict(size=10)),
            thickness=12, len=0.7, x=1.02,
            tickfont=dict(size=9),
        ),
        xgap=2, ygap=2,
    ))
    fig.update_layout(**base_layout(
        f"Median Premium Ratio by Category and Season ({year_range})",
        height=440,
    ))
    fig.update_layout(
        xaxis=dict(tickangle=-40, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
    )
    write_html(fig, "01_macro_heatmap")

    # ---- Sell-through line chart
    sellthrough = (
        df.groupby(["category", "season_year"])
        .agg(n=("lot_id", "count"), sold_n=("sold", "sum"))
    )
    sellthrough["sell_through"] = sellthrough["sold_n"] / sellthrough["n"]
    st = sellthrough["sell_through"].unstack().T
    st = st.reindex(sorted(st.index, key=_key))

    line_colors = [ACCENT_GOLD, ACCENT_CLARET, INK, SOFT_GOLD, MIDTONE]
    fig = go.Figure()
    for i, cat in enumerate(st.columns):
        fig.add_trace(go.Scatter(
            x=st.index, y=st[cat] * 100,
            mode="lines+markers",
            name=cat,
            line=dict(color=line_colors[i % len(line_colors)], width=1.8),
            marker=dict(size=6),
            hovertemplate=(
                f"<b>{cat}</b><br>"
                "%{x}<br>"
                "Sell-through: %{y:.1f}%<br>"
                "<extra></extra>"
            ),
        ))
    # Industry benchmark band 80-90% shaded
    fig.add_hrect(
        y0=80, y1=90,
        fillcolor=ACCENT_GOLD, opacity=0.06,
        layer="below", line_width=0,
    )
    fig.update_layout(**base_layout(
        f"Sell-Through Rate by Category Across Luxury Week Seasons ({year_range})",
        height=460,
    ))
    fig.update_layout(
        yaxis=dict(
            title="Sell-through rate (%)",
            range=[50, 105], ticksuffix="",
            tickvals=[50, 60, 70, 80, 90, 100],
        ),
        xaxis=dict(tickangle=-40, title="Sale season"),
    )
    write_html(fig, "01_macro_sellthrough")


# ---------------------------------------------------------------------------
# Section II — Hermès Birkin & Kelly
# ---------------------------------------------------------------------------

def build_birkin() -> None:
    print("\n[Section II] Hermès Birkin & Kelly")
    df = pd.read_csv(RAW_DIR / "handbags.csv")
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df = compute_premium(df)
    n = len(df)
    years = sorted(df["sale_date"].dt.year.dropna().unique().astype(int))
    year_range = f"{years[0]}\u2013{years[-1]}"

    features = ["leather", "hardware", "size_cm", "condition", "model"]
    ref_levels = {
        "leather": "Togo", "hardware": "GHW", "size_cm": 35,
        "condition": "Excellent", "model": "Birkin",
    }
    model = fit_premium_model(df, features, reference_levels=ref_levels)
    fig = coefficient_figure(
        model,
        f"Hermès Birkin & Kelly: Hammer Premium Drivers (n = {n}, {year_range})",
    )
    write_html(fig, "02_birkin_coefficients")


# ---------------------------------------------------------------------------
# Section III — HK-NY Corridor
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Section III — Asian buyer penetration (REAL DATA, not synthetic)
# ---------------------------------------------------------------------------

def build_buyer_penetration() -> None:
    """Asian-buyer share of Sotheby's buyers by luxury category.

    REAL data, sourced from Sotheby's December 2022 press release on the
    Buy Now Asia expansion. These are published figures, not synthesized,
    and are labelled as such in the chart and its on-page source line.
    Kept visually and structurally separate from the illustrative
    matched-pair scatter that follows it in Section III.
    """
    print("\n[Section III] Asian buyer penetration (real data)")

    # Published figures: Asian clients' share of buyers by category.
    # Source: Sotheby's press release, 2 December 2022.
    categories = [
        "Streetwear & Modern Collectables",
        "Handbags & Accessories",
        "Watches",
    ]
    shares = [57, 34, 33]

    # Colour the two categories relevant to this report's corridor in gold,
    # the streetwear bar (context only) in a muted tone.
    colors = [MIDTONE, ACCENT_GOLD, ACCENT_GOLD]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=shares,
        y=categories,
        orientation="h",
        marker=dict(color=colors, line=dict(color=INK, width=0.6)),
        text=[f"{s}%" for s in shares],
        textposition="outside",
        textfont=dict(family="Times New Roman, serif", size=13, color=INK),
        hovertemplate="<b>%{y}</b><br>Asian buyers: %{x}% of category<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(**base_layout(
        "Asian Clients' Share of Buyers, by Luxury Category",
        height=340,
    ))
    fig.update_layout(
        xaxis=dict(title="Share of category buyers (%)", range=[0, 70],
                   ticksuffix="%"),
        yaxis=dict(title=None, autorange="reversed"),
        bargap=0.4,
    )
    write_html(fig, "03_buyer_penetration")
def build_corridor() -> None:
    print("\n[Section III] HK-NY Corridor")
    df = pd.read_csv(RAW_DIR / "corridor.csv")
    df["hk_premium"] = df["hk_hammer"] / df["hk_mid_estimate"]
    df["ny_premium"] = df["ny_hammer"] / df["ny_mid_estimate"]
    df["log_ratio"] = np.log(df["hk_premium"] / df["ny_premium"])

    df["hk_sale_date"] = pd.to_datetime(df["hk_sale_date"], errors="coerce")
    df["ny_sale_date"] = pd.to_datetime(df["ny_sale_date"], errors="coerce")
    years = pd.concat([df["hk_sale_date"].dt.year, df["ny_sale_date"].dt.year]).dropna().astype(int)
    year_range = f"{years.min()}\u2013{years.max()}"

    cat_colors = {
        "Hermès Handbag": ACCENT_GOLD,
        "Vintage Rolex Daytona": ACCENT_CLARET,
        "Patek Philippe Calatrava": MIDTONE,
    }

    fig = go.Figure()
    for cat, color in cat_colors.items():
        sub = df[df["category"] == cat]
        fig.add_trace(go.Scatter(
            x=sub["ny_premium"], y=sub["hk_premium"],
            mode="markers",
            name=cat,
            marker=dict(color=color, size=10, line=dict(color="white", width=1)),
            customdata=np.column_stack([
                sub["reference_or_model"].values,
                sub["hk_premium"].values,
                sub["ny_premium"].values,
            ]),
            hovertemplate=(
                f"<b>{cat}</b><br>"
                "Ref/Model: %{customdata[0]}<br>"
                "HK premium: %{customdata[1]:.2f}\u00d7<br>"
                "NY premium: %{customdata[2]:.2f}\u00d7<br>"
                "<extra></extra>"
            ),
        ))

    pmin = max(0.3, min(df["hk_premium"].min(), df["ny_premium"].min()) * 0.9)
    pmax = min(8.0, max(df["hk_premium"].max(), df["ny_premium"].max()) * 1.1)
    fig.add_trace(go.Scatter(
        x=[pmin, pmax], y=[pmin, pmax],
        mode="lines", name="Equal premium (parity)",
        line=dict(color=INK, width=1, dash="dash"),
        hoverinfo="skip",
    ))

    fig.update_layout(**base_layout(
        f"HK\u2013NY Matched-Pair Premium Comparison (n = {len(df)}, {year_range})",
        height=560,
    ))
    fig.update_layout(
        xaxis=dict(
            title="New York premium (hammer / mid-estimate, log scale)",
            type="log", range=[np.log10(pmin), np.log10(pmax)],
        ),
        yaxis=dict(
            title="Hong Kong premium (hammer / mid-estimate, log scale)",
            type="log", range=[np.log10(pmin), np.log10(pmax)],
            scaleanchor="x", scaleratio=1,
        ),
    )
    write_html(fig, "03_corridor_paired")


# ---------------------------------------------------------------------------
# Section IV.1 — Daytona
# ---------------------------------------------------------------------------

def build_daytona() -> None:
    print("\n[Section IV.1] Daytona")
    df = pd.read_csv(RAW_DIR / "watches_daytona.csv")
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df = compute_premium(df)
    n = len(df)
    years = sorted(df["sale_date"].dt.year.dropna().unique().astype(int))
    year_range = f"{years[0]}\u2013{years[-1]}"

    features = ["reference", "dial_type", "bezel", "movement", "papers", "provenance"]
    ref_levels = {
        "reference": "6263", "dial_type": "Standard", "bezel": "Acrylic",
        "movement": "Valjoux 727", "papers": "None", "provenance": "Single owner",
    }
    model = fit_premium_model(df, features, reference_levels=ref_levels)
    fig = coefficient_figure(
        model,
        f"Vintage Rolex Daytona: Hammer Premium Drivers (n = {n}, {year_range})",
    )
    write_html(fig, "04_daytona_coefficients")


# ---------------------------------------------------------------------------
# Section IV.2 — Patek Calatrava
# ---------------------------------------------------------------------------

def build_patek() -> None:
    print("\n[Section IV.2] Patek Calatrava")
    df = pd.read_csv(RAW_DIR / "watches_patek.csv")
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df = compute_premium(df)
    n = len(df)
    years = sorted(df["sale_date"].dt.year.dropna().unique().astype(int))
    year_range = f"{years[0]}\u2013{years[-1]}"

    features = ["reference", "case_material", "dial_color",
                "extract_archives", "original_papers"]
    ref_levels = {
        "reference": "3444", "case_material": "Yellow gold",
        "dial_color": "Champagne",
        "extract_archives": "No", "original_papers": "No",
    }
    model = fit_premium_model(df, features, reference_levels=ref_levels)
    fig = coefficient_figure(
        model,
        f"Patek Philippe Calatrava: Hammer Premium Drivers (n = {n}, {year_range})",
    )
    write_html(fig, "05_patek_coefficients")


# ---------------------------------------------------------------------------
# Section IV.3 — DRC
# ---------------------------------------------------------------------------

GREAT_VINTAGES = {1990, 1999, 2005, 2009, 2015, 2018, 2019}

def build_drc() -> None:
    print("\n[Section IV.3] DRC Burgundy")
    df = pd.read_csv(RAW_DIR / "wine_drc.csv")
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df = compute_premium(df)
    df["vintage_tier"] = df["vintage"].apply(
        lambda v: "Great vintage" if int(v) in GREAT_VINTAGES else "Regular vintage"
    )
    n = len(df)
    years = sorted(df["sale_date"].dt.year.dropna().unique().astype(int))
    year_range = f"{years[0]}\u2013{years[-1]}"

    features = ["cuvee", "vintage_tier", "format", "source", "owc_intact"]
    ref_levels = {
        "cuvee": "Richebourg", "vintage_tier": "Regular vintage",
        "format": "750ml", "source": "Private cellar", "owc_intact": "No",
    }
    model = fit_premium_model(df, features, reference_levels=ref_levels)
    fig = coefficient_figure(
        model,
        f"DRC Burgundy: Hammer Premium Drivers (n = {n}, {year_range})",
    )
    write_html(fig, "06_drc_coefficients")


# ---------------------------------------------------------------------------
# Section IV.4 — Japanese Whisky
# ---------------------------------------------------------------------------

def build_whisky() -> None:
    print("\n[Section IV.4] Japanese Whisky")
    df = pd.read_csv(RAW_DIR / "whisky_japanese.csv")
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df = compute_premium(df)
    df["age_years"] = df["bottled_year"] - df["distilled_year"]
    df["age_tier"] = pd.cut(
        df["age_years"], bins=[0, 20, 30, 100],
        labels=["<20Y", "20-30Y", "30+Y"], include_lowest=True,
    ).astype(str)
    df["distill_era"] = df["distilled_year"].apply(
        lambda y: "Pre-1980 (golden era)" if int(y) < 1980 else "1980+"
    )
    n = len(df)
    years = sorted(df["sale_date"].dt.year.dropna().unique().astype(int))
    year_range = f"{years[0]}\u2013{years[-1]}"

    features = ["distillery", "cask_type", "label_series", "age_tier", "distill_era"]
    ref_levels = {
        "distillery": "Yamazaki", "cask_type": "Bourbon",
        "label_series": "Standard", "age_tier": "20-30Y",
        "distill_era": "1980+",
    }
    model = fit_premium_model(df, features, reference_levels=ref_levels)
    fig = coefficient_figure(
        model,
        f"Japanese Whisky: Hammer Premium Drivers (n = {n}, {year_range})",
    )
    write_html(fig, "07_whisky_coefficients")


# ---------------------------------------------------------------------------
# Section IV.5 — Sneakers
# ---------------------------------------------------------------------------

def _era_from_season(s: int) -> str:
    s = int(s)
    if s <= 1986:   return "MJ Rookie (1984\u20131986)"
    if s <= 1998:   return "Bulls Dynasty (1987\u20131998)"
    if s <= 2015:   return "Post-MJ (1999\u20132015)"
    return "Modern (2016+)"

def build_sneakers() -> None:
    print("\n[Section IV.5] Sneakers")
    df = pd.read_csv(RAW_DIR / "sneakers.csv")
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df = compute_premium(df)
    df["era"] = df["season"].apply(_era_from_season)
    df["game_worn"] = df["game_worn"].map({True: "Yes", False: "No", "True": "Yes", "False": "No"}).fillna("No")
    df["photo_matched"] = df["photo_matched"].map({True: "Yes", False: "No", "True": "Yes", "False": "No"}).fillna("No")
    df["signed"] = df["signed"].map({True: "Yes", False: "No", "True": "Yes", "False": "No"}).fillna("No")
    n = len(df)
    years = sorted(df["sale_date"].dt.year.dropna().unique().astype(int))
    year_range = f"{years[0]}\u2013{years[-1]}"

    features = ["player", "game_worn", "photo_matched", "signed", "condition", "era"]
    ref_levels = {
        "player": "Other", "game_worn": "No", "photo_matched": "No",
        "signed": "No", "condition": "Good",
        "era": "Post-MJ (1999\u20132015)",
    }
    model = fit_premium_model(df, features, reference_levels=ref_levels)
    fig = coefficient_figure(
        model,
        f"Game-Worn Sneakers: Hammer Premium Drivers (n = {n}, {year_range})",
    )
    write_html(fig, "08_sneakers_coefficients")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Generating Plotly HTML to {PLOTS_DIR}")
    build_macro()
    build_birkin()
    build_buyer_penetration()
    build_corridor()
    build_daytona()
    build_patek()
    build_drc()
    build_whisky()
    print(f"\nDone. Generated 9 Plotly HTML fragments in docs/assets/plots/")


if __name__ == "__main__":
    main()