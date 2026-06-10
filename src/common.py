"""Shared utilities: data loading, OLS regression, plotting, summary writing.

All analysis scripts import from here. Keeping shared logic in one place
means each category script (02..08) is short and focused on the features
that matter for that category.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

# Make project root importable when scripts are run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from config import PALETTE, FIGURES_DIR, SUMMARIES_DIR, RAW_DIR


# ---------------------------------------------------------------------------
# Pure-numpy OLS (no statsmodels dependency)
# ---------------------------------------------------------------------------

class OLSResults:
    """Minimal OLS results object, exposing the subset of the statsmodels
    Results API the rest of the codebase uses: ``params``, ``conf_int()``,
    ``rsquared``, ``rsquared_adj``, ``bse``, ``tvalues``, ``pvalues``,
    ``nobs``, ``summary()``.

    Math is textbook OLS:
        beta_hat = (X'X)^(-1) X'y
        sigma2   = RSS / (n - k)
        Var(b)   = sigma2 * (X'X)^(-1)
        SE(b_i)  = sqrt(diag(Var(b))_i)
        CI 95%   = b_i +/- t_{0.025, n-k} * SE(b_i)
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series):
        # Drop rows with any NaN in X or y (mirrors statsmodels missing='drop')
        mask = (~X.isna().any(axis=1)) & (~y.isna())
        X_clean = X.loc[mask].astype(float)
        y_clean = y.loc[mask].astype(float)

        self._feature_names = list(X_clean.columns)
        X_arr = X_clean.to_numpy()
        y_arr = y_clean.to_numpy()
        n, k = X_arr.shape

        # Solve normal equations with lstsq for numerical stability
        beta_hat, *_ = np.linalg.lstsq(X_arr, y_arr, rcond=None)
        y_hat = X_arr @ beta_hat
        resid = y_arr - y_hat
        rss = float(resid @ resid)
        dof = max(n - k, 1)
        sigma2 = rss / dof

        # Inverse of X'X (pseudo-inverse for safety against rank deficiency)
        xtx_inv = np.linalg.pinv(X_arr.T @ X_arr)
        var_beta = sigma2 * xtx_inv
        se = np.sqrt(np.maximum(np.diag(var_beta), 0.0))

        # R^2 and adjusted R^2
        y_mean = y_arr.mean() if len(y_arr) else 0.0
        tss = float(((y_arr - y_mean) ** 2).sum())
        r2 = 1.0 - rss / tss if tss > 0 else 0.0
        # Standard adjusted R^2 (k includes the intercept term)
        if n - k > 0 and tss > 0:
            adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k)
        else:
            adj_r2 = r2

        # t-values, p-values (two-sided), 95% CI
        with np.errstate(divide="ignore", invalid="ignore"):
            tvals = np.where(se > 0, beta_hat / se, 0.0)
        t_crit = scipy_stats.t.ppf(0.975, df=dof) if dof > 0 else 1.96
        ci_low = beta_hat - t_crit * se
        ci_high = beta_hat + t_crit * se
        pvals = 2 * (1 - scipy_stats.t.cdf(np.abs(tvals), df=dof))

        # Expose statsmodels-shaped fields
        self.params = pd.Series(beta_hat, index=self._feature_names)
        self.bse = pd.Series(se, index=self._feature_names)
        self.tvalues = pd.Series(tvals, index=self._feature_names)
        self.pvalues = pd.Series(pvals, index=self._feature_names)
        self._ci = pd.DataFrame(
            {0: ci_low, 1: ci_high}, index=self._feature_names
        )
        self.rsquared = float(r2)
        self.rsquared_adj = float(adj_r2)
        self.nobs = int(n)
        self.df_resid = int(dof)
        self.resid = pd.Series(resid, index=y_clean.index)
        self.fittedvalues = pd.Series(y_hat, index=y_clean.index)

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        """Return 95% CI as a DataFrame with columns 0 (lower) and 1 (upper).

        The ``alpha`` argument is accepted for API parity but only the default
        0.05 (i.e. 95% CI) is computed; this matches what the rest of the
        codebase uses.
        """
        if not np.isclose(alpha, 0.05):
            # Recompute on demand for other alphas
            dof = self.df_resid
            t_crit = scipy_stats.t.ppf(1 - alpha / 2, df=dof) if dof > 0 else 1.96
            lo = self.params - t_crit * self.bse
            hi = self.params + t_crit * self.bse
            return pd.DataFrame({0: lo, 1: hi})
        return self._ci.copy()

    def summary(self) -> str:
        """Compact text summary, mimicking the statsmodels summary header."""
        lines = [
            "OLS Regression Results",
            "=" * 60,
            f"  Observations: {self.nobs}",
            f"  R-squared:    {self.rsquared:.3f}",
            f"  Adj R-sq:     {self.rsquared_adj:.3f}",
            f"  Df residual:  {self.df_resid}",
            "-" * 60,
            f"  {'feature':<32} {'coef':>8} {'se':>8} {'t':>6} {'p':>6}",
            "-" * 60,
        ]
        for name in self._feature_names:
            lines.append(
                f"  {str(name)[:32]:<32} "
                f"{self.params[name]:>8.3f} "
                f"{self.bse[name]:>8.3f} "
                f"{self.tvalues[name]:>6.2f} "
                f"{self.pvalues[name]:>6.3f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


def _add_constant(X: pd.DataFrame) -> pd.DataFrame:
    """Prepend a 'const' column of 1s, matching statsmodels.add_constant."""
    out = X.copy()
    out.insert(0, "const", 1.0)
    return out


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

def apply_style() -> None:
    """Classical serif plot style. Call once at the top of each script."""
    mpl.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 110,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "font.serif": [
            "Garamond", "EB Garamond", "Times New Roman", "Times",
            "DejaVu Serif",
        ],
        "axes.edgecolor": PALETTE["midtone"],
        "axes.labelcolor": PALETTE["ink"],
        "axes.titlecolor": PALETTE["ink"],
        "axes.titleweight": "semibold",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": PALETTE["midtone"],
        "ytick.color": PALETTE["midtone"],
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "grid.color": "#E5E5E5",
        "grid.linewidth": 0.6,
        "axes.grid": True,
        "axes.axisbelow": True,
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(name: str) -> pd.DataFrame:
    """Load data/raw/<name>.csv. Raise a friendly error if missing."""
    path = RAW_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing data file: {path}\n"
            f"Run `python3 src/synthesize.py` to generate test data, or "
            f"populate this file with real Sotheby's lot data."
        )
    df = pd.read_csv(path)
    # Common normalization
    if "sale_date" in df.columns:
        df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Premium calculations
# ---------------------------------------------------------------------------

def compute_premium(df: pd.DataFrame) -> pd.DataFrame:
    """Add mid_estimate, premium_ratio, log_premium. Drops unsold lots.

    Premium ratio = hammer / mid-estimate, where mid = (low + high) / 2.
    Log-premium is used as the regression target so coefficients are
    symmetric above/below the estimate (a +20% beat and a -20% miss read
    as equal magnitudes).
    """
    df = df.copy()
    df = df[df["hammer_price"].notna() & (df["hammer_price"] > 0)].copy()
    df["mid_estimate"] = (df["low_estimate"] + df["high_estimate"]) / 2.0
    df["premium_ratio"] = df["hammer_price"] / df["mid_estimate"]
    df["log_premium"] = np.log(df["premium_ratio"])
    return df


def fit_premium_model(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str = "log_premium",
    reference_levels: dict | None = None,
):
    """Fit OLS on log-premium with one-hot encoded categoricals.

    Args:
        df: dataframe with features + target columns
        features: list of column names to use as predictors
        target: name of the target column (default log_premium)
        reference_levels: optional dict {feature_name: reference_value}
            specifying which level should be the dropped reference for
            each categorical. If a feature isn't in this dict, the
            alphabetically-first level is used (pandas default).

    Returns an :class:`OLSResults` object exposing ``params``, ``conf_int()``,
    ``rsquared``, ``rsquared_adj``, etc. Coefficients on dummy variables read
    as "log-premium gap vs the reference level".
    """
    reference_levels = reference_levels or {}
    X_raw = df[list(features)].copy()
    # For each column, decide if it should be treated as categorical.
    # Treat as categorical when: explicitly listed in reference_levels,
    # or dtype is object/string/categorical (i.e. anything non-numeric).
    for c in X_raw.columns:
        is_string_like = (
            X_raw[c].dtype == "object"
            or pd.api.types.is_string_dtype(X_raw[c])
            or isinstance(X_raw[c].dtype, pd.CategoricalDtype)
        )
        if c in reference_levels or is_string_like:
            ref = reference_levels.get(c)
            # Coerce to str so int-coded categories (e.g. reference "6263")
            # become string categories rather than numeric predictors.
            col = X_raw[c].astype(str)
            unique_vals = sorted(col.dropna().unique().tolist())
            if ref is not None and str(ref) in unique_vals:
                ordered = [str(ref)] + [v for v in unique_vals if v != str(ref)]
            else:
                ordered = unique_vals
            X_raw[c] = pd.Categorical(col, categories=ordered, ordered=False)
    X = pd.get_dummies(X_raw, drop_first=True, dtype=float)
    X = _add_constant(X)
    y = df[target]
    return OLSResults(X, y)

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_coefficients(
    model,
    title: str,
    output: Path,
    top_n: int | None = None,
) -> None:
    """Horizontal bar chart of regression coefficients with 95% CI.

    Coefficients are sorted by absolute magnitude. Gold = positive premium,
    claret = negative (discount). The reference category is implicit in
    the const term (not plotted).
    """
    params = model.params.copy()
    conf = model.conf_int().copy()
    if "const" in params.index:
        params = params.drop("const")
        conf = conf.drop("const")

    order = params.abs().sort_values(ascending=True).index
    params = params.loc[order]
    conf = conf.loc[order]

    if top_n is not None and len(params) > top_n:
        keep = params.abs().sort_values(ascending=False).index[:top_n]
        # preserve ascending-magnitude order within the kept subset
        order = [i for i in order if i in keep]
        params = params.loc[order]
        conf = conf.loc[order]

    fig, ax = plt.subplots(figsize=(9, max(4, len(params) * 0.36)))
    y = np.arange(len(params))
    err = np.array([
        (params.values - conf[0].values),
        (conf[1].values - params.values),
    ])
    colors = [PALETTE["gold"] if v > 0 else PALETTE["claret"]
              for v in params.values]

    ax.barh(y, params.values, color=colors,
            edgecolor=PALETTE["ink"], linewidth=0.6, alpha=0.92)
    ax.errorbar(params.values, y, xerr=err, fmt="none",
                ecolor=PALETTE["ink"], capsize=2.5, elinewidth=0.8)
    ax.axvline(0, color=PALETTE["ink"], linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([_humanize(i) for i in params.index], fontsize=9)
    ax.set_xlabel("Coefficient on log(Hammer / Mid-Estimate)")
    ax.set_title(title, loc="left")
    ax.grid(False, axis="y")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_combo_table(
    df: pd.DataFrame,
    title: str,
    output: Path,
) -> None:
    """Render a 'winning combinations' table as a figure for inclusion in the PDF."""
    fig, ax = plt.subplots(figsize=(9, max(2.5, 0.32 * (len(df) + 2))))
    ax.axis("off")
    cell = []
    for _, row in df.iterrows():
        line = []
        for col in df.columns:
            v = row[col]
            if isinstance(v, float):
                if col in ("mean", "median"):
                    line.append(f"{v:.2f}x")
                elif col == "count":
                    line.append(f"{int(v)}")
                else:
                    line.append(f"{v:.3f}")
            else:
                line.append(str(v))
        cell.append(line)

    tbl = ax.table(
        cellText=cell,
        colLabels=[_humanize(c) for c in df.columns],
        cellLoc="left",
        loc="upper left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.4)
    # Header styling
    for (r, c), cellobj in tbl.get_celld().items():
        cellobj.set_edgecolor(PALETTE["midtone"])
        cellobj.set_linewidth(0.4)
        if r == 0:
            cellobj.set_facecolor(PALETTE["parchment"])
            cellobj.set_text_props(weight="semibold", color=PALETTE["ink"])

    ax.set_title(title, loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def winning_combo_table(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    min_n: int = 3,
    top_k: int = 8,
) -> pd.DataFrame:
    """Group on feature combinations, return top-k by median premium.

    Filters out combinations with fewer than ``min_n`` observations so we
    don't headline a 'winning combo' that's really a single fluke lot.
    """
    g = (
        df.groupby(list(feature_cols))["premium_ratio"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    g = g[g["count"] >= min_n].sort_values("median", ascending=False)
    return g.head(top_k).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summary writing
# ---------------------------------------------------------------------------

def write_summary(name: str, payload: dict) -> None:
    """Write a JSON summary consumed by 09_report.py to build the PDF."""
    path = SUMMARIES_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def takeaway_pct(model, feature_name: str) -> float | None:
    """Return % premium implied by coefficient on a given dummy (exp(beta)-1)*100."""
    if feature_name not in model.params.index:
        return None
    return (np.exp(model.params[feature_name]) - 1) * 100


def top_driver_sentences(model, n: int = 5, pct_threshold: float = 5.0) -> list[str]:
    """Return sentences describing the n largest-magnitude coefficients.

    Skips coefficients with abs(%) below pct_threshold. Robust fallback
    when hardcoded labels (e.g. 'leather_Himalaya Niloticus') don't match
    the actual reference category chosen by the model.
    """
    params = model.params.copy()
    if "const" in params.index:
        params = params.drop("const")
    pct = (np.exp(params) - 1) * 100
    ordered = pct.reindex(pct.abs().sort_values(ascending=False).index)
    out = []
    for label, value in ordered.items():
        if abs(value) < pct_threshold:
            continue
        pretty = _humanize(label)
        out.append(f"{pretty}: {value:+.0f}% vs reference, all else equal.")
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _humanize(label) -> str:
    """Turn pandas dummy column names like 'leather_Niloticus' into 'Leather: Niloticus'."""
    s = str(label)
    if "_" in s:
        head, _, tail = s.partition("_")
        return f"{head.title()}: {tail}"
    return s
# ---------------------------------------------------------------------------
# Figure caption and source line (Sotheby's research-report convention)
# ---------------------------------------------------------------------------

def add_caption_and_source(
    fig,
    note: str | None = None,
    source: str = "Source: Sotheby's Past Auctions (sothebys.com); author's analysis.",
    note_y: float = -0.08,
    source_y: float = -0.14,
) -> None:
    """Add a left-aligned italic note + source line beneath a figure.

    Convention:
      - ``note`` (optional): methodology nuance, percentile clip, exclusions.
        Rendered in italic at ``note_y`` below the axes.
      - ``source``: data provenance. Always present. Rendered at ``source_y``
        in a slightly darker tone to anchor the figure as a citable artifact.

    Both are left-aligned (x=0) and use figure-relative coordinates so
    ``bbox_inches='tight'`` keeps them visible when saving.

    Standard pattern at the end of every figure script:

        add_caption_and_source(
            fig,
            note="Color scale clipped at 5th/95th percentile.",
        )
        fig.savefig(..., bbox_inches="tight")
    """
    if note:
        fig.text(
            0.0, note_y,
            f"Note: {note}",
            ha="left", va="top",
            fontsize=8, color=PALETTE["midtone"], style="italic",
            transform=fig.transFigure,
        )
    fig.text(
        0.0, source_y,
        source,
        ha="left", va="top",
        fontsize=8, color=PALETTE["ink"],
        transform=fig.transFigure,
    )