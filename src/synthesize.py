"""Generate synthetic but plausibly-distributed lot data for all eight
analyses. The synthetic distributions encode real-world market intuition
(Himalaya > Niloticus > Togo; Paul Newman dial >> standard dial; etc) so
the pipeline produces sensible-looking results even before you swap in
real Sotheby's data.

Outputs (into data/raw/):
    handbags.csv          - for 02_birkin.py
    watches_daytona.csv   - for 04_daytona.py
    watches_patek.csv     - for 05_patek.py
    wine_drc.csv          - for 06_drc.py
    whisky_japanese.csv   - for 07_whisky.py
    sneakers.csv          - for 08_sneakers.py
    macro.csv             - for 01_macro.py (derived: aggregates several)
    corridor.csv          - for 03_corridor.py (HK-NY matched pairs)
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import RAW_DIR, RANDOM_SEED


rng = np.random.default_rng(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _premium_from_log(n: int, log_mean: float, log_sd: float = 0.18) -> np.ndarray:
    """Draw n premium ratios from lognormal with given log-mean and log-sd."""
    return np.exp(rng.normal(loc=log_mean, scale=log_sd, size=n))


def _estimates(n: int, low_base: float = 8000, high_base: float = 14000,
               disp: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Sample paired (low, high) estimates that look like real auction
    pre-sale estimates: roughly lognormal in magnitude, with high ~ 1.5–2x low."""
    scale = np.exp(rng.normal(0, disp, n))
    low = low_base * scale
    high = high_base * scale * rng.uniform(1.0, 1.3, n)
    return low.round(-2), high.round(-2)


def _sale_dates(n: int, start: str = "2021-01-01", end: str = "2026-05-01") -> pd.Series:
    """Random dates uniformly between start and end."""
    start_ts = pd.Timestamp(start).value
    end_ts = pd.Timestamp(end).value
    ts = rng.uniform(start_ts, end_ts, n).astype("int64")
    return pd.to_datetime(ts)


def _season(d: pd.Timestamp) -> str:
    return "Spring" if d.month <= 6 else "Autumn"


# ---------------------------------------------------------------------------
# 1. Hermès Birkin & Kelly (handbags.csv)
# ---------------------------------------------------------------------------
# Encoded prior: exotic leathers >> calf leathers; diamond hardware huge;
# Pristine condition matters; Kelly Sellier slightly above Kelly Retourne.

def synth_handbags(n: int = 320) -> pd.DataFrame:
    leathers = rng.choice(
        ["Togo", "Clemence", "Epsom", "Chèvre", "Ostrich",
         "Niloticus crocodile", "Porosus crocodile", "Lizard",
         "Himalaya Niloticus"],
        size=n,
        p=[0.22, 0.16, 0.14, 0.08, 0.10, 0.10, 0.08, 0.04, 0.08],
    )
    hardware = rng.choice(
        ["PHW", "GHW", "Permabrass", "Diamond"],
        size=n, p=[0.45, 0.40, 0.10, 0.05],
    )
    sizes = rng.choice([25, 28, 30, 32, 35, 40], size=n,
                       p=[0.10, 0.15, 0.30, 0.15, 0.25, 0.05])
    models = rng.choice(
        ["Birkin", "Birkin Sellier", "Kelly", "Kelly Sellier", "Mini Kelly"],
        size=n, p=[0.40, 0.10, 0.25, 0.15, 0.10],
    )
    conditions = rng.choice(
        ["Pristine", "Excellent", "Very Good", "Good"],
        size=n, p=[0.20, 0.40, 0.30, 0.10],
    )

    # Log-premium starts at a slight overperformance baseline and accumulates
    # additive shifts per attribute. These numbers encode the prior knowledge.
    base = np.full(n, 0.05)
    leather_bonus = {
        "Togo": 0.00, "Clemence": 0.02, "Epsom": -0.05, "Chèvre": 0.00,
        "Ostrich": -0.10, "Niloticus crocodile": 0.18,
        "Porosus crocodile": 0.10, "Lizard": -0.05,
        "Himalaya Niloticus": 0.50,
    }
    hardware_bonus = {"PHW": 0.05, "GHW": 0.00, "Permabrass": 0.00, "Diamond": 0.45}
    size_bonus = {25: 0.10, 28: 0.02, 30: 0.06, 32: 0.00, 35: 0.04, 40: 0.18}
    model_bonus = {"Birkin": 0.04, "Birkin Sellier": 0.12, "Kelly": 0.00,
                   "Kelly Sellier": 0.15, "Mini Kelly": 0.08}
    cond_bonus = {"Pristine": 0.22, "Excellent": 0.05, "Very Good": -0.02, "Good": -0.25}

    log_p = base.copy()
    for i in range(n):
        log_p[i] += leather_bonus[leathers[i]]
        log_p[i] += hardware_bonus[hardware[i]]
        log_p[i] += size_bonus[sizes[i]]
        log_p[i] += model_bonus[models[i]]
        log_p[i] += cond_bonus[conditions[i]]
    log_p += rng.normal(0, 0.18, n)
    premium = np.exp(log_p)

    low, high = _estimates(n, low_base=10000, high_base=16000, disp=0.55)
    mid = (low + high) / 2
    hammer = mid * premium
    # 8% unsold
    unsold = rng.random(n) < 0.08
    hammer = np.where(unsold, np.nan, hammer.round(-2))

    dates = _sale_dates(n)
    locations = rng.choice(["New York", "Hong Kong", "Geneva"], size=n,
                           p=[0.45, 0.40, 0.15])
    buyer_region = rng.choice(["US", "Asia", "Europe", "Other"],
                              size=n, p=[0.30, 0.45, 0.20, 0.05])

    return pd.DataFrame({
        "lot_id":         [f"HB-{i+1001}" for i in range(n)],
        "sale_date":      dates,
        "sale_location":  locations,
        "brand":          ["Hermès"] * n,
        "model":          models,
        "leather":        leathers,
        "hardware":       hardware,
        "size_cm":        sizes,
        "year":           rng.integers(2014, 2025, n),
        "condition":      conditions,
        "low_estimate":   low.astype(int),
        "high_estimate":  high.astype(int),
        "hammer_price":   hammer,
        "buyer_region":   buyer_region,
    })


# ---------------------------------------------------------------------------
# 2. Vintage Rolex Daytona (watches_daytona.csv)
# ---------------------------------------------------------------------------
# Encoded prior: Paul Newman exotic dial dominates; full set massive;
# celebrity provenance enormous; stainless > gold for vintage Daytona.

def synth_daytona(n: int = 180) -> pd.DataFrame:
    refs = rng.choice(
        ["6239", "6241", "6262", "6263", "6264", "6265"],
        size=n, p=[0.25, 0.15, 0.10, 0.25, 0.10, 0.15],
    )
    dials = rng.choice(
        ["Standard", "Exotic 'Paul Newman'", "Tropical", "Sigma"],
        size=n, p=[0.55, 0.20, 0.15, 0.10],
    )
    bezels = rng.choice(["Steel", "Acrylic", "18K Gold"], size=n,
                        p=[0.55, 0.30, 0.15])
    movements = rng.choice(["Valjoux 72 unmodified", "Valjoux 727"],
                           size=n, p=[0.60, 0.40])
    papers = rng.choice(["None", "Partial papers", "Full set"], size=n,
                        p=[0.40, 0.35, 0.25])
    prov = rng.choice(["Single owner", "Named collection", "Celebrity"],
                      size=n, p=[0.70, 0.22, 0.08])

    base = np.full(n, 0.10)
    dial_bonus = {"Standard": 0.00, "Exotic 'Paul Newman'": 1.10,
                  "Tropical": 0.35, "Sigma": 0.10}
    bezel_bonus = {"Steel": 0.10, "Acrylic": 0.00, "18K Gold": -0.08}
    paper_bonus = {"None": 0.00, "Partial papers": 0.10, "Full set": 0.40}
    prov_bonus = {"Single owner": 0.00, "Named collection": 0.15, "Celebrity": 0.55}

    log_p = base.copy()
    for i in range(n):
        log_p[i] += dial_bonus[dials[i]]
        log_p[i] += bezel_bonus[bezels[i]]
        log_p[i] += paper_bonus[papers[i]]
        log_p[i] += prov_bonus[prov[i]]
    log_p += rng.normal(0, 0.20, n)
    premium = np.exp(log_p)

    low, high = _estimates(n, low_base=40000, high_base=70000, disp=0.6)
    mid = (low + high) / 2
    hammer = mid * premium
    unsold = rng.random(n) < 0.05
    hammer = np.where(unsold, np.nan, hammer.round(-2))

    dates = _sale_dates(n)
    locations = rng.choice(["New York", "Hong Kong", "Geneva"], size=n,
                           p=[0.40, 0.30, 0.30])
    buyer_region = rng.choice(["US", "Asia", "Europe", "Other"],
                              size=n, p=[0.35, 0.40, 0.20, 0.05])

    return pd.DataFrame({
        "lot_id":         [f"WD-{i+2001}" for i in range(n)],
        "sale_date":      dates,
        "sale_location":  locations,
        "reference":      refs,
        "dial_type":      dials,
        "bezel":          bezels,
        "movement":       movements,
        "papers":         papers,
        "provenance":     prov,
        "low_estimate":   low.astype(int),
        "high_estimate":  high.astype(int),
        "hammer_price":   hammer,
        "buyer_region":   buyer_region,
    })


# ---------------------------------------------------------------------------
# 3. Patek Philippe Calatrava (watches_patek.csv)
# ---------------------------------------------------------------------------
# Encoded prior: stainless steel case enormous (extremely rare for Patek);
# Patek archives extract material; original papers material;
# salmon dial > champagne > black.

def synth_patek(n: int = 160) -> pd.DataFrame:
    refs = rng.choice(
        ["96", "570", "2526", "2551", "2552", "3415", "3444"],
        size=n, p=[0.10, 0.12, 0.18, 0.16, 0.14, 0.14, 0.16],
    )
    case_mats = rng.choice(
        ["Yellow gold", "Pink gold", "White gold", "Platinum", "Stainless steel"],
        size=n, p=[0.40, 0.25, 0.18, 0.12, 0.05],
    )
    dial_colors = rng.choice(
        ["Champagne", "Silver", "Black", "Opaline", "Salmon"],
        size=n, p=[0.35, 0.25, 0.15, 0.15, 0.10],
    )
    archives = rng.choice(["Yes", "No"], size=n, p=[0.45, 0.55])
    papers = rng.choice(["Yes", "No"], size=n, p=[0.30, 0.70])

    base = np.full(n, 0.08)
    case_bonus = {"Yellow gold": 0.00, "Pink gold": -0.05, "White gold": -0.08,
                  "Platinum": 0.15, "Stainless steel": 0.50}
    dial_bonus = {"Champagne": 0.00, "Silver": -0.05, "Black": 0.18,
                  "Opaline": 0.15, "Salmon": 0.42}
    archive_bonus = {"Yes": 0.25, "No": 0.00}
    paper_bonus = {"Yes": 0.30, "No": 0.00}

    log_p = base.copy()
    for i in range(n):
        log_p[i] += case_bonus[case_mats[i]]
        log_p[i] += dial_bonus[dial_colors[i]]
        log_p[i] += archive_bonus[archives[i]]
        log_p[i] += paper_bonus[papers[i]]
    log_p += rng.normal(0, 0.20, n)
    premium = np.exp(log_p)

    low, high = _estimates(n, low_base=25000, high_base=45000, disp=0.55)
    mid = (low + high) / 2
    hammer = mid * premium
    unsold = rng.random(n) < 0.06
    hammer = np.where(unsold, np.nan, hammer.round(-2))

    dates = _sale_dates(n)
    locations = rng.choice(["New York", "Hong Kong", "Geneva"], size=n,
                           p=[0.35, 0.30, 0.35])
    buyer_region = rng.choice(["US", "Asia", "Europe", "Other"],
                              size=n, p=[0.30, 0.40, 0.25, 0.05])

    return pd.DataFrame({
        "lot_id":             [f"WP-{i+3001}" for i in range(n)],
        "sale_date":          dates,
        "sale_location":      locations,
        "reference":          refs,
        "case_material":      case_mats,
        "dial_color":         dial_colors,
        "extract_archives":   archives,
        "original_papers":    papers,
        "low_estimate":       low.astype(int),
        "high_estimate":      high.astype(int),
        "hammer_price":       hammer,
        "buyer_region":       buyer_region,
    })


# ---------------------------------------------------------------------------
# 4. DRC Burgundy (wine_drc.csv)
# ---------------------------------------------------------------------------
# Encoded prior: Romanée-Conti cuvée dominates; great vintages premium;
# magnum/jeroboam > 750ml; direct-from-domaine premium.

GREAT_VINTAGES = {1990, 1999, 2005, 2009, 2015, 2018, 2019}

def synth_drc(n: int = 200) -> pd.DataFrame:
    cuvees = rng.choice(
        ["Romanée-Conti", "La Tâche", "Richebourg", "Grands Échezeaux", "Échezeaux"],
        size=n, p=[0.20, 0.25, 0.20, 0.20, 0.15],
    )
    vintages = rng.integers(1985, 2021, n)
    formats = rng.choice(["750ml", "Magnum", "Jeroboam"],
                         size=n, p=[0.78, 0.18, 0.04])
    sources = rng.choice(["Private cellar", "Direct from Domaine", "Acker auction"],
                         size=n, p=[0.55, 0.25, 0.20])
    owc = rng.choice(["Yes", "No"], size=n, p=[0.45, 0.55])

    base = np.full(n, 0.05)
    cuvee_bonus = {"Romanée-Conti": 0.22, "La Tâche": 0.10, "Richebourg": 0.00,
                   "Grands Échezeaux": -0.05, "Échezeaux": -0.10}
    format_bonus = {"750ml": 0.00, "Magnum": 0.10, "Jeroboam": 0.20}
    source_bonus = {"Private cellar": 0.00, "Direct from Domaine": 0.20,
                    "Acker auction": 0.05}
    owc_bonus = {"Yes": 0.08, "No": 0.00}

    log_p = base.copy()
    for i in range(n):
        log_p[i] += cuvee_bonus[cuvees[i]]
        log_p[i] += format_bonus[formats[i]]
        log_p[i] += source_bonus[sources[i]]
        log_p[i] += owc_bonus[owc[i]]
        if int(vintages[i]) in GREAT_VINTAGES:
            log_p[i] += 0.10
    log_p += rng.normal(0, 0.20, n)
    premium = np.exp(log_p)

    low, high = _estimates(n, low_base=12000, high_base=20000, disp=0.5)
    mid = (low + high) / 2
    hammer = mid * premium
    unsold = rng.random(n) < 0.07
    hammer = np.where(unsold, np.nan, hammer.round(-2))

    dates = _sale_dates(n)
    locations = rng.choice(["New York", "Hong Kong"], size=n, p=[0.55, 0.45])
    buyer_region = rng.choice(["US", "Asia", "Europe", "Other"],
                              size=n, p=[0.30, 0.45, 0.20, 0.05])

    return pd.DataFrame({
        "lot_id":         [f"WN-{i+4001}" for i in range(n)],
        "sale_date":      dates,
        "sale_location":  locations,
        "cuvee":          cuvees,
        "vintage":        vintages,
        "format":         formats,
        "source":         sources,
        "owc_intact":     owc,
        "low_estimate":   low.astype(int),
        "high_estimate":  high.astype(int),
        "hammer_price":   hammer,
        "buyer_region":   buyer_region,
    })


# ---------------------------------------------------------------------------
# 5. Japanese Whisky (whisky_japanese.csv)
# ---------------------------------------------------------------------------
# Encoded prior: Karuizawa (closed) massive premium; Ichiro's Card Series;
# pre-1980 distillation; mizunara cask.

def synth_whisky(n: int = 180) -> pd.DataFrame:
    distilleries = rng.choice(
        ["Karuizawa", "Hanyu Ichiro's", "Yamazaki", "Hibiki", "Hakushu", "Yoichi"],
        size=n, p=[0.18, 0.15, 0.22, 0.20, 0.13, 0.12],
    )
    distill_years = rng.integers(1965, 2010, n)
    bottle_years = distill_years + rng.integers(15, 35, n)
    casks = rng.choice(
        ["Sherry", "Bourbon", "Mizunara", "Refill"],
        size=n, p=[0.30, 0.40, 0.15, 0.15],
    )
    abv = np.round(rng.uniform(43, 62, n), 1)
    labels = rng.choice(
        ["Standard", "Ichiro's Card Series", "Vintage release", "Cask strength"],
        size=n, p=[0.50, 0.12, 0.20, 0.18],
    )

    base = np.full(n, 0.08)
    distillery_bonus = {
        "Karuizawa": 0.45, "Hanyu Ichiro's": 0.20, "Yamazaki": 0.05,
        "Hibiki": 0.00, "Hakushu": 0.00, "Yoichi": -0.05,
    }
    cask_bonus = {"Sherry": 0.10, "Bourbon": 0.00, "Mizunara": 0.22, "Refill": -0.05}
    label_bonus = {"Standard": 0.00, "Ichiro's Card Series": 0.32,
                   "Vintage release": 0.12, "Cask strength": 0.10}

    log_p = base.copy()
    for i in range(n):
        log_p[i] += distillery_bonus[distilleries[i]]
        log_p[i] += cask_bonus[casks[i]]
        log_p[i] += label_bonus[labels[i]]
        if distill_years[i] < 1980:
            log_p[i] += 0.18
    log_p += rng.normal(0, 0.22, n)
    premium = np.exp(log_p)

    low, high = _estimates(n, low_base=4000, high_base=8000, disp=0.7)
    mid = (low + high) / 2
    hammer = mid * premium
    unsold = rng.random(n) < 0.10
    hammer = np.where(unsold, np.nan, hammer.round(-2))

    dates = _sale_dates(n)
    locations = rng.choice(["Hong Kong", "New York", "London"], size=n,
                           p=[0.55, 0.30, 0.15])
    buyer_region = rng.choice(["US", "Asia", "Europe", "Other"],
                              size=n, p=[0.20, 0.60, 0.15, 0.05])

    return pd.DataFrame({
        "lot_id":         [f"WK-{i+5001}" for i in range(n)],
        "sale_date":      dates,
        "sale_location":  locations,
        "distillery":     distilleries,
        "distilled_year": distill_years,
        "bottled_year":   bottle_years,
        "cask_type":      casks,
        "abv":            abv,
        "label_series":   labels,
        "low_estimate":   low.astype(int),
        "high_estimate":  high.astype(int),
        "hammer_price":   hammer,
        "buyer_region":   buyer_region,
    })


# ---------------------------------------------------------------------------
# 6. Game-worn / Photo-matched Sneakers (sneakers.csv)
# ---------------------------------------------------------------------------

def synth_sneakers(n: int = 140) -> pd.DataFrame:
    players = rng.choice(
        ["Michael Jordan", "Kobe Bryant", "LeBron James", "Magic Johnson",
         "Larry Bird", "Other"],
        size=n, p=[0.45, 0.18, 0.15, 0.08, 0.07, 0.07],
    )
    models = rng.choice(
        ["Air Jordan 1", "Air Jordan 11", "Air Jordan 13", "Nike Air Ship",
         "Converse", "Adidas Crazy 8"],
        size=n, p=[0.30, 0.18, 0.12, 0.10, 0.15, 0.15],
    )
    seasons = rng.integers(1984, 2024, n)
    game_worn = rng.choice([True, False], size=n, p=[0.65, 0.35])
    photo_matched = np.where(game_worn,
                             rng.choice([True, False], size=n, p=[0.45, 0.55]),
                             False)
    signed = rng.choice([True, False], size=n, p=[0.40, 0.60])
    conditions = rng.choice(
        ["Pristine", "Very Good", "Good", "Heavily worn"],
        size=n, p=[0.10, 0.30, 0.35, 0.25],
    )

    base = np.full(n, 0.05)
    player_bonus = {
        "Michael Jordan": 0.25, "Kobe Bryant": 0.15, "LeBron James": 0.10,
        "Magic Johnson": 0.00, "Larry Bird": -0.05, "Other": -0.15,
    }
    cond_bonus = {"Pristine": 0.10, "Very Good": 0.05,
                  "Good": -0.05, "Heavily worn": -0.35}

    log_p = base.copy()
    for i in range(n):
        log_p[i] += player_bonus[players[i]]
        log_p[i] += cond_bonus[conditions[i]]
        if game_worn[i]:
            log_p[i] += 0.20
        if photo_matched[i]:
            log_p[i] += 0.45
        if signed[i]:
            log_p[i] += 0.22
        # MJ rookie-era kicker
        if players[i] == "Michael Jordan" and seasons[i] <= 1986:
            log_p[i] += 0.22
    log_p += rng.normal(0, 0.22, n)
    premium = np.exp(log_p)

    low, high = _estimates(n, low_base=6000, high_base=12000, disp=0.6)
    mid = (low + high) / 2
    hammer = mid * premium
    unsold = rng.random(n) < 0.09
    hammer = np.where(unsold, np.nan, hammer.round(-2))

    dates = _sale_dates(n)
    locations = rng.choice(["New York", "Hong Kong"], size=n, p=[0.70, 0.30])
    buyer_region = rng.choice(["US", "Asia", "Europe", "Other"],
                              size=n, p=[0.55, 0.25, 0.15, 0.05])

    return pd.DataFrame({
        "lot_id":         [f"SN-{i+6001}" for i in range(n)],
        "sale_date":      dates,
        "sale_location":  locations,
        "player":         players,
        "model":          models,
        "season":         seasons,
        "game_worn":      game_worn,
        "photo_matched":  photo_matched,
        "signed":         signed,
        "condition":      conditions,
        "low_estimate":   low.astype(int),
        "high_estimate":  high.astype(int),
        "hammer_price":   hammer,
        "buyer_region":   buyer_region,
    })


# ---------------------------------------------------------------------------
# 7. Macro aggregate (macro.csv)
# ---------------------------------------------------------------------------
# Combines the category-level data into one long table, plus simulates
# Jewels and Wine/Spirits not covered by the deep-dives, plus encodes a
# 'sold' flag for sell-through analysis.

def synth_macro(handbags, daytona, patek, drc, whisky, sneakers,
                n_extra_jewels: int = 200) -> pd.DataFrame:
    rows = []

    # Handbags
    for _, r in handbags.iterrows():
        rows.append(dict(
            lot_id=r["lot_id"], sale_date=r["sale_date"],
            category="Handbags",
            low_estimate=r["low_estimate"], high_estimate=r["high_estimate"],
            hammer_price=r["hammer_price"],
        ))
    # Watches (combine daytona + patek under "Watches")
    for _, r in pd.concat([daytona, patek]).iterrows():
        rows.append(dict(
            lot_id=r["lot_id"], sale_date=r["sale_date"],
            category="Watches",
            low_estimate=r["low_estimate"], high_estimate=r["high_estimate"],
            hammer_price=r["hammer_price"],
        ))
    # Wine
    for _, r in drc.iterrows():
        rows.append(dict(
            lot_id=r["lot_id"], sale_date=r["sale_date"],
            category="Wine",
            low_estimate=r["low_estimate"], high_estimate=r["high_estimate"],
            hammer_price=r["hammer_price"],
        ))
    # Spirits (whisky)
    for _, r in whisky.iterrows():
        rows.append(dict(
            lot_id=r["lot_id"], sale_date=r["sale_date"],
            category="Spirits",
            low_estimate=r["low_estimate"], high_estimate=r["high_estimate"],
            hammer_price=r["hammer_price"],
        ))
    # Modern Collectibles (sneakers)
    for _, r in sneakers.iterrows():
        rows.append(dict(
            lot_id=r["lot_id"], sale_date=r["sale_date"],
            category="Modern Collectibles",
            low_estimate=r["low_estimate"], high_estimate=r["high_estimate"],
            hammer_price=r["hammer_price"],
        ))

    df = pd.DataFrame(rows)
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df["sale_year"] = df["sale_date"].dt.year
    df["sale_season"] = df["sale_date"].apply(_season)
    df["sold"] = df["hammer_price"].notna()
    return df


# ---------------------------------------------------------------------------
# 8. HK-NY matched pairs (corridor.csv)
# ---------------------------------------------------------------------------
# Synthesize comparable lots that appeared in both HK and NY, with
# different premium realizations. Encoded prior: handbags premium higher
# in HK; vintage Daytona premium higher in NY; Patek roughly parity.

def synth_corridor(n_pairs: int = 80) -> pd.DataFrame:
    categories = rng.choice(
        ["Hermès Handbag", "Vintage Rolex Daytona", "Patek Philippe Calatrava"],
        size=n_pairs, p=[0.40, 0.35, 0.25],
    )
    hk_bias = {
        "Hermès Handbag": 0.10,
        "Vintage Rolex Daytona": -0.15,
        "Patek Philippe Calatrava": 0.02,
    }
    refs = []
    for c in categories:
        if c == "Hermès Handbag":
            refs.append(rng.choice(["Birkin 30", "Birkin 35", "Kelly 28", "Kelly 32"]))
        elif c == "Vintage Rolex Daytona":
            refs.append(rng.choice(["6239", "6241", "6263", "6265"]))
        else:
            refs.append(rng.choice(["570", "2526", "3415", "3444"]))

    rows = []
    for i in range(n_pairs):
        c = categories[i]
        bias = hk_bias[c]
        # Same mid-estimate band; different realized premium
        mid = float(rng.uniform(30000, 200000))
        low = mid * 0.85
        high = mid * 1.18
        # Common shock + asymmetric venue noise
        common = rng.normal(0.10, 0.15)
        hk_premium = np.exp(common + bias + rng.normal(0, 0.25))
        ny_premium = np.exp(common + rng.normal(0, 0.25))
        rows.append(dict(
            pair_id=f"P{i+7001}",
            reference_or_model=refs[i],
            category=c,
            hk_lot_id=f"HK-{i+8001}", hk_hammer=mid * hk_premium,
            hk_mid_estimate=mid,
            hk_sale_date=_sale_dates(1)[0],
            ny_lot_id=f"NY-{i+9001}", ny_hammer=mid * ny_premium,
            ny_mid_estimate=mid,
            ny_sale_date=_sale_dates(1)[0],
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating synthetic data into {RAW_DIR}")

    handbags = synth_handbags()
    daytona  = synth_daytona()
    patek    = synth_patek()
    drc      = synth_drc()
    whisky   = synth_whisky()
    sneakers = synth_sneakers()

    handbags.to_csv(RAW_DIR / "handbags.csv", index=False)
    daytona.to_csv(RAW_DIR / "watches_daytona.csv", index=False)
    patek.to_csv(RAW_DIR / "watches_patek.csv", index=False)
    drc.to_csv(RAW_DIR / "wine_drc.csv", index=False)
    whisky.to_csv(RAW_DIR / "whisky_japanese.csv", index=False)
    sneakers.to_csv(RAW_DIR / "sneakers.csv", index=False)

    macro = synth_macro(handbags, daytona, patek, drc, whisky, sneakers)
    macro.to_csv(RAW_DIR / "macro.csv", index=False)

    corridor = synth_corridor()
    corridor.to_csv(RAW_DIR / "corridor.csv", index=False)

    print(f"  wrote {len(handbags):>4} rows → handbags.csv")
    print(f"  wrote {len(daytona):>4} rows → watches_daytona.csv")
    print(f"  wrote {len(patek):>4} rows → watches_patek.csv")
    print(f"  wrote {len(drc):>4} rows → wine_drc.csv")
    print(f"  wrote {len(whisky):>4} rows → whisky_japanese.csv")
    print(f"  wrote {len(sneakers):>4} rows → sneakers.csv")
    print(f"  wrote {len(macro):>4} rows → macro.csv")
    print(f"  wrote {len(corridor):>4} rows → corridor.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()