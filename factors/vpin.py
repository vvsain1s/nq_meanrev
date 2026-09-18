"""
factors/vpin.py

Tier 2 factor: VPIN (Volume-Synchronized Probability of Informed Trading),
per Easley, Lopez de Prado, O'Hara -- built on VOLUME-CLOCK bucketing and
Bulk Volume Classification (BVC), not the tick rule.

SCOPE: this module implements the bucketing + VPIN calculation only. It does
NOT implement the "exhaustion divergence" condition from the Phase 1 spec
(VPIN plateauing/rolling over while price fails to extend) -- that's Tier 3
execution-trigger logic (Condition A: dVPIN/dt <= 0), which belongs in the
state machine, not here. This file gives you VPIN(t) as a time series; the
state machine decides what to do with its derivative.

Formulas (from Phase 1 research, Easley/Lopez de Prado/O'Hara):
    Volume bucket size V is fixed (contracts per bucket). Each trade's
    volume is assigned to buckets by cumulative volume, so bucket
    boundaries fall mid-trade when a single trade's volume straddles two
    buckets (this module SPLITS such trades across buckets rather than
    assigning them wholesale to one bucket -- a common source of VPIN
    implementation bugs is failing to split, which skews bucket volumes).

    Bulk Volume Classification per bucket tau:
        V_tau^BUY  = V * Phi( (P_tau - P_{tau-1}) / sigma_dP )
        V_tau^SELL = V - V_tau^BUY

    where Phi is the standard normal CDF, P_tau is the bucket's closing
    price, and sigma_dP is the standard deviation of price changes
    (estimated over a trailing window of bucket-to-bucket price changes).

    VPIN over the trailing n buckets:
        VPIN = (1/n) * sum_{tau=1}^{n} |V_tau^SELL - V_tau^BUY| / V

CAVEAT (flagged per the Phase 1 research, not glossed over here): VPIN's
predictive power is contested in the academic literature -- some critiques
argue it's largely a mechanical proxy for volatility/trading intensity
rather than a clean "toxicity" signal. Treat this as ONE INPUT to the
Tier 2 exhaustion condition, never a standalone trigger, per the original
research synthesis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def volume_bucket_trades(trades: pd.DataFrame, bucket_size: float) -> pd.DataFrame:
    """
    Aggregate tick/trade-level data into fixed-size volume buckets.

    trades: DataFrame with columns:
        price  -- trade price
        volume -- trade size (contracts/shares), must be > 0
    Ordered by time (index order is assumed to be trade sequence -- this
    function does not sort by a timestamp column, since trade DataFrames
    from most feeds already arrive time-ordered; if yours doesn't, sort it
    before calling this).

    bucket_size: V, the fixed volume per bucket (same units as `volume`).
    Choosing V is a Phase 3 calibration question (tie it to the
    instrument's typical trade-arrival volume, per the Phase 1/2 spec) --
    this function does not pick V for you.

    A single trade whose volume straddles a bucket boundary is SPLIT: the
    portion before the boundary goes to the closing bucket, the remainder
    starts the next bucket. This matters because failing to split
    systematically over/under-fills buckets near boundaries, which biases
    the BVC price-change measurement.

    Returns a DataFrame, one row per completed bucket, with columns:
        bucket_id, close_price, volume (== bucket_size for every completed
        bucket; the final partial bucket, if trades run out mid-bucket, is
        DROPPED -- VPIN needs completed buckets only).
    """
    if bucket_size <= 0:
        raise ValueError("bucket_size must be > 0")
    if (trades["volume"] <= 0).any():
        raise ValueError("volume_bucket_trades: found non-positive trade volume")

    prices = trades["price"].to_numpy(dtype=float)
    volumes = trades["volume"].to_numpy(dtype=float)

    bucket_closes: list[float] = []
    running_volume = 0.0
    last_price = None

    for price, vol in zip(prices, volumes):
        remaining = vol
        last_price = price
        while remaining > 0:
            space_left = bucket_size - running_volume
            take = min(space_left, remaining)
            running_volume += take
            remaining -= take
            last_price = price
            if running_volume >= bucket_size - 1e-9:
                bucket_closes.append(last_price)
                running_volume = 0.0
    # any leftover partial bucket (running_volume > 0 at the end) is
    # intentionally dropped -- see docstring.

    return pd.DataFrame({
        "bucket_id": np.arange(len(bucket_closes)),
        "close_price": bucket_closes,
        "volume": bucket_size,
    })


def classify_bucket_volume(
    buckets: pd.DataFrame,
    sigma_window: int = 50,
) -> pd.DataFrame:
    """
    Apply Bulk Volume Classification to each bucket, splitting its volume
    into buy-initiated and sell-initiated portions.

    buckets: output of volume_bucket_trades() (needs close_price, volume).

    sigma_window: number of trailing bucket-to-bucket price changes used to
    estimate sigma_dP (the normalizing volatility in the BVC formula).
    Uses an expanding window until sigma_window buckets are available, then
    a trailing rolling window -- the first sigma_window buckets will have a
    noisier sigma_dP estimate as a result; this is inherent to needing a
    price-change history before BVC can run, not a bug to fix here.

    Returns buckets with two new columns: buy_volume, sell_volume.
    """
    df = buckets.copy()
    price_changes = df["close_price"].diff()

    sigma_dP = price_changes.rolling(window=sigma_window, min_periods=2).std()
    # backfill the initial NaN sigma estimates with the first valid value
    # rather than leaving NaN (which would make early-session BVC undefined)
    sigma_dP = sigma_dP.bfill()

    z = price_changes / sigma_dP
    buy_fraction = norm.cdf(z.fillna(0.0))  # first bucket has no prior price -> treat as neutral (0.5 via cdf(0))

    df["buy_volume"] = df["volume"] * buy_fraction
    df["sell_volume"] = df["volume"] - df["buy_volume"]
    return df


def compute_vpin(classified_buckets: pd.DataFrame, n: int = 50) -> pd.Series:
    """
    Rolling VPIN over the trailing n buckets:
        VPIN_tau = (1/n) * sum_{i=tau-n+1}^{tau} |sell_i - buy_i| / volume_i

    classified_buckets: output of classify_bucket_volume() (needs
    buy_volume, sell_volume, volume columns).

    n: number of trailing buckets averaged -- this is the same n referenced
    in the Phase 1/2 spec (illustrative default there: 50, matching Easley
    et al.'s original parameterization) and is itself a Phase 3 calibration
    target, not a fixed constant.

    Returns a pandas Series (same index as classified_buckets) of VPIN
    values; the first n-1 entries are NaN since a full trailing window
    isn't available yet.
    """
    imbalance = (classified_buckets["sell_volume"] - classified_buckets["buy_volume"]).abs()
    normalized = imbalance / classified_buckets["volume"]
    return normalized.rolling(window=n, min_periods=n).mean()


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n_trades = 20_000
    price_path = 500 + np.cumsum(rng.normal(0, 0.02, n_trades))
    trade_volumes = rng.integers(1, 15, n_trades).astype(float)

    demo_trades = pd.DataFrame({"price": price_path, "volume": trade_volumes})

    buckets = volume_bucket_trades(demo_trades, bucket_size=500.0)
    print(f"Bucketed {n_trades} trades into {len(buckets)} volume buckets")

    classified = classify_bucket_volume(buckets, sigma_window=50)
    vpin_series = compute_vpin(classified, n=50)

    print(f"VPIN: min={vpin_series.min():.4f} max={vpin_series.max():.4f} "
          f"mean={vpin_series.mean():.4f} (last 5): {vpin_series.tail(5).to_numpy()}")
