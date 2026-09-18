"""
factors/kyle_lambda.py

Tier 2 factor: Kyle's Lambda (1985), the linear price-impact coefficient --
lambda = Cov(dP, Q) / Var(Q), estimated via rolling regression on the same
volume-bucketed data used for VPIN.

SCOPE: this module computes lambda_t (a rolling estimate) and the Depth
Exhaustion condition from the Phase 1/2 spec (|dP_t| > kappa * lambda_t *
Q_t). It does NOT implement Tier 3's tick-level execution trigger (that
needs OFI/CVD, a separate not-yet-built module) -- Depth Exhaustion firing
is a Tier 2 gate, not the final entry signal.

Formula:
    Kyle (1985): Delta P_t = lambda * Q_t + epsilon_t
    where Q_t is signed order flow (net buy volume - sell volume) in a
    bucket, and Delta P_t is that bucket's price change.

    lambda_hat = Cov(Delta P, Q) / Var(Q), estimated over a trailing
    rolling window of buckets (volume-clock, not time-clock -- same
    bucketing rationale as VPIN: comparable "information time" across
    buckets, not comparable wall-clock time).

Depth Exhaustion condition (Phase 1/2 spec):
    DepthExhaustion_t = 1 if |Delta P_t| > kappa * lambda_hat_t * Q_t

kappa (>1) is explicitly a Phase 3 calibration hyperparameter in the
original spec, not a fixed constant -- this module requires the caller to
pass it, it does not default to a "reasonable-sounding" value that would
quietly become a de facto standard nobody chose on purpose.

METHODOLOGICAL CAVEAT (flag this, don't let it hide): Q here comes from
BVC classification (factors.vpin.classify_bucket_volume), which itself
derives buy/sell volume FROM the bucket's own price change via the normal
CDF of standardized dP. That means Q and dP are correlated partly BY
CONSTRUCTION, not purely because of genuine price-impact/liquidity
dynamics -- BVC-derived Kyle's Lambda is a real, commonly-used estimator
in the literature, but it is not the same thing as lambda estimated from
a true trade-classification scheme (tick rule, Lee-Ready) where Q is
independent of the price-change measurement. Don't over-interpret small
absolute lambda_hat differences as pure liquidity signal without keeping
this construction effect in mind.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_signed_order_flow(classified_buckets: pd.DataFrame) -> pd.Series:
    """
    Q_t per bucket = buy_volume - sell_volume (signed net order flow),
    from the same BVC-classified buckets used for VPIN
    (factors.vpin.classify_bucket_volume output).
    """
    return classified_buckets["buy_volume"] - classified_buckets["sell_volume"]


def rolling_kyle_lambda(
    classified_buckets: pd.DataFrame,
    window: int = 50,
) -> pd.Series:
    """
    Rolling Kyle's Lambda: lambda_hat_t = Cov(dP, Q) / Var(Q) over the
    trailing `window` buckets.

    classified_buckets: output of factors.vpin.classify_bucket_volume()
    (needs close_price, buy_volume, sell_volume).

    window: number of trailing buckets used per regression. Same
    volume-clock rationale as VPIN's n -- this is a Phase 3 calibration
    target, default here (50) just matches VPIN's illustrative default
    for consistency, not because it's been fit to anything.

    Returns a pandas Series of lambda_hat (same index), NaN for the first
    window-1 entries (no full window yet) and also NaN wherever the
    trailing window's Var(Q) is exactly 0 (can't estimate lambda from zero
    order-flow variance -- flagged as NaN rather than silently returning
    inf or 0, which would be misleading in either direction).
    """
    Q = compute_signed_order_flow(classified_buckets)
    dP = classified_buckets["close_price"].diff()

    def _lambda_for_window(idx_end: int) -> float:
        start = idx_end - window + 1
        if start < 0:
            return np.nan
        q_slice = Q.iloc[start:idx_end + 1]
        dp_slice = dP.iloc[start:idx_end + 1]
        # dP's first overall entry is NaN (no prior price); drop paired NaNs
        valid = dp_slice.notna()
        q_slice, dp_slice = q_slice[valid], dp_slice[valid]
        if len(q_slice) < 2:
            return np.nan
        var_q = q_slice.var(ddof=1)
        if var_q == 0 or np.isnan(var_q):
            return np.nan
        cov_qp = np.cov(dp_slice, q_slice, ddof=1)[0, 1]
        return cov_qp / var_q

    return pd.Series(
        [_lambda_for_window(i) for i in range(len(classified_buckets))],
        index=classified_buckets.index,
        name="kyle_lambda",
    )


def depth_exhaustion(
    classified_buckets: pd.DataFrame,
    lambda_hat: pd.Series,
    kappa: float,
) -> pd.Series:
    """
    Depth Exhaustion condition (Phase 1/2 spec):
        DepthExhaustion_t = 1 if |dP_t| > kappa * lambda_hat_t * Q_t, else 0

    kappa: REQUIRED, no default -- per the Phase 1/2 spec this is a
    backtest-calibrated hyperparameter ("materially exceeds model-implied
    impact"), and defaulting it to a specific number here would smuggle an
    unvalidated assumption into what's supposed to be a Phase 3 fitting
    target.

    Returns a pandas Series of 0/1 (int), NaN wherever lambda_hat is NaN
    (can't evaluate the condition without a valid lambda estimate for that
    bucket).
    """
    Q = compute_signed_order_flow(classified_buckets)
    dP = classified_buckets["close_price"].diff()

    threshold = (kappa * lambda_hat * Q).abs()
    exhausted = dP.abs() > threshold
    result = exhausted.astype("Int64")  # nullable int, so NaN can propagate
    result[lambda_hat.isna() | dP.isna()] = pd.NA
    return result


if __name__ == "__main__":
    from factors.vpin import volume_bucket_trades, classify_bucket_volume

    rng = np.random.default_rng(7)
    n_trades = 20_000
    price_path = 500 + np.cumsum(rng.normal(0, 0.02, n_trades))
    trade_volumes = rng.integers(1, 15, n_trades).astype(float)
    demo_trades = pd.DataFrame({"price": price_path, "volume": trade_volumes})

    buckets = volume_bucket_trades(demo_trades, bucket_size=500.0)
    classified = classify_bucket_volume(buckets, sigma_window=50)

    lam = rolling_kyle_lambda(classified, window=50)
    print(f"Kyle's Lambda: min={lam.min():.6f} max={lam.max():.6f} "
          f"mean={lam.mean():.6f} (last 5): {lam.tail(5).to_numpy()}")

    exhaustion = depth_exhaustion(classified, lam, kappa=1.5)
    fired = int((exhaustion == 1).sum())
    total_valid = int(exhaustion.notna().sum())
    print(f"Depth exhaustion fired on {fired}/{total_valid} valid buckets "
          f"({fired / total_valid:.1%})")
