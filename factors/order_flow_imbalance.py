"""
factors/order_flow_imbalance.py

Tier 3 factor: Order Flow Imbalance (OFI), per Cont, Kukanov, Stoikov (2014)
"The Price Impact of Order Book Events" -- the L2 depth-based flow-imbalance
metric that Phase 1 research identified as leading price change with
R^2 > 60% at short bars, versus CVD's coarser, execution-only signal.

SCOPE: this module computes OFI (L2) and CVD (trade-print fallback), plus
the two standalone Tier 3 execution-trigger CONDITIONS from the spec
(Condition A: VPIN plateau/rollover, Condition B: OFI sign flip near the
extreme). It does NOT implement the state machine that sequences these
into EXECUTION_ARMED, entry price, stop, or target -- that's
state_machine/, not yet built. This file answers "is condition A/B true
right now," not "what do we do about it."

L2 OFI formula (Cont-Kukanov-Stoikov 2014), per L2 top-of-book event n:

    bid contribution:
        +q^B_n              if P^B_n  > P^B_{n-1}   (bid price improved up)
        q^B_n - q^B_{n-1}   if P^B_n == P^B_{n-1}   (same level, size changed)
        -q^B_{n-1}          if P^B_n  < P^B_{n-1}   (bid price stepped down)

    ask contribution:
        -q^A_{n-1}          if P^A_n  > P^A_{n-1}   (ask price stepped up --
                                                       offers retreated, bullish)
        q^A_n - q^A_{n-1}   if P^A_n == P^A_{n-1}   (same level, size changed)
        +q^A_n              if P^A_n  < P^A_{n-1}   (ask price improved down --
                                                       more aggressive selling)

    OFI_event = bid_contribution - ask_contribution

This module's sign convention was validated (see `_self_validate()` below)
against three synthetic scenarios where the correct sign is obvious by
construction: an aggressive buy sweep (ask price steps up as it's
consumed) must produce POSITIVE OFI; new resting sell size posted at an
unchanged ask must produce NEGATIVE OFI; a bid price stepping up must
produce POSITIVE OFI. If you ever edit the contribution formulas, rerun
`_self_validate()` -- a flipped sign here would silently invert every
downstream Tier 3 trigger.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# L2 Order Flow Imbalance (Cont-Kukanov-Stoikov)
# --------------------------------------------------------------------------

def compute_ofi_l2(book_updates: pd.DataFrame) -> pd.Series:
    """
    Compute per-event OFI from top-of-book L2 snapshots.

    book_updates: DataFrame, one row per book-update event, ordered by
    time, with columns:
        bid_price, bid_size  -- best bid level
        ask_price, ask_size  -- best ask level

    Returns a pandas Series of per-event OFI (same index, first row NaN --
    no prior snapshot to diff against). Sum/cumsum this over whatever
    window you need (e.g. tick-level trigger checks within N ticks of an
    extreme, per Tier 3 Condition B) -- this function deliberately returns
    the raw per-event series rather than pre-aggregating, since the right
    aggregation window is a Tier 3/state-machine decision, not this
    module's.
    """
    required = {"bid_price", "bid_size", "ask_price", "ask_size"}
    missing = required - set(book_updates.columns)
    if missing:
        raise ValueError(f"book_updates missing required columns: {missing}")

    bp, bs = book_updates["bid_price"].to_numpy(dtype=float), book_updates["bid_size"].to_numpy(dtype=float)
    ap, asz = book_updates["ask_price"].to_numpy(dtype=float), book_updates["ask_size"].to_numpy(dtype=float)

    n = len(book_updates)
    ofi = np.full(n, np.nan)

    for i in range(1, n):
        if bp[i] > bp[i - 1]:
            bid_contrib = bs[i]
        elif bp[i] == bp[i - 1]:
            bid_contrib = bs[i] - bs[i - 1]
        else:
            bid_contrib = -bs[i - 1]

        if ap[i] > ap[i - 1]:
            ask_contrib = -asz[i - 1]
        elif ap[i] == ap[i - 1]:
            ask_contrib = asz[i] - asz[i - 1]
        else:
            ask_contrib = asz[i]

        ofi[i] = bid_contrib - ask_contrib

    return pd.Series(ofi, index=book_updates.index, name="ofi")


# --------------------------------------------------------------------------
# CVD fallback (tick-rule based), per the spec's explicit fallback clause:
# "CVD Delta-Imbalance fallback if L2 unavailable"
# --------------------------------------------------------------------------

def compute_cvd_tick_rule(trades: pd.DataFrame) -> pd.Series:
    """
    Cumulative Volume Delta via the classic tick rule (Lee-Ready style,
    simplified): a trade is classified buy-initiated if its price is above
    the previous trade's price, sell-initiated if below, and inherits the
    previous trade's direction if the price is unchanged.

    trades: DataFrame with columns price, volume, ordered by time.

    Returns a pandas Series of CUMULATIVE volume delta (running sum of
    signed volume), same index. The first trade has no prior price to
    compare against and is arbitrarily classified as buy-initiated (a
    standard tick-rule convention for the very first observation) --
    this only affects the series' starting offset, not its shape.

    CAVEAT (per Phase 2 data-spec discussion): CVD is explicitly the
    FALLBACK here, not the preferred signal -- it only tells you what
    already executed, not resting-liquidity changes, so it's a lagging,
    coarser proxy for what compute_ofi_l2() captures directly. Use CVD
    only when genuine L2 depth isn't available from the feed.
    """
    price = trades["price"].to_numpy(dtype=float)
    volume = trades["volume"].to_numpy(dtype=float)
    n = len(trades)

    direction = np.zeros(n)
    direction[0] = 1.0
    for i in range(1, n):
        if price[i] > price[i - 1]:
            direction[i] = 1.0
        elif price[i] < price[i - 1]:
            direction[i] = -1.0
        else:
            direction[i] = direction[i - 1]

    signed_volume = direction * volume
    return pd.Series(np.cumsum(signed_volume), index=trades.index, name="cvd")


# --------------------------------------------------------------------------
# Tier 3 trigger CONDITIONS (not the full state-machine trigger -- see
# module docstring)
# --------------------------------------------------------------------------

def condition_a_vpin_rollover(vpin_series: pd.Series, k: int = 5) -> pd.Series:
    """
    Condition A: dVPIN/dt <= 0 over the trailing k VPIN observations, i.e.
    toxicity has plateaued or is rolling over rather than still climbing.

    vpin_series: output of factors.vpin.compute_vpin() (bucket-indexed).
    k: trailing lookback in BUCKET-UPDATES, not ticks -- per spec, "small,
    e.g. 3-5 bucket-updates." Phase 3 calibration target, not fixed.

    Returns a boolean Series (True where the condition holds), NaN-safe:
    positions without k valid trailing VPIN values are False (condition
    cannot be evaluated yet, treated as "not satisfied" rather than
    raising).
    """
    if k < 2:
        raise ValueError("k must be >= 2 to compute a derivative")
    delta = vpin_series.diff(k - 1)
    return (delta <= 0).fillna(False)


def condition_b_ofi_flip(
    ofi_series: pd.Series,
    price_series: pd.Series,
    extreme_price: float,
    sweep_direction: int,
    n_ticks: int,
) -> pd.Series:
    """
    Condition B: OFI sign flips counter to the sweep direction, evaluated
    only while price is within n_ticks of the swept extreme.

    ofi_series: per-event OFI (factors.order_flow_imbalance.compute_ofi_l2
    output) or a CVD delta series (compute_cvd_tick_rule, differenced) as
    the fallback -- either works here since this function only needs sign,
    not magnitude.
    price_series: same index as ofi_series, the price at each event.
    extreme_price: the swept high (for a downward-sweep-then-reversal
    setup) or swept low (for an upward sweep), in price units.
    sweep_direction: +1 if the sweep ran price UP (so a bullish extreme --
    a swept high -- and we're now watching for SELL-side flow, i.e.
    negative OFI, to confirm reversal), -1 if the sweep ran price DOWN
    (swept low, watching for positive/buy-side OFI to confirm reversal).
    n_ticks: proximity window in PRICE UNITS (not literal exchange ticks --
    pass whatever tick-to-price conversion your instrument needs before
    calling this; this function takes a price distance, not a tick count,
    to stay instrument-agnostic).

    Returns a boolean Series: True where (a) price is within n_ticks of
    extreme_price AND (b) OFI's sign is counter to sweep_direction (i.e.
    the reversal-confirming sign).
    """
    if sweep_direction not in (1, -1):
        raise ValueError("sweep_direction must be +1 or -1")

    near_extreme = (price_series - extreme_price).abs() <= n_ticks
    # counter-trend sign: sweep up (+1) needs negative OFI to confirm
    # reversal; sweep down (-1) needs positive OFI.
    counter_trend_sign = -sweep_direction
    flip_confirmed = np.sign(ofi_series) == counter_trend_sign

    return (near_extreme & flip_confirmed).fillna(False)


# --------------------------------------------------------------------------
# Self-validation: verify OFI's sign convention against synthetic scenarios
# where the correct sign is unambiguous by construction.
# --------------------------------------------------------------------------

def _self_validate() -> None:
    # Scenario 1: aggressive buy sweep -- ask price steps UP as resting
    # offers get consumed, bid unchanged. Must be OFI > 0.
    buy_sweep = pd.DataFrame({
        "bid_price": [100.00, 100.00],
        "bid_size":  [50, 50],
        "ask_price": [100.25, 100.50],   # ask stepped up (consumed)
        "ask_size":  [40, 35],
    })
    ofi = compute_ofi_l2(buy_sweep)
    assert ofi.iloc[1] > 0, f"buy-sweep scenario should give OFI > 0, got {ofi.iloc[1]}"

    # Scenario 2: new resting sell size posted at an UNCHANGED ask price
    # (more selling interest, no price move) -- bearish, must be OFI < 0.
    new_sell_size = pd.DataFrame({
        "bid_price": [100.00, 100.00],
        "bid_size":  [50, 50],
        "ask_price": [100.25, 100.25],   # unchanged
        "ask_size":  [40, 70],           # more size posted
    })
    ofi = compute_ofi_l2(new_sell_size)
    assert ofi.iloc[1] < 0, f"new-resting-sell scenario should give OFI < 0, got {ofi.iloc[1]}"

    # Scenario 3: bid price steps UP (buyer stepped up aggressively),
    # ask unchanged -- bullish, must be OFI > 0.
    bid_step_up = pd.DataFrame({
        "bid_price": [100.00, 100.25],   # bid improved
        "bid_size":  [50, 45],
        "ask_price": [100.50, 100.50],
        "ask_size":  [40, 40],
    })
    ofi = compute_ofi_l2(bid_step_up)
    assert ofi.iloc[1] > 0, f"bid-step-up scenario should give OFI > 0, got {ofi.iloc[1]}"


_self_validate()


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    n = 200
    bid = 100 + np.cumsum(rng.choice([-0.25, 0, 0.25], size=n, p=[0.2, 0.6, 0.2]))
    ask = bid + rng.choice([0.25, 0.5], size=n)
    demo_book = pd.DataFrame({
        "bid_price": bid,
        "bid_size": rng.integers(10, 100, n).astype(float),
        "ask_price": ask,
        "ask_size": rng.integers(10, 100, n).astype(float),
    })
    ofi_series = compute_ofi_l2(demo_book)
    print(f"L2 OFI: mean={ofi_series.mean():.2f} std={ofi_series.std():.2f} "
          f"(last 5): {ofi_series.tail(5).to_numpy()}")

    demo_trades = pd.DataFrame({
        "price": 100 + np.cumsum(rng.normal(0, 0.05, n)),
        "volume": rng.integers(1, 20, n).astype(float),
    })
    cvd = compute_cvd_tick_rule(demo_trades)
    print(f"CVD (tick rule): final={cvd.iloc[-1]:.1f} (last 5): {cvd.tail(5).to_numpy()}")
