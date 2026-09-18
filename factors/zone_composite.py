"""
factors/zone_composite.py

Tier 1 zone composite: takes the per-contract output of
factors.options_greeks.aggregate_chain_exposure() and derives:

    1. Call Wall / Put Wall       -- find_walls()
    2. Zero-Gamma Flip level (S*) -- compute_gex_profile() + find_zero_gamma_flip()
    3. w_charm(t) time-of-day weight -- w_charm()
    4. Confluence score + Z_t band  -- confluence_score() + build_zone()

SCOPE: this is still Tier 1 only. It does not include Tier 2 (VPIN, Kyle's
Lambda), Tier 3 (OFI/execution trigger), or the state machine. "IV walls /
skew analysis" was never part of the original Phase 1 spec and is NOT
implemented here -- flagged explicitly so it doesn't quietly become an
assumed requirement.

Depends on: factors.options_greeks (gamma/vanna/charm, aggregate_chain_exposure)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factors.options_greeks import gamma, vanna, charm, REQUIRED_COLUMNS


# --------------------------------------------------------------------------
# 1. Call Wall / Put Wall
# --------------------------------------------------------------------------

def find_walls(per_contract: pd.DataFrame) -> dict:
    """
    Call Wall / Put Wall: the strike with the largest dollar-gamma
    concentration on the call side and the put side respectively.

    Takes the `per_contract` DataFrame returned by
    options_greeks.aggregate_chain_exposure() (must have columns:
    strike, option_type, dollar_gex).

    Returns:
        {
            "call_wall": {"strike": float, "dollar_gex": float} or None,
            "put_wall":  {"strike": float, "dollar_gex": float} or None,
            "by_strike": pd.DataFrame  # dollar_gex summed per strike, both legs
        }

    Note: "wall" here means largest MAGNITUDE concentration at that strike
    on that side -- call wall is the strike with the most positive call-side
    dollar_gex, put wall is the strike with the most negative put-side
    dollar_gex (puts contribute negative dollar_gex under the dealer sign
    convention, so "most negative" = "largest put-side concentration").
    """
    df = per_contract.copy()
    df.columns = [c.lower() for c in df.columns]
    is_call = df["option_type"].astype(str).str.lower().str.startswith("c")

    calls_by_strike = df[is_call].groupby("strike")["dollar_gex"].sum()
    puts_by_strike = df[~is_call].groupby("strike")["dollar_gex"].sum()

    call_wall = None
    if not calls_by_strike.empty:
        strike = calls_by_strike.idxmax()
        call_wall = {"strike": float(strike), "dollar_gex": float(calls_by_strike.max())}

    put_wall = None
    if not puts_by_strike.empty:
        strike = puts_by_strike.idxmin()
        put_wall = {"strike": float(strike), "dollar_gex": float(puts_by_strike.min())}

    by_strike = df.groupby("strike")["dollar_gex"].sum().reset_index().sort_values("strike")

    return {"call_wall": call_wall, "put_wall": put_wall, "by_strike": by_strike}


# --------------------------------------------------------------------------
# 2. Zero-Gamma Flip level (S*)
# --------------------------------------------------------------------------

def compute_gex_profile(
    chain: pd.DataFrame,
    spot_grid: np.ndarray | None = None,
    grid_pct_range: float = 0.05,
    grid_points: int = 101,
    r: float = 0.0,
    q: float = 0.0,
) -> pd.DataFrame:
    """
    Reprice total dollar GEX across a grid of HYPOTHETICAL spot prices,
    holding each contract's strike/OI/IV/DTE fixed. This is what's actually
    needed to find the zero-gamma flip -- a single dollar_gex number at the
    CURRENT spot (from aggregate_chain_exposure) cannot tell you where GEX
    crosses zero, since that requires knowing GEX's shape as a function of S.

    chain: same schema as options_greeks.aggregate_chain_exposure() expects
    (spot, strike, dte, iv, open_interest, option_type). The chain's own
    "spot" column is IGNORED for this repricing -- only strike/dte/iv/OI/
    option_type are used; spot_grid supplies the S values instead.

    spot_grid: explicit array of spot prices to evaluate at. If None,
    defaults to +/- grid_pct_range around the chain's current spot
    (its first row's spot value), with grid_points steps.

    CAVEAT (flagged, not buried): this holds IV FIXED as spot moves, which
    ignores the vol smile shifting as spot moves (a real market effect,
    "sticky strike" vs "sticky delta" assumptions differ here). This is the
    standard simplification used in public zero-gamma calculators, but it
    is an approximation, not an exact repricing -- don't treat S* as more
    precise than the IV-fixed assumption allows.

    Returns a DataFrame with columns: spot, dollar_gex (total, all contracts).
    """
    missing = REQUIRED_COLUMNS - {c.lower() for c in chain.columns}
    if missing:
        raise ValueError(f"chain DataFrame missing required columns: {missing}")

    df = chain.copy()
    df.columns = [c.lower() for c in df.columns]

    if spot_grid is None:
        current_spot = float(df["spot"].iloc[0])
        spot_grid = np.linspace(
            current_spot * (1 - grid_pct_range),
            current_spot * (1 + grid_pct_range),
            grid_points,
        )

    K = df["strike"].to_numpy(dtype=float)
    T = df["dte"].to_numpy(dtype=float) / 365.0
    sigma = df["iv"].to_numpy(dtype=float)
    OI = df["open_interest"].to_numpy(dtype=float)
    otype = df["option_type"].to_numpy()
    is_call = np.char.lower(otype.astype(str)).astype("U1") == "c"
    sign = np.where(is_call, 1.0, -1.0)

    if np.any(T <= 0):
        raise ValueError(
            "compute_gex_profile: found dte <= 0 -- filter 0DTE-at-close "
            "rows before calling, same caveat as aggregate_chain_exposure()."
        )

    totals = np.empty(len(spot_grid))
    for i, S_hyp in enumerate(spot_grid):
        S_arr = np.full_like(K, S_hyp)
        g = gamma(S_arr, K, T, sigma, r, q)
        dollar_gex = g * OI * 100 * S_arr ** 2 * 0.01 * sign
        totals[i] = dollar_gex.sum()

    return pd.DataFrame({"spot": spot_grid, "dollar_gex": totals})


def find_zero_gamma_flip(gex_profile: pd.DataFrame) -> float | None:
    """
    Given a GEX-vs-spot profile (from compute_gex_profile), find S* -- the
    spot price where cumulative dollar GEX crosses zero -- via linear
    interpolation between the two grid points that bracket the sign change.

    Returns None if the profile never changes sign across the grid (e.g.
    GEX is positive or negative everywhere in range -- widen grid_pct_range
    in compute_gex_profile() and retry).

    If GEX crosses zero MORE than once across the grid, returns the flip
    nearest to the middle of the grid (closest to current spot), since
    that's the economically relevant one for near-term hedging behavior --
    but logs nothing about the other crossings, so a caller who needs the
    full crossing structure should inspect gex_profile directly rather than
    relying on this function's single return value.
    """
    spot = gex_profile["spot"].to_numpy()
    gex = gex_profile["dollar_gex"].to_numpy()

    sign_changes = np.where(np.diff(np.sign(gex)) != 0)[0]
    if len(sign_changes) == 0:
        return None

    mid_idx = len(spot) // 2
    best_i = sign_changes[np.argmin(np.abs(sign_changes - mid_idx))]

    s0, s1 = spot[best_i], spot[best_i + 1]
    g0, g1 = gex[best_i], gex[best_i + 1]
    # linear interpolation for the zero crossing
    flip = s0 + (0 - g0) * (s1 - s0) / (g1 - g0)
    return float(flip)


# --------------------------------------------------------------------------
# 3. w_charm(t) -- convex time-of-day weight ramp (Phase 1 spec)
# --------------------------------------------------------------------------

def w_charm(
    minutes_since_open: float,
    minutes_total_session: float,
    w_min: float = 1.0,
    w_max: float = 2.5,
    p: float = 2.0,
) -> float:
    """
    w_charm(t) = w_min + (w_max - w_min) * clip[(t - t_start)/(t_close - t_start), 0, 1]^p

    Convex (p > 1) ramp that up-weights 0DTE charm's contribution to the
    confluence score later in the session, per the Phase 1 spec. t_start is
    implicitly 0 here (minutes_since_open already measures from session
    open) -- pass 0 for minutes_since_open at the open, and
    minutes_total_session at the close.

    w_min/w_max/p are explicitly flagged in the Phase 1 spec as tunable
    hyperparameters to fit empirically in backtest, not fixed constants --
    the defaults here (1.0, 2.5, 2.0) are the illustrative values from that
    spec, not calibrated values.
    """
    if minutes_total_session <= 0:
        raise ValueError("minutes_total_session must be > 0")
    frac = np.clip(minutes_since_open / minutes_total_session, 0.0, 1.0)
    return w_min + (w_max - w_min) * (frac ** p)


# --------------------------------------------------------------------------
# 4. Confluence score + Z_t band
# --------------------------------------------------------------------------

def _normalize(value: float, session_min: float, session_max: float) -> float:
    """
    Normalize a value to [-1, 1] using the trailing session's own min/max
    range, per the Phase 1 spec ("normalize each component to [-1,1] via
    its own trailing-session range before summing -- otherwise gamma
    dollar-notional dwarfs vanna/charm").

    session_min/session_max must be tracked externally (e.g. a rolling
    running min/max of that metric across the current session) -- this
    function does not maintain that state itself, since it depends on the
    live daemon's running history, not on a single chain snapshot.

    Returns 0.0 if session_max == session_min (no range yet, e.g. first
    tick of the day) rather than dividing by zero.
    """
    if session_max == session_min:
        return 0.0
    midpoint = (session_max + session_min) / 2
    half_range = (session_max - session_min) / 2
    return float(np.clip((value - midpoint) / half_range, -1.0, 1.0))


def confluence_score(
    dollar_gex: float,
    dollar_vanna: float,
    dollar_charm: float,
    gex_session_range: tuple[float, float],
    vanna_session_range: tuple[float, float],
    charm_session_range: tuple[float, float],
    w_charm_value: float,
) -> float:
    """
    Confluence(S) = GEX_norm(S) + VannaScore_norm(S) + w_charm(t)*CharmScore_norm(S)

    Each *_session_range is a (session_min, session_max) tuple for that
    metric, tracked externally across the session (see _normalize's
    docstring -- this function is stateless by design, the caller owns the
    running min/max history).
    """
    gex_norm = _normalize(dollar_gex, *gex_session_range)
    vanna_norm = _normalize(dollar_vanna, *vanna_session_range)
    charm_norm = _normalize(dollar_charm, *charm_session_range)
    return gex_norm + vanna_norm + w_charm_value * charm_norm


def build_zone(
    per_contract: pd.DataFrame,
    gex_profile: pd.DataFrame,
    confluence_by_strike: pd.Series,
    confluence_threshold: float,
) -> dict:
    """
    Assemble Z_t = [P_lower, P_upper] from the pieces above.

    confluence_by_strike: a pandas Series indexed by strike, giving that
    strike's confluence score (the caller computes this per-strike using
    confluence_score() above -- this function does not compute per-strike
    confluence itself, since the session-range normalization inputs differ
    per metric and are the caller's responsibility to track).

    Z_t is defined here as the nearest strikes below/above the current spot
    where |confluence| exceeds confluence_threshold, per the Phase 1 spec
    ("nearest strike below/above spot at which |Confluence(S)| exceeds a
    threshold tau_conf"). tau_conf itself is NOT chosen by this function --
    it's a Phase 3 backtest-calibrated hyperparameter, passed in by the
    caller.

    Returns:
        {
            "P_lower": float or None,
            "P_upper": float or None,
            "call_wall": dict or None,
            "put_wall": dict or None,
            "zero_gamma_flip": float or None,
        }
    """
    walls = find_walls(per_contract)
    flip = find_zero_gamma_flip(gex_profile)
    current_spot = float(per_contract["spot"].iloc[0])

    strikes = confluence_by_strike.index.to_numpy(dtype=float)
    scores = confluence_by_strike.to_numpy(dtype=float)
    qualifying = strikes[np.abs(scores) > confluence_threshold]

    below = qualifying[qualifying < current_spot]
    above = qualifying[qualifying > current_spot]

    P_lower = float(below.max()) if len(below) else None
    P_upper = float(above.min()) if len(above) else None

    return {
        "P_lower": P_lower,
        "P_upper": P_upper,
        "call_wall": walls["call_wall"],
        "put_wall": walls["put_wall"],
        "zero_gamma_flip": flip,
    }


if __name__ == "__main__":
    from factors.options_greeks import aggregate_chain_exposure

    demo_chain = pd.DataFrame({
        "spot": [500.0] * 6,
        "strike": [480.0, 480.0, 495.0, 495.0, 510.0, 510.0],
        "dte": [30] * 6,
        "iv": [0.22, 0.24, 0.19, 0.21, 0.18, 0.20],
        "open_interest": [6000, 4000, 10000, 8000, 12000, 9000],
        "option_type": ["C", "P", "C", "P", "C", "P"],
    })

    result = aggregate_chain_exposure(demo_chain)
    walls = find_walls(result["per_contract"])
    print("Call wall:", walls["call_wall"])
    print("Put wall: ", walls["put_wall"])

    profile = compute_gex_profile(demo_chain, grid_pct_range=0.15, grid_points=161)
    flip = find_zero_gamma_flip(profile)
    print(f"Zero-gamma flip (S*): {flip}")

    w = w_charm(minutes_since_open=350, minutes_total_session=390)
    print(f"w_charm at 350/390 min into session: {w:.3f}")
