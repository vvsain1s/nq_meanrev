"""
factors/options_greeks.py

Tier 1 factor engine: contract-level Black-Scholes Greeks (Gamma, Vanna, Charm)
and chain-wide dollar exposure aggregation (GEX, dollar Vanna, dollar Charm)
under the standard dealer positioning convention.

SCOPE: this module is Tier 1 math ONLY. It does not touch Tier 2 (VPIN,
Kyle's Lambda), Tier 3 (OFI/execution trigger), the state machine, or the
w_charm(t) time-of-day weighting from the Phase 1 spec -- those are separate,
not-yet-built modules.

Formulas (standard Black-Scholes-Merton, continuous dividend/cost-of-carry q):
    d1 = [ln(S/K) + (r - q + sigma^2/2) * T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    Gamma (same for call/put) = e^(-qT) * phi(d1) / (S * sigma * sqrt(T))
    Vanna (same for call/put) = -e^(-qT) * phi(d1) * d2 / sigma
    Charm (= -dDelta/dT, i.e. delta decay per unit of calendar time):
        call: q*e^(-qT)*N(d1)  - e^(-qT)*phi(d1) * [ (r-q+sigma^2/2)/(sigma*sqrt(T)) - d1/(2T) ]
        put:  charm_call - q*e^(-qT)

Charm's call/put formulas were derived here from first principles (d(Delta)/dT
of the standard BSM delta) rather than copied from a secondary source, BECAUSE
charm sign-convention writeups vary across vendors/blogs and a copied formula
with a silent sign error would break Tier 1 without any error message. The
`_self_validate()` function at the bottom of this file checks every closed-form
formula against a finite-difference numerical derivative at import time -- if
you ever edit this math, that check will fail loudly rather than silently
producing wrong dealer-positioning signals.

Dollar exposure aggregation (per confirmed public GEX/VEX/CHEX conventions):
    GEX_contract   = Gamma * OI * 100 * S^2 * 0.01 * sign
    VEX_contract   = Vanna * OI * 100 * S       * sign
    CHEX_contract  = Charm * OI * 100 * S       * sign

Dealer sign convention (per spec): dealers are assumed short customer calls,
long customer puts.
    calls: sign = +1  (dealer is short calls -> short gamma/vanna/charm on
                        that leg is the OPPOSITE of the raw option's greek,
                        BUT the public GEX/VEX/CHEX convention already bakes
                        "dealer short calls" into using +1 for calls -- see
                        docstring note in aggregate_chain_exposure() below)
    puts:  sign = -1

Caveat (flag explicitly, don't bury it): this call+/put- convention is a
market-wide simplifying assumption about AGGREGATE dealer inventory, not a
verified fact about any specific dealer's book. It is empirically reasonable
for index/ETF options (SPX, QQQ, SPY, and by extension NQ/ES-linked options)
but can be wrong for single names with unusual institutional positioning.
Treat GEX/VEX/CHEX as a well-established INDUSTRY MODEL, not ground truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

# --------------------------------------------------------------------------
# Contract-level Greeks
# --------------------------------------------------------------------------

def _d1_d2(S: np.ndarray, K: np.ndarray, T: np.ndarray, sigma: np.ndarray,
           r: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Shared d1/d2 computation. T in years, sigma annualized IV (decimal)."""
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def gamma(S, K, T, sigma, r: float = 0.0, q: float = 0.0):
    """
    Black-Scholes Gamma (per share). Identical for calls and puts.

    S: spot, K: strike, T: time to expiry in YEARS, sigma: annualized IV
    (decimal, e.g. 0.20 for 20%), r: risk-free rate, q: continuous dividend
    / cost-of-carry yield. Vectorized -- pass numpy arrays or pandas Series.
    """
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    d1, _ = _d1_d2(S, K, T, sigma, r, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vanna(S, K, T, sigma, r: float = 0.0, q: float = 0.0):
    """
    Black-Scholes Vanna (per share) = dDelta/dVol = dVega/dSpot.
    Identical for calls and puts (a consequence of put-call parity: the
    call-put delta difference is a constant w.r.t. sigma).
    """
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    d1, d2 = _d1_d2(S, K, T, sigma, r, q)
    return -np.exp(-q * T) * norm.pdf(d1) * d2 / sigma


def charm(S, K, T, sigma, option_type, r: float = 0.0, q: float = 0.0):
    """
    Black-Scholes Charm (per share) = -dDelta/dT_expiry = dDelta/dt_calendar,
    i.e. "how much delta decays as one day of calendar time passes." This
    DOES differ between calls and puts (unlike gamma/vanna).

    option_type: array-like of 'C'/'call' or 'P'/'put' (case-insensitive),
    same length as S/K/T/sigma.
    """
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    option_type = np.asarray(option_type)
    is_call = np.char.lower(option_type.astype(str)).astype("U1") == "c"

    d1, _ = _d1_d2(S, K, T, sigma, r, q)
    disc = np.exp(-q * T)
    term = disc * norm.pdf(d1) * ((r - q + 0.5 * sigma ** 2) / (sigma * np.sqrt(T)) - d1 / (2 * T))

    charm_call = q * disc * norm.cdf(d1) - term
    charm_put = charm_call - q * disc
    return np.where(is_call, charm_call, charm_put)


# --------------------------------------------------------------------------
# Chain-wide dollar exposure aggregation
# --------------------------------------------------------------------------

REQUIRED_COLUMNS = {"spot", "strike", "dte", "iv", "open_interest", "option_type"}


def aggregate_chain_exposure(
    chain: pd.DataFrame,
    r: float = 0.0,
    q: float = 0.0,
) -> dict:
    """
    Process an options chain DataFrame and return chain-wide Dollar GEX,
    Dollar Vanna, and Dollar Charm under the standard dealer positioning
    convention (dealers short customer calls, long customer puts).

    Expected columns (case-insensitive, matched by name):
        spot          -- underlying price (same for every row; if it varies
                          row-to-row this function uses each row's own value,
                          which is fine for a single time-sliced snapshot)
        strike        -- option strike
        dte           -- days to expiry (converted to years internally)
        iv            -- annualized implied volatility, DECIMAL (0.20, not 20)
        open_interest -- contracts outstanding
        option_type   -- 'C'/'call' or 'P'/'put'

    Returns:
        {
            "dollar_gex": float,    # total dollar gamma exposure (per 1% move)
            "dollar_vanna": float,  # total dollar vanna exposure (per 1% vol move)
            "dollar_charm": float,  # total dollar charm exposure (per day)
            "per_contract": pd.DataFrame  # row-level gamma/vanna/charm + signed $ values, for strike-level GEX profiles
        }

    Sign convention note: the public GEX/VEX/CHEX formula uses sign=+1 for
    calls and sign=-1 for puts DIRECTLY on the raw (long) option greek --
    this is the standard convention because it's calibrated to express "what
    dealers must do to stay hedged," not the dealer's own P&L greek. A dealer
    short a call has to hedge as if THEY held the call's gamma (they buy/sell
    the same shares a long-call holder's delta-hedge would require, just to
    flatten their own book) -- so the sign convention operates on the raw
    option greek with a market-wide +call/-put convention, not by flipping
    the sign of the greek itself to represent "dealer's short position."
    This matches every source checked (SpotGamma, FlashAlpha CHEX/VEX docs,
    expireworthless.com) and is NOT something to "simplify" by naively
    negating the greek for calls -- that would produce the opposite of every
    public GEX implementation and silently invert your gamma-flip level.
    """
    missing = REQUIRED_COLUMNS - {c.lower() for c in chain.columns}
    if missing:
        raise ValueError(f"chain DataFrame missing required columns: {missing}")

    df = chain.copy()
    df.columns = [c.lower() for c in df.columns]

    S = df["spot"].to_numpy(dtype=float)
    K = df["strike"].to_numpy(dtype=float)
    T = df["dte"].to_numpy(dtype=float) / 365.0
    sigma = df["iv"].to_numpy(dtype=float)
    OI = df["open_interest"].to_numpy(dtype=float)
    otype = df["option_type"].to_numpy()

    if np.any(T <= 0):
        raise ValueError(
            "aggregate_chain_exposure: found dte <= 0 (expired/0-DTE-at-close "
            "rows). Filter these before calling, or the T=0 division will "
            "produce inf/nan -- 0DTE contracts need same-day-expiry handling "
            "this function does not attempt."
        )

    g = gamma(S, K, T, sigma, r, q)
    v = vanna(S, K, T, sigma, r, q)
    c = charm(S, K, T, sigma, otype, r, q)

    is_call = np.char.lower(otype.astype(str)).astype("U1") == "c"
    sign = np.where(is_call, 1.0, -1.0)

    dollar_gex_contract = g * OI * 100 * S ** 2 * 0.01 * sign
    dollar_vanna_contract = v * OI * 100 * S * sign
    dollar_charm_contract = c * OI * 100 * S * sign

    per_contract = df.copy()
    per_contract["gamma"] = g
    per_contract["vanna"] = v
    per_contract["charm"] = c
    per_contract["dollar_gex"] = dollar_gex_contract
    per_contract["dollar_vanna"] = dollar_vanna_contract
    per_contract["dollar_charm"] = dollar_charm_contract

    return {
        "dollar_gex": float(dollar_gex_contract.sum()),
        "dollar_vanna": float(dollar_vanna_contract.sum()),
        "dollar_charm": float(dollar_charm_contract.sum()),
        "per_contract": per_contract,
    }


# --------------------------------------------------------------------------
# Self-validation: closed-form Greeks vs. finite-difference numerical
# derivatives. Runs on import (cheap, a handful of scalar evaluations) and
# raises immediately if any formula's sign or magnitude is wrong. This is
# the safety net referenced in the module docstring -- don't remove it
# without replacing it with an equivalent check.
# --------------------------------------------------------------------------

def _bs_price(S, K, T, sigma, option_type, r=0.0, q=0.0):
    d1, d2 = _d1_d2(S, K, T, sigma, r, q)
    if option_type.lower().startswith("c"):
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def _bs_delta(S, K, T, sigma, option_type, r=0.0, q=0.0):
    d1, _ = _d1_d2(S, K, T, sigma, r, q)
    if option_type.lower().startswith("c"):
        return np.exp(-q * T) * norm.cdf(d1)
    return np.exp(-q * T) * (norm.cdf(d1) - 1)


def _self_validate(tol: float = 1e-3) -> None:
    S, K, T, sigma, r, q = 500.0, 505.0, 30 / 365.0, 0.18, 0.04, 0.0
    h_S, h_sigma, h_T = 0.01, 1e-4, 1e-5

    for opt in ("call", "put"):
        # Gamma: d(delta)/dS via central difference
        delta_up = _bs_delta(S + h_S, K, T, sigma, opt, r, q)
        delta_dn = _bs_delta(S - h_S, K, T, sigma, opt, r, q)
        gamma_numeric = (delta_up - delta_dn) / (2 * h_S)
        gamma_closed = float(gamma(S, K, T, sigma, r, q))
        assert abs(gamma_closed - gamma_numeric) < tol, (
            f"Gamma mismatch ({opt}): closed={gamma_closed:.6f} "
            f"numeric={gamma_numeric:.6f}"
        )

        # Vanna: d(delta)/dsigma via central difference
        delta_up = _bs_delta(S, K, T, sigma + h_sigma, opt, r, q)
        delta_dn = _bs_delta(S, K, T, sigma - h_sigma, opt, r, q)
        vanna_numeric = (delta_up - delta_dn) / (2 * h_sigma)
        vanna_closed = float(vanna(S, K, T, sigma, r, q))
        assert abs(vanna_closed - vanna_numeric) < tol, (
            f"Vanna mismatch ({opt}): closed={vanna_closed:.6f} "
            f"numeric={vanna_numeric:.6f}"
        )

        # Charm: -d(delta)/dT via central difference (T = time to expiry;
        # charm is defined as delta decay per unit of CALENDAR time, i.e.
        # -dDelta/dT).
        delta_up = _bs_delta(S, K, T + h_T, sigma, opt, r, q)
        delta_dn = _bs_delta(S, K, T - h_T, sigma, opt, r, q)
        charm_numeric = -(delta_up - delta_dn) / (2 * h_T)
        charm_closed = float(charm(S, K, T, sigma, np.array([opt]), r, q)[0])
        assert abs(charm_closed - charm_numeric) < tol, (
            f"Charm mismatch ({opt}): closed={charm_closed:.6f} "
            f"numeric={charm_numeric:.6f}"
        )


_self_validate()


if __name__ == "__main__":
    # Small worked example, matching the style of the public GEX writeups
    # this module's formulas were checked against.
    demo_chain = pd.DataFrame({
        "spot": [500.0] * 4,
        "strike": [495.0, 495.0, 505.0, 505.0],
        "dte": [30, 30, 30, 30],
        "iv": [0.19, 0.21, 0.18, 0.20],
        "open_interest": [10000, 8000, 12000, 9000],
        "option_type": ["C", "P", "C", "P"],
    })
    result = aggregate_chain_exposure(demo_chain)
    print(f"Dollar GEX:   ${result['dollar_gex']:,.0f}")
    print(f"Dollar Vanna: ${result['dollar_vanna']:,.0f}")
    print(f"Dollar Charm: ${result['dollar_charm']:,.0f}")
    print("\nPer-contract detail:")
    print(result["per_contract"][["strike", "option_type", "gamma", "vanna", "charm", "dollar_gex"]])
