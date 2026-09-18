"""
lse_options_client.py

Wraps the `lse-data` SDK's options endpoints for Tier 1 factor inputs
(GEX, Vanna, Charm need: strike, expiry, OI, IV, gamma/vanna/charm or the
raw greeks to derive them, per-underlying).

Confirmed SDK surface (from PyPI/GitHub docs, verify against your installed
version with `pip show lse-data`):
    client.options(underlying, type=..., max_dte=...)   -> chain snapshot
    client.options_flow(underlying, min_premium=...)     -> individual prints
    client.option_candles(underlying, strike=, expiry=, type=) -> 1-min bars
    client.options_underlyings()                         -> available names

IMPORTANT: the docs describe options() as returning "the latest price,
implied volatility, greeks, and volume/premium traded today" -- it does NOT
explicitly confirm an open-interest (OI) field in that payload, which your
GEX formula requires (GEX = OI * gamma * 100 * S). Run
schema_inspect_lse_options.py (sibling script) before writing any factor
math against an assumed `oi` column -- confirm the field exists and get its
actual update cadence rather than assuming.

Requires:
    pip install 'lse-data[frames]'
    export LSE_API_KEY="your_key"
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from lse import LSE

logger = logging.getLogger(__name__)

NQ_UNDERLYING = "NQ"   # confirm exact ticker convention via options_underlyings()
ES_UNDERLYING = "ES"


class LSEOptionsClient:
    def __init__(self, cache_dir: str | Path = "./data_cache/lse_options"):
        api_key = os.environ.get("LSE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LSE_API_KEY not set. export LSE_API_KEY='your_key' "
                "(persist it in ~/.bashrc or your VPS service env file)."
            )
        self.client = LSE(api_key=api_key)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def list_underlyings(self) -> list[str]:
        """Confirm the exact ticker string LSE uses for NQ/ES options first."""
        return self.client.options_underlyings()

    def get_chain_snapshot(
        self,
        underlying: str,
        option_type: str | None = None,
        max_dte: int | None = None,
    ) -> pd.DataFrame:
        """
        One row per contract: price, IV, greeks, volume/premium today.
        This is your Tier 1 raw input -- GEX/Vanna/Charm normalization
        happens downstream in factors/gamma_exposure.py, NOT here. This
        client stays a thin, dumb data-fetch layer.
        """
        kwargs = {}
        if option_type is not None:
            kwargs["type"] = option_type
        if max_dte is not None:
            kwargs["max_dte"] = max_dte

        df = self.client.options(underlying, **kwargs)

        missing_expected = {"strike", "expiry"} - set(df.columns)
        if missing_expected:
            logger.warning(
                "chain snapshot missing expected columns %s -- schema may "
                "have changed, re-run schema_inspect_lse_options.py",
                missing_expected,
            )
        if "oi" not in df.columns and "open_interest" not in df.columns:
            logger.warning(
                "NO OPEN INTEREST FIELD FOUND in chain snapshot. GEX math "
                "requires OI -- confirm the actual field name/availability "
                "before wiring this into factors/gamma_exposure.py."
            )
        return df

    def get_0dte_slice(self, underlying: str) -> pd.DataFrame:
        """Convenience wrapper: chain filtered to same-day expiry only,
        for the Tier 1 charm/vanna weighting that's 0DTE-specific."""
        chain = self.get_chain_snapshot(underlying, max_dte=0)
        return chain

    def snapshot_to_parquet(self, underlying: str, label: str) -> Path:
        """Persist a timestamped snapshot -- this is what builds your
        point-in-time store for backtesting (Layer A). Never overwrite;
        each snapshot is a distinct point-in-time observation."""
        df = self.get_chain_snapshot(underlying)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        path = self.cache_dir / f"{underlying}_{label}_{ts}.parquet"
        df.to_parquet(path)
        logger.info("wrote snapshot: %s (%d rows)", path.name, len(df))
        return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = LSEOptionsClient()

    print("Available underlyings (confirm NQ/ES naming convention):")
    print(client.list_underlyings())

    chain = client.get_chain_snapshot(NQ_UNDERLYING, max_dte=45)
    print(f"\nColumns returned: {list(chain.columns)}")
    print(chain.head())
