"""
databento_futures_client.py

Pass 2 of the two-pass ingestion plan: pull CME Globex MBP-10 (market-by-price,
10 levels) L2 data ONLY for the specific ~30-minute windows flagged by a prior
Pass-1 L1/OHLCV backtest as candidate Tier-2 "armed" periods.

This deliberately does NOT do a bulk historical pull. Databento historical
usage is metered, and MBP-10 is one of the more expensive schemas per the
$125 free-credit budget, so the whole point of Pass 1 is to shrink the set
of windows you ever pay to fetch at L2 resolution.

Requires:
    pip install databento
    export DATABENTO_API_KEY="db-xxxxxxxxxxxx"

Docs verified against Databento's own quickstart / NautilusTrader Databento
catalog how-to as of this writing. If Databento changes schema names or
continuous-contract symbology conventions, this will need updating -- always
check `client.metadata.list_schemas(dataset=...)` before assuming a schema
name is still valid.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import databento as db
import pandas as pd

logger = logging.getLogger(__name__)

DATASET = "GLBX.MDP3"          # CME Globex MDP 3.0 -- covers NQ/MNQ futures
SCHEMA = "mbp-10"              # 10-level market-by-price (our L2 depth proxy)
DEFAULT_SYMBOL = "NQ.n.0"      # front-month continuous contract, "n"othing gaps
STYPE_IN = "continuous"


@dataclass(frozen=True)
class SniperWindow:
    """One Pass-1-flagged window we want L2 data for."""
    start: datetime   # UTC
    end: datetime      # UTC
    label: str         # e.g. the Tier-2 arm event id, for traceability


class DatabentoFuturesClient:
    """
    Thin wrapper around databento.Historical scoped to the sniper-pull
    workflow: given a list of narrow time windows, fetch MBP-10 for each,
    write to local .dbn.zst files, and load into pandas only on demand
    (MBP-10 frames get large fast -- don't materialize everything at once).
    """

    def __init__(self, cache_dir: str | Path = "./data_cache/databento_mbp10"):
        api_key = os.environ.get("DATABENTO_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DATABENTO_API_KEY not set. On a VPS/mobile SSH session: "
                "export DATABENTO_API_KEY='db-xxxx' (add to ~/.bashrc or "
                "your systemd service env file so it persists)."
            )
        self.client = db.Historical(api_key)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, window: SniperWindow, symbol: str) -> Path:
        fname = f"{symbol.replace('.', '_')}_{window.label}_{window.start:%Y%m%dT%H%M}.dbn.zst"
        return self.cache_dir / fname

    def estimate_cost(self, window: SniperWindow, symbol: str = DEFAULT_SYMBOL) -> float:
        """
        Always check cost before pulling -- Databento bills per request even
        if you've fetched overlapping data before. Returns cost in USD.
        """
        cost = self.client.metadata.get_cost(
            dataset=DATASET,
            symbols=[symbol],
            stype_in=STYPE_IN,
            schema=SCHEMA,
            start=window.start,
            end=window.end,
        )
        return float(cost)

    def fetch_window(
        self,
        window: SniperWindow,
        symbol: str = DEFAULT_SYMBOL,
        force_refetch: bool = False,
    ) -> Path:
        """
        Fetch one sniper window's MBP-10 data to disk, skipping the request
        entirely if it's already cached locally (protects your free credit
        from accidental re-runs).
        """
        path = self._cache_path(window, symbol)
        if path.exists() and not force_refetch:
            logger.info("cache hit, skipping fetch: %s", path.name)
            return path

        est = self.estimate_cost(window, symbol)
        logger.info(
            "fetching %s [%s -> %s] est. cost=$%.4f",
            window.label, window.start, window.end, est,
        )

        self.client.timeseries.get_range(
            dataset=DATASET,
            symbols=[symbol],
            stype_in=STYPE_IN,
            schema=SCHEMA,
            start=window.start,
            end=window.end,
            path=path,
        )
        return path

    def fetch_all(
        self,
        windows: list[SniperWindow],
        symbol: str = DEFAULT_SYMBOL,
        budget_usd: float | None = None,
    ) -> list[Path]:
        """
        Fetch every window, but stop early (and log clearly) if a cumulative
        budget_usd is set and would be exceeded -- protects the $125 free
        credit across a long list of Pass-1-flagged windows.
        """
        paths: list[Path] = []
        spent = 0.0
        for w in windows:
            est = self.estimate_cost(w, symbol)
            if budget_usd is not None and spent + est > budget_usd:
                logger.warning(
                    "stopping: window %s would push spend to $%.4f > budget $%.2f",
                    w.label, spent + est, budget_usd,
                )
                break
            paths.append(self.fetch_window(w, symbol))
            spent += est
        logger.info("fetch_all complete: %d windows, ~$%.4f spent", len(paths), spent)
        return paths

    @staticmethod
    def load_mbp10(path: Path) -> pd.DataFrame:
        """Load a cached .dbn.zst file into a pandas DataFrame for factor calcs."""
        store = db.DBNStore.from_file(path)
        return store.to_df()


if __name__ == "__main__":
    # Minimal smoke test: fetch 2 minutes of NQ MBP-10 for a known-liquid
    # window. Run this once to confirm credentials + schema access before
    # wiring in real Pass-1-flagged windows.
    logging.basicConfig(level=logging.INFO)
    client = DatabentoFuturesClient()
    test_window = SniperWindow(
        start=datetime(2024, 6, 10, 14, 30),
        end=datetime(2024, 6, 10, 14, 32),
        label="smoketest",
    )
    p = client.fetch_window(test_window)
    df = client.load_mbp10(p)
    print(df.head())
    print(f"rows: {len(df)}, columns: {list(df.columns)}")
