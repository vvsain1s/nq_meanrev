"""
schema_inspect_lse_options.py

Answers the question the docs don't: what is the ACTUAL update cadence of
OI, IV, and greeks in the lse-data options feed -- 1-min, 5-min, EOD-only?

Method: poll the same chain snapshot repeatedly over a short window and
diff consecutive pulls per-contract. If a column's values are identical
across N consecutive 1-minute polls and then jump, that's your empirical
cadence -- don't trust vendor docs on this, confirm it yourself, since your
whole Tier 1 dynamic-recompute schedule (1-5 min) depends on the data
actually refreshing that often.

Run this for at least 20-30 minutes during RTH before writing any factor
math that assumes a specific refresh rate.

Usage (from a VPS/SSH session, runs in the foreground -- use `screen` or
`tmux` if running over Termius so it survives a dropped mobile connection):

    python schema_inspect_lse_options.py --underlying NQ --minutes 30
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone

import pandas as pd

from lse_options_client import LSEOptionsClient, NQ_UNDERLYING

logger = logging.getLogger(__name__)


def _diff_snapshot(prev: pd.DataFrame, curr: pd.DataFrame, key_cols: list[str]) -> dict:
    """
    For each tracked column, report whether ANY row changed vs. the
    previous poll. Returns {column_name: changed_bool}.
    """
    if prev.empty or curr.empty:
        return {}

    merged = prev.merge(curr, on=key_cols, suffixes=("_prev", "_curr"), how="inner")
    if merged.empty:
        logger.warning("no overlapping contracts between polls -- chain may have rolled")
        return {}

    result = {}
    candidate_cols = [c for c in curr.columns if c not in key_cols]
    for col in candidate_cols:
        prev_col, curr_col = f"{col}_prev", f"{col}_curr"
        if prev_col in merged.columns and curr_col in merged.columns:
            changed = not merged[prev_col].equals(merged[curr_col])
            result[col] = changed
    return result


def run(underlying: str, minutes: int, poll_interval_sec: int = 60) -> None:
    client = LSEOptionsClient()
    key_cols_candidates = ["strike", "expiry", "type"]

    prev_snapshot: pd.DataFrame = pd.DataFrame()
    change_log: list[dict] = []

    n_polls = max(2, (minutes * 60) // poll_interval_sec)
    logger.info("polling %s every %ds for ~%d polls (%d min)", underlying, poll_interval_sec, n_polls, minutes)

    for i in range(n_polls):
        ts = datetime.now(timezone.utc)
        curr = client.get_chain_snapshot(underlying)
        key_cols = [c for c in key_cols_candidates if c in curr.columns]

        if not key_cols:
            logger.error(
                "none of expected key columns %s present in response -- "
                "actual columns: %s. Adjust key_cols_candidates and rerun.",
                key_cols_candidates, list(curr.columns),
            )
            return

        if not prev_snapshot.empty:
            diffs = _diff_snapshot(prev_snapshot, curr, key_cols)
            diffs["_timestamp"] = ts.isoformat()
            diffs["_poll_index"] = i
            change_log.append(diffs)
            logger.info("poll %d @ %s: changed columns = %s", i, ts.isoformat(),
                        [c for c, changed in diffs.items() if changed is True])

        prev_snapshot = curr
        if i < n_polls - 1:
            time.sleep(poll_interval_sec)

    report = pd.DataFrame(change_log)
    out_path = f"lse_options_cadence_report_{underlying}_{datetime.now():%Y%m%d_%H%M}.csv"
    report.to_csv(out_path, index=False)

    print("\n=== CADENCE SUMMARY ===")
    for col in report.columns:
        if col.startswith("_"):
            continue
        change_rate = report[col].mean() if report[col].dtype == bool else None
        if change_rate is not None:
            print(f"  {col:20s} changed in {change_rate*100:5.1f}% of polls "
                  f"({'~every poll' if change_rate > 0.8 else 'infrequent -- check report CSV for exact timing'})")
    print(f"\nFull per-poll detail written to: {out_path}")
    print("Look at OI / iv / gamma / delta specifically -- that answers "
          "your w_charm(t) and dynamic Z_t recompute scheduling question.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--underlying", default=NQ_UNDERLYING)
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--poll-interval-sec", type=int, default=60)
    args = parser.parse_args()
    run(args.underlying, args.minutes, args.poll_interval_sec)
