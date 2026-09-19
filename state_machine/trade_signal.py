"""
state_machine/trade_signal.py

Shared TradeSignal schema, per the Layer G requirement that the live
daemon's Discord alert content and the Nautilus backtest's logged trade
must be byte-for-byte the same function output -- both import this
dataclass rather than each defining their own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TradeSignal:
    """
    One EXECUTION_ARMED event's full detail -- everything needed to place
    the trade AND everything needed to log/alert it identically.
    """
    timestamp: datetime
    entry_price: float
    stop_price: float           # fixed 10-15 NQ points beyond the swept wick
    target_price: float         # nearer of session VWAP or session POC
    swept_extreme_price: float  # the wick this setup anchors to
    sweep_direction: int        # +1 (swept high, expecting reversal down) or -1
    zone_p_lower: float | None
    zone_p_upper: float | None
    call_wall: float | None
    put_wall: float | None
    zero_gamma_flip: float | None
    vpin_at_trigger: float | None
    kyle_lambda_at_trigger: float | None
    ofi_at_trigger: float | None
    notes: str = ""
    tags: dict = field(default_factory=dict)  # open-ended extra context

    def to_discord_embed_dict(self) -> dict:
        """
        Common formatting for the Discord Execution Ping (Layer G) --
        both the live daemon and anything replaying signals for visual
        verification should call THIS, not hand-format their own message,
        so the alert content and the logged signal never drift apart.
        """
        return {
            "title": "Execution Trigger",
            "fields": [
                {"name": "Entry", "value": f"{self.entry_price:.2f}"},
                {"name": "Stop", "value": f"{self.stop_price:.2f}"},
                {"name": "Target", "value": f"{self.target_price:.2f}"},
                {"name": "Swept Extreme", "value": f"{self.swept_extreme_price:.2f}"},
                {"name": "Zone", "value": f"[{self.zone_p_lower}, {self.zone_p_upper}]"},
            ],
            "timestamp": self.timestamp.isoformat(),
        }
