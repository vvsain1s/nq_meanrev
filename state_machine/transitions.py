"""
state_machine/transitions.py

The actual state machine: sequences the Tier 1/2/3 factor modules into
IDLE -> HTF_ZONE -> LTF_ARMED -> LTF_EXHAUSTED -> EXECUTION_ARMED ->
IN_POSITION, per the architecture spec.

SCOPE AND KNOWN GAPS (flagged explicitly, not glossed over):
  1. VWAP / session POC are NOT computed anywhere in this codebase yet.
     TradeSignal.target_price needs "nearer of session VWAP or session
     POC" -- this module accepts them as externally-supplied inputs to
     on_tick() rather than faking a placeholder calculation. Whoever
     wires this into the live daemon needs to build that calculation (it
     needs a running volume-at-price histogram for POC, and a running
     cumulative(price*volume)/cumulative(volume) for VWAP) before
     EXECUTION_ARMED signals are actually usable.
  2. Swept-extreme detection is DELIBERATELY SIMPLE here: this class
     tracks running session high/low as "the extreme" rather than
     detecting a genuine sweep-and-reversal candlestick pattern. A real
     implementation needs sweep-specific logic (e.g. "price broke the
     prior N-bar high by at least X ticks and closed back below it") --
     that pattern-detection logic does not exist in this codebase and is
     NOT invented here as a guess.
  3. This class is fed pre-computed snapshots via on_tick() -- it does
     NOT fetch data itself, does not know about lse-data/Databento, and
     is not wired into live/render_app.py's daemon_loop. That wiring is
     a separate step (see render_app.py's TODO comments once this lands).

Grandfathering (Layer I, approved decision from Phase 2): if Z_t shifts
mid-flight while a setup is armed/exhausted, the ALREADY-ARMED zone is
kept rather than re-validated against the new zone.

Zone-exit behavior (per spec, and per a bug caught by this module's own
test run -- see git history / conversation for the specifics): LTF_ARMED
resets to HTF_ZONE (not IDLE) if spot leaves the armed zone before
exhaustion fires. LTF_EXHAUSTED does NOT re-check zone membership at all
once entered -- a sweep is defined as price moving BEYOND the zone band,
so requiring "still inside the zone" during the exhaustion/reversal watch
would contradict the scenario Tier 3 exists to detect. Only the dwell
timer can move LTF_EXHAUSTED back to LTF_ARMED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from state_machine.states import EngineState, LTF_EXHAUSTED_MAX_DWELL_SECONDS
from state_machine.trade_signal import TradeSignal


@dataclass
class SessionRangeTracker:
    """
    Running min/max for one metric across the session, feeding the
    normalization step in factors.zone_composite.confluence_score()
    (which explicitly requires the caller to track this history --
    see that module's docstring). One instance per metric (GEX, Vanna,
    Charm).
    """
    running_min: float | None = None
    running_max: float | None = None

    def update(self, value: float) -> tuple[float, float]:
        if self.running_min is None or value < self.running_min:
            self.running_min = value
        if self.running_max is None or value > self.running_max:
            self.running_max = value
        return self.running_min, self.running_max


@dataclass
class MarketSnapshot:
    """
    One tick/update's worth of pre-computed inputs. The daemon loop is
    responsible for calling the factor modules and assembling this --
    this class is a plain data container, it does not call
    factors.options_greeks, factors.vpin, etc. itself.
    """
    timestamp: datetime
    spot: float

    # Tier 1 (from factors.zone_composite, already computed by caller)
    dollar_gex: float
    dollar_vanna: float
    dollar_charm: float
    call_wall: float | None
    put_wall: float | None
    zero_gamma_flip: float | None
    zone_p_lower: float | None
    zone_p_upper: float | None

    # Tier 2 (from factors.vpin / factors.kyle_lambda)
    vpin: float | None
    vpin_condition_a: bool  # factors.order_flow_imbalance.condition_a_vpin_rollover
    depth_exhaustion: bool  # factors.kyle_lambda.depth_exhaustion, this tick

    # Tier 3
    ofi_condition_b: bool  # factors.order_flow_imbalance.condition_b_ofi_flip

    # External gates
    econ_mute: bool  # True during the [-15, +15] min high-impact-event window
    in_maintenance_window: bool  # from live.render_app.in_cme_maintenance_window

    # Session extremes (see module docstring gap #2 -- simple running
    # high/low, not genuine sweep-pattern detection)
    session_high: float
    session_low: float

    # NOT computed anywhere yet (see module docstring gap #1) -- pass
    # None until a VWAP/POC module exists; on_tick() will refuse to
    # produce an EXECUTION_ARMED TradeSignal without these.
    session_vwap: float | None = None
    session_poc: float | None = None


class StateMachine:
    def __init__(
        self,
        confluence_threshold: float,
        stop_offset_points: float,
        max_dwell_seconds: float = LTF_EXHAUSTED_MAX_DWELL_SECONDS,
    ):
        if not (10.0 <= stop_offset_points <= 15.0):
            raise ValueError(
                f"stop_offset_points={stop_offset_points} outside the spec's "
                f"fixed 10-15 NQ point range -- pass a value in that range "
                f"or update the spec first, don't silently accept anything."
            )
        self.confluence_threshold = confluence_threshold
        self.stop_offset_points = stop_offset_points
        self.max_dwell_seconds = max_dwell_seconds

        self.state = EngineState.IDLE
        self.armed_zone: tuple[float, float] | None = None  # grandfathered zone
        self.exhausted_at: datetime | None = None
        self.swept_extreme: float | None = None
        self.sweep_direction: int | None = None

        self.gex_range = SessionRangeTracker()
        self.vanna_range = SessionRangeTracker()
        self.charm_range = SessionRangeTracker()

    def _reset_to_idle(self) -> None:
        self.state = EngineState.IDLE
        self.armed_zone = None
        self.exhausted_at = None
        self.swept_extreme = None
        self.sweep_direction = None

    def on_tick(self, snap: MarketSnapshot) -> TradeSignal | None:
        """
        Advance the state machine by one tick. Returns a TradeSignal only
        on the tick where EXECUTION_ARMED actually fires; None otherwise.
        """
        # Layer H: economic mute and maintenance windows force IDLE
        # unconditionally, regardless of what state we were in -- this is
        # a hard interrupt, not a normal transition.
        if snap.econ_mute or snap.in_maintenance_window:
            if self.state != EngineState.IDLE:
                self._reset_to_idle()
            return None

        # update session normalization ranges every tick regardless of state
        self.gex_range.update(snap.dollar_gex)
        self.vanna_range.update(snap.dollar_vanna)
        self.charm_range.update(snap.dollar_charm)

        in_zone = (
            snap.zone_p_lower is not None
            and snap.zone_p_upper is not None
            and snap.zone_p_lower <= snap.spot <= snap.zone_p_upper
        )

        if self.state == EngineState.IDLE:
            if in_zone:
                self.state = EngineState.HTF_ZONE
                # grandfather this zone -- see module docstring
                self.armed_zone = (snap.zone_p_lower, snap.zone_p_upper)
            return None

        if self.state == EngineState.HTF_ZONE:
            # grandfathering: check membership against the ARMED zone, not
            # whatever zone_composite recomputed this tick (Layer I,
            # approved decision)
            still_in_armed_zone = (
                self.armed_zone is not None
                and self.armed_zone[0] <= snap.spot <= self.armed_zone[1]
            )
            if not still_in_armed_zone:
                self._reset_to_idle()
                return None
            if snap.depth_exhaustion:
                self.state = EngineState.LTF_ARMED
            return None

        if self.state == EngineState.LTF_ARMED:
            # Depth exhaustion is checked FIRST, before the zone-exit
            # reset: a sweep and exhaustion firing happen on the SAME tick
            # in the real scenario (price moves beyond the zone exactly as
            # it exhausts available depth), so checking zone-exit first
            # would wrongly reset a tick that should have armed. Once
            # exhaustion fires, the setup moves to LTF_EXHAUSTED and stops
            # re-checking zone membership entirely (a sweep leaving the
            # band is the scenario Tier 3 watches for, not a disqualifier).
            if snap.depth_exhaustion:
                self.state = EngineState.LTF_EXHAUSTED
                self.exhausted_at = snap.timestamp
                # capture the swept extreme + direction at the moment of
                # exhaustion (see module docstring gap #2 -- session
                # high/low as a proxy, not genuine sweep detection)
                if snap.spot >= snap.session_high:
                    self.swept_extreme = snap.session_high
                    self.sweep_direction = 1
                else:
                    self.swept_extreme = snap.session_low
                    self.sweep_direction = -1
                return None

            # Per spec: "if spot leaves Z_t before execution, reset to
            # HTF_ZONE" -- NOT all the way to IDLE. Only reached if
            # exhaustion did NOT fire this tick (see above).
            still_in_armed_zone = (
                self.armed_zone is not None
                and self.armed_zone[0] <= snap.spot <= self.armed_zone[1]
            )
            if not still_in_armed_zone:
                self.state = EngineState.HTF_ZONE
            return None

        if self.state == EngineState.LTF_EXHAUSTED:
            if self.exhausted_at is not None:
                dwell = (snap.timestamp - self.exhausted_at).total_seconds()
                if dwell > self.max_dwell_seconds:
                    # decay back to LTF_ARMED, not all the way to IDLE --
                    # per spec, a stale exhaustion re-arms rather than
                    # discarding the whole setup
                    self.state = EngineState.LTF_ARMED
                    self.exhausted_at = None
                    self.swept_extreme = None
                    self.sweep_direction = None
                    return None

            if snap.vpin_condition_a and snap.ofi_condition_b:
                return self._fire_execution(snap)
            return None

        if self.state == EngineState.EXECUTION_ARMED:
            # single-tick transitional state -- on_tick always moves this
            # straight to IN_POSITION the tick after firing, since actual
            # order-fill confirmation is an execution-layer concern this
            # class doesn't own.
            self.state = EngineState.IN_POSITION
            return None

        if self.state == EngineState.IN_POSITION:
            # stop/target/time-based exit is NOT implemented here --
            # that's a separate concern (position management), flagged
            # rather than guessed at.
            return None

        return None

    def _fire_execution(self, snap: MarketSnapshot) -> TradeSignal:
        if snap.session_vwap is None or snap.session_poc is None:
            raise RuntimeError(
                "on_tick: EXECUTION_ARMED conditions were met, but "
                "session_vwap/session_poc are None -- no VWAP/POC "
                "calculation exists in this codebase yet (see module "
                "docstring gap #1). Refusing to fabricate a target price "
                "rather than silently producing a wrong TradeSignal."
            )
        if self.swept_extreme is None or self.sweep_direction is None:
            raise RuntimeError("on_tick: fired without a captured swept extreme -- internal state bug.")

        self.state = EngineState.EXECUTION_ARMED

        entry_price = snap.spot
        stop_price = (
            self.swept_extreme - self.stop_offset_points
            if self.sweep_direction == 1
            else self.swept_extreme + self.stop_offset_points
        )
        target_price = min(
            (snap.session_vwap, snap.session_poc),
            key=lambda p: abs(p - entry_price),
        )

        return TradeSignal(
            timestamp=snap.timestamp,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            swept_extreme_price=self.swept_extreme,
            sweep_direction=self.sweep_direction,
            zone_p_lower=self.armed_zone[0] if self.armed_zone else None,
            zone_p_upper=self.armed_zone[1] if self.armed_zone else None,
            call_wall=snap.call_wall,
            put_wall=snap.put_wall,
            zero_gamma_flip=snap.zero_gamma_flip,
            vpin_at_trigger=snap.vpin,
            kyle_lambda_at_trigger=None,  # caller may attach if it tracks this separately
            ofi_at_trigger=None,
            notes="Fired via StateMachine.on_tick",
        )
