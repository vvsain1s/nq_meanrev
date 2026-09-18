"""
render_app.py

Entry point for Render deployment. Render's free web-service tier runs one
process, so the daemon can't be a separate worker -- it runs as an asyncio
background task inside the same FastAPI process that serves health checks.

State is IN-MEMORY ONLY for now (explicit decision, not an oversight): a
Render restart resets to IDLE and naturally re-arms on the next Z_t entry.
No external persistence wired in yet -- revisit if restart frequency turns
out to be high enough that resets are costing real missed setups.

Render start command (set in Render dashboard, not in code):
    uvicorn live.render_app:app --host 0.0.0.0 --port $PORT

Required Render environment variables (set in dashboard, NOT in a .env file
committed to the repo):
    DATABENTO_API_KEY
    LSE_API_KEY
    DISCORD_WEBHOOK_URL          (execution pings / alerts channel)
    DISCORD_WEBHOOK_URL_BRIEFING (morning briefing channel, can be same URL)
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum, auto

from fastapi import FastAPI
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("nq_meanrev.daemon")


# --------------------------------------------------------------------------
# In-memory state machine status (Tier 1/2/3 states from the spec).
# This is the SAME state enum the Nautilus backtest strategy should import
# from state_machine/states.py in the real build -- duplicated here only as
# a placeholder until that module exists, so this file is runnable standalone.
# --------------------------------------------------------------------------
class EngineState(Enum):
    IDLE = auto()
    HTF_ZONE = auto()
    LTF_ARMED = auto()
    LTF_EXHAUSTED = auto()
    EXECUTION_ARMED = auto()
    IN_POSITION = auto()


class DaemonStatus:
    """Process-wide, in-memory status snapshot. No disk, no external DB.
    Lost on restart by design (see module docstring)."""

    def __init__(self) -> None:
        self.state: EngineState = EngineState.IDLE
        self.last_tick_at: datetime | None = None
        self.started_at: datetime = datetime.now(timezone.utc)
        self.last_signal: dict | None = None
        self.loop_error_count: int = 0
        self.last_error: str | None = None

    def as_dict(self) -> dict:
        return {
            "state": self.state.name,
            "started_at": self.started_at.isoformat(),
            "last_tick_at": self.last_tick_at.isoformat() if self.last_tick_at else None,
            "last_signal": self.last_signal,
            "loop_error_count": self.loop_error_count,
            "last_error": self.last_error,
            "uptime_seconds": (datetime.now(timezone.utc) - self.started_at).total_seconds(),
        }


status = DaemonStatus()

# Required env vars -- fail loudly at import time if any are missing, rather
# than silently degrading (a live daemon silently missing a Discord webhook
# is worse than one that refuses to start).
REQUIRED_ENV_VARS = ["DATABENTO_API_KEY", "LSE_API_KEY", "DISCORD_WEBHOOK_URL"]


def check_required_env() -> list[str]:
    missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        logger.error("MISSING required env vars: %s -- set these in Render's "
                     "dashboard under Environment, not in code or a committed file.", missing)
    return missing


def in_cme_maintenance_window(now: datetime) -> bool:
    """
    17:00-18:00 EST daily maintenance + weekend closure, per Layer G spec.
    NOTE: uses naive EST offset for clarity -- swap for zoneinfo("America/New_York")
    in the real build to handle DST correctly across March/November.
    """
    # placeholder logic -- replace with proper America/New_York zoneinfo handling
    # before this goes live; flagging explicitly rather than shipping a DST bug.
    return False


async def daemon_loop() -> None:
    """
    The actual trading loop. Currently a skeleton: real Tier 1/2/3 evaluation
    (importing factors/ and state_machine/) gets wired in here once those
    modules exist. This loop's job right now is to prove the FastAPI +
    background-task + Render architecture works end to end.
    """
    logger.info("daemon_loop starting")
    while True:
        try:
            now = datetime.now(timezone.utc)

            if in_cme_maintenance_window(now):
                if status.state != EngineState.IDLE:
                    logger.info("entering maintenance window -- forcing IDLE")
                status.state = EngineState.IDLE
                await asyncio.sleep(60)
                continue

            # --- placeholder for real Tier 1/2/3 evaluation ---
            # zone = compute_htf_zone(...)
            # if spot in zone: status.state = EngineState.HTF_ZONE
            # ... etc, per the state machine spec
            status.last_tick_at = now

            await asyncio.sleep(5)  # placeholder tick interval

        except Exception as exc:  # noqa: BLE001 -- daemon must never die silently
            status.loop_error_count += 1
            status.last_error = str(exc)
            logger.exception("daemon_loop error (count=%d)", status.loop_error_count)
            await asyncio.sleep(10)  # backoff before retrying


@asynccontextmanager
async def lifespan(app: FastAPI):
    missing = check_required_env()
    if missing:
        # Don't crash the process -- Render will just restart-loop it. Instead
        # surface the problem clearly via /health so an uptime monitor or you
        # checking from your phone sees it immediately.
        status.last_error = f"missing env vars: {missing}"

    task = asyncio.create_task(daemon_loop())
    logger.info("daemon background task launched")
    yield
    task.cancel()
    logger.info("daemon background task cancelled on shutdown")


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"service": "nq_meanrev", "status": "running"}


@app.get("/health")
async def health():
    missing = check_required_env()
    body = status.as_dict()
    body["missing_env_vars"] = missing
    ok = not missing and status.loop_error_count < 5
    return JSONResponse(content=body, status_code=200 if ok else 503)
