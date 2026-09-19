"""
state_machine/states.py

The state enum for the 3-tier state machine, per the Phase 2 architecture
spec. This is the SINGLE canonical definition -- live/render_app.py
currently has its own duplicate EngineState enum (written as an explicit
placeholder before this module existed); that duplicate should be replaced
with an import from here once this module is wired into the daemon. Not
done automatically as part of this module, since it's a live-daemon edit
that deserves its own deploy/verify step, not a silent side effect of
adding this file.
"""

from enum import Enum, auto


class EngineState(Enum):
    IDLE = auto()
    HTF_ZONE = auto()
    LTF_ARMED = auto()
    LTF_EXHAUSTED = auto()
    EXECUTION_ARMED = auto()
    IN_POSITION = auto()


# Dwell-time limit for LTF_EXHAUSTED before it decays back to LTF_ARMED,
# per the architecture spec ("max dwell time... if Condition A/B don't fire
# quickly after exhaustion, the setup is stale"). Spec gave an illustrative
# range (30-60 sec) -- this is a Phase 3 calibration target like the other
# thresholds in this codebase, not a fixed constant; exposed as a module
# constant so it's easy to find and override, not because 45s is "correct."
LTF_EXHAUSTED_MAX_DWELL_SECONDS: float = 45.0
