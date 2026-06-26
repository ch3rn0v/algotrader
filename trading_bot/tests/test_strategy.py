"""State-machine completeness and gate tests (SPEC 6.3, 14)."""
from __future__ import annotations

from src.core.strategy import (
    BANDS,
    GATES,
    REGIMES,
    TRANSITION_TABLE,
    initial_state,
    step,
)

PARAMS = {"max_position_lots": 5, "entry_cooldown_bars": 2}


def test_transition_table_complete_36_entries_no_fallthrough():
    expected = set()
    for prev in (-1, 0, 1):
        for regime in REGIMES:
            for band in BANDS:
                for gate in GATES:
                    expected.add((prev, regime, band, gate))
    assert len(expected) == 36
    # Every combination present; no KeyError, no implicit "hold previous".
    for key in expected:
        assert key in TRANSITION_TABLE, f"missing transition for {key}"
    assert set(TRANSITION_TABLE.keys()) == expected


def test_transition_table_matches_spec_rows():
    # Spot-check the rationale rows from SPEC 6.3.
    assert TRANSITION_TABLE[(0, "up", "at_lower", "open")] == 1     # enter long
    assert TRANSITION_TABLE[(0, "up", "at_lower", "blocked")] == 0  # cooldown
    assert TRANSITION_TABLE[(0, "down", "at_upper", "open")] == -1  # enter short
    assert TRANSITION_TABLE[(1, "up", "at_upper", "open")] == 0     # target hit, exit long
    assert TRANSITION_TABLE[(1, "up", "at_lower", "open")] == 1     # hold long
    assert TRANSITION_TABLE[(1, "down", "inside", "open")] == 0     # regime flip exit
    assert TRANSITION_TABLE[(-1, "down", "at_lower", "open")] == 0  # target hit, exit short
    assert TRANSITION_TABLE[(-1, "down", "at_upper", "open")] == -1 # hold short
    assert TRANSITION_TABLE[(-1, "up", "inside", "open")] == 0      # regime flip exit


def _enter_long(state):
    # up regime (fast>slow), close at lower band, gate open -> +1
    return step(state, close=90.0, fast_ewm=10.0, slow_ewm=9.0, bb_lo=95.0, bb_up=110.0,
               bb_mid=100.0, in_session=True, params=PARAMS)


def test_gate_blocks_until_cooldown_and_midcross():
    state = initial_state()
    sig, target, regime, band, state = _enter_long(state)
    assert sig == 1 and target == 5.0 and state["gate"] == "open"

    # Exit at upper band (target hit): +1 -> 0, gate becomes blocked.
    sig, _, _, _, state = step(state, close=115.0, fast_ewm=10.0, slow_ewm=9.0, bb_lo=95.0,
                               bb_up=110.0, bb_mid=100.0, in_session=True, params=PARAMS)
    assert sig == 0 and state["gate"] == "blocked" and state["bars_since_exit"] == 0
    assert state["crossed_mid_since_exit"] is False

    # Next bar: still above mid (no cross), cooldown not satisfied -> still blocked,
    # and even an at_lower touch must NOT re-enter.
    sig, _, _, _, state = step(state, close=112.0, fast_ewm=10.0, slow_ewm=9.0, bb_lo=95.0,
                               bb_up=110.0, bb_mid=100.0, in_session=True, params=PARAMS)
    assert sig == 0 and state["gate"] == "blocked" and state["bars_since_exit"] == 1

    # Bar that touches lower band while gate blocked: no entry.
    sig, _, _, _, state = step(state, close=90.0, fast_ewm=10.0, slow_ewm=9.0, bb_lo=95.0,
                               bb_up=110.0, bb_mid=100.0, in_session=True, params=PARAMS)
    # close 112 -> 90 crosses mid 100 (sign change), and bars_since_exit now 2 >= cooldown 2.
    assert sig == 0  # gate was 'blocked' when THIS signal was computed
    assert state["gate"] == "open"  # gate re-opens AFTER this bar (cross + cooldown met)
    assert state["crossed_mid_since_exit"] is True

    # Following bar at lower band with gate now open -> re-enter long.
    sig, _, _, _, state = step(state, close=90.0, fast_ewm=10.0, slow_ewm=9.0, bb_lo=95.0,
                               bb_up=110.0, bb_mid=100.0, in_session=True, params=PARAMS)
    assert sig == 1


def test_no_reentry_without_midcross_even_after_cooldown():
    state = initial_state()
    _, _, _, _, state = _enter_long(state)
    # Exit.
    _, _, _, _, state = step(state, close=115.0, fast_ewm=10.0, slow_ewm=9.0, bb_lo=95.0,
                             bb_up=110.0, bb_mid=100.0, in_session=True, params=PARAMS)
    # Several bars all staying ABOVE mid (no re-cross): cooldown satisfied but no cross.
    for _ in range(5):
        _, _, _, _, state = step(state, close=112.0, fast_ewm=10.0, slow_ewm=9.0, bb_lo=111.0,
                                 bb_up=120.0, bb_mid=100.0, in_session=True, params=PARAMS)
    assert state["gate"] == "blocked"
    assert state["crossed_mid_since_exit"] is False


def test_out_of_session_leaves_state_untouched():
    state = initial_state()
    _, _, _, _, state = _enter_long(state)
    before = dict(state)
    sig, target, regime, band, after = step(state, close=90.0, fast_ewm=10.0, slow_ewm=9.0,
                                             bb_lo=95.0, bb_up=110.0, bb_mid=100.0,
                                             in_session=False, params=PARAMS)
    assert sig == 0 and target == 0.0 and regime is None and band is None
    assert after == before  # state carried unchanged across the gap
