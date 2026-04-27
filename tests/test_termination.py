import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.termination import TerminationReason, classify


def make_state(
    *,
    x: float = 0.0,
    y: float = 0.5,
    vx: float = 0.0,
    vy: float = 0.0,
    theta: float = 0.0,
    theta_dot: float = 0.0,
    left_leg: float = 0.0,
    right_leg: float = 0.0,
) -> np.ndarray:
    return np.array(
        [x, y, vx, vy, theta, theta_dot, left_leg, right_leg],
        dtype=np.float32,
    )


class TestTerminationReasonEnum(unittest.TestCase):
    def test_values_are_lowercase_strings(self):
        self.assertEqual(TerminationReason.LAND.value, "land")
        self.assertEqual(TerminationReason.CRASH.value, "crash")
        self.assertEqual(TerminationReason.OUT_OF_VIEW.value, "out_of_view")
        self.assertEqual(TerminationReason.SLEEP.value, "sleep")
        self.assertEqual(TerminationReason.TRUNCATED.value, "truncated")
        self.assertEqual(TerminationReason.UNKNOWN.value, "unknown")

    def test_enum_inherits_from_str(self):
        self.assertEqual(TerminationReason.LAND, "land")
        self.assertEqual(str(TerminationReason.LAND.value), "land")


class TestClassify(unittest.TestCase):
    def test_truncated_wins_over_terminated(self):
        state = make_state(left_leg=1.0, right_leg=1.0)
        self.assertEqual(
            classify({}, reward=100.0, state=state, terminated=True, truncated=True),
            TerminationReason.TRUNCATED,
        )

    def test_truncated_wins_over_landing_signals(self):
        state = make_state(left_leg=1.0, right_leg=1.0)
        self.assertEqual(
            classify({}, reward=100.0, state=state, terminated=False, truncated=True),
            TerminationReason.TRUNCATED,
        )

    def test_neither_terminated_nor_truncated_is_unknown(self):
        self.assertEqual(
            classify({}, reward=0.0, state=make_state(), terminated=False, truncated=False),
            TerminationReason.UNKNOWN,
        )

    def test_land_with_both_legs_and_high_reward(self):
        state = make_state(left_leg=1.0, right_leg=1.0, vy=-0.05)
        self.assertEqual(
            classify({}, reward=100.0, state=state, terminated=True, truncated=False),
            TerminationReason.LAND,
        )

    def test_land_requires_strong_positive_reward(self):
        state = make_state(left_leg=1.0, right_leg=1.0)
        self.assertNotEqual(
            classify({}, reward=10.0, state=state, terminated=True, truncated=False),
            TerminationReason.LAND,
        )

    def test_land_requires_both_legs(self):
        state = make_state(left_leg=1.0, right_leg=0.0)
        self.assertNotEqual(
            classify({}, reward=100.0, state=state, terminated=True, truncated=False),
            TerminationReason.LAND,
        )

    def test_crash_with_negative_reward_no_legs_inside_frame(self):
        state = make_state(x=0.1, vy=-0.5)
        self.assertEqual(
            classify({}, reward=-100.0, state=state, terminated=True, truncated=False),
            TerminationReason.CRASH,
        )

    def test_out_of_view_left_edge(self):
        state = make_state(x=-1.5)
        self.assertEqual(
            classify({}, reward=-100.0, state=state, terminated=True, truncated=False),
            TerminationReason.OUT_OF_VIEW,
        )

    def test_out_of_view_right_edge(self):
        state = make_state(x=1.5)
        self.assertEqual(
            classify({}, reward=-100.0, state=state, terminated=True, truncated=False),
            TerminationReason.OUT_OF_VIEW,
        )

    def test_out_of_view_takes_priority_over_crash(self):
        state = make_state(x=1.2)
        self.assertEqual(
            classify({}, reward=-100.0, state=state, terminated=True, truncated=False),
            TerminationReason.OUT_OF_VIEW,
        )

    def test_sleep_at_rest_without_leg_contact(self):
        state = make_state(x=0.2, vx=0.0, vy=0.0, theta_dot=0.0)
        self.assertEqual(
            classify({}, reward=0.0, state=state, terminated=True, truncated=False),
            TerminationReason.SLEEP,
        )

    def test_sleep_not_triggered_when_velocity_high(self):
        state = make_state(x=0.2, vx=0.5)
        self.assertEqual(
            classify({}, reward=0.0, state=state, terminated=True, truncated=False),
            TerminationReason.UNKNOWN,
        )

    def test_unknown_when_no_heuristic_matches(self):
        state = make_state(x=0.2, vx=0.5, vy=-0.2, theta_dot=0.3)
        self.assertEqual(
            classify({}, reward=5.0, state=state, terminated=True, truncated=False),
            TerminationReason.UNKNOWN,
        )

    def test_info_override_recognized_value(self):
        self.assertEqual(
            classify(
                {"termination_reason": "land"},
                reward=-100.0,
                state=make_state(),
                terminated=True,
                truncated=False,
            ),
            TerminationReason.LAND,
        )

    def test_info_override_invalid_value_falls_back_to_heuristics(self):
        state = make_state(left_leg=1.0, right_leg=1.0)
        self.assertEqual(
            classify(
                {"termination_reason": "bogus"},
                reward=100.0,
                state=state,
                terminated=True,
                truncated=False,
            ),
            TerminationReason.LAND,
        )

    def test_invalid_state_shape_raises(self):
        with self.assertRaises(ValueError):
            classify({}, reward=0.0, state=np.zeros(7), terminated=True, truncated=False)


if __name__ == "__main__":
    unittest.main()
