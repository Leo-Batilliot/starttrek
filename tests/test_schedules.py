import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.schedules import linear_epsilon


class TestLinearEpsilon(unittest.TestCase):
    def test_start_at_step_zero(self):
        self.assertEqual(linear_epsilon(0, start=1.0, end=0.05, decay_steps=100), 1.0)

    def test_negative_step_returns_start(self):
        self.assertEqual(linear_epsilon(-10, start=1.0, end=0.05, decay_steps=100), 1.0)

    def test_end_at_decay_steps(self):
        self.assertEqual(linear_epsilon(100, start=1.0, end=0.05, decay_steps=100), 0.05)

    def test_clamped_beyond_decay_steps(self):
        self.assertEqual(linear_epsilon(10_000, start=1.0, end=0.05, decay_steps=100), 0.05)

    def test_midpoint(self):
        self.assertAlmostEqual(
            linear_epsilon(50, start=1.0, end=0.0, decay_steps=100), 0.5
        )

    def test_monotonic_decreasing(self):
        vals = [linear_epsilon(s, start=1.0, end=0.05, decay_steps=100) for s in range(0, 120, 5)]
        for a, b in zip(vals, vals[1:]):
            self.assertGreaterEqual(a, b)

    def test_increasing_schedule(self):
        # start < end is also valid (linear interpolation, not assumed decay).
        self.assertAlmostEqual(
            linear_epsilon(25, start=0.0, end=1.0, decay_steps=100), 0.25
        )

    def test_constant_when_start_equals_end(self):
        for s in (0, 1, 50, 999):
            self.assertEqual(linear_epsilon(s, start=0.3, end=0.3, decay_steps=100), 0.3)

    def test_decay_steps_one(self):
        self.assertEqual(linear_epsilon(0, start=1.0, end=0.0, decay_steps=1), 1.0)
        self.assertEqual(linear_epsilon(1, start=1.0, end=0.0, decay_steps=1), 0.0)

    def test_invalid_decay_steps_raises(self):
        with self.assertRaises(ValueError):
            linear_epsilon(0, start=1.0, end=0.0, decay_steps=0)
        with self.assertRaises(ValueError):
            linear_epsilon(0, start=1.0, end=0.0, decay_steps=-5)

    def test_returns_float(self):
        self.assertIsInstance(linear_epsilon(10, start=1, end=0, decay_steps=100), float)


if __name__ == "__main__":
    unittest.main()
