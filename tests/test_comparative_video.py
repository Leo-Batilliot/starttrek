import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import comparative_video as cv


class TestComparativeVideo(unittest.TestCase):
    def test_header_shape_and_dtype(self):
        h = cv._header(1800, [("A", -110.0, "crash"),
                              ("B", 261.0, "land"),
                              ("C", 270.0, "land")])
        self.assertEqual(h.shape, (cv._HEADER_H, 1800, 3))
        self.assertEqual(h.dtype, np.uint8)

    def test_three_agents_declared(self):
        labels = [a[0] for a in cv._AGENTS]
        self.assertEqual(len(cv._AGENTS), 3)
        self.assertIn("DQN entraine", labels)
        # only the trained agent carries a checkpoint
        self.assertEqual([bool(a[2]) for a in cv._AGENTS], [False, False, True])

    def test_abs_resolves_under_root(self):
        self.assertEqual(cv._abs("a/b.txt"), _ROOT / "a" / "b.txt")


if __name__ == "__main__":
    unittest.main()
