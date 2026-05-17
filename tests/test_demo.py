import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import demo  # noqa: E402

_CKPT = _ROOT / "assets" / "checkpoints" / "dqn_best.pt"


class TestDemo(unittest.TestCase):
    def test_missing_checkpoint_returns_error(self):
        argv = ["demo.py", "--checkpoint", str(_ROOT / "nope.pt"), "--episodes", "1"]
        with patch.object(sys, "argv", argv):
            self.assertEqual(demo.main(), 84)

    @unittest.skipUnless(_CKPT.exists(), "backed-up checkpoint not present")
    def test_live_demo_one_episode(self):
        argv = ["demo.py", "--episodes", "1", "--seed", "0"]
        with patch.object(sys, "argv", argv):
            self.assertEqual(demo.main(), 0)


if __name__ == "__main__":
    unittest.main()
