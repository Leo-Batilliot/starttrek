import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.video import save_video


def _frames(n: int, h: int = 32, w: int = 48, dtype=np.uint8) -> list[np.ndarray]:
    out = []
    for i in range(n):
        f = np.zeros((h, w, 3), dtype=dtype)
        f[:, :, 0] = (i * 8) % 256
        out.append(f)
    return out


class TestSaveVideo(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_writes_mp4_file(self):
        path = self.dir / "out.mp4"
        save_video(_frames(30), path)
        self.assertTrue(path.exists(), "MP4 file should be written")
        self.assertGreater(path.stat().st_size, 0, "MP4 file should not be empty")

    def test_mp4_is_readable(self):
        import imageio.v2 as iio

        path = self.dir / "readable.mp4"
        save_video(_frames(30), path)
        read_back = iio.mimread(str(path))
        self.assertGreaterEqual(len(read_back), 1, "should read at least one frame")
        self.assertEqual(read_back[0].shape[-1], 3, "frames should be RGB")

    def test_creates_parent_directory(self):
        path = self.dir / "nested" / "deep" / "out.mp4"
        save_video(_frames(10), path)
        self.assertTrue(path.exists())

    def test_empty_frames_raises(self):
        with self.assertRaises(ValueError):
            save_video([], self.dir / "x.mp4")

    def test_bad_shape_raises(self):
        bad = [np.zeros((32, 48, 4), dtype=np.uint8)]
        with self.assertRaises(ValueError):
            save_video(bad, self.dir / "x.mp4")

    def test_inconsistent_shape_raises(self):
        frames = _frames(3, h=32, w=48)
        frames.append(np.zeros((33, 48, 3), dtype=np.uint8))
        with self.assertRaises(ValueError):
            save_video(frames, self.dir / "x.mp4")

    def test_fps_zero_raises(self):
        with self.assertRaises(ValueError):
            save_video(_frames(10), self.dir / "x.mp4", fps=0)

    def test_missing_extension_raises(self):
        with self.assertRaises(ValueError):
            save_video(_frames(10), self.dir / "noextension")

    def test_returns_resolved_path(self):
        path = self.dir / "ret.mp4"
        out = save_video(_frames(10), path)
        self.assertEqual(Path(out), path)


if __name__ == "__main__":
    unittest.main()
