"""HDR loading and the training configuration contract."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import torch

from envmap.field import detect_projection, load_envmap_field
from fit.evaluation import load_latlong_raster
from utils.config import parse_training_cli, resolve_training_config
from utils.image_io import load_rgb_hdr, save_image
from utils.logging import prepare_run_directories


class HDRInputTests(unittest.TestCase):
    @staticmethod
    def _write_hdr(path, values):
        written = cv2.imwrite(str(path), np.ascontiguousarray(values[..., ::-1]))
        if not written:
            raise RuntimeError(f"could not create test image: {path}")

    def test_exr_and_hdr_load_as_finite_fp32_rgb(self):
        values = np.linspace(0.0, 2.0, 4 * 4 * 3, dtype=np.float32).reshape(4, 4, 3)
        with tempfile.TemporaryDirectory() as directory:
            for suffix in (".exr", ".hdr"):
                with self.subTest(suffix=suffix):
                    path = Path(directory) / f"input{suffix}"
                    self._write_hdr(path, values)
                    loaded = load_rgb_hdr(path)
                    self.assertEqual(loaded.dtype, np.float32)
                    self.assertEqual(loaded.shape, (4, 4, 3))
                    tolerance = 2e-3 if suffix == ".exr" else 2e-2
                    self.assertTrue(
                        np.allclose(loaded, values, rtol=tolerance, atol=tolerance)
                    )

    def test_negative_radiance_is_clamped_once_for_fitting(self):
        values = np.ones((4, 4, 3), dtype=np.float32)
        values[1, 2] = (-2.0, 0.5, -0.25)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "negative.exr"
            self._write_hdr(path, values)
            self.assertLess(float(load_rgb_hdr(path).min()), 0.0)
            field = load_envmap_field(
                directory,
                path.name,
                torch.float32,
                "cpu",
                projection_name="octahedral",
            )
            self.assertEqual(float(field.values.min()), 0.0)
            self.assertEqual(float(field.values[1, 2, 1]), 0.5)

    def test_loader_rejects_non_rgb_non_fp32_and_nonfinite_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.exr"
            path.touch()
            invalid = (
                np.ones((4, 4), dtype=np.float32),
                np.ones((4, 4, 3), dtype=np.uint16),
                np.full((4, 4, 3), np.nan, dtype=np.float32),
            )
            for values in invalid:
                with self.subTest(shape=values.shape, dtype=values.dtype), mock.patch(
                    "utils.image_io.cv2.imread", return_value=values
                ), self.assertRaises(ValueError):
                    load_rgb_hdr(path)


class ConfigTests(unittest.TestCase):
    def test_preset_budget_and_cli_override_precedence(self):
        values = parse_training_cli(["--config", "cfgs/p_only.yaml", "--budget", "2500", "--overwrite_output"])
        resolved = resolve_training_config(values)
        self.assertEqual((resolved.asg_footprint, resolved.asg_learn_p, resolved.asg_learn_q),
                         ("superelliptic", True, False))
        self.assertEqual((resolved.num_gaussians, resolved.densify_max_gaussians), (875, 2500))
        self.assertTrue(resolved.overwrite_output)
        values = parse_training_cli(["--budget", "2500", "--num_gaussians", "17"])
        self.assertEqual(resolve_training_config(values).num_gaussians, 17)

    def test_unknown_training_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown training"):
            resolve_training_config({"typo_num_step": 5})

    def test_default_cli_is_the_paper_ours_setup(self):
        resolved = resolve_training_config(parse_training_cli([]))
        self.assertEqual(resolved.asg_footprint, "superelliptic")
        self.assertEqual((resolved.num_steps, resolved.evaluation_subdivision), (10000, 9))
        self.assertEqual((resolved.num_gaussians, resolved.densify_max_gaussians), (3500, 10000))

    def test_positional_input_detects_the_projection_unless_given(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, shape in (("square", (8, 8)), ("wide", (8, 16)), ("odd", (8, 24))):
                save_image(torch.ones(*shape, 3), str(root / f"{name}.exr"))
            values = parse_training_cli([str(root / "square.exr")])
            self.assertEqual((values["data_root"], values["input_path"], values["envmap_projection"]),
                             (str(root), "square.exr", "octahedral"))
            self.assertEqual(parse_training_cli([str(root / "wide.exr")])["envmap_projection"], "latlong")
            values = parse_training_cli([str(root / "square.exr"), "--latlong-reference", "data/ll.exr"])
            self.assertEqual(values["latlong_reference"], "data/ll.exr")  # kept as given, like the input
            self.assertIsNone(parse_training_cli([str(root / "square.exr")])["latlong_reference"])
            with self.assertRaisesRegex(ValueError, "must be a 1024x512 lat-long map"):
                load_latlong_raster(root / "wide.exr")  # train.py checks the reference before fitting
            values = parse_training_cli(["--data_root", str(root), "--input_path", "wide.exr"])
            self.assertEqual(values["envmap_projection"], "latlong")
            values = parse_training_cli([str(root / "square.exr"), "--envmap_projection", "latlong"])
            self.assertEqual(values["envmap_projection"], "latlong")
            with mock.patch("sys.stderr"):
                for argv in ([str(root / "odd.exr")], [str(root / "missing.exr")],
                             [str(root / "square.exr"), "--input_path", "wide.exr"]):
                    with self.subTest(argv=argv), self.assertRaises(SystemExit):
                        parse_training_cli(argv)
        self.assertEqual(parse_training_cli([])["envmap_projection"], "octahedral")
        with self.assertRaisesRegex(ValueError, "cannot infer"):
            detect_projection(512, 1000)

    def test_run_directory_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            prepare_run_directories(path)
            marker = path / "marker"
            marker.write_text("owned", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_run_directories(path)
            prepare_run_directories(path, overwrite=True)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
