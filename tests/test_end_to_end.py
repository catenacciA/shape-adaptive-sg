"""End-to-end fitting, reconstruction, encoding and evaluation on tiny inputs."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from evaluate import evaluate_artifact, parse_args as parse_evaluate_args
from fit.evaluation import project
from fit.trainer import EnvmapTrainer
from model.asg import render_asg_proposal_aggregate
from model.serialization import load_native_inference_artifact
from reconstruct import main as reconstruct_main
from tests.helpers import lobe_target, tiny_config, tiny_envmap
from utils.config import training_config_defaults
from utils.image_io import load_rgb_hdr, save_image
from utils.provenance import sha256_file

FIT_FILES = {"inference.npz", "inference.json", "config.yaml", "metrics.json", "kernels.png",
             "representation.skc", "reconstruction.exr"}
TINY_FIT = dict(
    log_level="ERROR", seed=123, envmap_projection="octahedral",
    num_gaussians=12, densify_max_gaussians=24, num_steps=30,
    densify_start_step=5, densify_interval=5, densify_stop_fraction=0.8,
    batch_size=512, grad_accum_steps=1, renderer_chunk_size=8,
    log_every=30, evaluation_subdivision=3, debug_artifacts=False,
)
METHODS = {
    "ours": dict(asg_footprint="superelliptic", asg_learn_p=True, asg_learn_q=True),
    "asg": dict(asg_footprint="gaussian"),
    "p_only": dict(asg_footprint="superelliptic", asg_learn_p=True, asg_learn_q=False),
    "q_only": dict(asg_footprint="superelliptic", asg_learn_p=False, asg_learn_q=True),
}


def setUpModule():
    # Fits evaluate PU21-LPIPS; stub the pretrained network so tests stay offline.
    patcher = patch("fit.metrics._evaluate_lpips", return_value=.125)
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


class PresetFitTests(unittest.TestCase):
    """Every preset fits with insertion and keeps its shape family."""

    device = "cpu"

    def check(self, method):
        preset = METHODS[method]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_image(lobe_target(), str(root / "target.exr"))
            cfg = training_config_defaults()
            cfg.update(TINY_FIT, **preset, device=self.device, data_root=str(root),
                       input_path="target.exr", log_dir=str(root / method))
            trainer = EnvmapTrainer(SimpleNamespace(**cfg))
            trainer.optimize()
            arrays = dict(np.load(root / method / "inference.npz"))
        count = len(arrays["local_coefficients"])
        # The three-lobe target is badly underfit by 12 primitives.
        self.assertEqual(count, trainer.num_gaussians)
        self.assertLess(TINY_FIT["num_gaussians"], count)
        self.assertLessEqual(count, TINY_FIT["densify_max_gaussians"])
        self.assertTrue(all(np.isfinite(value).all() for value in arrays.values()))
        self.assertTrue((arrays["local_coefficients"] >= 0).all())
        if preset["asg_footprint"] == "gaussian":
            self.assertFalse({"local_p", "local_q"} & arrays.keys())
            return
        # Frozen exponents stay at the Gaussian member (p, q) = (1, 2) for every
        # row, inserted ones included; learned exponents move.
        for name, learned, gaussian in (("local_p", preset["asg_learn_p"], 1.0),
                                        ("local_q", preset["asg_learn_q"], 2.0)):
            frozen = bool((arrays[name] == gaussian).all())
            self.assertNotEqual(frozen, learned, name)

    def test_ours(self):
        self.check("ours")

    def test_asg(self):
        self.check("asg")

    def test_p_only(self):
        self.check("p_only")

    def test_q_only(self):
        self.check("q_only")


class FitArtifactTests(unittest.TestCase):
    def test_fit_save_load_evaluate_and_encode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latlong = root / "latlong.exr"
            save_image(torch.rand(32, 64, 3), str(latlong))
            original = root / "original_latlong.exr"  # the paper's image reference: an original lat-long map
            save_image(torch.rand(512, 1024, 3), str(original))
            for kind in ("asg", "ours"):
                with self.subTest(kind=kind):
                    cfg = tiny_config(root, f"cfgs/{kind}.yaml", kind, num_gaussians=3, densify_enabled=False,
                                      densify_max_gaussians=None, num_steps=1, batch_size=16,
                                      geodesic_subdivision=0, log_level="ERROR",
                                      latlong_reference=str(original) if kind == "asg" else None)
                    trainer = EnvmapTrainer(SimpleNamespace(**cfg))
                    trainer.optimize()
                    run = root / kind
                    self.assertEqual({p.name for p in run.iterdir()}, FIT_FILES)
                    manifest = json.loads((run / "inference.json").read_text())
                    self.assertEqual(manifest["provenance"]["config_sha256"], sha256_file(run / "config.yaml"))
                    native = load_native_inference_artifact(run / "inference.json")
                    dirs = trainer.evaluation_context.quadrature.directions
                    torch.testing.assert_close(native.evaluate_directions(dirs),
                                               trainer.evaluate_directions(dirs).detach(), atol=2e-5, rtol=2e-5)
                    metrics = json.loads((run / "metrics.json").read_text())
                    self.assertEqual(metrics["evaluation"]["relative_l2"]["subdivision"], 1)
                    image = metrics["evaluation"]["image"]
                    if kind == "asg":
                        self.assertEqual((image["reference"], image["reference_file"], image["target_sha256"]),
                                         ("prepared_latlong", original.name, sha256_file(original)))
                    else:
                        self.assertEqual(image["reference"], "reprojected_reference")
                    self.assertEqual(metrics["pu21_lpips"], .125)
                    runtime = metrics["trainer_runtime_sec"]  # disjoint phases partition the total
                    self.assertTrue(all(value >= 0 for value in runtime.values()))
                    self.assertAlmostEqual(runtime.pop("total"), sum(runtime.values()), delta=0.01)
                    self.assertEqual(metrics["evaluation"]["image"]["reconstruction_sha256"],
                                     sha256_file(run / "reconstruction.exr"))

                    def evaluate(fit=run, reference=None, **extra):
                        return evaluate_artifact(SimpleNamespace(**{
                            "fit": fit, "reference": reference, "latlong_reference": None, "device": "cpu",
                            "subdivision": None, "chunk_size": 8192, "image_height": 512, **extra}))

                    # By default, re-evaluating a fit reproduces metrics.json, lat-long reference included.
                    reevaluated = evaluate()
                    for key in ("relative_l2", "pu21_psnr", "pu21_ssim", "pu21_lpips", "evaluation", "artifact"):
                        self.assertEqual(metrics[key], reevaluated[key], key)
                    self.assertTrue(reevaluated["reference_is_fit_source"])

                    prepared = evaluate(latlong_reference=latlong, image_height=32)  # replaces the fit's
                    self.assertEqual(prepared["relative_l2"], reevaluated["relative_l2"])
                    image = prepared["evaluation"]["image"]
                    self.assertEqual((image["reference"], image["target_sha256"]), ("prepared_latlong", sha256_file(latlong)))

                    skc = run / "representation.skc"
                    self.assertEqual(metrics["codec"], {"file": skc.name, "bytes": skc.stat().st_size})
                    decoded = evaluate(fit=skc)
                    self.assertEqual(decoded["codec"]["bytes"], 28 + math.ceil((104 if kind == "asg" else 120) * 3 / 8))
                    self.assertEqual(decoded["artifact"]["sha256"], sha256_file(skc))
                    self.assertTrue(decoded["reference_is_fit_source"])
                    self.assertLess(abs(decoded["relative_l2"] - reevaluated["relative_l2"]), 0.01)

                    args = parse_evaluate_args([str(skc), "--latlong-reference", str(latlong)])
                    self.assertEqual((args.fit, args.reference, args.latlong_reference), (skc, None, latlong))

                    for source, projection in ((run, "latlong"), (skc, "octahedral")):
                        output = root / f"{kind}_{projection}.exr"
                        reconstruct_main([str(source), str(output), "--projection", projection, "--height", "8",
                                          "--device", "cpu"])  # same device as the expected rendering
                        image = load_rgb_hdr(output)
                        self.assertEqual(image.shape, (8, 16 if projection == "latlong" else 8, 3))
                        self.assertTrue((image >= 0).all())
                    expected = project(native.evaluate_directions, "latlong", 8, 64).numpy()
                    np.testing.assert_allclose(load_rgb_hdr(root / f"{kind}_latlong.exr"), expected, rtol=1e-6)

                    # Any reference with a valid layout is accepted; its projection follows its shape.
                    # An explicit --reference also replaces the fit's lat-long reference.
                    other = root / "other_octa.exr"
                    save_image(torch.rand(8, 8, 3) + .1, str(other))
                    against_other = evaluate(reference=other)
                    self.assertFalse(against_other["reference_is_fit_source"])
                    self.assertEqual(against_other["reference_sha256"], sha256_file(other))
                    self.assertEqual(against_other["evaluation"]["image"]["reference"], "reprojected_reference")
                    self.assertNotAlmostEqual(against_other["relative_l2"], reevaluated["relative_l2"], places=3)
                    bad = root / "bad_layout.exr"
                    save_image(torch.ones(8, 24, 3), str(bad))
                    with self.assertRaisesRegex(ValueError, "cannot infer"):
                        evaluate(reference=bad)
            # The default references must still be the maps the fit was evaluated against.
            save_image(torch.rand(512, 1024, 3), str(original))
            with self.assertRaisesRegex(ValueError, "has changed since the fit; pass --latlong-reference"):
                evaluate(fit=root / "asg")
            save_image(2 * tiny_envmap(), str(root / "tiny.exr"))
            with self.assertRaisesRegex(ValueError, "has changed since the fit; pass --reference"):
                evaluate()

    def test_failure_before_diagnostic_does_not_mark_fit_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            cfg=tiny_config(root,'cfgs/asg.yaml','failed',num_gaussians=2,densify_enabled=False,densify_max_gaussians=None,
                             num_steps=1,batch_size=8,log_level='ERROR')
            trainer=EnvmapTrainer(SimpleNamespace(**cfg))
            with patch('fit.trainer.save_kernel_plot',side_effect=RuntimeError('diagnostic failed')):
                with self.assertRaisesRegex(RuntimeError,'diagnostic failed'):
                    trainer.optimize()
            self.assertFalse((root/'failed/metrics.json').exists())


class ContinuationTests(unittest.TestCase):
    def test_artifacts_do_not_change_densification_or_next_optimizer_update(self):
        outputs = []
        with tempfile.TemporaryDirectory() as directory:
            for artifacts in (False, True):
                cfg = tiny_config(
                    Path(directory),
                    "cfgs/asg.yaml",
                    str(artifacts),
                    num_gaussians=3,
                    densify_max_gaussians=6,
                    batch_size=16,
                    densify_start_step=1,
                    densify_interval=2,
                    densify_add_fraction=0.5,
                    debug_artifacts=artifacts,
                )
                trainer = EnvmapTrainer(SimpleNamespace(**cfg))
                trainer._optimization_step(1)
                # Give the live trainer a known missing positive lobe, so this
                # checks artifact invariance through a successful append.
                bank = trainer.kernel_bank
                dirs = trainer.training_dataset.dirs
                lobe = bank.initialize_densification_proposals(
                    dirs[:1],
                    dirs.new_tensor([0.3]),
                    dirs.new_tensor([math.asin(0.3)]),
                    dirs.new_tensor([[0.5, 0.4, 0.3]]),
                )
                trainer.training_dataset.target = (
                    bank.render_directions_chunked(dirs, 16)
                    + render_asg_proposal_aggregate(dirs, lobe)
                ).detach()
                trainer._densify_from_spherical_residual(1)
                trainer._optimization_step(2)
                self.assertLessEqual(trainer.kernel_bank.num_kernels, 6)
                event = trainer.residual_densification_events[0]
                self.assertEqual(event["status"], "inserted")
                self.assertEqual(event["resulting_count"], trainer.kernel_bank.num_kernels)
                outputs.append(
                    (
                        copy.deepcopy(trainer.kernel_bank.state_dict()),
                        copy.deepcopy(trainer.optimizer.state_dict()),
                    )
                )

        def identical(a, b):
            if isinstance(a, torch.Tensor):
                self.assertTrue(torch.equal(a, b))
            elif isinstance(a, dict):
                self.assertEqual(a.keys(), b.keys())
                for key in a:
                    identical(a[key], b[key])
            elif isinstance(a, (tuple, list)):
                self.assertEqual(len(a), len(b))
                for left, right in zip(a, b):
                    identical(left, right)
            else:
                self.assertEqual(a, b)

        identical(*outputs)


if __name__ == "__main__":
    unittest.main()
