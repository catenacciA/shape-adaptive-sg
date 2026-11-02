"""Runtime evaluation for native ASG artifacts."""

import torch
import torch.nn.functional as F

from model.asg import ASGFootprint, BetaFootprint, render_mixture


class NativeInferenceRepresentation:
    """Evaluate a serialized positive ASG mixture at arbitrary directions."""

    def __init__(self, manifest, arrays, *, device="cpu", dtype=torch.float32):
        self.manifest = manifest
        self.device = torch.device(device)
        self.dtype = dtype
        self.arrays = {
            name: torch.as_tensor(value).to(device=self.device, dtype=dtype)
            for name, value in arrays.items()
        }
        self.feat_dim = int(manifest["feature_dimension"])
        self.asg_footprint = None
        if manifest["asg_footprint"] == "superelliptic":
            self.asg_footprint = ASGFootprint.from_exponents(
                self.arrays["local_p"], self.arrays["local_q"], manifest.get("asg_normalize_shape", True))
        elif manifest["asg_footprint"] == "beta":
            self.asg_footprint = BetaFootprint.from_power(self.arrays["local_beta"])

    def evaluate_directions(self, directions):
        directions = F.normalize(torch.as_tensor(directions, device=self.device, dtype=self.dtype), dim=1)
        a = self.arrays
        return render_mixture(
            directions, a["local_z"], a["local_x"], a["local_y"], a["local_lambda"], a["local_mu"],
            a["local_coefficients"], self.asg_footprint,
        )
