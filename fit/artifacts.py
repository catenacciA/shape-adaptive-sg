import json
import os

import numpy as np

from utils.provenance import validate_artifact_reference


def _rgb(image):
    if image.shape[-1] >= 3:
        return image[..., :3]
    return image[..., :1].expand(*image.shape[:-1], 3)


def tonemap_for_display(image, gamma=2.2):
    image = image.clamp(min=0.0)
    return (image / (1.0 + image)).pow(1.0 / max(float(gamma), 1e-6))


def rgb_for_display(image, gamma=2.2):
    return _rgb(tonemap_for_display(image, gamma=gamma))


def write_metrics(log_dir, record):
    """Atomically write a fit's metrics.json, bound to its saved inference artifact."""
    validate_artifact_reference(os.path.join(log_dir, "inference.json"), record["artifact"])
    metrics_path = os.path.join(log_dir, "metrics.json")
    with open(metrics_path + ".tmp", "w", encoding="utf-8") as file:
        json.dump(record, file, indent=2, allow_nan=False)
        file.write("\n")
    os.replace(metrics_path + ".tmp", metrics_path)
    return record


def save_densification_proposals(
    log_dir, event_id, diagnostics, *, committed_start=None
):
    """Persist bounded candidate/selected-row arrays, never full quadrature data."""
    relative_path = os.path.join("densification_proposals", f"{event_id}.npz")
    path = os.path.join(log_dir, relative_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays = {name: value.detach().cpu().numpy() for name, value in diagnostics.items()}
    count = len(arrays["selected_sample_indices"])
    arrays["committed_model_rows"] = (
        np.empty(0, dtype=np.int64)
        if committed_start is None
        else np.arange(committed_start, committed_start + count, dtype=np.int64)
    )
    np.savez_compressed(path, **arrays)
    record = {"path": relative_path, "schema_version": 5}
    return record


def format_summary(record):
    """Short human-readable summary of an evaluation record (metrics.json or evaluate.py)."""
    relative_l2, psnr = record["relative_l2"], record["pu21_psnr"]
    relative_l2 = "undefined" if relative_l2 is None else f"{relative_l2:.4g}"
    psnr = psnr if isinstance(psnr, str) else f"{psnr:.2f}"
    lines = [f"relative L2 {relative_l2} | PU21-PSNR {psnr} dB | "
             f"PU21-SSIM {record['pu21_ssim']:.4f} | PU21-LPIPS {record['pu21_lpips']:.4f}"]
    image = record["evaluation"]["image"]
    lines.append("PU21 metrics against " + (image["reference_file"] if image["reference"] == "prepared_latlong"
                                          else "the reference map resampled to lat-long"))
    codec = record.get("codec")
    if codec:
        lines.append(f"{codec['file']}: {codec['bytes']} bytes "
                     f"({codec['bytes'] / max(record['num_kernels'], 1):.1f} B/primitive)")
    return "\n".join(lines)
