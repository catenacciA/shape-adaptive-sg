import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch

HDR_INPUT_FORMATS = {".exr", ".hdr"}
OUTPUT_IMAGE_FORMATS = {".exr", ".png"}


def load_rgb_hdr(load_path):
    """Load one finite FP32 RGB EXR/HDR image as HWC data."""
    load_path = os.fspath(load_path)
    if not os.path.isfile(load_path):
        raise FileNotFoundError(f"HDR input file does not exist: '{load_path}'")
    extension = os.path.splitext(load_path)[1].lower()
    if extension not in HDR_INPUT_FORMATS:
        raise ValueError("HDR input must be one RGB .exr or .hdr file")
    image = cv2.imread(load_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read HDR image '{load_path}'")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(
            f"HDR input must have exactly three channels, got shape {image.shape}"
        )
    if image.dtype != np.float32:
        raise ValueError(f"HDR input must be FP32, got {image.dtype}")
    image = np.ascontiguousarray(image[..., ::-1])
    if not np.isfinite(image).all():
        raise ValueError("HDR input contains non-finite values")
    return image


def to_output_format(image, image_format, gamma):  # adapted from Image-GS (MIT, see LICENSE)
    if image_format not in OUTPUT_IMAGE_FORMATS:
        raise ValueError(f"Invalid image format: {image_format}")
    if len(image.shape) not in [2, 3]:
        raise ValueError(f"Invalid image format: shape = {image.shape}")
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().clone().numpy()
    if len(image.shape) == 3 and image.shape[2] not in [1, 3]:
        image = image.transpose(1, 2, 0)
        if image.shape[2] not in [1, 3]:
            raise ValueError(f"Invalid image format: shape = {image.shape}")
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = image.squeeze(axis=2)
    image = image.astype(np.float32)
    if gamma is not None:
        image = np.power(image, 1.0 / gamma)
    if image_format == ".png":
        image = np.clip(image, 0.0, 1.0)
        image = (65535.0 * image).astype(np.uint16)
    return image


def save_image(image, save_path, gamma=None):
    image_format = os.path.splitext(save_path)[1].lower()
    image = to_output_format(image, image_format, gamma)
    if len(image.shape) == 3:
        image = image[..., ::-1]
    if not cv2.imwrite(save_path, image):
        raise OSError(f"Could not write image '{save_path}'")
