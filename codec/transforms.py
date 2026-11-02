"""Reference CPU/NumPy float64 geometry and scalar transforms for SKC v1."""

import numpy as np


FRAME_TOLERANCE = 1e-5
P_BOUNDS = (1.0, 4.0)
Q_BOUNDS = (1.25, 16.0)


def normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def validate_frames(z, x, y):
    """Reject invalid frames, then remove only roundoff-sized frame drift.

    Returns fresh float64 z, x, y arrays; never changes caller-owned values.
    This validator is also used by the native NPZ loader.
    """
    vectors = [np.asarray(v, dtype=np.float64) for v in (z, x, y)]
    if any(v.ndim != 2 or v.shape[1] != 3 for v in vectors):
        raise ValueError("ASG frames must have shape [N, 3]")
    if any(v.shape != vectors[0].shape or not np.isfinite(v).all() for v in vectors):
        raise ValueError("ASG frames must have matching, finite vectors")
    z, x, y = vectors
    frame = np.stack((x, y, z), axis=-1)
    gram = np.swapaxes(frame, -1, -2) @ frame
    if (np.any(np.abs(gram - np.eye(3)) > FRAME_TOLERANCE)
            or np.any(np.linalg.det(frame) <= 0)):
        raise ValueError("ASG frames must be orthonormal and right handed")
    z = normalize(z)
    x = normalize(x - np.sum(x * z, axis=-1, keepdims=True) * z)
    return z, x, np.cross(z, x)


def tangent_basis(z):
    """The native Duff basis, including its z[2] >= 0 chart convention."""
    z = normalize(z)
    sign = np.where(z[:, 2] >= 0, 1.0, -1.0)
    a = -1.0 / (sign + z[:, 2])
    b = z[:, 0] * z[:, 1] * a
    t1 = np.stack((1 + sign * z[:, 0] ** 2 * a, sign * b, -sign * z[:, 0]), axis=1)
    t2 = np.stack((b, sign + z[:, 1] ** 2 * a, -z[:, 1]), axis=1)
    return normalize(t1), normalize(t2)


def oct_coordinates(z):
    projected = z / np.abs(z).sum(axis=1, keepdims=True)
    xy = projected[:, :2]
    folded = (1 - np.abs(xy[:, ::-1])) * np.where(xy >= 0, 1.0, -1.0)
    return np.where((z[:, 2] < 0)[:, None], folded, xy)


def oct_direction(xy):
    z = 1 - np.abs(xy).sum(axis=1)
    folded = (1 - np.abs(xy[:, ::-1])) * np.where(xy >= 0, 1.0, -1.0)
    return normalize(np.column_stack((np.where((z < 0)[:, None], folded, xy), z)))


def round_unsigned(value):
    return np.floor(value + 0.5).astype(np.uint32)


def encode_direction(z, bits):
    return round_unsigned(np.clip((oct_coordinates(z) + 1) * 0.5, 0, 1) * ((1 << bits) - 1))


def decode_direction(code, bits):
    return oct_direction(code.astype(np.float64) * (2.0 / ((1 << bits) - 1)) - 1)


def transport_axis(z, new_z, x):
    """Apply the shortest rotation z -> new_z (Rodrigues, near identity)."""
    cross = np.cross(z, new_z)
    dot = np.sum(z * new_z, axis=1, keepdims=True)
    if np.any(dot <= -1 + 1e-12):
        raise ValueError("antipodal direction transport is undefined")
    return x + np.cross(cross, x) + np.cross(cross, np.cross(cross, x)) / (1 + dot)


def encode_angle(z, x, bits):
    t1, t2 = tangent_basis(z)
    theta = np.mod(np.arctan2(np.sum(x * t2, axis=1), np.sum(x * t1, axis=1)), np.pi)
    return round_unsigned(theta * ((1 << bits) / np.pi)) % (1 << bits)


def decode_frame(z, code, bits):
    theta = code.astype(np.float64) * (np.pi / (1 << bits))
    t1, t2 = tangent_basis(z)
    x = np.cos(theta)[:, None] * t1 + np.sin(theta)[:, None] * t2
    return x, np.cross(z, x)


def encode_log(values, low, high, bits, *, zero=False):
    maximum = (1 << bits) - 1 - int(zero)
    values = np.asarray(values, dtype=np.float64)
    positive = values > 0
    safe = np.where(positive, values, low)
    if low == high:
        code = np.zeros(values.shape, dtype=np.uint32)
    else:
        unit = (np.log2(safe) - np.log2(low)) / (np.log2(high) - np.log2(low))
        code = round_unsigned(np.clip(unit, 0, 1) * maximum)
    return np.where(positive, code + 1, 0).astype(np.uint32) if zero else code


def decode_log(code, low, high, bits, *, zero=False):
    maximum = (1 << bits) - 1 - int(zero)
    index = np.maximum(code.astype(np.int64) - int(zero), 0)
    if low == high:
        result = np.full(code.shape, low, dtype=np.float64)
    else:
        result = np.exp2(np.log2(low) + index * ((np.log2(high) - np.log2(low)) / maximum))
        result = np.clip(result, low, high)
        result = np.where(index == 0, low, np.where(index == maximum, high, result))
    return np.where(code == 0, 0, result) if zero else result
