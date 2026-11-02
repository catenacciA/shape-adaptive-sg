"""Encode physical RGB ASG arrays to SKC; decode to physical FP32 arrays."""

import numpy as np

from codec.bitstream import BITS, HEADER, Header, field_widths, pack_records, read_header, unpack_records
from codec.transforms import (
    P_BOUNDS, Q_BOUNDS, decode_direction, decode_frame, decode_log,
    encode_angle, encode_direction, encode_log, transport_axis, validate_frames,
)


def _physical_arrays(arrays, method):
    expected = {"local_z": 3, "local_x": 3, "local_y": 3,
                "local_lambda": None, "local_mu": None, "local_coefficients": 3}
    if method != "asg":
        expected.update(local_p=None, local_q=None)
    if set(arrays) != set(expected):
        raise ValueError("SKC method does not match native inference arrays")
    result = {}
    count = None
    for name, channels in expected.items():
        value = np.asarray(arrays[name])
        if (not np.issubdtype(value.dtype, np.floating) or value.ndim != (1 if channels is None else 2)
                or (channels is not None and value.shape[1] != channels)
                or not np.isfinite(value).all()):
            raise ValueError(f"invalid SKC physical array: {name}")
        count = value.shape[0] if count is None else count
        if value.shape[0] != count:
            raise ValueError("inconsistent SKC primitive counts")
        if name in ("local_p", "local_q"):
            low, high = P_BOUNDS if name == "local_p" else Q_BOUNDS
            if np.any((value < low) | (value > high)):
                raise ValueError("SKC shape outside supported bounds")
        if ((method == "radial" and name == "local_q" and np.any(value != 2))
                or (method == "contour" and name == "local_p" and np.any(value != 1))):
            raise ValueError("SKC frozen shape parameters must match exact defaults")
        # V1's physical interchange domain is FP32, including header endpoints.
        with np.errstate(over="ignore", under="ignore"):
            fp32 = value.astype(np.float32)
        if not np.isfinite(fp32).all() or np.any((value != 0) & (fp32 == 0)):
            raise ValueError("SKC physical values must be representable in FP32")
        result[name] = fp32.astype(np.float64)
    for name in ("local_lambda", "local_mu"):
        if np.any(result[name] <= 0):
            raise ValueError("SKC concentrations must be positive")
    if np.any(result["local_coefficients"] < 0):
        raise ValueError("SKC coefficients must be nonnegative")
    return result, count


def encode_arrays(arrays, *, method):
    """Return deterministic bytes; caller must explicitly identify the variant."""
    widths = field_widths(method)
    a, count = _physical_arrays(arrays, method)
    z, x, y = validate_frames(a["local_z"], a["local_x"], a["local_y"])
    l, m = a["local_lambda"], a["local_mu"]
    swap = l > m
    x = np.where(swap[:, None], y, x)  # Other axis becomes -x: handedness retained.
    l, m = np.minimum(l, m), np.maximum(l, m)
    coefficient = a["local_coefficients"]
    positive = coefficient[coefficient > 0]
    header = Header(method, count,
                    float(l.min()) if count else 5.0, float(m.max()) if count else 5.0,
                    float(positive.min()) if positive.size else 1.0,
                    float(positive.max()) if positive.size else 1.0)
    header.validate()
    d, t, s, c, p, q = BITS
    direction = encode_direction(z, d)
    decoded_z = decode_direction(direction, d)
    angle = encode_angle(decoded_z, transport_axis(z, decoded_z, x), t)
    columns = [direction, angle[:, None],
               encode_log(l, header.scale_min, header.scale_max, s)[:, None],
               encode_log(m, header.scale_min, header.scale_max, s)[:, None],
               encode_log(coefficient, header.coefficient_min, header.coefficient_max, c, zero=True)]
    if method in ("radial", "full"):
        columns.append(encode_log(a["local_p"], *P_BOUNDS, p)[:, None])
    if method in ("contour", "full"):
        columns.append(encode_log(a["local_q"], *Q_BOUNDS, q)[:, None])
    data = header.pack() + pack_records(np.column_stack(columns), widths)
    if len(data) != header.file_bytes:
        raise AssertionError("SKC packed size mismatch")
    return data


def decode_arrays(data):
    """Validate a complete stream and return (Header, native FP32 array dict)."""
    header = read_header(data)
    widths = field_widths(header.method)
    rows = np.empty((header.count, len(widths)), dtype=np.uint32)
    for i, row in enumerate(unpack_records(memoryview(data)[HEADER.size:], widths, header.count)):
        rows[i] = row
    d, t, s, c, p, q = BITS
    z = decode_direction(rows[:, :2], d)
    x, y = decode_frame(z, rows[:, 2], t)
    arrays = dict(local_z=z, local_x=x, local_y=y,
                  local_lambda=decode_log(rows[:, 3], header.scale_min, header.scale_max, s),
                  local_mu=decode_log(rows[:, 4], header.scale_min, header.scale_max, s),
                  local_coefficients=decode_log(rows[:, 5:8], header.coefficient_min,
                                                header.coefficient_max, c, zero=True))
    if np.any(rows[:, 3] > rows[:, 4]):
        raise ValueError("SKC concentration fields are not canonically ordered")
    if header.method != "asg":
        arrays["local_p"] = (decode_log(rows[:, 8], *P_BOUNDS, p)
                             if header.method in ("radial", "full") else np.ones(header.count))
        arrays["local_q"] = (decode_log(rows[:, -1], *Q_BOUNDS, q)
                             if header.method in ("contour", "full") else np.full(header.count, 2.0))
    return header, {name: value.astype(np.float32) for name, value in arrays.items()}
