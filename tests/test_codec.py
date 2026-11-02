"""Fixed-rate primitive codec (Sec. 3.5): wire format, quantizers and decoded rendering."""

import math
import struct
import unittest

import numpy as np
import torch

from codec.bitstream import METHODS, pack_records, unpack_records
from codec.native import decode_native, encode_native
from codec.spherical import decode_arrays, encode_arrays
from codec.transforms import (
    P_BOUNDS, Q_BOUNDS, decode_direction, decode_frame, decode_log, encode_angle,
    encode_direction, encode_log, tangent_basis, transport_axis, validate_frames,
)
from model.inference import NativeInferenceRepresentation
from model.primitives import build_asg_tangent_frame


def arrays(count=7, method="asg"):
    rng = np.random.default_rng(71)
    axis = torch.from_numpy(rng.normal(size=(count, 3)))
    theta = torch.from_numpy(rng.uniform(-10, 10, count))
    x, y, z = build_asg_tangent_frame(axis, theta)
    a = {"local_z": z.numpy().astype(np.float32), "local_x": x.numpy().astype(np.float32),
         "local_y": y.numpy().astype(np.float32),
         "local_lambda": np.geomspace(5, 1000, count).astype(np.float32),
         "local_mu": np.geomspace(300, 8, count).astype(np.float32),
         "local_coefficients": np.exp(rng.uniform(-5, 1, (count, 3))).astype(np.float32)}
    if method != "asg":
        a["local_p"] = np.ones(count, dtype=np.float32) if method == "contour" else np.linspace(1, 4, count, dtype=np.float32)
        a["local_q"] = np.full(count, 2, dtype=np.float32) if method == "radial" else np.geomspace(1.25, 16, count).astype(np.float32)
    return a


def native(a, method):
    return NativeInferenceRepresentation({"kernel_type": "asg", "feature_dimension": 3,
        "num_local_kernels": len(a["local_z"]),
        "asg_footprint": "gaussian" if method == "asg" else "superelliptic"}, a)


class BitstreamTests(unittest.TestCase):
    def test_golden_empty_and_one_primitive_streams(self):
        empty = bytes.fromhex("534b430001000000000000000000a0400000a0400000803f0000803f")
        self.assertEqual(encode_arrays(arrays(0), method="asg"), empty)
        a = {"local_z": np.array([[0, 0, -1]], dtype=np.float32),
             "local_x": np.array([[1, 0, 0]], dtype=np.float32),
             "local_y": np.array([[0, -1, 0]], dtype=np.float32),
             "local_lambda": np.array([5], dtype=np.float32),
             "local_mu": np.array([5], dtype=np.float32),
             "local_coefficients": np.ones((1, 3), dtype=np.float32)}
        one = bytes.fromhex("534b430001000000010000000000a0400000a0400000803f0000803f"
                            "ffffffff000000001000011000")
        self.assertEqual(encode_arrays(a, method="asg"), one)
        _, decoded = decode_arrays(one)
        for name in a:
            np.testing.assert_array_equal(decoded[name], a[name])

    def test_cross_byte_packing_has_independent_bit_reference(self):
        widths = (3, 10, 1, 18)
        rows = [[5, 1023, 1, 0x31234], [2, 7, 0, 11], [0, 0, 0, 1]]
        bits = ''.join(''.join(str((v >> k) & 1) for k in range(w))
                       for row in rows for v, w in zip(row, widths))
        expected = bytes(sum(int(bit) << j for j, bit in enumerate(bits[i:i + 8]))
                         for i in range(0, len(bits), 8))
        packed = pack_records(rows, widths)
        self.assertEqual(packed, expected)
        self.assertEqual(list(unpack_records(packed, widths, 3)), rows)
        with self.assertRaises(ValueError):
            pack_records([[8, 0, 0, 0]], widths)

    def test_stream_size_is_eq8_and_encoding_is_deterministic(self):
        # Eq. (8): B_ASG(N) = 28 + ceil(104 N / 8), B_Ours(N) = 28 + ceil(120 N / 8).
        record_bits = {"asg": 104, "radial": 112, "contour": 112, "full": 120}
        for method in METHODS:
            for count in (0, 1, 3, 8):
                with self.subTest(method=method, count=count):
                    a = arrays(count, method)
                    before = {k: v.copy() for k, v in a.items()}
                    data = encode_arrays(a, method=method)
                    self.assertEqual(data, encode_arrays(a, method=method))
                    self.assertEqual(len(data), 28 + math.ceil(record_bits[method] * count / 8))
                    h, decoded = decode_arrays(data)
                    self.assertEqual(h.record_bits, record_bits[method])
                    self.assertEqual(decoded["local_z"].shape, (count, 3))
                    for k in a:
                        np.testing.assert_array_equal(a[k], before[k])
                    self.assertTrue(all(np.isfinite(v).all() for v in decoded.values()))
                    self.assertTrue(np.all(decoded["local_lambda"] <= decoded["local_mu"]))

    def test_malformed_and_every_byte_truncation(self):
        data = encode_arrays(arrays(3, "full"), method="full")
        for cut in range(len(data)):
            with self.subTest(cut=cut), self.assertRaises(ValueError):
                decode_arrays(data[:cut])
        for offset, byte in [(0, 0), (4, 2), (5, 255), (6, 255), (7, 1)]:
            broken = bytearray(data)
            broken[offset] = byte
            with self.assertRaises(ValueError):
                decode_arrays(broken)
        for offset in (12, 16, 20, 24):
            for value in (0.0, -1.0, float("nan"), float("inf")):
                broken = bytearray(data)
                struct.pack_into("<f", broken, offset, value)
                with self.assertRaises(ValueError):
                    decode_arrays(broken)
        broken = bytearray(data)
        struct.pack_into("<I", broken, 8, 0xFFFFFFFF)
        with self.assertRaises(ValueError):
            decode_arrays(broken)
        with self.assertRaises(ValueError):
            decode_arrays(data + b"\0")


class TransformTests(unittest.TestCase):
    def test_octahedral_roundtrip_axes_folds_signed_zero_and_random(self):
        rng = np.random.default_rng(123)
        z = np.concatenate((np.eye(3), -np.eye(3), [[0., -0., -1.], [1., 1., 0.]],
                            rng.normal(size=(10000, 3))))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        for bits in (16, 18):
            decoded = decode_direction(encode_direction(z, bits), bits)
            angle = np.arctan2(np.linalg.norm(np.cross(z, decoded), axis=1), np.sum(z * decoded, axis=1))
            self.assertLess(angle.max(), 5 / ((1 << bits) - 1))
            np.testing.assert_allclose(np.linalg.norm(decoded, axis=1), 1, atol=1e-14)

    def test_equator_crossing_transports_orientation_in_both_directions(self):
        for sign in (-1, 1):
            z = np.array([[.6, .8, sign * 1e-8]])
            z /= np.linalg.norm(z, axis=1, keepdims=True)
            new_z = z.copy()
            new_z[:, 2] *= -1
            t1, t2 = tangent_basis(z)
            x = math.cos(.37) * t1 + math.sin(.37) * t2
            moved = transport_axis(z, new_z, x)
            code = encode_angle(new_z, moved, 12)
            x1, y1 = decode_frame(new_z, code, 12)
            self.assertLess(min(np.linalg.norm(x1 - moved), np.linalg.norm(x1 + moved)),
                            math.pi / 8192 + 1e-12)
            validate_frames(new_z, x1, y1)
            # Reusing the old chart angle would have produced a large error.
            old_code = encode_angle(z, x, 12)
            wrong, _ = decode_frame(new_z, old_code, 12)
            self.assertGreater(np.linalg.norm(wrong - moved), .5)

    def test_logarithmic_bounds_zero_constants_and_extremes(self):
        tiny = float(np.nextafter(np.float32(0), np.float32(1)))
        large = float(np.finfo(np.float32).max)
        for low, high in ((tiny, large), (1e-10, 4.2), (5, 1e6), P_BOUNDS, Q_BOUNDS, (2, 2)):
            for zero in (False, True):
                values = np.geomspace(low, high, 100)
                if zero:
                    values = np.r_[0, values]
                decoded = decode_log(encode_log(values, low, high, 16, zero=zero), low, high, 16, zero=zero)
                self.assertTrue(np.isfinite(decoded).all())
                np.testing.assert_equal(decoded[-1], high)
                self.assertEqual(decoded[1 if zero else 0], low)
                if zero:
                    self.assertEqual(decoded[0], 0)
                positive = values > 0
                bound = math.expm1(math.log(high / low) / (2 * (65535 - int(zero)))) + 1e-12
                self.assertLessEqual(np.max(np.abs(decoded[positive] / values[positive] - 1)), bound)

    def test_invalid_frames_and_physical_domains(self):
        a = arrays(3)
        for field, value in (("local_lambda", 0), ("local_mu", -1), ("local_coefficients", -1),
                             ("local_coefficients", np.nan), ("local_x", 0)):
            bad = {k: v.copy() for k, v in a.items()}
            bad[field][0] = value
            with self.assertRaises(ValueError):
                encode_arrays(bad, method="asg")
        bad = {**a, "local_y": -a["local_y"]}
        with self.assertRaisesRegex(ValueError, "right handed"):
            encode_arrays(bad, method="asg")
        for method, field in (("radial", "local_q"), ("contour", "local_p")):
            bad = arrays(3, method)
            bad[field][0] += .01
            with self.assertRaises(ValueError):
                encode_arrays(bad, method=method)
        for field, value in (("local_p", .99), ("local_p", 4.01),
                             ("local_q", 1.24), ("local_q", 16.01)):
            bad = arrays(3, "full")
            bad[field][0] = value
            with self.assertRaises(ValueError):
                encode_arrays(bad, method="full")
        bad = arrays(1, "radial")
        bad["local_q"] = np.array([2 + 1e-12])
        with self.assertRaisesRegex(ValueError, "exact"):
            encode_arrays(bad, method="radial")

    def test_actual_hdr_stream_extrema_and_zero_channels(self):
        a = arrays(3)
        tiny, large = np.nextafter(np.float32(0), np.float32(1)), np.finfo(np.float32).max
        a["local_coefficients"][:] = [[tiny, 0, large], [1e-10, 1, 100], [0, 0, 0]]
        _, decoded = decode_arrays(encode_arrays(a, method="asg"))
        self.assertEqual(decoded["local_coefficients"][0, 0], tiny)
        self.assertEqual(decoded["local_coefficients"][0, 2], large)
        np.testing.assert_array_equal(decoded["local_coefficients"][a["local_coefficients"] == 0], 0)
        self.assertTrue(np.isfinite(decoded["local_coefficients"]).all())


class NativeCodecTests(unittest.TestCase):
    def test_decoded_gaussian_member_and_dtype_contract(self):
        a = arrays(3, "full")
        a["local_p"][:] = 1
        a["local_q"][:] = 2
        shape = decode_native(encode_arrays(a, method="full"), dtype=torch.float64)
        # Share the decoded geometry and coefficients; only the tiny q rounding
        # distinguishes the decoded Gaussian member from the ordinary branch.
        plain = NativeInferenceRepresentation({"feature_dimension": 3, "asg_footprint": "gaussian"},
            {k: v for k, v in shape.arrays.items() if k not in ("local_p", "local_q")}, dtype=torch.float64)
        dirs = torch.randn(1000, 3, generator=torch.Generator().manual_seed(83), dtype=torch.float64)
        expected, actual = plain.evaluate_directions(dirs), shape.evaluate_directions(dirs)
        self.assertLess(float(torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)), .001)
        with self.assertRaises(ValueError):
            decode_native(encode_arrays(arrays(1), method="asg"), dtype=torch.float16)

    def test_decoded_rendering_error_is_small_for_all_variants(self):
        dirs = torch.randn(4000, 3, generator=torch.Generator().manual_seed(19))
        for method in METHODS:
            original = native(arrays(7, method), method)
            reference = original.evaluate_directions(dirs)
            data = encode_native(original, method=method)
            decoded = decode_native(data)
            result = decoded.evaluate_directions(dirs)
            self.assertTrue(bool(torch.isfinite(result).all()))
            self.assertTrue(bool((result >= 0).all()))
            relative = torch.linalg.vector_norm(result - reference) / torch.linalg.vector_norm(reference)
            self.assertLess(float(relative), .006, method)
            _, expected = decode_arrays(data)
            for k, v in decoded.arrays.items():
                np.testing.assert_array_equal(v.numpy(), expected[k])
            if method == "radial":
                self.assertTrue(bool((decoded.arrays["local_q"] == 2).all()))
            if method == "contour":
                self.assertTrue(bool((decoded.arrays["local_p"] == 1).all()))


if __name__ == "__main__":
    unittest.main()
