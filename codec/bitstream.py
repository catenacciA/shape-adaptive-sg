"""SKC v1 wire format of the paper's fixed-rate code (Sec. 3.5, Eq. 8).

A 28-byte header is followed by unpadded records. Per primitive: octahedral
direction 2x16 bits, tangent angle mod pi 12 bits, two log concentrations
2x12 bits, three log RGB coefficients 3x12 bits (code 0 = zero), and for the
shape-adaptive variants a log radial exponent p and/or contour exponent q,
8 bits each. Records are 104 bits (ASG) and 120 bits (Ours).
"""

from dataclasses import dataclass
import math
import struct


MAGIC = b"SKC\0"
VERSION = 1
HEADER = struct.Struct("<4sBBBBI4f")
METHODS = ("asg", "radial", "contour", "full")
# Coordinate, angle, concentration, coefficient, p, q bits.
BITS = (16, 12, 12, 12, 8, 8)
FORMAT_ID = 0  # header byte 6; reserved for future precision formats


def field_widths(method):
    if method not in METHODS:
        raise ValueError("unknown SKC method")
    d, t, s, c, p, q = BITS
    widths = (d, d, t, s, s, c, c, c)
    if method in ("radial", "full"):
        widths += (p,)
    if method in ("contour", "full"):
        widths += (q,)
    return widths


@dataclass(frozen=True)
class Header:
    method: str
    count: int
    scale_min: float
    scale_max: float
    coefficient_min: float
    coefficient_max: float

    @property
    def record_bits(self):
        return sum(field_widths(self.method))

    @property
    def file_bytes(self):
        return HEADER.size + (self.count * self.record_bits + 7) // 8

    def validate(self):
        field_widths(self.method)
        if type(self.count) is not int or not 0 <= self.count <= 0xFFFFFFFF:
            raise ValueError("invalid SKC primitive count")
        for low, high in ((self.scale_min, self.scale_max),
                          (self.coefficient_min, self.coefficient_max)):
            if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
                raise ValueError("SKC endpoints must be finite, positive, and ordered")

    def pack(self):
        self.validate()
        try:
            packed = HEADER.pack(MAGIC, VERSION, METHODS.index(self.method),
                                 FORMAT_ID, 0, self.count,
                                 self.scale_min, self.scale_max,
                                 self.coefficient_min, self.coefficient_max)
            if any(v <= 0 or not math.isfinite(v) for v in HEADER.unpack(packed)[6:]):
                raise ValueError("SKC endpoints must fit positive FP32")
            return packed
        except (OverflowError, struct.error) as exc:
            raise ValueError("SKC endpoints must fit FP32") from exc


def read_header(data):
    """Check the entire file length and padding before allocating decoded rows."""
    if len(data) < HEADER.size:
        raise ValueError("truncated SKC header")
    magic, version, method, fmt, reserved, count, *ends = HEADER.unpack_from(data)
    if magic != MAGIC or version != VERSION:
        raise ValueError("unsupported SKC magic or version")
    if method >= len(METHODS) or fmt != FORMAT_ID or reserved:
        raise ValueError("invalid SKC method, format, or reserved byte")
    header = Header(METHODS[method], count, *ends)
    header.validate()
    if len(data) != header.file_bytes:
        raise ValueError("SKC file length does not match primitive count")
    remainder = (count * header.record_bits) % 8
    if remainder and data[-1] >> remainder:
        raise ValueError("nonzero SKC padding bits")
    return header


def pack_records(rows, widths):
    """Field and byte order are both LSB first; there is no row padding."""
    output = bytearray()
    buffer = occupied = 0
    for row in rows:
        if len(row) != len(widths):
            raise ValueError("wrong SKC record field count")
        for value, bits in zip(row, widths):
            integer = int(value)
            if integer != value or not 0 <= integer < 1 << bits:
                raise ValueError("SKC symbol outside field range")
            buffer |= integer << occupied
            occupied += bits
            while occupied >= 8:
                output.append(buffer & 255)
                buffer >>= 8
                occupied -= 8
    if occupied:
        output.append(buffer)
    return bytes(output)


def unpack_records(payload, widths, count):
    buffer = occupied = offset = 0
    for _ in range(count):
        row = []
        for bits in widths:
            while occupied < bits:
                if offset >= len(payload):
                    raise ValueError("truncated SKC record")
                buffer |= payload[offset] << occupied
                occupied += 8
                offset += 1
            row.append(buffer & ((1 << bits) - 1))
            buffer >>= bits
            occupied -= bits
        yield row
