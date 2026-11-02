"""SKC v1: post-training scalar coding of native RGB spherical kernels.

Every fit writes its bitstream as representation.skc; reconstruct.py and
evaluate.py read it like a run directory.
"""

from codec.spherical import decode_arrays, encode_arrays

__all__ = ["encode_arrays", "decode_arrays"]
