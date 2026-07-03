import mlx.core as mx
import pytest

from mlx_train_perf.core.chunked import QuantSpec
from mlx_train_perf.core.kernel.launch import forward_quantized
from mlx_train_perf.errors import UnsupportedHeadError


def test_rejects_unsupported_bits_or_group() -> None:
    q = QuantSpec(w_q=mx.zeros((8, 8), dtype=mx.uint32), scales=mx.ones((8, 1)),
                  biases=mx.zeros((8, 1)), group_size=64, bits=8)  # bits=8 unsupported
    with pytest.raises(UnsupportedHeadError):
        forward_quantized(mx.zeros((4, 64)), q, mx.zeros((4,), dtype=mx.int32),
                          row_tiles=4, tile=256, rate_macs_per_s=1e13)


def test_rejects_d_not_multiple_of_64() -> None:
    q = QuantSpec(w_q=mx.zeros((8, 12), dtype=mx.uint32), scales=mx.ones((8, 2)),
                  biases=mx.zeros((8, 2)), group_size=64, bits=4)
    with pytest.raises(UnsupportedHeadError):  # d=96, not a multiple of 64
        forward_quantized(mx.zeros((4, 96)), q, mx.zeros((4,), dtype=mx.int32),
                          row_tiles=4, tile=256, rate_macs_per_s=1e13)
