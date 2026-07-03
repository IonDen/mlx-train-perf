"""Regression guard for the MLX affine int4/gs64 layout the quant kernel decodes in MSL."""
import mlx.core as mx


def test_int4_gs64_shapes_and_low_nibble_first_order() -> None:
    mx.random.seed(17)
    w = mx.random.normal((8, 64)).astype(mx.bfloat16)
    w_q, scales, biases = mx.quantize(w, group_size=64, bits=4)
    assert w_q.shape == (8, 8) and w_q.dtype == mx.uint32     # noqa: PT018 — 8 nibbles/uint32
    assert scales.shape == (8, 1) and biases.shape == (8, 1)  # noqa: PT018 — one group/row here
    w_dq = mx.dequantize(w_q, scales, biases, group_size=64, bits=4)
    for k in (0, 1, 7, 8, 63):  # positions crossing nibble and uint32 boundaries
        packed = int(w_q[0, k // 8].item())
        nib = (packed >> (4 * (k % 8))) & 0xF
        manual = float(scales[0, 0].item()) * nib + float(biases[0, 0].item())
        assert abs(manual - float(w_dq[0, k].item())) < 1e-2  # bf16 scale/bias rounding
