import mlx.core as mx
import pytest

from mlx_train_perf.core.chunked import QuantSpec
from mlx_train_perf.core.kernel.launch import forward, forward_quantized
from mlx_train_perf.core.naive import naive_linear_ce

pytestmark = pytest.mark.metal
GENEROUS_RATE = 1e13


def _qdata(n: int, d: int, v: int) -> tuple[mx.array, QuantSpec, mx.array, mx.array]:
    mx.random.seed(13)
    hidden = mx.random.normal((n, d)).astype(mx.bfloat16)
    w = (mx.random.normal((v, d)) * 0.05).astype(mx.bfloat16)
    targets = mx.random.randint(0, v, (n,))
    targets[0] = 0
    targets[1] = v - 1
    w_q, scales, biases = mx.quantize(w, group_size=64, bits=4)
    q = QuantSpec(w_q=w_q, scales=scales, biases=biases, group_size=64, bits=4)
    w_dq = mx.dequantize(w_q, scales, biases, group_size=64, bits=4)
    return hidden, q, targets, w_dq


def _qdata_fp32(n: int, d: int, v: int) -> tuple[mx.array, QuantSpec, mx.array, mx.array]:
    """Same construction as `_qdata`, but fp32 hidden/weights -> fp32 scales/biases: no
    bf16 intermediate anywhere in the dequant path. This is the exact ablation that
    isolated the 8e-3-class gap in test_parity_vs_dequantize_then_kernel_oracle to
    mx.dequantize's OWN bf16 rounding rather than to our nibble/group/stride math."""
    mx.random.seed(13)
    hidden = mx.random.normal((n, d)).astype(mx.float32)
    w = (mx.random.normal((v, d)) * 0.05).astype(mx.float32)
    targets = mx.random.randint(0, v, (n,))
    targets[0] = 0
    targets[1] = v - 1
    w_q, scales, biases = mx.quantize(w, group_size=64, bits=4)
    q = QuantSpec(w_q=w_q, scales=scales, biases=biases, group_size=64, bits=4)
    w_dq = mx.dequantize(w_q, scales, biases, group_size=64, bits=4)
    return hidden, q, targets, w_dq


@pytest.mark.parametrize("row_tiles", [2, 4])
@pytest.mark.parametrize(("n", "d", "v", "tile"), [
    (64, 64, 1024, 256), (64, 128, 1024, 333), (65, 192, 1027, 256), (128, 64, 8192, 1024),
])
def test_parity_vs_dequantize_then_kernel_oracle(
    n: int, d: int, v: int, tile: int, row_tiles: int,
) -> None:
    hidden, q, targets, w_dq = _qdata(n, d, v)
    lse_o, tgt_o = forward(hidden, w_dq, targets, row_tiles=row_tiles, tile=tile,
                           rate_macs_per_s=GENEROUS_RATE)
    lse_q, tgt_q = forward_quantized(hidden, q, targets, row_tiles=row_tiles, tile=tile,
                                     rate_macs_per_s=GENEROUS_RATE)
    # RED at 1e-6 measured up to 2.8682e-3 (n=65,d=192,v=1027,tile=256) — NOT a nibble/
    # group/row-stride bug: tests/test_quant_layout.py independently pins the layout, and
    # an fp32-head ablation of this exact grid (scales/biases fp32, no bf16 intermediate)
    # measured an EXACT 0.0 diff. Root cause: mx.dequantize's own MSL kernel computes
    # `scale * nib + bias` in bf16 arithmetic (T=bf16 in Apple's shipped
    # quantized.h::affine_dequantize) and both oracles here (`forward`'s dense kernel and
    # `naive_linear_ce`) read that already-bf16-rounded w_dq buffer; our mtp_dq4 casts to
    # float BEFORE the multiply-add, so it reconstructs the dequantized value MORE
    # faithfully than the bf16-materialized oracle it's compared against. Pin the measured
    # worst (2.8682e-3) at a ~2.8x margin (same convention as test_chunked.py's bf16 pins).
    assert mx.abs((lse_q - tgt_q) - (lse_o - tgt_o)).max().item() < 8e-3
    # INDEPENDENT anchor: the line above shares the whole MMA/launcher/chaining code with
    # its oracle — a bug living in both sides is invisible to it. The naive fp32 reference
    # is outside that code entirely. Measured worst is identical to the line above (both
    # oracles are dominated by the same bf16-dequant rounding, not by their own drift from
    # each other — `forward` vs `naive_linear_ce` alone is ~1e-6, per test_kernel_parity.py).
    ref = naive_linear_ce(hidden.astype(mx.float32), w_dq.astype(mx.float32), targets)
    assert mx.abs((lse_q - tgt_q) - ref).max().item() < 8e-3


@pytest.mark.parametrize("row_tiles", [2, 4])
@pytest.mark.parametrize(("n", "d", "v", "tile"), [
    (64, 64, 1024, 256), (64, 128, 1024, 333), (65, 192, 1027, 256), (128, 64, 8192, 1024),
])
def test_dequant_correctness_lock_fp32_head_no_bf16_rounding(
    n: int, d: int, v: int, tile: int, row_tiles: int,
) -> None:
    """DEQUANT-CORRECTNESS LOCK — bf16-rounding-free anchor for nibble/group/row-stride
    math. `test_parity_vs_dequantize_then_kernel_oracle`'s 8e-3 gate is desensitized: both
    sides there read the SAME bf16-rounded `w_dq` (mx.dequantize itself rounds to bf16
    in-kernel — see that test's comment), so an indexing bug small enough to hide under
    bf16 rounding noise would pass it undetected. With fp32 scales/biases (no bf16
    intermediate anywhere), the ablation that root-caused the 8e-3 gap measured an EXACT
    0.0 diff on this exact grid — pin 1e-5-class (not 0.0 literally, to tolerate fp32
    reduction-order noise across MLX versions/hardware) so a REAL layout bug (wrong
    nibble shift, wrong group index, wrong row stride) trips this test even though it
    would slip past the bf16-desensitized gate above.
    """
    hidden, q, targets, w_dq = _qdata_fp32(n, d, v)
    lse_o, tgt_o = forward(hidden, w_dq, targets, row_tiles=row_tiles, tile=tile,
                           rate_macs_per_s=GENEROUS_RATE)
    lse_q, tgt_q = forward_quantized(hidden, q, targets, row_tiles=row_tiles, tile=tile,
                                     rate_macs_per_s=GENEROUS_RATE)
    assert mx.abs((lse_q - tgt_q) - (lse_o - tgt_o)).max().item() < 1e-5
