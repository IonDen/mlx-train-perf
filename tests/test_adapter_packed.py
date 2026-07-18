"""`make_packed_loss_fn` -- the packed-sequence trainer loss (spec §3.2, §8.4, §8.5).

The packed loss walks the trunk block-by-block to thread a `PackedMask` carrier (the
top-level model call hardcodes the "causal" string), so it REQUIRES the flash-attention
wrapper -- stock SDPA would hand the carrier to
`mx.fast.scaled_dot_product_attention` and die opaquely (review F5). The walk is exactly
`embed_tokens -> layers(h, mask, cache) -> norm`, re-confirmed against the installed
mlx-lm==0.31.3 (mlx==0.32.0) source this venv: llama.py:173-197 (LlamaModel.__call__),
qwen2.py:137-155 (Qwen2Model.__call__), qwen3.py:142-160 (Qwen3Model.__call__) -- all three
`embed_tokens -> for layer: layer(h, mask, c) -> norm`, no extra scaling.

Tolerances:
- `PACKED_ROPE_TOL = 2e-2` -- the headline packed-vs-unpacked parity bound. Gate A measured
  the RoPE offset drift of a 512-token segment at pack offset 3584 vs 0 (bf16, rope base 1e4)
  at max |ΔO| = 1.5625e-2 (exactly 1 bf16 ULP); rounded up one significant digit. Consumed
  ONLY here, never by kernel-parity pins.
- `_WALK_EQUIV_TOL` -- the walk-equivalence bound (measured below), NOT the RoPE tolerance:
  a single segment spanning the row sits at identical positions to the unpacked run, so the
  only drift is the packed-vs-causal kernel at identical positions.
"""
import math

import mlx.core as mx
import pytest
from mlx import nn

pytest.importorskip("mlx_lm")

from test_attention_wrapper import _tiny_llama_hd64

from mlx_train_perf.adapters.mlx_lm import make_loss_fn, make_packed_loss_fn, split_model
from mlx_train_perf.attention import wrapper as _wrapper_mod
from mlx_train_perf.attention.segments import PackedMask
from mlx_train_perf.attention.wrapper import enable_flash_attention
from mlx_train_perf.core.loss import linear_cross_entropy
from mlx_train_perf.data.packing import build_row
from mlx_train_perf.errors import AdapterError

# Gate A (bf16) measured max |ΔO| 1.5625e-2 (1 bf16 ULP), rounded up one sig-fig.
PACKED_ROPE_TOL = 2e-2

# Walk-equivalence: measured worst |diff| single-segment packed vs make_loss_fn's causal
# path at IDENTICAL positions (mlx 0.32.0, fp32, tiny hd64): 0.0 for BOTH reference
# (math_attention 1-seg == plain causal, bit-identical) AND kernel (the packed-vs-causal
# kernel is bit-identical at n=24, single all-same segment). Guarded at 1e-6 -- far under any
# real walk defect (a wrong mask/norm/denominator is O(0.1-1)), above float-ULP noise.
_WALK_EQUIV_TOL = 1e-6


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _packed_arrays(
    entries: list[tuple[list[int], int]], pack_len: int
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """One packed row (B=1) as the 4-array tuple `packed_iterate_batches` yields."""
    row, seg_id, seg_start, loss_mask = build_row(entries, pack_len)
    return (
        mx.array([row], dtype=mx.int32),
        mx.array([seg_id], dtype=mx.int32),
        mx.array([seg_start], dtype=mx.int32),
        mx.array([loss_mask], dtype=mx.bool_),
    )


def _unpacked_batch(
    entries: list[tuple[list[int], int]], width: int
) -> tuple[mx.array, mx.array]:
    """The same sequences as an UNPACKED (B, width+1) batch + `(offset, length)` lengths for
    `make_loss_fn` -- each row is its tokens right-padded with the pad id 0 (the target-only
    slot after the last real token is the 0 stock's `1 + ...` row sizing guarantees)."""
    rows = [list(tokens) + [0] * (width + 1 - len(tokens)) for tokens, _ in entries]
    lengths = [[offset, len(tokens)] for tokens, offset in entries]
    return mx.array(rows, dtype=mx.int32), mx.array(lengths, dtype=mx.int32)


def _packed_loss_from_emb(
    model: nn.Module,
    emb: mx.array,
    targets: mx.array,
    pm: PackedMask,
    loss_mask: mx.array,
    *,
    impl: str,
) -> mx.array:
    """`make_packed_loss_fn`'s walk, entered from a differentiable embedding tensor (mirrors
    the Gate-B `_packed_walk_scalar` idiom) so `mx.grad` wrt the per-position embedding is
    reachable -- the positional pad-gradient property (spec §8.4) cannot be read from a
    grad wrt token ids (a gather) or the shared-across-positions embedding weight."""
    _, head = split_model(model)
    h = emb
    for layer in model.model.layers:
        h = layer(h, pm, None)
    hidden = model.model.norm(h)
    nll = linear_cross_entropy(
        hidden, head, targets, impl=impl, reduction="none", validate_targets=False  # type: ignore[arg-type]
    )
    ntoks = loss_mask.sum()
    return (nll * loss_mask).astype(mx.float32).sum() / ntoks


# The three-sequence packed batch reused by the headline + flip-honesty tests: offsets
# 0/2/0 exercise the `max(offset, 1) - 1` supervised-window lower bound (segment 1 has a
# 2-token unsupervised prompt). Costs sum(len+1) = 21 <= pack_len+1 = 25, leaving a pad tail.
_PACK_LEN = 24
_ENTRIES_OFFSETS = (0, 2, 0)
_ENTRIES_LENS = (5, 6, 7)


def _three_sequences() -> list[tuple[list[int], int]]:
    mx.random.seed(7)
    entries: list[tuple[list[int], int]] = []
    for ln, off in zip(_ENTRIES_LENS, _ENTRIES_OFFSETS, strict=True):
        toks = mx.random.randint(1, 256, (ln,)).tolist()  # 1.. : never the pad id 0
        entries.append((toks, off))
    return entries


def _parity_losses(
    *, attn_impl: str, ce_impl: str, bf16: bool
) -> tuple[float, float, int, int]:
    """Packed loss (one row) vs unpacked loss (B=3) on the SAME wrapped model. Returns
    (packed_loss, unpacked_loss, packed_ntoks, unpacked_ntoks)."""
    mx.random.seed(0)
    model = _tiny_llama_hd64()
    if bf16:
        model.set_dtype(mx.bfloat16)
    enable_flash_attention(
        model, impl=attn_impl, seq_len=_PACK_LEN, batch_size=1, packed=True
    )
    mx.eval(model.parameters())

    entries = _three_sequences()
    batch_p, seg_id, seg_start, loss_mask = _packed_arrays(entries, _PACK_LEN)
    batch_u, lengths = _unpacked_batch(entries, _PACK_LEN)

    packed_fn = make_packed_loss_fn(model, impl=ce_impl)  # type: ignore[arg-type]
    unpacked_fn = make_loss_fn(model, impl=ce_impl)  # type: ignore[arg-type]
    lp, np_ = packed_fn(model, batch_p, seg_id, seg_start, loss_mask)
    lu, nu = unpacked_fn(model, batch_u, lengths)
    mx.eval(lp, np_, lu, nu)
    return lp.item(), lu.item(), int(np_.item()), int(nu.item())


# ---------------------------------------------------------------------------
# 1. construction fail-fast
# ---------------------------------------------------------------------------


def test_construction_refuses_unwrapped_model() -> None:
    """A model whose attention was never wrapped is refused at CONSTRUCTION with an
    `AdapterError` naming `enable_flash_attention` -- the packed carrier is interpretable
    only by the wrapper (spec §3.2 / review F5); stock SDPA would die opaquely mid-run."""
    model = _tiny_llama_hd64()  # NOT enable_flash_attention'd
    with pytest.raises(AdapterError, match="enable_flash_attention"):
        make_packed_loss_fn(model)


def test_construction_refuses_unsupported_family() -> None:
    """The family gate mirrors `make_loss_fn`: an unsupported architecture is refused at
    construction (via `split_model`), before the wrapper check."""
    with pytest.raises(AdapterError, match="unsupported model architecture"):
        make_packed_loss_fn(nn.Linear(4, 4))


# ---------------------------------------------------------------------------
# 2. walk-equivalence: one segment spanning the row == make_loss_fn's causal path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("attn_impl", "ce_impl"),
    [("reference", "chunked"), pytest.param("kernel", "kernel", marks=pytest.mark.metal)],
)
def test_walk_equivalence_single_segment(attn_impl: str, ce_impl: str) -> None:
    """A SINGLE segment spanning the whole row is block-diagonal-causal == plain causal at
    identical positions, so `make_packed_loss_fn` must reproduce `make_loss_fn` (which walks
    the SAME wrapped model with the "causal" string) -- proving the manual block walk
    reproduces `inner(x)` + mask + denominator. Drift is kernel-level only (packed-vs-causal
    kernel, identical RoPE positions), so it is pinned to its OWN measured constant
    `_WALK_EQUIV_TOL`, NOT the RoPE tolerance. ntoks are integer-equal by construction."""
    mx.random.seed(0)
    model = _tiny_llama_hd64()  # fp32 -- tightest measurement
    enable_flash_attention(model, impl=attn_impl, seq_len=_PACK_LEN, batch_size=1)
    mx.eval(model.parameters())

    tokens = mx.random.randint(1, 256, (_PACK_LEN,)).tolist()
    entries = [(tokens, 0)]  # one segment, seq_len == pack_len -> spans the row, no tail
    batch, seg_id, seg_start, loss_mask = _packed_arrays(entries, _PACK_LEN)
    lengths = mx.array([[0, _PACK_LEN]], dtype=mx.int32)

    packed_fn = make_packed_loss_fn(model, impl=ce_impl)  # type: ignore[arg-type]
    unpacked_fn = make_loss_fn(model, impl=ce_impl)  # type: ignore[arg-type]
    lp, np_ = packed_fn(model, batch, seg_id, seg_start, loss_mask)
    lu, nu = unpacked_fn(model, batch, lengths)
    mx.eval(lp, np_, lu, nu)

    assert int(np_.item()) == int(nu.item()) == _PACK_LEN
    diff = abs(lp.item() - lu.item())
    assert diff < _WALK_EQUIV_TOL, (
        f"{attn_impl}/{ce_impl} single-segment walk vs make_loss_fn |diff| {diff:.3e} "
        f"exceeds {_WALK_EQUIV_TOL:.0e}"
    )


# ---------------------------------------------------------------------------
# 3. packed-vs-unpacked parity -- THE headline (spec §8.5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("attn_impl", "ce_impl"),
    [("reference", "chunked"), pytest.param("kernel", "kernel", marks=pytest.mark.metal)],
)
def test_packed_vs_unpacked_parity(attn_impl: str, ce_impl: str) -> None:
    """THE headline (spec §8.5): 3 short sequences packed into one row vs the same 3 as an
    unpacked B=3 batch through `make_loss_fn`, on the SAME bf16 wrapped model. The §4
    separator-slot construction makes the supervised-token multisets IDENTICAL -> `ntoks`
    EXACTLY equal (integer assert), and the loss matches within `PACKED_ROPE_TOL` (the only
    residual is RoPE offset drift -- segments sit at absolute positions 0 / 6 / 13 packed vs
    0 unpacked -- which this bound is measured from, plus the packed-vs-causal kernel drift).
    Simultaneously proves masking, boundary algebra, and the RoPE no-reset claim. Observed on
    this tiny fixture (mlx 0.32.0, bf16, rope base 1e4, small offsets): reference/chunked
    2.29e-4, kernel/kernel 5.00e-4 -- both far inside the conservative 2e-2 design ceiling."""
    lp, lu, np_, nu = _parity_losses(attn_impl=attn_impl, ce_impl=ce_impl, bf16=True)

    assert np_ == nu, f"packed ntoks {np_} != unpacked ntoks {nu} (multiset mismatch)"
    diff = abs(lp - lu)
    assert diff < PACKED_ROPE_TOL, (
        f"{attn_impl}/{ce_impl} packed {lp:.5f} vs unpacked {lu:.5f} |diff| {diff:.3e} "
        f"exceeds PACKED_ROPE_TOL {PACKED_ROPE_TOL:.0e}"
    )


# ---------------------------------------------------------------------------
# 4. pad-tail: finite loss (no NaN) + zero gradient through pad positions (spec §8.4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("attn_impl", "ce_impl"),
    [("reference", "chunked"), pytest.param("kernel", "kernel", marks=pytest.mark.metal)],
)
def test_pad_tail_finite_and_zero_pad_gradient(attn_impl: str, ce_impl: str) -> None:
    """A packed row with a long pad tail: the loss is FINITE (the gapless-coverage invariant
    keeps every pad row attending at least itself -> no all-`-inf` softmax row -> no NaN,
    spec §4), and `mx.grad` wrt the per-position embedding has EXACTLY-zero rows at every pad
    position -- the segment's own trailing pad slot AND the whole tail segment, none of which
    a supervised position attends (block-diagonal) or supervises (`loss_mask` False). Run
    against BOTH the reference oracle and the kernel attention path (spec §8.4)."""
    pack_len = 32
    seg_len = 4
    mx.random.seed(0)
    model = _tiny_llama_hd64()
    enable_flash_attention(model, impl=attn_impl)  # eager grad: lazy calibration is fine
    mx.eval(model.parameters())

    tokens = mx.random.randint(1, 256, (seg_len,)).tolist()
    entries = [(tokens, 0)]  # one short segment; positions [seg_len, pack_len) are pad
    batch, seg_id, seg_start, loss_mask = _packed_arrays(entries, pack_len)

    # Forward finiteness through the REAL public function.
    loss_fn = make_packed_loss_fn(model, impl=ce_impl)  # type: ignore[arg-type]
    loss, ntoks = loss_fn(model, batch, seg_id, seg_start, loss_mask)
    mx.eval(loss, ntoks)
    assert math.isfinite(loss.item()), "pad tail produced a non-finite loss (NaN leak)"
    assert int(ntoks.item()) == seg_len  # offset 0 -> the whole segment is supervised

    # Zero gradient through pad positions.
    inputs, targets = batch[:, :-1], batch[:, 1:]
    pm = PackedMask(seg_id=seg_id, seg_start=seg_start)
    emb = model.model.embed_tokens(inputs)
    mx.eval(emb)
    grad_emb = mx.grad(
        lambda e: _packed_loss_from_emb(model, e, targets, pm, loss_mask, impl=ce_impl)
    )(emb)
    mx.eval(grad_emb)

    row_norm = mx.sqrt((grad_emb.astype(mx.float32) ** 2).sum(axis=-1))[0]  # (pack_len,)
    pad_max = row_norm[seg_len:].max().item()  # pad slot + tail segment
    sup_max = row_norm[:seg_len].max().item()  # the supervised segment
    assert pad_max == 0.0, f"{attn_impl}: nonzero gradient at pad positions (max {pad_max:.3e})"
    assert sup_max > 0.0, f"{attn_impl}: no gradient reached the supervised segment"


# ---------------------------------------------------------------------------
# 5. flip-parity honesty (acceptance criterion 3): the suite detects dropped segments
# ---------------------------------------------------------------------------


def test_flip_honesty_packed_parity_detects_dropped_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cross-contamination detector: with the wrapper's mask resolution monkeypatched to DROP
    the packed carrier (every mask -> plain causal, `segments=None`), the 3 packed segments
    become one causal chain -- segments 1 and 2 attend the prior segments -- so the headline
    parity assertion MUST fail. Baseline (unpatched) parity holds first; the recorded
    divergence proves the suite would catch a wrapper that silently drops segments."""
    # Baseline: parity holds (reference/chunked, bf16 -- default lane, no metal).
    lp0, lu0, np0, nu0 = _parity_losses(attn_impl="reference", ce_impl="chunked", bf16=True)
    assert np0 == nu0
    assert abs(lp0 - lu0) < PACKED_ROPE_TOL, "baseline packed parity is already broken"

    # Flip: force every resolved mask to plain causal (PackedMask -> True -> segments=None).
    monkeypatch.setattr(_wrapper_mod, "_resolve_mask", lambda _mask: True)
    lp1, lu1, np1, nu1 = _parity_losses(attn_impl="reference", ce_impl="chunked", bf16=True)

    assert np1 == nu1, "ntoks come from loss_mask, not attention -- the flip must not move them"
    divergence = abs(lp1 - lu1)
    assert divergence > PACKED_ROPE_TOL, (
        f"dropping segments left the packed loss within tolerance (|diff| {divergence:.3e}) -- "
        "the parity test would NOT catch cross-contamination"
    )
