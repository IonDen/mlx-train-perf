"""T12 -- `enable_flash_attention` mlx-lm integration wrapper (spec §4.1 amended, §5, §9 P4).

Family detection + typed enable-time refusals, the call-time causal-mask contract, LoRA
attach-into-wrapper, the make_loss_fn coexistence, an mlx-lm attention-surface drift pin,
the compiled-train pre-calibration contract (kernel rate caches warm before the trace), and
-- subprocess-isolated -- one real gc=True compiled train() run.

Head-dim discipline: `enable_flash_attention` gates head_dim to the kernel's {64, 96, 128}
UNCONDITIONALLY (a training wrapper whose reason to exist is the kernel), so every enabled
model here is built at head_dim=64 -- NOT the head_dim=16 `tests/test_worker_train_step.py::
_tiny_llama` helper, which is reused only as the head_dim-refusal fixture (exactly the shape
the gate must reject).
"""
import inspect
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.optimizers as optim
import pytest
from mlx import nn
from mlx.utils import tree_flatten

pytest.importorskip("mlx_lm")

from mlx_lm.models import llama, qwen2, qwen3
from mlx_lm.models.base import create_attention_mask
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.trainer import default_loss
from mlx_lm.tuner.utils import linear_to_lora_layers

# Reused as the head_dim-refusal fixture only (head_dim=16 < the kernel's supported set).
from test_worker_train_step import _tiny_llama

from mlx_train_perf.adapters.mlx_lm import make_loss_fn
from mlx_train_perf.attention.api import flash_attention
from mlx_train_perf.attention.kernel.dispatch import select_bwd_tiles, select_fwd_tile
from mlx_train_perf.attention.kernel.launch import (
    _BWD_DKV_RATE_CACHE,
    _BWD_DQ_RATE_CACHE,
    _FWD_RATE_CACHE,
    _n_bucket,
)
from mlx_train_perf.attention.reference import math_attention
from mlx_train_perf.attention.segments import PackedMask
from mlx_train_perf.attention.wrapper import (
    FlashAttentionWrapper,
    _resolve_mask,
    enable_flash_attention,
)
from mlx_train_perf.bench.worker import _run_train_steps, _synthetic_train_examples
from mlx_train_perf.errors import AttentionInputError, UnsupportedAttentionError

_GC_CHILD = Path(__file__).parent / "_attention_wrapper_gc_child.py"
_HEAD_DIM = 64


# ---------------------------------------------------------------------------
# tiny head_dim=64 model builders (llama + qwen3)
# ---------------------------------------------------------------------------


def _tiny_llama_hd64() -> llama.Model:
    args = llama.ModelArgs(
        model_type="llama", hidden_size=128, num_hidden_layers=2, intermediate_size=256,
        num_attention_heads=2, num_key_value_heads=1, vocab_size=256, rms_norm_eps=1e-5,
        rope_theta=10000.0, tie_word_embeddings=False, head_dim=_HEAD_DIM,
    )
    return llama.Model(args)


def _tiny_qwen3_hd64() -> qwen3.Model:
    args = qwen3.ModelArgs(
        model_type="qwen3", hidden_size=128, num_hidden_layers=2, intermediate_size=256,
        num_attention_heads=2, num_key_value_heads=1, vocab_size=256, rms_norm_eps=1e-5,
        rope_theta=1000000.0, head_dim=_HEAD_DIM, max_position_embeddings=2048,
        tie_word_embeddings=False,
    )
    return qwen3.Model(args)


def _tiny_qwen2_hd64() -> qwen2.Model:
    # qwen2 derives head_dim = hidden_size // num_attention_heads (no head_dim field), so
    # 128 / 2 = 64 lands in the kernel's {64, 96, 128}. qwen2's q/k/v carry bias=True
    # (llama/qwen3 do not); the wrapper holds the nn.Linear submodules directly, so the bias
    # is applied transparently.
    args = qwen2.ModelArgs(
        model_type="qwen2", hidden_size=128, num_hidden_layers=2, intermediate_size=256,
        num_attention_heads=2, num_key_value_heads=1, vocab_size=256, rms_norm_eps=1e-5,
        rope_theta=1000000.0, max_position_embeddings=2048, tie_word_embeddings=False,
    )
    return qwen2.Model(args)


_FAMILIES = {
    "llama": _tiny_llama_hd64, "qwen2": _tiny_qwen2_hd64, "qwen3": _tiny_qwen3_hd64
}


def _ids(vocab: int, b: int, length: int, *, seed: int = 0) -> mx.array:
    mx.random.seed(seed)
    return mx.random.randint(0, vocab, (b, length))


def _packed_layout(seg_lens: list[int], b: int) -> tuple[mx.array, mx.array]:
    """(B, N) int32 seg_id/seg_start for a fixed segment-length list, shared across batch rows
    (mirrors tests/test_attention_api.py::_packed_layout -- the PackedMask contract: seg_id
    contiguous ascending, seg_start each position's segment-start index)."""
    seg_id_row: list[int] = []
    seg_start_row: list[int] = []
    start = 0
    for sid, ln in enumerate(seg_lens):
        seg_id_row += [sid] * ln
        seg_start_row += [start] * ln
        start += ln
    seg_id = mx.array([seg_id_row] * b, dtype=mx.int32)
    seg_start = mx.array([seg_start_row] * b, dtype=mx.int32)
    return seg_id, seg_start


def _wrapper_ref_forward(
    wrapper: FlashAttentionWrapper, x: mx.array, *, segments: PackedMask | None
) -> mx.array:
    """Independent oracle for the wrapper's forward (llama, no qk-norm): reproduces the
    project -> reshape -> RoPE steps on the wrapper's OWN held submodules, then routes the
    attention through `math_attention(segments=)` (the block-diagonal-causal oracle) instead
    of `flash_attention`. Distinguishes segments-forwarded from segments-dropped: with
    `segments` set the oracle is block-diagonal, so a wrapper that dropped the mask would not
    match it."""
    b, length, _ = x.shape
    q = wrapper.q_proj(x).reshape(b, length, wrapper.n_heads, -1).transpose(0, 2, 1, 3)
    k = wrapper.k_proj(x).reshape(b, length, wrapper.n_kv_heads, -1).transpose(0, 2, 1, 3)
    v = wrapper.v_proj(x).reshape(b, length, wrapper.n_kv_heads, -1).transpose(0, 2, 1, 3)
    q = wrapper.rope(q)
    k = wrapper.rope(k)
    o = math_attention(q, k, v, scale=wrapper.scale, causal=True, segments=segments)
    o = o.transpose(0, 2, 1, 3).reshape(b, length, -1)
    return wrapper.o_proj(o)


# ---------------------------------------------------------------------------
# 1. wrapper output parity against the stock attention module (both impls)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", list(_FAMILIES))
@pytest.mark.parametrize(
    "impl", ["reference", pytest.param("kernel", marks=pytest.mark.metal)]
)
def test_wrapper_output_matches_stock_attention_module(family: str, impl: str) -> None:
    """Full-model forward through stock SDPA vs the enabled wrapper: identical weights (the
    wrapper holds the ORIGINAL projections), differing only in the attention call. Weights are
    seeded below, so the measured worst |diff| reproduces (mlx 0.32.0, fp32, tiny 1x8,
    head_dim=64): reference 8.345e-07 (llama, qwen2) / 8.941e-07 (qwen3); kernel 9.537e-07
    (llama) / 1.073e-06 (qwen2) / 1.013e-06 (qwen3) -- the kernel's fp32-accumulate flash
    forward vs mlx's fused SDPA. Pin 2e-6 (~1.86x over the measured worst, qwen2 kernel
    1.073e-06)."""
    mx.random.seed(0)  # deterministic weights, so the measured worst-diff below reproduces
    model = _FAMILIES[family]()
    ids = _ids(model.args.vocab_size, 1, 8)
    out_stock = model(ids)
    mx.eval(out_stock)

    enable_flash_attention(model, impl=impl)
    out_ours = model(ids)
    mx.eval(out_ours)

    worst = mx.abs(out_ours - out_stock).max().item()
    assert worst < 2e-6, f"{family}/{impl} worst |diff| {worst:.3e} exceeds 2e-6"


# ---------------------------------------------------------------------------
# 2. enable-time refusals (all UnsupportedAttentionError)
# ---------------------------------------------------------------------------


def test_enable_refuses_mixed_layer_types_llama() -> None:
    args = llama.ModelArgs(
        model_type="llama", hidden_size=128, num_hidden_layers=2, intermediate_size=256,
        num_attention_heads=2, num_key_value_heads=1, vocab_size=256, rms_norm_eps=1e-5,
        rope_theta=10000.0, tie_word_embeddings=False, head_dim=_HEAD_DIM,
        layer_types=["full_attention", "sliding_attention"], sliding_window=8,
    )
    model = llama.Model(args)
    with pytest.raises(UnsupportedAttentionError, match="sliding"):
        enable_flash_attention(model)


def test_enable_refuses_unknown_family() -> None:
    with pytest.raises(UnsupportedAttentionError, match="unsupported model architecture"):
        enable_flash_attention(nn.Linear(4, 4))


def test_enable_refuses_head_dim_outside_supported_set() -> None:
    """`_tiny_llama` is head_dim=16 (64 hidden / 4 heads) -- below the kernel's {64,96,128}.
    The gate refuses it regardless of impl (a training wrapper whose reason to exist is the
    kernel)."""
    with pytest.raises(UnsupportedAttentionError, match="head_dim"):
        enable_flash_attention(_tiny_llama())


def test_enable_refuses_configured_dropout() -> None:
    model = _tiny_llama_hd64()
    model.args.attention_dropout = 0.1  # config carries a nonzero dropout
    with pytest.raises(UnsupportedAttentionError, match="dropout"):
        enable_flash_attention(model)


# ---------------------------------------------------------------------------
# 3. call-time mask contract (the STRING "causal" is THE supported case)
# ---------------------------------------------------------------------------


def test_resolve_mask_accepts_string_none_packedmask_refuses_array() -> None:
    assert _resolve_mask("causal") is True
    assert _resolve_mask(None) is True
    seg_id, seg_start = _packed_layout([2, 2], b=1)
    pm = PackedMask(seg_id=seg_id, seg_start=seg_start)
    assert _resolve_mask(pm) is pm  # a PackedMask passes through untouched
    with pytest.raises(AttentionInputError, match="causal"):
        _resolve_mask("sliding")
    with pytest.raises(AttentionInputError, match="array attention masks"):
        _resolve_mask(mx.zeros((4, 4)))


@pytest.mark.parametrize(
    "impl", ["reference", pytest.param("kernel", marks=pytest.mark.metal)]
)
def test_wrapper_forwards_packed_mask_into_flash_attention_segments(impl: str) -> None:
    """A `PackedMask` mask routes through the wrapper into `flash_attention(segments=)`: the
    wrapper output matches the hand-built block-diagonal oracle (project -> RoPE ->
    math_attention(segments=) -> o_proj) AND differs from the plain-causal output -- proving
    the segments actually reached the attention rather than being silently dropped. Measured
    worst |diff| vs the oracle (mlx 0.32.0, fp32, tiny 1x16, head_dim=64): reference
    0.000e+00 (bit-identical -- flash_attention_reference and math_attention share a code
    path), kernel ~9e-07 (fp32-accumulate flash vs materialized oracle). Pin 2e-6, matching
    the sibling wrapper/API packed pins."""
    model = _tiny_llama_hd64()
    enable_flash_attention(model, impl=impl)
    wrapper = model.model.layers[0].self_attn
    assert isinstance(wrapper, FlashAttentionWrapper)

    b, length = 1, 16
    x = mx.random.normal((b, length, model.args.hidden_size))
    mx.eval(x)
    seg_id, seg_start = _packed_layout([9, 7], b=b)  # two segments
    pm = PackedMask(seg_id=seg_id, seg_start=seg_start)

    out = wrapper(x, mask=pm)
    oracle = _wrapper_ref_forward(wrapper, x, segments=pm)
    out_causal = wrapper(x, mask="causal")
    mx.eval(out, oracle, out_causal)

    worst = mx.abs(out - oracle).max().item()
    assert worst < 2e-6, f"{impl} wrapper packed forward worst |diff| {worst:.3e} exceeds 2e-6"
    assert not mx.array_equal(out, out_causal).item(), (
        "packed and plain-causal outputs are identical -- the PackedMask never reached the "
        "attention call (segments silently dropped)"
    )


def test_call_time_guard_accepts_causal_string_refuses_array_mask() -> None:
    """The wrapper's __call__ mask guard, exercised on a real enabled module (impl=reference,
    default lane). `"causal"` and `None` run; an `mx.array` mask and a cache both refuse."""
    model = _tiny_llama_hd64()
    enable_flash_attention(model, impl="reference")
    wrapper = model.model.layers[0].self_attn
    assert isinstance(wrapper, FlashAttentionWrapper)
    x = mx.random.normal((1, 8, model.args.hidden_size))
    mx.eval(x)

    out_causal = wrapper(x, mask="causal")
    out_none = wrapper(x, mask=None)
    mx.eval(out_causal, out_none)
    assert out_causal.shape == x.shape
    assert out_none.shape == x.shape

    with pytest.raises(AttentionInputError, match="array attention masks"):
        wrapper(x, mask=mx.zeros((8, 8)))
    with pytest.raises(AttentionInputError, match="training-only"):
        wrapper(x, cache=object())


# ---------------------------------------------------------------------------
# 4. compiled-train pre-calibration contract (T5-review, binding)
# ---------------------------------------------------------------------------


@pytest.mark.metal
def test_enable_prewarms_rate_caches_no_calibration_in_compiled_trace() -> None:
    """Enabling with a `seq_len` hint pre-warms all three kernel rate caches (fwd + bwd dQ +
    bwd dK/dV) for the training shape, so a subsequently COMPILED grad step traces with warm
    caches and never runs a host-synced calibration inside the compiled region. Isolated by
    popping the target keys first, so this genuinely proves ENABLE warms them (not a prior
    test)."""
    model = _tiny_llama_hd64()
    attn = model.model.layers[0].self_attn
    hq, hkv = attn.n_heads, attn.n_kv_heads
    dtype = model.model.norm.weight.dtype
    n, b = 32, 1

    fwd_tile = select_fwd_tile(n, _HEAD_DIM)
    dq_tile, dkv_tile = select_bwd_tiles(n, _HEAD_DIM)
    nb = _n_bucket(n)
    # key tail gains a trailing `packed` bool (0.4.0); enable pre-warms the non-packed path.
    fkey = (_HEAD_DIM, str(dtype), True, b, hq, nb, fwd_tile.variant, fwd_tile.d_slab, False)
    dqkey = (_HEAD_DIM, str(dtype), True, b, hq, nb, dq_tile.variant, dq_tile.d_slab, False)
    dkvkey = (_HEAD_DIM, str(dtype), True, b, hq, nb, dkv_tile.variant, dkv_tile.d_slab, False)
    for cache, key in (
        (_FWD_RATE_CACHE, fkey), (_BWD_DQ_RATE_CACHE, dqkey), (_BWD_DKV_RATE_CACHE, dkvkey)
    ):
        cache.pop(key, None)

    enable_flash_attention(model, impl="kernel", seq_len=n, batch_size=b)

    assert fkey in _FWD_RATE_CACHE, "enable did not pre-warm the forward rate cache"
    assert dqkey in _BWD_DQ_RATE_CACHE, "enable did not pre-warm the dQ backward rate cache"
    assert dkvkey in _BWD_DKV_RATE_CACHE, "enable did not pre-warm the dK/dV rate cache"
    sizes = (len(_FWD_RATE_CACHE), len(_BWD_DQ_RATE_CACHE), len(_BWD_DKV_RATE_CACHE))

    q = mx.zeros((b, hq, n, _HEAD_DIM), dtype=dtype)
    k = mx.zeros((b, hkv, n, _HEAD_DIM), dtype=dtype)
    v = mx.zeros((b, hkv, n, _HEAD_DIM), dtype=dtype)

    def loss(q_: mx.array) -> mx.array:
        return flash_attention(q_, k, v, scale=1.0 / _HEAD_DIM**0.5, causal=True,
                               impl="kernel").sum()

    g = mx.compile(mx.grad(loss))(q)  # a host-synced calibration in-trace would raise here
    mx.eval(g)

    assert (len(_FWD_RATE_CACHE), len(_BWD_DQ_RATE_CACHE), len(_BWD_DKV_RATE_CACHE)) == sizes


@pytest.mark.metal
def test_enable_packed_prewarms_packed_keyed_rate_caches() -> None:
    """`enable_flash_attention(packed=True, seq_len=...)` additionally warms the THREE
    packed-keyed rate caches (fwd + bwd dQ + bwd dK/dV) -- the same shape a packed training
    forward hits, keyed with the trailing `packed` component True -- so a subsequently
    compiled packed step traces warm. One `segments=`-carrying forward suffices: all three
    `calibrated_*_rate(packed=True)` probes run in `flash_attention`'s Python body. Isolated by
    popping the packed keys first, so this proves ENABLE warms them."""
    model = _tiny_llama_hd64()
    attn = model.model.layers[0].self_attn
    hq = attn.n_heads
    dtype = model.model.norm.weight.dtype
    n, b = 32, 1

    fwd_tile = select_fwd_tile(n, _HEAD_DIM)
    dq_tile, dkv_tile = select_bwd_tiles(n, _HEAD_DIM)
    nb = _n_bucket(n)
    # trailing `packed` component True -> the packed-keyed rate slots.
    fkey = (_HEAD_DIM, str(dtype), True, b, hq, nb, fwd_tile.variant, fwd_tile.d_slab, True)
    dqkey = (_HEAD_DIM, str(dtype), True, b, hq, nb, dq_tile.variant, dq_tile.d_slab, True)
    dkvkey = (_HEAD_DIM, str(dtype), True, b, hq, nb, dkv_tile.variant, dkv_tile.d_slab, True)
    for cache, key in (
        (_FWD_RATE_CACHE, fkey), (_BWD_DQ_RATE_CACHE, dqkey), (_BWD_DKV_RATE_CACHE, dkvkey)
    ):
        cache.pop(key, None)

    enable_flash_attention(model, impl="kernel", seq_len=n, batch_size=b, packed=True)

    assert fkey in _FWD_RATE_CACHE, "packed enable did not pre-warm the packed forward rate"
    assert dqkey in _BWD_DQ_RATE_CACHE, "packed enable did not pre-warm the packed dQ rate"
    assert dkvkey in _BWD_DKV_RATE_CACHE, "packed enable did not pre-warm the packed dK/dV rate"


def test_enable_packed_without_seq_len_succeeds() -> None:
    """`packed=True` with NO `seq_len` hint enables without error -- eager / `mx.grad` callers
    calibrate the packed rates lazily and harmlessly on the first packed attention call (the
    documented lazy-calibration path; the compiled-train contract only binds when a hint is
    given)."""
    model = _tiny_llama_hd64()
    enable_flash_attention(model, impl="kernel", packed=True)  # no seq_len -> no pre-warm
    assert isinstance(model.model.layers[0].self_attn, FlashAttentionWrapper)


# ---------------------------------------------------------------------------
# 5. LoRA attaches to the wrapped projections and they receive gradients
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", list(_FAMILIES))
def test_lora_attaches_to_wrapped_projections(family: str) -> None:
    """`linear_to_lora_layers` discovers `self_attn.q_proj` INSIDE the wrapper (by module-tree
    path) and replaces it, and the wrapper's attribute-lookup `__call__` runs the injected
    adapter -- proven by a nonzero grad on the LoRA `lora_b` after one step (base q_proj.weight
    is frozen; lora_a's grad is zero at step 1 by LoRA's zero-init of lora_b, a real property,
    so the discriminating signal is lora_b). Parametrized across families so qwen2's
    bias-carrying q_proj (llama/qwen3 have none) flows through LoRA attach + backward -- the
    one integration path qwen2 does not share with the llama representative."""
    model = _FAMILIES[family]()
    enable_flash_attention(model, impl="reference")
    mx.random.seed(0)
    model.freeze()
    linear_to_lora_layers(model, -1, {"rank": 4, "dropout": 0.0, "scale": 20.0})
    mx.eval(model.parameters())

    q_proj = model.model.layers[0].self_attn.q_proj
    assert isinstance(q_proj, LoRALinear), "LoRA did not attach to the wrapped q_proj"

    ids = _ids(model.args.vocab_size, 1, 8)

    def loss(m: nn.Module, x: mx.array) -> mx.array:
        return m(x).sum()

    grads = nn.value_and_grad(model, loss)(model, ids)[1]
    flat = dict(tree_flatten(grads))
    lora_b_keys = [k for k in flat if "self_attn.q_proj.lora_b" in k]
    lora_a_keys = [k for k in flat if "self_attn.q_proj.lora_a" in k]
    assert lora_b_keys, f"no q_proj LoRA lora_b grad in tree: {sorted(flat)}"
    assert lora_a_keys, f"no q_proj LoRA lora_a grad in tree: {sorted(flat)}"
    worst_b = max(mx.abs(flat[k]).max().item() for k in lora_b_keys)
    assert worst_b > 0.0, "wrapped q_proj LoRA adapter received no gradient (forward skipped it)"


# ---------------------------------------------------------------------------
# 6. coexistence with the 0.1.0 make_loss_fn CE adapter (spec §5)
# ---------------------------------------------------------------------------


def test_wrapper_composes_with_make_loss_fn() -> None:
    """Enabled attention wrapper + the 0.1.0 linear-CE loss adapter in one step (both pure-MLX
    paths here: attention impl='reference', CE impl='chunked')."""
    model = _tiny_llama_hd64()
    enable_flash_attention(model, impl="reference")
    loss_fn = make_loss_fn(model, impl="chunked")

    seq_len = 8
    batch = _ids(model.args.vocab_size, 1, seq_len + 1)
    lengths = mx.array([[0, seq_len]])
    loss, ntoks = loss_fn(model, batch, lengths)
    mx.eval(loss, ntoks)
    assert math.isfinite(loss.item())
    assert ntoks.item() > 0


# ---------------------------------------------------------------------------
# 7. mlx-lm attention-surface drift pin (spec §5 drift lane)
# ---------------------------------------------------------------------------


def test_mlx_lm_attention_shape_drift_pin() -> None:
    """Pins the installed mlx-lm attention-module surface the wrapper depends on -- so an
    mlx-lm refactor fails loudly HERE, not inside a training run."""
    for build in (_tiny_llama_hd64, _tiny_qwen2_hd64, _tiny_qwen3_hd64):
        attn = build().model.layers[0].self_attn
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "rope"):
            assert hasattr(attn, name), f"{type(attn).__module__} attn missing {name}"
        assert hasattr(attn, "n_heads")
        assert hasattr(attn, "n_kv_heads")
        assert isinstance(attn.scale, float)

    # qwen3 adds per-head q/k RMSNorm; llama and qwen2 do not (both take the wrapper's
    # `_has_qk_norm == False` branch).
    assert hasattr(_tiny_qwen3_hd64().model.layers[0].self_attn, "q_norm")
    assert not hasattr(_tiny_llama_hd64().model.layers[0].self_attn, "q_norm")
    assert not hasattr(_tiny_qwen2_hd64().model.layers[0].self_attn, "q_norm")

    # __call__ arity: (self, x, mask, cache) in all three families.
    for attn_cls in (llama.Attention, qwen2.Attention, qwen3.Attention):
        params = list(inspect.signature(attn_cls.__call__).parameters)
        assert params == ["self", "x", "mask", "cache"], (attn_cls, params)

    # The mask sentinels the wrapper's call-time guard maps (T12 review carry-forward):
    # the training path (N>1, no cache) hands the wrapper the STRING "causal"; N==1
    # returns None. If a future mlx-lm changes either, the wrapper would fail at
    # TRAINING time (AttentionInputError) -- this pin fails the drift lane instead.
    assert create_attention_mask(mx.zeros((1, 4, 8)), None) == "causal"
    assert create_attention_mask(mx.zeros((1, 1, 8)), None) is None


# ---------------------------------------------------------------------------
# 8. gc=True compiled train() -- SUBPROCESS-ISOLATED (gotcha 13)
# ---------------------------------------------------------------------------


@pytest.mark.metal
def test_wrapped_model_trains_under_compiled_train_gc_true() -> None:
    """One real mlx_lm `train()` (grad_checkpoint=True, kernel impl, 2 iters) on an enabled
    tiny llama, run in a CHILD process. mlx_lm's grad_checkpoint patches
    `type(layer).__call__` at the CLASS level and never reverts (gotcha 13), so this gc=True
    site must be subprocess-isolated -- the other two in-process gc=True sites are
    `tests/test_worker_train_step.py` and `tests/_composition_gc_child.py`. Verdict on the
    child's stdout."""
    proc = subprocess.run(
        [sys.executable, str(_GC_CHILD)], capture_output=True, text=True, timeout=600,
        check=False,
    )
    assert proc.returncode == 0, f"child failed:\n{proc.stdout}\n{proc.stderr}"
    assert "WRAPPER_GC_OK" in proc.stdout, proc.stdout
    losses = _parse_losses(proc.stdout)
    assert len(losses) == 2, losses
    assert all(math.isfinite(x) for x in losses), losses
    assert losses[1] <= losses[0] * 2.0, f"loss not decreasing-or-close: {losses}"


# ---------------------------------------------------------------------------
# 9. loss-curve differential regression -- the cheap standing proxy for acceptance 3
# ---------------------------------------------------------------------------


@pytest.mark.metal
def test_loss_curve_matches_stock_on_tiny_model() -> None:
    """SAME tiny model / data / seed trained through stock SDPA vs the flash kernel
    (impl='kernel'), gc=False, per-step loss trajectories match within a measured pin. The
    committed standing proxy for acceptance 3 (loss-curve parity) that gates every PR, not
    just T13's one campaign. Measured (mlx 0.32.0, fp32, tiny, 2 steps): step 1 bit-identical,
    worst per-step |diff| 4.768e-07 -> pin 2e-6 (~4.2x; kernel fp32-accumulate vs mlx SDPA
    through 2 optimizer steps)."""
    seq_len, batch, steps = 16, 1, 2
    stock = _tiny_lora_train_losses(
        enable_flash=False, seq_len=seq_len, batch=batch, steps=steps
    )
    ours = _tiny_lora_train_losses(
        enable_flash=True, seq_len=seq_len, batch=batch, steps=steps
    )
    assert len(stock) == len(ours) == steps
    worst = max(abs(a - b) for a, b in zip(stock, ours, strict=True))
    assert worst < 2e-6, (
        f"loss-curve worst per-step |diff| {worst:.3e} exceeds 2e-6: {stock} vs {ours}"
    )


def _tiny_lora_train_losses(
    *, enable_flash: bool, seq_len: int, batch: int, steps: int
) -> list[float]:
    mx.random.seed(0)  # identical weights across both arms (build draws RNG)
    model = _tiny_llama_hd64()
    if enable_flash:
        # pre-warm at the training shape so the compiled train() step traces warm caches;
        # calibration uses LOCAL RNG keys, so it does not disturb the seeded weight/LoRA init.
        enable_flash_attention(model, impl="kernel", seq_len=seq_len, batch_size=batch)
    mx.random.seed(100)  # identical LoRA init across both arms
    model.freeze()
    linear_to_lora_layers(model, -1, {"rank": 4, "dropout": 0.0, "scale": 20.0})
    mx.eval(model.parameters())

    examples = _synthetic_train_examples(
        vocab_size=model.args.vocab_size, seq_len=seq_len, num_examples=batch, seed=0
    )
    reports = _run_train_steps(
        model, optim.Adam(learning_rate=1e-4), default_loss, examples,
        batch=batch, seq_len=seq_len, steps=steps, grad_checkpoint=False,
    )
    return [float(r["train_loss"]) for r in reports]


def _parse_losses(stdout: str) -> list[float]:
    marker = "WRAPPER_GC_OK losses="
    line = next(ln for ln in stdout.splitlines() if marker in ln)
    payload: Any = line.split(marker, 1)[1].strip()
    return [float(x) for x in payload.strip("[]").split(",") if x.strip()]
