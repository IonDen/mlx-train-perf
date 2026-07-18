"""Gate B (spec §9c / §9f) -- compile-behavior GO/NO-GO probes for the packed attention path.

Four probes, all `@metal`, against a tiny 2-layer wrapped llama (`_tiny_llama_hd64`,
head_dim=64) enabled with `packed=True` so the rate caches are pre-warmed:

1. **Single trace across differing layouts** -- a compiled step over
   `(batch, seg_id, seg_start, loss_mask)` traced ONCE across two batches with DIFFERENT
   natural segment layouts (same shapes); a Python-side counter inside the compiled fn
   increments only during trace, so it lands at 1.
2. **No constant capture** -- the two layouts produce DIFFERENT losses, each matching its own
   eager computation (the `seg_id`/`seg_start` primal threading, not a frozen first layout).
3. **Zero host-sync** -- the three packed-keyed rate caches gain NO new keys during the
   compiled call (pre-warm keyed them; an in-trace calibration would insert one).
4. **gc peak parity (§9f)** -- the manual packed block-walk's grad-checkpointed peak ≈ the
   stock causal-walk peak, both under a SINGLE class-level `grad_checkpoint` patch (proves the
   patch fires for direct block calls).

This is the Gate-B rehearsal of the Task-10 adapter's manual block walk: the walk
(embed → blocks(h, PackedMask, None) → norm → a simple masked loss) is hand-rolled inside the
tests because the adapter does not exist yet.
"""
import mlx.core as mx
import pytest
from mlx.utils import tree_map

pytest.importorskip("mlx_lm")

from mlx_lm.tuner.trainer import grad_checkpoint

# Reuse the tiny head_dim=64 builder + packed-layout / id helpers from the sibling wrapper
# tests (same idioms; DRY -- test_attention_wrapper.py itself imports from test_worker_train_step).
from test_attention_wrapper import _ids, _packed_layout, _tiny_llama_hd64

from mlx_train_perf.attention.kernel.launch import (
    _BWD_DKV_RATE_CACHE,
    _BWD_DQ_RATE_CACHE,
    _FWD_RATE_CACHE,
)
from mlx_train_perf.attention.segments import PackedMask
from mlx_train_perf.attention.wrapper import enable_flash_attention

_L = 32  # probes 1-3: small context, fast
_B = 1


def _masked_lse_loss(model: object, h: mx.array, loss_mask: mx.array) -> mx.array:
    """A simple masked scalar loss over the trunk output: masked mean of the row logsumexp of
    the lm-head logits. Content-sensitive (unlike a norm-squared, which RMSNorm flattens), so
    two segment layouts yield measurably different losses -- the signal probe 2 needs."""
    logits = model.lm_head(h)  # type: ignore[attr-defined]
    lse = mx.logsumexp(logits, axis=-1)
    return (lse * loss_mask).sum() / mx.maximum(loss_mask.sum(), mx.array(1.0))


def _packed_walk_scalar(
    model: object, batch: mx.array, seg_id: mx.array, seg_start: mx.array, loss_mask: mx.array
) -> mx.array:
    """The Task-10 pattern, hand-rolled: embed → every block with a `PackedMask(seg_id,
    seg_start)` mask → final norm → masked loss. `seg_id`/`seg_start` reach `flash_attention`
    as custom_function primals (threaded, never captured)."""
    inner = model.model  # type: ignore[attr-defined]
    h = inner.embed_tokens(batch)
    pm = PackedMask(seg_id=seg_id, seg_start=seg_start)
    for layer in inner.layers:
        h = layer(h, pm, None)
    return _masked_lse_loss(model, inner.norm(h), loss_mask)


# ---------------------------------------------------------------------------
# Probe 1 -- one trace across two different segment layouts (same shapes)
# ---------------------------------------------------------------------------


@pytest.mark.metal
def test_compiled_packed_step_traces_once_across_differing_layouts() -> None:
    """§9c(i): fixed `(B, L)` buffers make the packed step shape-stable, so two batches with
    DIFFERENT natural segment layouts hit the SAME compiled trace. A Python-side counter inside
    the compiled fn increments only while tracing (mx.compile replays -- never re-runs the
    Python body -- on a shape-matching call), so it lands at 1 after both calls."""
    model = _tiny_llama_hd64()
    enable_flash_attention(model, impl="kernel", seq_len=_L, batch_size=_B, packed=True)
    mx.eval(model.parameters())

    ids = _ids(model.args.vocab_size, _B, _L, seed=1)
    loss_mask = mx.ones((_B, _L))
    seg_a = _packed_layout([_L // 2, _L - _L // 2], b=_B)  # [16, 16]
    seg_b = _packed_layout([5, _L - 5], b=_B)              # [5, 27]

    traces: list[int] = []

    def step(batch: mx.array, seg_id: mx.array, seg_start: mx.array, mask: mx.array) -> mx.array:
        traces.append(1)  # Python side effect: runs during trace, not on replay
        return _packed_walk_scalar(model, batch, seg_id, seg_start, mask)

    compiled = mx.compile(step)
    l1 = compiled(ids, seg_a[0], seg_a[1], loss_mask)
    l2 = compiled(ids, seg_b[0], seg_b[1], loss_mask)
    mx.eval(l1, l2)

    assert len(traces) == 1, f"expected a single trace across the two layouts, got {len(traces)}"


# ---------------------------------------------------------------------------
# Probe 2 -- no constant capture (layout is threaded, not frozen)
# ---------------------------------------------------------------------------


@pytest.mark.metal
def test_compiled_packed_step_reflects_each_layout_no_constant_capture() -> None:
    """§9c(ii): the two layouts produce DIFFERENT losses (same tokens -- the ONLY variable is
    the segment masking), and each compiled loss matches its own EAGER computation. If
    `seg_id`/`seg_start` were closure-captured (frozen at the first trace), the second compiled
    call would silently reuse the first layout and diverge from its eager value. Measured (mlx
    0.32.0, tiny hd64, fp32, L=32): |l1 - l2| = 6.30e-3 (layout genuinely moves the loss);
    compiled vs eager = 0.0 (bit-identical) each. Pins: >1e-3 apart, <1e-4 vs eager."""
    model = _tiny_llama_hd64()
    enable_flash_attention(model, impl="kernel", seq_len=_L, batch_size=_B, packed=True)
    mx.eval(model.parameters())

    ids = _ids(model.args.vocab_size, _B, _L, seed=1)
    loss_mask = mx.ones((_B, _L))
    seg_a = _packed_layout([_L // 2, _L - _L // 2], b=_B)  # [16, 16]
    seg_b = _packed_layout([5, _L - 5], b=_B)              # [5, 27]

    def step(batch: mx.array, seg_id: mx.array, seg_start: mx.array, mask: mx.array) -> mx.array:
        return _packed_walk_scalar(model, batch, seg_id, seg_start, mask)

    compiled = mx.compile(step)  # `model` is captured, NOT a traced input
    l1 = compiled(ids, seg_a[0], seg_a[1], loss_mask)
    l2 = compiled(ids, seg_b[0], seg_b[1], loss_mask)
    e1 = _packed_walk_scalar(model, ids, seg_a[0], seg_a[1], loss_mask)
    e2 = _packed_walk_scalar(model, ids, seg_b[0], seg_b[1], loss_mask)
    mx.eval(l1, l2, e1, e2)

    assert abs(l1.item() - l2.item()) > 1e-3, (
        f"the two layouts produced near-identical losses ({l1.item()} vs {l2.item()}) -- the "
        "segment masking did not affect the output"
    )
    assert abs(l1.item() - e1.item()) < 1e-4, f"layout A: compiled {l1.item()} != eager {e1.item()}"
    assert abs(l2.item() - e2.item()) < 1e-4, (
        f"layout B: compiled {l2.item()} != eager {e2.item()} -- the second call reused a "
        "captured first layout (seg_id/seg_start frozen into the trace)"
    )


# ---------------------------------------------------------------------------
# Probe 3 -- zero host-sync: no new packed-keyed rate-cache keys during the compiled call
# ---------------------------------------------------------------------------


@pytest.mark.metal
def test_compiled_packed_step_adds_no_rate_cache_keys() -> None:
    """§9c(iii): `enable_flash_attention(packed=True, seq_len, batch_size)` pre-warms the three
    packed-keyed rate caches, so a compiled packed forward finds every rate already cached and
    runs NO in-trace calibration -- which would host-sync (`mx.eval` of the detached probe) AND
    insert a key. Asserting the three cache key sets are unchanged across the compiled call is
    the real signal (per the T6 measurement a cold cache calibrates in-trace rather than
    raising, so crash-absence alone would not prove it)."""
    model = _tiny_llama_hd64()
    enable_flash_attention(model, impl="kernel", seq_len=_L, batch_size=_B, packed=True)
    mx.eval(model.parameters())

    ids = _ids(model.args.vocab_size, _B, _L, seed=2)
    loss_mask = mx.ones((_B, _L))
    seg_id, seg_start = _packed_layout([_L // 2, _L - _L // 2], b=_B)

    caches = (_FWD_RATE_CACHE, _BWD_DQ_RATE_CACHE, _BWD_DKV_RATE_CACHE)
    before = tuple(set(c) for c in caches)

    def step(batch: mx.array, sid: mx.array, sst: mx.array, mask: mx.array) -> mx.array:
        return _packed_walk_scalar(model, batch, sid, sst, mask)

    compiled = mx.compile(step)  # `model` is captured, NOT a traced input
    out = compiled(ids, seg_id, seg_start, loss_mask)
    mx.eval(out)

    after = tuple(set(c) for c in caches)
    added = tuple(len(a - b) for a, b in zip(after, before, strict=True))
    assert after == before, (
        "the compiled packed step added rate-cache keys -> an in-trace calibration host-synced "
        f"(pre-warm missed the shape): fwd +{added[0]}, dq +{added[1]}, dkv +{added[2]}"
    )


# ---------------------------------------------------------------------------
# Probe 4 -- gc peak parity: manual packed walk ≈ stock causal walk (§9f)
# ---------------------------------------------------------------------------


@pytest.mark.metal
def test_gc_peak_parity_manual_walk_matches_stock_walk() -> None:
    """§9f: with gradient checkpointing, the manual PACKED block-walk peak ≈ the stock CAUSAL
    block-walk peak -- proving the class-level `grad_checkpoint` patch fires for direct block
    calls (the Task-10 walk gets checkpointing for free).

    METHODOLOGY (all mandatory, this is a GATE):
    - `grad_checkpoint(block)` is applied EXACTLY ONCE and both arms are measured under that one
      class-level patch (gotcha 13: the patch mutates `type(block).__call__` and never reverts;
      re-invoking it per arm nests checkpoints and corrupts the peak). Restored in `finally`.
    - `mx.eval(model.parameters())` before both windows (gotcha 14). `grad_checkpoint`'s
      `checkpointed_fn` does `model.update(traced params)` INSIDE the compiled grad, leaving
      `model.parameters()` holding traced arrays afterward -- so concrete params are snapshotted
      once and restored before each arm.
    - Warmup output stays alive through the measured window; `mx.synchronize()` before every
      `clear_cache()`/snapshot (gotcha 15). Peaks are allocation-deterministic (reproducible to
      <0.1% across runs).

    MEASURED (mlx 0.32.0, tiny 2-layer hd64, B=1, L=2048, fp32, grad wrt input embeddings):
    walk 118.5 MB / stock 119.8 MB → ratio 0.988 (the small deficit is the packed 5-primal path
    vs the causal 3-primal path; on this tiny model checkpointing costs a little peak rather than
    saving it, so "loses checkpointing" reads as a DROP, not a spike). Failure modes measured on
    this exact fixture: a walk that SILENTLY LOSES checkpointing (unpatched block) → 0.91; a
    NESTED checkpoint (grad_checkpoint applied twice) → 1.25. The [0.95, 1.05] band passes 0.988
    and rejects both (its lower edge 0.95 is the midpoint of 0.988 and 0.91)."""
    length, batch_size = 2048, 1
    model = _tiny_llama_hd64()
    enable_flash_attention(model, impl="kernel", seq_len=length, batch_size=batch_size,
                           packed=True)
    ids = _ids(model.args.vocab_size, batch_size, length, seed=1)
    seg_id, seg_start = _packed_layout(
        [length // 3, length // 3, length - 2 * (length // 3)], b=batch_size
    )
    packed = PackedMask(seg_id=seg_id, seg_start=seg_start)
    loss_mask = mx.ones((batch_size, length))
    emb = model.model.embed_tokens(ids)  # grad target: a float input flowing through the blocks
    mx.eval(emb)

    def walk(emb_: mx.array) -> mx.array:
        h = emb_
        for layer in model.model.layers:
            h = layer(h, packed, None)  # PackedMask → flash_attention(segments=)
        return _masked_lse_loss(model, model.model.norm(h), loss_mask)

    def stock(emb_: mx.array) -> mx.array:
        h = emb_
        for layer in model.model.layers:
            h = layer(h, "causal", None)  # the stock mask → flash_attention(segments=None)
        return _masked_lse_loss(model, model.model.norm(h), loss_mask)

    block = model.model.layers[0]
    original_call = type(block).__call__
    grad_checkpoint(block)  # applied ONCE; both arms measured under this single class patch
    mx.eval(model.parameters())
    saved = tree_map(mx.array,model.parameters())

    def peak_of(fn: object) -> int:
        model.update(tree_map(mx.array,saved))  # restore concrete params
        mx.eval(model.parameters())
        grad_fn = mx.compile(mx.grad(fn))  # type: ignore[arg-type]
        warm = grad_fn(emb)
        mx.eval(warm)              # warmup/trace OUTSIDE the measured window
        mx.synchronize()
        mx.clear_cache()
        mx.reset_peak_memory()
        out = grad_fn(emb)
        mx.eval(out)
        mx.synchronize()
        peak = int(mx.get_peak_memory())
        del warm, out             # released only after the snapshot
        return peak

    try:
        walk_peak = peak_of(walk)
        stock_peak = peak_of(stock)
    finally:
        type(block).__call__ = original_call  # never leave the class patched (gotcha 13)

    ratio = walk_peak / stock_peak
    assert 0.95 <= ratio <= 1.05, (
        f"gc peak parity broken: manual packed walk {walk_peak} vs stock causal walk "
        f"{stock_peak} (ratio {ratio:.4f}) outside [0.95, 1.05] -- the packed walk lost "
        "checkpointing (ratio drops) or a nested-checkpoint corruption crept in (ratio spikes)"
    )
