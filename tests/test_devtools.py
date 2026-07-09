import builtins

import mlx.core as mx
import pytest

from mlx_train_perf.attention.kernel.source import build_fwd_mma_source
from mlx_train_perf.core.kernel.source import (
    QUANT_HELPERS,
    build_backward_dhidden_mma_source,
    build_backward_dhidden_source,
    build_backward_dw_source,
    build_dense_source,
    build_quant_source,
)
from mlx_train_perf.devtools.regpressure import (
    _prepare_msl,
    _strip_banner_and_fences,
    compiled_ceiling,
)
from mlx_train_perf.errors import MissingDependencyError

# ---------------------------------------------------------------------------
# Pure text-transform helpers: GPU-free, pyobjc-free -- always run in the default lane.
# ---------------------------------------------------------------------------


def test_strip_banner_and_fences_removes_banner_and_markdown_fences() -> None:
    raw = "Generated source code for `mtp_probe`:\n```\nline one\nline two\n```\n"
    assert _strip_banner_and_fences(raw) == "line one\nline two"


def test_strip_banner_and_fences_is_noop_without_a_banner_line() -> None:
    raw = "```\nno banner here\n```"
    assert _strip_banner_and_fences(raw) == "no banner here"


def test_strip_banner_and_fences_only_strips_bare_fence_lines() -> None:
    # A ``` occurring mid-line (not its own line) must NOT be stripped -- only lines whose
    # stripped content is exactly the fence marker are capture noise.
    raw = "Generated source code for `mtp_probe`:\n```\nuint x = 1; // ```not a fence```\n```\n"
    assert _strip_banner_and_fences(raw) == "uint x = 1; // ```not a fence```"


def test_prepare_msl_prepends_the_minimal_jit_prelude() -> None:
    raw = "Generated source code for `mtp_probe`:\n```\nbody text\n```\n"
    result = _prepare_msl(raw)
    assert result.startswith("#include <metal_stdlib>\n")
    assert "using namespace metal;" in result
    assert "typedef bfloat bfloat16_t;" in result
    assert result.endswith("body text")


# ---------------------------------------------------------------------------
# compiled_ceiling: the public probe entry point
# ---------------------------------------------------------------------------


def test_missing_pyobjc_is_precise_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real = builtins.__import__

    def fake(name: str, *a: object, **k: object) -> object:
        if name == "Metal":
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(MissingDependencyError) as ei:
        compiled_ceiling("out[0] = 1.0f;")
    assert "probe" in str(ei.value)


def test_input_names_and_inputs_must_be_supplied_together() -> None:
    # Both are pure validation failures -- neither reaches the `import Metal` guard, so
    # this stays in the default lane (no GPU/pyobjc needed to observe the ValueError).
    with pytest.raises(ValueError, match="supplied together"):
        compiled_ceiling("out[0] = 1.0f;", input_names=["a"])
    with pytest.raises(ValueError, match="supplied together"):
        compiled_ceiling("out[0] = 1.0f;", inputs=[mx.zeros((1,))])


def test_output_contract_args_must_be_supplied_together() -> None:
    # Same validated-triple pattern as input_names/inputs -- a partial output override is
    # ambiguous (which of the 3 pieces the caller forgot is not recoverable).
    with pytest.raises(ValueError, match="supplied together"):
        compiled_ceiling("out[0] = 1.0f;", output_names=["out"])
    with pytest.raises(ValueError, match="supplied together"):
        compiled_ceiling("out[0] = 1.0f;", output_shapes=[(1,)])
    with pytest.raises(ValueError, match="supplied together"):
        compiled_ceiling("out[0] = 1.0f;", output_dtypes=[mx.float32])


@pytest.mark.metal
def test_dense_v2e_ceiling_matches_spike_measurement() -> None:
    # spike-measured compiled ceilings: v2e (RT=4) -> 448, v2d (RT=2) -> 640
    # NOTE: compiled_ceiling wraps a bare function shell around the body; reconcile the
    # wrapper so shapes/template refs resolve (port the spike script's shell verbatim).
    assert compiled_ceiling(build_dense_source(4)) == 448
    assert compiled_ceiling(build_dense_source(2)) == 640


@pytest.mark.metal
def test_quantized_kernel_ceiling_is_plausible() -> None:
    # The quantized kernel has a DIFFERENT 8-input contract (hidden, wq, sc, bi, targets,
    # offs, lse_in, tgt_in) than the dense default (6 inputs, no wq/sc/bi) -- this is the
    # capability the parameterized `input_names`/`inputs` args exist for. No number from
    # this repo's measured history is hard-pinned here: that history was never measured
    # through this exact probe path, only through the earlier (different kernel family)
    # E3 experiment. The observed value is logged for the record.
    n, d, v = 8, 64, 16  # d must be a group_size(64) multiple for a valid quantized probe
    mx.random.seed(1)
    hidden = mx.random.normal((n, d)).astype(mx.bfloat16)
    w = (mx.random.normal((v, d)) * 0.05).astype(mx.bfloat16)
    wq, sc, bi = mx.quantize(w, group_size=64, bits=4)
    targets = mx.random.randint(0, v, (n,))
    offs = mx.array([0, v], dtype=mx.uint32)
    lse = mx.full((n,), float("-inf"), dtype=mx.float32)
    tgt = mx.zeros((n,), dtype=mx.float32)
    mx.eval(hidden, wq, sc, bi, targets, offs, lse, tgt)
    ceiling = compiled_ceiling(
        build_quant_source(4),
        header=QUANT_HELPERS,
        input_names=["hidden", "wq", "sc", "bi", "targets", "offs", "lse_in", "tgt_in"],
        inputs=[hidden, wq, sc, bi, targets, offs, lse, tgt],
    )
    print(f"quantized RT=4 compiled ceiling (observed): {ceiling}")
    assert 0 < ceiling <= 1024


@pytest.mark.metal
def test_backward_dhidden_kernel_ceiling_is_plausible() -> None:
    # The v0-correct backward kernel has a DIFFERENT contract on BOTH sides: inputs add
    # lse/cotangent/d_hidden_in (fixed fp32, not templated on T), and the OUTPUT is a
    # single (rows, d) fp32 d_hidden_out -- not the dense forward's (lse_out, tgt_out)
    # pair. This is the capability the parameterized output_names/output_shapes/
    # output_dtypes args exist for (same precedent as the quantized input contract above).
    n, d, v = 8, 32, 16
    mx.random.seed(2)
    hidden = mx.random.normal((n, d)).astype(mx.bfloat16)
    w = (mx.random.normal((v, d)) * 0.05).astype(mx.bfloat16)
    targets = mx.random.randint(0, v, (n,))
    offs = mx.array([0, v], dtype=mx.uint32)
    lse = mx.full((n,), float("-inf"), dtype=mx.float32)
    cotangent = mx.full((n,), 1.0 / n, dtype=mx.float32)
    d_hidden_in = mx.zeros((n, d), dtype=mx.float32)
    mx.eval(hidden, w, targets, offs, lse, cotangent, d_hidden_in)
    ceiling = compiled_ceiling(
        build_backward_dhidden_source(4),
        input_names=["hidden", "w", "targets", "offs", "lse", "cotangent", "d_hidden_in"],
        inputs=[hidden, w, targets, offs, lse, cotangent, d_hidden_in],
        output_names=["d_hidden_out"],
        output_shapes=[(n, d)],
        output_dtypes=[mx.float32],
    )
    print(f"backward d_hidden RT=4 compiled ceiling (observed): {ceiling}")
    assert 0 < ceiling <= 1024


@pytest.mark.metal
def test_backward_dw_kernel_ceiling_is_plausible() -> None:
    # The d_w kernel's output is ATOMIC-typed (`device atomic<float>*`, not plain
    # `device float*`) -- this is the capability `atomic_outputs=True` exists for. Same
    # (hidden, w, targets, offs, lse, cotangent) input contract as d_hidden minus
    # d_hidden_in (d_w needs no cross-tile accumulator chain -- see source.py's derivation
    # comment), and the output is a single (v, d) fp32 d_w_out.
    n, d, v = 8, 32, 16
    mx.random.seed(2)
    hidden = mx.random.normal((n, d)).astype(mx.bfloat16)
    w = (mx.random.normal((v, d)) * 0.05).astype(mx.bfloat16)
    targets = mx.random.randint(0, v, (n,))
    offs = mx.array([0, v], dtype=mx.uint32)
    lse = mx.full((n,), float("-inf"), dtype=mx.float32)
    cotangent = mx.full((n,), 1.0 / n, dtype=mx.float32)
    mx.eval(hidden, w, targets, offs, lse, cotangent)
    ceiling = compiled_ceiling(
        build_backward_dw_source(4),
        input_names=["hidden", "w", "targets", "offs", "lse", "cotangent"],
        inputs=[hidden, w, targets, offs, lse, cotangent],
        output_names=["d_w_out"],
        output_shapes=[(v, d)],
        output_dtypes=[mx.float32],
        atomic_outputs=True,
    )
    print(f"backward d_w RT=4 compiled ceiling (observed): {ceiling}")
    assert 0 < ceiling <= 1024


@pytest.mark.metal
def test_backward_dhidden_mma_kernel_ceiling_is_plausible() -> None:
    # The d_hidden MMA kernel (Task 16b step 4) shares the v0 backward's exact contract --
    # inputs (hidden, w, targets, offs, lse, cotangent, d_hidden_in) and a single (rows, d)
    # fp32 d_hidden_out -- so it probes through the same parameterized path. Measured 448 at
    # RT=4 -- the forward's own ceiling, NO drop (reusing the C tiles in place for the
    # gradient coefficients keeps per-lane register state flat). The number is a register-
    # pressure telltale only, never a rate verdict (see the module docstring) -- logged here.
    n, d, v = 8, 32, 16
    mx.random.seed(2)
    hidden = mx.random.normal((n, d)).astype(mx.bfloat16)
    w = (mx.random.normal((v, d)) * 0.05).astype(mx.bfloat16)
    targets = mx.random.randint(0, v, (n,))
    offs = mx.array([0, v], dtype=mx.uint32)
    lse = mx.full((n,), float("-inf"), dtype=mx.float32)
    cotangent = mx.full((n,), 1.0 / n, dtype=mx.float32)
    d_hidden_in = mx.zeros((n, d), dtype=mx.float32)
    mx.eval(hidden, w, targets, offs, lse, cotangent, d_hidden_in)
    ceiling = compiled_ceiling(
        build_backward_dhidden_mma_source(4),
        input_names=["hidden", "w", "targets", "offs", "lse", "cotangent", "d_hidden_in"],
        inputs=[hidden, w, targets, offs, lse, cotangent, d_hidden_in],
        output_names=["d_hidden_out"],
        output_shapes=[(n, d)],
        output_dtypes=[mx.float32],
    )
    print(f"backward d_hidden MMA RT=4 compiled ceiling (observed): {ceiling}")
    assert 0 < ceiling <= 1024


@pytest.mark.metal
@pytest.mark.parametrize(
    ("head_dim", "d_slab"),
    [(64, None), (96, None), (128, None), (128, 128)],
    ids=["hd64-default", "hd96-default", "hd128-default", "hd128-SHIPPED-slab128"],
)
def test_flash_fwd_mma_ceiling_stays_in_the_mma_class(head_dim: int, d_slab: int | None) -> None:
    # The 4x4 simdgroup-matrix flash-attention forward (rung 2, register-resident P@V O-path
    # with D-slabbing) shares the v0 scalar body's (q, k, v, qoffs, scale_in) -> (o_out, l_out)
    # contract, so it probes through the same parameterized path. `None` = the builder's
    # register-safe DEFAULT slab (32); the (128, 128) case is the FLAGSHIP config the rung-3
    # dispatch table actually SHIPS at the measured saturation bucket (single-pass full-D C_o)
    # -- the shipped kernel's own pressure telltale must be pinned, not just the default's.
    # Rung 2 removed ALL threadgroup memory (register C_o accumulator + simd_shuffle softmax);
    # live per-lane state is the 16 QK/P tiles (32 fp32/lane) plus the RT*(D_SLAB/8) C_o tiles.
    # MEASURED (mlx 0.32.0, M1 Max): a regpressure sweep over slab widths {16,32,48,64,head_dim}
    # read a FLAT 384 at EVERY width AND head dim -- maxTotalThreadsPerThreadgroup cannot
    # distinguish "fits" from "spilled-to-fit" (user-metal-kernels), so the ceiling is
    # NON-DISCRIMINATING here and the d_slab choice comes from the SATURATION RATE SWEEP
    # (rung 2b artifacts: slab128 = 1462.7 G, +57% over the default 32 -- the spill arithmetic
    # over-predicted). The value is a register-pressure telltale only, never a rate verdict
    # (module docstring); the rung contract's bar is "restructure if a config collapses below
    # ~256", and the measured 384 clears it, so pin >= 256.
    b, hq, hkv, n = 1, 8, 8, 8
    scale = 1.0 / (head_dim ** 0.5)
    mx.random.seed(2)
    q = mx.random.normal((b, hq, n, head_dim)).astype(mx.bfloat16)
    k = mx.random.normal((b, hkv, n, head_dim)).astype(mx.bfloat16)
    v = mx.random.normal((b, hkv, n, head_dim)).astype(mx.bfloat16)
    qoffs = mx.array([0, n], dtype=mx.uint32)
    scale_in = mx.array([scale], dtype=mx.float32)
    mx.eval(q, k, v, qoffs, scale_in)
    kwargs = {} if d_slab is None else {"d_slab": d_slab}
    ceiling = compiled_ceiling(
        build_fwd_mma_source(head_dim, causal=True, **kwargs),
        input_names=["q", "k", "v", "qoffs", "scale_in"],
        inputs=[q, k, v, qoffs, scale_in],
        output_names=["o_out", "l_out"],
        output_shapes=[(b, hq, n, head_dim), (b, hq, n)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )
    print(f"flash fwd MMA head_dim={head_dim} d_slab={d_slab} ceiling (observed): {ceiling}")
    assert 256 <= ceiling <= 1024
