import mlx.core as mx
import pytest

from mlx_train_perf.core.guards import clamped_caps
from mlx_train_perf.errors import PlanInputError
from mlx_train_perf.plan.calibration import load_calibration
from mlx_train_perf.plan.estimate import (
    FitPoint,
    ModelShape,
    TrainConfig,
    _lora_bytes,
    _loss_bytes,
    _optimizer_bytes,
    estimate_peak,
    fit_memory_coeffs,
    plan_fit,
)


def _shape() -> ModelShape:  # micro "llama": V=1000, D=64, L=2, I=128, 4h/2kv, untied
    return ModelShape(vocab=1000, hidden=64, layers=2, intermediate=128, heads=4,
                      kv_heads=2, tied=False, quant_bits=None, quant_group=None)


def test_param_count_hand_computed() -> None:
    s = _shape()
    # embed 1000*64 + head 1000*64
    # + 2*(attn 64*64*(2+2*0.5)=64*64*3 + mlp 3*64*128 + norms 2*64) + 64
    expected = 64000 + 64000 + 2 * (12288 + 24576 + 128) + 64
    assert s.param_count() == expected


def test_loss_component_dominates_naive_vanishes_kernel() -> None:
    s = _shape()
    calib = load_calibration()
    naive_cfg = TrainConfig(batch=2, seq_len=512, dtype="bfloat16", lora_rank=8,
                            lora_layers=2, grad_checkpoint=True, impl="naive")
    kernel_cfg = TrainConfig(batch=2, seq_len=512, dtype="bfloat16", lora_rank=8,
                             lora_layers=2, grad_checkpoint=True, impl="kernel")
    _, comp_n = estimate_peak(s, naive_cfg, calib)
    _, comp_k = estimate_peak(s, kernel_cfg, calib)
    # task-13 review item 3: the naive coefficient is now calibrated (measured against
    # a persisted gate artifact -- see Calibration.naive_loss_bytes_per_nv), replacing
    # the brief's literal "x2" which under-modeled it by ~1.9x at production shape.
    assert comp_n["loss"] == int(calib.naive_loss_bytes_per_nv * 2 * 512 * 1000)
    assert comp_k["loss"] < comp_n["loss"] // 100


def test_refuses_and_suggests() -> None:
    s = _shape()
    cfg = TrainConfig(batch=4096, seq_len=8192, dtype="bfloat16", lora_rank=8,
                      lora_layers=2, grad_checkpoint=True, impl="kernel")
    # Budget above the ~1.6 GB base transient but far below this huge batch/seq config.
    report = plan_fit(s, cfg, budget_bytes=3 * 1024**3)
    assert not report.fits
    assert report.is_estimate
    assert report.suggestion is not None
    assert report.suggestion.batch < 4096


def test_calibration_has_provenance() -> None:
    calib = load_calibration()
    for key in ("machine", "macos", "mlx_version", "measured_date"):
        assert calib.provenance[key]


def test_from_config_reads_hf_llama_keys_matches_hand_shape() -> None:
    config = {
        "vocab_size": 1000, "hidden_size": 64, "num_hidden_layers": 2,
        "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
        "tie_word_embeddings": False,
    }
    assert ModelShape.from_config(config) == _shape()


def test_from_config_defaults_kv_heads_and_reads_quantization_block() -> None:
    # HF omits `num_key_value_heads` for plain (non-GQA) attention -- MHA fallback is
    # num_attention_heads. Quantization metadata mirrors the `quantization` block
    # mlx_lm.convert writes (group_size/bits) -- see adapters/mlx_lm.py's verified notes.
    config = {
        "vocab_size": 1000, "hidden_size": 64, "num_hidden_layers": 2,
        "intermediate_size": 128, "num_attention_heads": 4,
        "quantization": {"group_size": 64, "bits": 4},
    }
    s = ModelShape.from_config(config)
    assert s.kv_heads == 4
    assert s.tied is False  # HF's own default when tie_word_embeddings is absent
    assert s.quant_bits == 4
    assert s.quant_group == 64


def test_unknown_dtype_raises_plan_input_error() -> None:
    s = _shape()
    calib = load_calibration()
    cfg = TrainConfig(batch=1, seq_len=1, dtype="int8", lora_rank=8, lora_layers=2,
                      grad_checkpoint=True, impl="naive")
    with pytest.raises(PlanInputError):
        estimate_peak(s, cfg, calib)


def test_default_budget_uses_this_projects_clamped_wired_cap() -> None:
    s = _shape()
    cfg = TrainConfig(batch=1, seq_len=128, dtype="bfloat16", lora_rank=8, lora_layers=2,
                      grad_checkpoint=True, impl="kernel")
    report = plan_fit(s, cfg)
    dev_max = int(mx.device_info()["max_recommended_working_set_size"])
    wired, _ = clamped_caps(dev_max)
    assert report.budget_bytes == wired


def test_weights_component_prices_whole_model_at_quantized_rate() -> None:
    """task-13 review item 2 (controller ruling, supersedes an earlier head-only
    reading): a quantized ModelShape prices ALL params at the quantized rate (body AND
    head), not just the V*D head matrix -- real mlx-community 4-bit checkpoints
    quantize the whole model, and pricing only the head would add a phantom
    dtype_size-per-param cost for the (dominant) body layers.

    Expected value is INDEPENDENTLY hand-derived using the exact same literal building
    blocks as test_param_count_hand_computed (64000, 12288, 24576, 128, 64) -- not a
    call to s.param_count() or any other code path -- so a regression in the
    implementation can't accidentally still satisfy this assertion.
    """
    s = ModelShape(vocab=1000, hidden=64, layers=2, intermediate=128, heads=4,
                   kv_heads=2, tied=False, quant_bits=4, quant_group=64)
    calib = load_calibration()
    cfg = TrainConfig(batch=1, seq_len=1, dtype="bfloat16", lora_rank=8, lora_layers=2,
                      grad_checkpoint=True, impl="kernel")
    _, comp = estimate_peak(s, cfg, calib)
    p_total = 64000 + 64000 + 2 * (12288 + 24576 + 128) + 64  # == 202048
    rate = 4 / 8 + 4 / 64                                     # bits/8 + 4/group == 0.5625
    expected = int(p_total * rate)                            # == 113652
    assert comp["weights"] == expected


def test_chunked_loss_scales_with_fixed_tile_not_vocab() -> None:
    s = _shape()
    calib = load_calibration()
    cfg = TrainConfig(batch=2, seq_len=512, dtype="bfloat16", lora_rank=8, lora_layers=2,
                      grad_checkpoint=True, impl="chunked")
    _, comp = estimate_peak(s, cfg, calib)
    assert comp["loss"] == 2 * 512 * 8192 * 4 * 3  # n * chunk_tile * 4 bytes * 3 buffers


def test_unknown_impl_raises_plan_input_error() -> None:
    s = _shape()
    calib = load_calibration()
    cfg = TrainConfig(batch=1, seq_len=1, dtype="bfloat16", lora_rank=8, lora_layers=2,
                      grad_checkpoint=True, impl="auto")
    with pytest.raises(PlanInputError):
        estimate_peak(s, cfg, calib)


def test_full_ft_adds_d_w_term_to_loss() -> None:
    s = _shape()
    calib = load_calibration()
    lora_cfg = TrainConfig(batch=2, seq_len=512, dtype="bfloat16", lora_rank=8,
                           lora_layers=2, grad_checkpoint=True, impl="kernel")
    full_ft_cfg = TrainConfig(batch=2, seq_len=512, dtype="bfloat16", lora_rank=0,
                              lora_layers=0, grad_checkpoint=True, impl="kernel")
    _, comp_lora = estimate_peak(s, lora_cfg, calib)
    _, comp_full = estimate_peak(s, full_ft_cfg, calib)
    d_w = s.vocab * s.hidden * 4 * 2
    assert comp_full["loss"] == comp_lora["loss"] + d_w


def test_suggestion_falls_through_to_seq_len_when_batch_already_floored() -> None:
    # batch=1 already -- the halving phase has nothing to try, so a fits=False verdict
    # must fall through straight to the seq_len-stepping phase.
    s = _shape()
    cfg = TrainConfig(batch=1, seq_len=8192, dtype="bfloat16", lora_rank=8, lora_layers=2,
                      grad_checkpoint=True, impl="kernel")
    # Budget above base + the seq-1024 floor config, below the seq-8192 config (whose
    # O(N^2) attention term dominates) -- so the fix must step seq_len down, not batch.
    report = plan_fit(s, cfg, budget_bytes=3 * 1024**3)
    assert not report.fits
    assert report.suggestion is not None
    assert report.suggestion.batch == 1
    assert 1024 <= report.suggestion.seq_len < 8192


def test_suggestion_is_none_when_nothing_fits_even_at_floor() -> None:
    s = _shape()
    cfg = TrainConfig(batch=1, seq_len=8192, dtype="bfloat16", lora_rank=8, lora_layers=2,
                      grad_checkpoint=True, impl="kernel")
    report = plan_fit(s, cfg, budget_bytes=1000)
    assert not report.fits
    assert report.suggestion is None


def test_naive_loss_within_15pct_of_measured_gate_artifact() -> None:
    """task-13 review item 3: reconciles the naive coefficient against the persisted
    mlx-train-perf-spike/results/gate_naive_n8192.json artifact (reference-only: read
    for its recorded numbers, never executed by this project). That gate ran
    naive_linear_ce under mx.value_and_grad(argnums=(0, 1)) -- a TRAINABLE head, i.e.
    lora_rank=0 in this planner's terms (gate_trainstep.py's "naive" condition; its
    nonzero d_w_checksum/d_w_corner fields confirm the head gradient was computed) --
    at n=8192, V=151936, D=4096, bf16 hidden, measuring marginal_peak_gb=18.547 (GiB,
    per that script's own `round((peak - active_before) / 1024**3, 3)`).

    Derivation of the calibrated coefficient (see calibration_data.json /
    Calibration.naive_loss_bytes_per_nv): converting to bytes, measured ~=
    18.547 * 1024**3 ~= 19,914,689,610. Holding the (separate, unchanged by this item)
    d_w term V*D*4*2 = 4,978,638,848 fixed, the remaining ~14,936,050,762 bytes divided
    by n*V = 1,244,659,712 gives ~12.0 bytes per (n, V) pair. This is an EMPIRICAL FIT to
    this one anchor, not a validated buffer-by-buffer decomposition -- at this exact
    shape 2*D == n, so the d_w term and the n*V coefficient are not separately
    identifiable from this point alone (see
    test_naive_estimate_is_conservative_at_smaller_n for the cross-check against a
    second artifact at a different n). Body-shape fields (layers/intermediate/heads/
    kv_heads) are irrelevant to the "loss" component and set to 1 to keep this test
    isolated to exactly what the artifact measured.
    """
    shape = ModelShape(vocab=151936, hidden=4096, layers=1, intermediate=1, heads=1,
                       kv_heads=1, tied=False, quant_bits=None, quant_group=None)
    calib = load_calibration()
    cfg = TrainConfig(batch=1, seq_len=8192, dtype="bfloat16", lora_rank=0, lora_layers=0,
                      grad_checkpoint=True, impl="naive")
    _, comp = estimate_peak(shape, cfg, calib)
    measured_bytes = 18.547 * 1024**3
    assert abs(comp["loss"] - measured_bytes) / measured_bytes < 0.15


def test_naive_estimate_is_conservative_at_smaller_n() -> None:
    """task-13 re-review: the naive coefficient (Calibration.naive_loss_bytes_per_nv)
    is fit to ONE production-shape anchor (n=8192, gate_naive_n8192.json) and is known
    NOT to extrapolate linearly -- the sibling artifact gate_naive_n2048.json (same
    code path, reference-only, never executed by this project) measures n=2048,
    V=151936, D=4096, bf16, marginal_peak_gb=4.057 (~4,356,170,580 bytes), while this
    planner's formula (12.0*n*V + the d_w term) predicts ~8,712,617,984 bytes there --
    about 2x too high, and a single non-negative linear coefficient cannot fit both the
    n=2048 and n=8192 anchors exactly (solving for both forces a negative intercept).
    Pinning the direction of that miss: the estimate must OVER-predict, never
    under-predict, at a shape smaller than the calibration anchor -- the safe direction
    for a planner whose job is to steer callers away from the discouraged naive path,
    not reassure them it fits when it might not.
    """
    shape = ModelShape(vocab=151936, hidden=4096, layers=1, intermediate=1, heads=1,
                       kv_heads=1, tied=False, quant_bits=None, quant_group=None)
    calib = load_calibration()
    cfg = TrainConfig(batch=1, seq_len=2048, dtype="bfloat16", lora_rank=0, lora_layers=0,
                      grad_checkpoint=True, impl="naive")
    _, comp = estimate_peak(shape, cfg, calib)
    measured_bytes = 4.057 * 1024**3
    assert comp["loss"] >= measured_bytes


# ---------------------------------------------------------------------------
# The corrected memory model: a LINEAR activation term (gc-aware) + a QUADRATIC
# attention-backward term (one layer, gc-independent -- mlx's O(N^2) SDPA backward
# materializes one (N,N) at a time). These tests pin the SHAPE + measured accuracy.
# ---------------------------------------------------------------------------


def test_predicted_peak_matches_measured_qwen3_8b_within_tolerance() -> None:
    """Measured-vs-predicted acceptance: anchor the corrected model to the North-Star
    measurement it was fit against -- Qwen3-8B-4bit, seq 8192, grad_checkpoint=True,
    kernel, MEASURED total peak 25.68 GB (`_artifacts/northstar_context_sweep/`, session
    435c2ef). The model (with the committed calibration) predicts within 15%, and
    OVER-predicts (the safe direction for a fit planner). The whole point of the O(N^2)
    correction is here: a purely-linear activation model grossly UNDER-predicts at this
    context, which is exactly the (unsafe) failure this task fixed."""
    qwen = ModelShape(vocab=151936, hidden=4096, layers=36, intermediate=12288, heads=32,
                      kv_heads=8, tied=False, quant_bits=4, quant_group=64)
    calib = load_calibration()
    cfg = TrainConfig(batch=1, seq_len=8192, dtype="bfloat16", lora_rank=8, lora_layers=36,
                      grad_checkpoint=True, impl="kernel")
    peak, _ = estimate_peak(qwen, cfg, calib)
    measured_bytes = 25.68 * 1024**3
    assert abs(peak - measured_bytes) / measured_bytes < 0.15
    assert peak >= measured_bytes  # over-predicts -- the safe direction for a fit planner


def test_attention_component_scales_quadratically_with_seq() -> None:
    """The `"attention"` component is `a_quad * batch*heads*seq^2` -- doubling seq_len
    must ~4x it (O(N^2)), which is the whole point of the correction. Approx (not exact)
    because `estimate_peak` int-floors each component."""
    s = _shape()
    calib = load_calibration()
    cfg_n = TrainConfig(batch=1, seq_len=512, dtype="bfloat16", lora_rank=8, lora_layers=2,
                        grad_checkpoint=True, impl="kernel")
    cfg_2n = TrainConfig(batch=1, seq_len=1024, dtype="bfloat16", lora_rank=8, lora_layers=2,
                         grad_checkpoint=True, impl="kernel")
    _, comp_n = estimate_peak(s, cfg_n, calib)
    _, comp_2n = estimate_peak(s, cfg_2n, calib)
    # rel tolerance absorbs `estimate_peak`'s per-component int-floor (a few bytes on
    # values of millions) -- the SHAPE (4x vs 2x) is what this pins.
    assert comp_2n["attention"] == pytest.approx(4 * comp_n["attention"], rel=1e-5)
    # And it is the term that grows fastest with context: attention grows 4x while the
    # linear activations term only doubles.
    assert comp_2n["activations"] == pytest.approx(2 * comp_n["activations"], rel=1e-5)


def test_grad_checkpoint_changes_only_the_linear_activation_term() -> None:
    """grad_checkpoint selects the linear coefficient (`_ckpt` vs `_full`, ~45x apart)
    but does NOT change the attention term (the O(N^2) backward is one-layer either way)."""
    s = _shape()
    calib = load_calibration()
    cfg_ck = TrainConfig(batch=1, seq_len=1024, dtype="bfloat16", lora_rank=8, lora_layers=2,
                         grad_checkpoint=True, impl="kernel")
    cfg_no = TrainConfig(batch=1, seq_len=1024, dtype="bfloat16", lora_rank=8, lora_layers=2,
                         grad_checkpoint=False, impl="kernel")
    _, comp_ck = estimate_peak(s, cfg_ck, calib)
    _, comp_no = estimate_peak(s, cfg_no, calib)
    assert comp_no["attention"] == comp_ck["attention"]  # gc-independent
    assert comp_no["activations"] > comp_ck["activations"]  # full stores more than ckpt
    assert comp_no["activations"] / comp_ck["activations"] == pytest.approx(
        calib.act_bytes_per_token_hidden_layer_full
        / calib.act_bytes_per_token_hidden_layer_ckpt, rel=1e-5)


def test_base_component_is_the_calibration_base_transient() -> None:
    """review item: `components["base"]` is new in the O(N^2) rework and nothing else
    pins it -- the measured-anchor test stays green with base dropped entirely (mutation-
    verified during review: anchor error moves 8.6% -> 2.2%, still passing), so a
    refactor that forgets to add it to the components dict would ship silently."""
    s = _shape()
    calib = load_calibration()
    cfg = TrainConfig(batch=1, seq_len=512, dtype="bfloat16", lora_rank=8, lora_layers=2,
                      grad_checkpoint=True, impl="kernel")
    _, comp = estimate_peak(s, cfg, calib)
    assert comp["base"] == int(calib.base_transient_bytes)


def test_activation_and_attention_components_hand_computed() -> None:
    """Exact-value pins for the two components the rework added. The ratio tests above
    are invariant to a dropped constant factor (losing `shape.hidden` from the linear
    term still doubles with seq and keeps the ckpt/full ratio -- mutation-verified during
    review), so each term gets one hand-computed absolute value: drivers as literals from
    `_shape()` (batch=1, seq=512, hidden=64, layers=2, heads=4 -- all powers of two, so
    the product is float-exact), coefficient from the committed calibration."""
    s = _shape()
    calib = load_calibration()
    cfg = TrainConfig(batch=1, seq_len=512, dtype="bfloat16", lora_rank=8, lora_layers=2,
                      grad_checkpoint=True, impl="kernel")
    _, comp = estimate_peak(s, cfg, calib)
    assert comp["activations"] == int(
        calib.act_bytes_per_token_hidden_layer_ckpt * (1 * 512 * 64 * 2))
    assert comp["attention"] == int(
        calib.attn_bytes_per_head_token2 * (1 * 4 * 512 * 512))


# ---------------------------------------------------------------------------
# fit_memory_coeffs: OLS fit of (base, a_lin_ckpt, a_quad) from grad_checkpoint=True
# kernel-impl `run_train_step` peaks + a_lin_full from grad_checkpoint=False peaks.
# Every point here is SYNTHETIC (generated forward from known true coefficients via the
# exact same model + `_lora_bytes`/`_optimizer_bytes`/`_loss_bytes` the fit itself
# subtracts) -- no real artifact, no model, no MLX device. optimizer_bytes_per_param is
# analytic (not fitted). The controller runs this against real artifacts after the
# production runs.
# ---------------------------------------------------------------------------


def _synthesize_fit_point(
    *, shape: ModelShape, calib, batch: int, seq_len: int, lora_rank: int,
    lora_layers: int, grad_checkpoint: bool, base: float, a_lin_ckpt: float,
    a_lin_full: float, a_quad: float,
) -> FitPoint:
    cfg = TrainConfig(batch=batch, seq_len=seq_len, dtype="bfloat16", lora_rank=lora_rank,
                      lora_layers=lora_layers, grad_checkpoint=grad_checkpoint, impl="kernel")
    analytic = (_lora_bytes(cfg, shape) + _optimizer_bytes(cfg, shape, calib)
                + _loss_bytes(cfg, shape, calib))
    a_lin = a_lin_ckpt if grad_checkpoint else a_lin_full
    x_lin = batch * seq_len * shape.hidden * shape.layers
    x_quad = batch * shape.heads * seq_len ** 2
    marginal = base + a_lin * x_lin + a_quad * x_quad + analytic
    return FitPoint(shape=shape, cfg=cfg, marginal_peak_bytes=marginal)


def test_fit_memory_coeffs_recovers_known_coefficients() -> None:
    """Classic fit-recovery: synthesize FitPoints FORWARD from KNOWN coefficients, fit
    BACKWARD, assert recovery. gc=True points recover (base, a_lin_ckpt, a_quad); the
    gc=False point recovers a_lin_full."""
    shape = _shape()
    calib = load_calibration()
    base, a_lin_ckpt, a_lin_full, a_quad = 1.5e9, 4.0, 180.0, 9.0

    def synth(seq_len: int, gc: bool) -> FitPoint:
        return _synthesize_fit_point(
            shape=shape, calib=calib, batch=1, seq_len=seq_len, lora_rank=8, lora_layers=2,
            grad_checkpoint=gc, base=base, a_lin_ckpt=a_lin_ckpt, a_lin_full=a_lin_full,
            a_quad=a_quad)

    points = [synth(512, True), synth(1024, True), synth(2048, True), synth(1024, False)]
    c = fit_memory_coeffs(points, calib=calib)
    assert c["base_transient_bytes"] == pytest.approx(base, rel=1e-6)
    assert c["act_bytes_per_token_hidden_layer_ckpt"] == pytest.approx(a_lin_ckpt, rel=1e-6)
    assert c["act_bytes_per_token_hidden_layer_full"] == pytest.approx(a_lin_full, rel=1e-6)
    assert c["attn_bytes_per_head_token2"] == pytest.approx(a_quad, rel=1e-6)


def test_fit_memory_coeffs_without_gc_false_points_keeps_calib_full() -> None:
    """With no grad_checkpoint=False points, `a_lin_full` is not identifiable -- the fit
    falls back to the existing calib value rather than inventing one."""
    shape = _shape()
    calib = load_calibration()

    def synth(seq_len: int) -> FitPoint:
        return _synthesize_fit_point(
            shape=shape, calib=calib, batch=1, seq_len=seq_len, lora_rank=8, lora_layers=2,
            grad_checkpoint=True, base=1e9, a_lin_ckpt=3.0, a_lin_full=999.0, a_quad=5.0)

    c = fit_memory_coeffs([synth(512), synth(1024), synth(2048)], calib=calib)
    assert (c["act_bytes_per_token_hidden_layer_full"]
            == calib.act_bytes_per_token_hidden_layer_full)


def test_fit_memory_coeffs_rejects_fewer_than_three_gc_true_points() -> None:
    shape = _shape()
    calib = load_calibration()

    def synth(seq_len: int) -> FitPoint:
        return _synthesize_fit_point(
            shape=shape, calib=calib, batch=1, seq_len=seq_len, lora_rank=8, lora_layers=2,
            grad_checkpoint=True, base=1.0, a_lin_ckpt=1.0, a_lin_full=1.0, a_quad=1.0)

    with pytest.raises(PlanInputError, match="at least 3"):
        fit_memory_coeffs([synth(512), synth(1024)], calib=calib)


def test_fit_memory_coeffs_rejects_single_seq_len_gc_true() -> None:
    """Three gc=True points but all the SAME seq_len cannot separate the linear from the
    quadratic term (both driven only by seq) -- needs >= 2 distinct seq_len."""
    shape = _shape()
    calib = load_calibration()

    def synth(batch: int) -> FitPoint:
        return _synthesize_fit_point(
            shape=shape, calib=calib, batch=batch, seq_len=512, lora_rank=8, lora_layers=2,
            grad_checkpoint=True, base=1.0, a_lin_ckpt=1.0, a_lin_full=1.0, a_quad=1.0)

    with pytest.raises(PlanInputError, match="distinct seq_len"):
        fit_memory_coeffs([synth(1), synth(2), synth(4)], calib=calib)


def test_fit_memory_coeffs_rejects_two_distinct_seq_len_gc_true() -> None:
    """Three gc=True points at only TWO distinct seq_len (batch fixed -- the actual
    calibration regime) give a rank-2 design for the 3 unknowns (base, linear, quad).
    `np.linalg.lstsq` does NOT raise on rank deficiency -- it returns the minimum-norm
    solution, which here is garbage (verified during review: wildly wrong values
    including a negative a_quad). The fit must refuse loudly instead."""
    shape = _shape()
    calib = load_calibration()

    def synth(seq_len: int) -> FitPoint:
        return _synthesize_fit_point(
            shape=shape, calib=calib, batch=1, seq_len=seq_len, lora_rank=8, lora_layers=2,
            grad_checkpoint=True, base=1.5e9, a_lin_ckpt=4.0, a_lin_full=180.0, a_quad=9.0)

    with pytest.raises(PlanInputError, match="distinct seq_len"):
        fit_memory_coeffs([synth(1024), synth(1024), synth(2048)], calib=calib)


def test_fit_memory_coeffs_accepts_two_seq_len_when_batch_variation_restores_rank() -> None:
    """The guard is about design-matrix RANK, not a seq_len head-count: with batch
    varying, two distinct seq_len CAN span (constant, linear, quadratic) -- x_lin and
    x_quad scale together in batch but differently in seq_len, so e.g. (b=1,s=512),
    (b=2,s=512), (b=1,s=1024) is full-rank and must fit exactly."""
    shape = _shape()
    calib = load_calibration()
    base, a_lin_ckpt, a_quad = 1.5e9, 4.0, 9.0

    def synth(batch: int, seq_len: int) -> FitPoint:
        return _synthesize_fit_point(
            shape=shape, calib=calib, batch=batch, seq_len=seq_len, lora_rank=8,
            lora_layers=2, grad_checkpoint=True, base=base, a_lin_ckpt=a_lin_ckpt,
            a_lin_full=180.0, a_quad=a_quad)

    c = fit_memory_coeffs([synth(1, 512), synth(2, 512), synth(1, 1024)], calib=calib)
    assert c["base_transient_bytes"] == pytest.approx(base, rel=1e-6)
    assert c["act_bytes_per_token_hidden_layer_ckpt"] == pytest.approx(a_lin_ckpt, rel=1e-6)
    assert c["attn_bytes_per_head_token2"] == pytest.approx(a_quad, rel=1e-6)


def test_fit_memory_coeffs_rejects_non_kernel_impl() -> None:
    shape = _shape()
    calib = load_calibration()
    cfg = TrainConfig(batch=1, seq_len=512, dtype="bfloat16", lora_rank=8, lora_layers=2,
                      grad_checkpoint=True, impl="naive")
    points = [FitPoint(shape=shape, cfg=cfg, marginal_peak_bytes=float(i)) for i in (1, 2, 3)]
    with pytest.raises(PlanInputError, match="kernel"):
        fit_memory_coeffs(points, calib=calib)


# ---------------------------------------------------------------------------
# Task 14: the flash-attention branch. `TrainConfig.attention` selects between the
# stock O(N^2) attention-backward term (a_quad*batch*heads*seq^2, unchanged) and the
# flash O(N) term (analytic O/L saved state + a_flash*batch*heads*seq). Driver form
# settled from the T13 single-op scaling: growth is linear in N (single-op 2048->4096
# == exactly 2.0x); the 4096->8192 super-linearity is the confirmed budget-bounded
# dispatch-split additive, not a growth-law change, so the planner models a pure linear
# term and leans on overhead_frac for the over-predict-safe cushion.
# ---------------------------------------------------------------------------


def test_attention_defaults_to_stock_and_uses_the_quadratic_term() -> None:
    """A TrainConfig built without an explicit `attention` keeps the 0.1.0 behavior:
    the stock O(N^2) attention term (a_quad*batch*heads*seq^2)."""
    s = _shape()
    calib = load_calibration()
    cfg = TrainConfig(batch=1, seq_len=512, dtype="bfloat16", lora_rank=8, lora_layers=2,
                      grad_checkpoint=True, impl="kernel")
    assert cfg.attention == "stock"
    _, comp = estimate_peak(s, cfg, calib)
    assert comp["attention"] == int(
        calib.attn_bytes_per_head_token2 * (1 * 4 * 512 * 512))


def test_flash_attention_changes_only_the_attention_component() -> None:
    """`attention="flash"` replaces the O(N^2) term with the O(N) flash term and changes
    NOTHING else (mirror `test_grad_checkpoint_changes_only_the_linear_activation_term`).
    At long context (seq=8192, the flagship) the O(N) flash term is far below the O(N^2)
    stock term -- the whole point of the branch. (At SHORT context the two are comparable,
    or flash is even slightly larger: its fitted linear coefficient captures the real
    backward transient, and the quadratic term has not yet dominated -- an honest property,
    not a regression.)"""
    s = _shape()
    calib = load_calibration()
    stock = TrainConfig(batch=1, seq_len=8192, dtype="bfloat16", lora_rank=8, lora_layers=2,
                        grad_checkpoint=True, impl="kernel", attention="stock")
    flash = TrainConfig(batch=1, seq_len=8192, dtype="bfloat16", lora_rank=8, lora_layers=2,
                        grad_checkpoint=True, impl="kernel", attention="flash")
    _, comp_stock = estimate_peak(s, stock, calib)
    _, comp_flash = estimate_peak(s, flash, calib)
    for key in ("weights", "base", "lora", "optimizer", "activations", "loss"):
        assert comp_flash[key] == comp_stock[key]
    assert comp_flash["attention"] < comp_stock["attention"]


def test_flash_attention_component_hand_computed() -> None:
    """Exact-value pin (0.1.0 pin style): the flash attention component is the analytic
    O/L saved state (O = batch*seq*hidden*dtype_bytes, bf16=2; L = batch*heads*seq*4 fp32)
    plus a_flash*batch*heads*seq. Drivers are powers of two from `_shape()` (batch=1,
    seq=512, hidden=64, heads=4) so the products are float-exact; a_flash comes from the
    committed calibration."""
    s = _shape()
    calib = load_calibration()
    cfg = TrainConfig(batch=1, seq_len=512, dtype="bfloat16", lora_rank=8, lora_layers=2,
                      grad_checkpoint=True, impl="kernel", attention="flash")
    _, comp = estimate_peak(s, cfg, calib)
    o_saved = 1 * 512 * 64 * 2       # attention output O in compute dtype (bf16)
    l_saved = 1 * 4 * 512 * 4        # logsumexp L in fp32
    expected = int(o_saved + l_saved
                   + calib.attn_bytes_per_head_token_flash * (1 * 4 * 512))
    assert comp["attention"] == expected


def test_flash_attention_scales_linearly_with_seq() -> None:
    """The flash attention component is O(N) in seq -- doubling seq_len ~2x it (vs the
    stock term's 4x). Approx (not exact) because `estimate_peak` int-floors each
    component."""
    s = _shape()
    calib = load_calibration()
    cfg_n = TrainConfig(batch=1, seq_len=512, dtype="bfloat16", lora_rank=8, lora_layers=2,
                        grad_checkpoint=True, impl="kernel", attention="flash")
    cfg_2n = TrainConfig(batch=1, seq_len=1024, dtype="bfloat16", lora_rank=8, lora_layers=2,
                         grad_checkpoint=True, impl="kernel", attention="flash")
    _, comp_n = estimate_peak(s, cfg_n, calib)
    _, comp_2n = estimate_peak(s, cfg_2n, calib)
    assert comp_2n["attention"] == pytest.approx(2 * comp_n["attention"], rel=1e-5)


def test_unknown_attention_raises_plan_input_error() -> None:
    s = _shape()
    calib = load_calibration()
    cfg = TrainConfig(batch=1, seq_len=512, dtype="bfloat16", lora_rank=8, lora_layers=2,
                      grad_checkpoint=True, impl="kernel", attention="bogus")
    with pytest.raises(PlanInputError, match="attention"):
        estimate_peak(s, cfg, calib)


def _synthesize_flash_fit_point(
    *, shape: ModelShape, calib, batch: int, seq_len: int, lora_rank: int,
    lora_layers: int, grad_checkpoint: bool, a_flash: float,
) -> FitPoint:
    """Synthesize a flash FitPoint FORWARD from a known a_flash, holding base/a_lin at
    the calibration's own values (which the flash fit holds fixed). marginal = base +
    a_lin*x_lin + a_flash*x_flash + analytic_small + analytic_O/L."""
    cfg = TrainConfig(batch=batch, seq_len=seq_len, dtype="bfloat16", lora_rank=lora_rank,
                      lora_layers=lora_layers, grad_checkpoint=grad_checkpoint, impl="kernel",
                      attention="flash")
    analytic = (_lora_bytes(cfg, shape) + _optimizer_bytes(cfg, shape, calib)
                + _loss_bytes(cfg, shape, calib))
    a_lin = (calib.act_bytes_per_token_hidden_layer_ckpt if grad_checkpoint
             else calib.act_bytes_per_token_hidden_layer_full)
    x_lin = batch * seq_len * shape.hidden * shape.layers
    x_flash = batch * shape.heads * seq_len
    o_l = batch * seq_len * shape.hidden * 2 + batch * shape.heads * seq_len * 4
    marginal = calib.base_transient_bytes + a_lin * x_lin + a_flash * x_flash + analytic + o_l
    return FitPoint(shape=shape, cfg=cfg, marginal_peak_bytes=marginal)


def test_fit_memory_coeffs_recovers_a_flash_from_flash_points() -> None:
    """Fit-recovery for the flash coefficient: synthesize flash FitPoints FORWARD from a
    known a_flash (base/a_lin held at calib), fit BACKWARD by 1-variable residual OLS,
    assert recovery. Flash-only points do NOT require the stock rank guard (a_flash is a
    through-origin 1-variable fit needing only >= 1 point, den>0)."""
    shape = _shape()
    calib = load_calibration()
    a_flash = 12345.0
    pts = [_synthesize_flash_fit_point(shape=shape, calib=calib, batch=1, seq_len=s,
                                       lora_rank=8, lora_layers=2, grad_checkpoint=True,
                                       a_flash=a_flash)
           for s in (512, 1024, 2048)]
    c = fit_memory_coeffs(pts, calib=calib)
    assert c["attn_bytes_per_head_token_flash"] == pytest.approx(a_flash, rel=1e-6)


def test_fit_memory_coeffs_recovers_a_flash_from_a_single_flash_point() -> None:
    """The flash fit is identifiable from a single flash point (den = x_flash^2 > 0) --
    the T13 campaign anchors it from as few as two Qwen points."""
    shape = _shape()
    calib = load_calibration()
    a_flash = 9999.0
    pts = [_synthesize_flash_fit_point(shape=shape, calib=calib, batch=1, seq_len=8192,
                                       lora_rank=8, lora_layers=2, grad_checkpoint=True,
                                       a_flash=a_flash)]
    c = fit_memory_coeffs(pts, calib=calib)
    assert c["attn_bytes_per_head_token_flash"] == pytest.approx(a_flash, rel=1e-6)


def test_fit_memory_coeffs_flash_only_keeps_stock_coefficients() -> None:
    """The real T13 campaign refits ONLY a_flash from a flash-only manifest -- the stock
    coefficients (base, a_lin, a_quad) must be preserved unchanged from `calib`, never
    silently reset by a flash-only fit."""
    shape = _shape()
    calib = load_calibration()
    pts = [_synthesize_flash_fit_point(shape=shape, calib=calib, batch=1, seq_len=s,
                                       lora_rank=8, lora_layers=2, grad_checkpoint=True,
                                       a_flash=5000.0)
           for s in (512, 1024, 2048)]
    c = fit_memory_coeffs(pts, calib=calib)
    assert c["base_transient_bytes"] == calib.base_transient_bytes
    assert c["act_bytes_per_token_hidden_layer_ckpt"] == calib.act_bytes_per_token_hidden_layer_ckpt
    assert c["act_bytes_per_token_hidden_layer_full"] == calib.act_bytes_per_token_hidden_layer_full
    assert c["attn_bytes_per_head_token2"] == calib.attn_bytes_per_head_token2


def test_fit_memory_coeffs_without_flash_points_keeps_calib_flash() -> None:
    """With no flash points, `a_flash` is not identifiable -- the fit falls back to the
    existing calib value rather than inventing one (mirrors the a_lin_full fallback)."""
    shape = _shape()
    calib = load_calibration()

    def synth(seq_len: int) -> FitPoint:
        return _synthesize_fit_point(
            shape=shape, calib=calib, batch=1, seq_len=seq_len, lora_rank=8, lora_layers=2,
            grad_checkpoint=True, base=1e9, a_lin_ckpt=3.0, a_lin_full=999.0, a_quad=5.0)

    c = fit_memory_coeffs([synth(512), synth(1024), synth(2048)], calib=calib)
    assert (c["attn_bytes_per_head_token_flash"]
            == calib.attn_bytes_per_head_token_flash)


def test_fit_memory_coeffs_flash_fit_ols_default_is_byte_identical() -> None:
    """`flash_fit` defaults to `"ols"` -- omitting the new keyword must reproduce the
    shipped 0.4.0 behavior exactly (Task 8, 0.5.0)."""
    shape = _shape()
    calib = load_calibration()
    pts = [_synthesize_flash_fit_point(shape=shape, calib=calib, batch=1, seq_len=s,
                                       lora_rank=8, lora_layers=2, grad_checkpoint=True,
                                       a_flash=12345.0)
           for s in (512, 1024, 2048)]
    c_default = fit_memory_coeffs(pts, calib=calib)
    c_explicit_ols = fit_memory_coeffs(pts, calib=calib, flash_fit="ols")
    assert c_default == c_explicit_ols


def test_fit_memory_coeffs_flash_fit_envelope_matches_max_ratio() -> None:
    """`flash_fit="envelope"` (Task 8, 0.5.0): `a_flash = max over flash points of
    (flash_residual(p) / x_flash(p))`, reusing the SAME private closures the `"ols"`
    path uses. Three flash FitPoints are synthesized FORWARD from three DIFFERENT true
    a_flash values, so OLS's weighted-average solution provably diverges from the
    envelope's max -- both expected values are derived here directly from the
    synthetic per-point (x_flash, true a_flash) pairs, independent of
    `fit_memory_coeffs`'s own implementation (noiseless forward construction makes
    `flash_residual(p) == true_a_flash_i * x_flash(p)` exactly, so OLS's closed form
    reduces to a weighted average of the true values and the envelope reduces to their
    max)."""
    shape = _shape()
    calib = load_calibration()
    seq_lens = (512, 1024, 2048)
    true_a_flash = (1_000.0, 1_500.0, 50_000.0)   # last point is the "under-predicted anchor"
    pts = [
        _synthesize_flash_fit_point(shape=shape, calib=calib, batch=1, seq_len=s,
                                    lora_rank=8, lora_layers=2, grad_checkpoint=True,
                                    a_flash=a)
        for s, a in zip(seq_lens, true_a_flash, strict=True)
    ]
    x = [1 * shape.heads * s for s in seq_lens]   # x_flash(p) = batch * heads * seq_len
    expected_ols = (sum(xi ** 2 * ai for xi, ai in zip(x, true_a_flash, strict=True))
                    / sum(xi ** 2 for xi in x))
    expected_envelope = max(true_a_flash)
    assert expected_ols != expected_envelope   # construction sanity: OLS != envelope

    c_ols = fit_memory_coeffs(pts, calib=calib, flash_fit="ols")
    c_env = fit_memory_coeffs(pts, calib=calib, flash_fit="envelope")
    assert c_ols["attn_bytes_per_head_token_flash"] == pytest.approx(expected_ols, rel=1e-6)
    assert (c_env["attn_bytes_per_head_token_flash"]
            == pytest.approx(expected_envelope, rel=1e-6))


def test_predicted_peak_one_sided_and_bounded_qwen3_8b_flash() -> None:
    """Measured-vs-predicted acceptance for the flash branch under the 0.5.0 ENVELOPE
    contract. The 0.5.0 refit (anchors to seq 12288, BOTH loss impls) showed the old
    OLS coefficient UNDER-predicted the stock-loss flash arm from 8192 up (-2.1 GiB at
    8192, -4.1 at 12288); the envelope fit covers the worst measured arm instead, so
    the FUSED-loss anchor here reads deliberately conservative rather than accurate.
    Contract pinned: predicted >= measured (never under), AND predicted <= 1.5x
    measured (measured ratio ~1.41 at this anchor under the envelope calibration --
    ~6% margin, own measurement, never inherited). Anchor: Qwen3-8B-4bit, seq 8192,
    gc=True, kernel, attention=flash, MEASURED total 12.7462 GB
    (`_artifacts/bench_train_step_flash/..._seq8192_ours.json`)."""
    qwen = ModelShape(vocab=151936, hidden=4096, layers=36, intermediate=12288, heads=32,
                      kv_heads=8, tied=False, quant_bits=4, quant_group=64)
    calib = load_calibration()
    cfg = TrainConfig(batch=1, seq_len=8192, dtype="bfloat16", lora_rank=8, lora_layers=36,
                      grad_checkpoint=True, impl="kernel", attention="flash")
    peak, _ = estimate_peak(qwen, cfg, calib)
    measured_bytes = 12.7462 * 1024**3
    assert peak >= measured_bytes          # never under -- the planner's core promise
    assert peak <= measured_bytes * 1.5    # bounded conservatism (measured 1.405)


def test_flash_cross_model_validation_on_llama3b() -> None:
    """Cross-model validation of the Qwen-fitted a_flash on Llama-3.2-3B-Instruct-4bit --
    a DIFFERENT heads/(hidden*layers) ratio than the identification model, so this checks
    generality, not fit. The Qwen-fitted coefficient bounds the measured Llama-3B flash
    train-step TOTAL peak (seq 8192, gc=True, kernel;
    `_artifacts/bench_train_step_flash_llama3b/..._seq8192_ours.json`, total 7.5133 GB)
    under the 0.5.0 ENVELOPE contract: one-sided (never under) with bounded
    conservatism. The 0.5.0 refit covers the worst measured loss arm, so the fused-loss
    anchor here reads conservative; the cross-model ratio is WIDER than the
    identification model's (measured 1.566 here vs 1.405 on Qwen under the envelope
    calibration -- Llama's heads/(hidden*layers) ratio amplifies the shared
    coefficient), so this test pins its OWN bound (own measurement, never inherited)."""
    llama = ModelShape(vocab=128256, hidden=3072, layers=28, intermediate=8192, heads=24,
                       kv_heads=8, tied=True, quant_bits=4, quant_group=64)
    calib = load_calibration()
    cfg = TrainConfig(batch=1, seq_len=8192, dtype="bfloat16", lora_rank=8, lora_layers=28,
                      grad_checkpoint=True, impl="kernel", attention="flash")
    peak, _ = estimate_peak(llama, cfg, calib)
    measured_bytes = 7.5133 * 1024**3
    assert peak >= measured_bytes          # never under, cross-model too
    assert peak <= measured_bytes * 1.7    # bounded conservatism (measured 1.566)


def test_envelope_flash_fit_guards_a_degenerate_point_instead_of_raw_zerodivisionerror() -> None:
    """Checkpoint C review fix (item 4, Low): the envelope generator
    (`max(flash_residual(p) / x_flash(p) for p in flash_points)`) divides by
    `x_flash(p)` per point with no per-point guard. The aggregate
    `den = sum(x_flash(p)**2) > 0` check just above only bounds the SUM of squares --
    a manifest with one degenerate (batch=0) point mixed with a well-formed one still
    passes that check (`den` is dominated by the well-formed point) but raises a raw
    `ZeroDivisionError` inside the envelope `max()` itself, deep in a generator, naming
    nothing about which point caused it. RED (before this fix): confirmed by a scratch
    run raising `ZeroDivisionError: float division by zero` for exactly the points
    below. The fix raises `PlanInputError` naming the offending point instead."""
    shape = _shape()
    calib = load_calibration()
    normal = FitPoint(
        shape=shape,
        cfg=TrainConfig(batch=1, seq_len=512, dtype="bfloat16", lora_rank=8, lora_layers=2,
                        grad_checkpoint=True, impl="kernel", attention="flash"),
        marginal_peak_bytes=1e9,
    )
    zero_batch = FitPoint(
        shape=shape,
        cfg=TrainConfig(batch=0, seq_len=1024, dtype="bfloat16", lora_rank=8, lora_layers=2,
                        grad_checkpoint=True, impl="kernel", attention="flash"),
        marginal_peak_bytes=1e9,
    )
    with pytest.raises(PlanInputError, match="seq_len=1024, batch=0"):
        fit_memory_coeffs([normal, zero_batch], calib=calib, flash_fit="envelope")


def test_flash_never_under_predicts_stock_loss_anchors() -> None:
    """Checkpoint C review fix (item 1, High): the two never-under-predict contract
    tests above (`test_predicted_peak_one_sided_and_bounded_qwen3_8b_flash`,
    `test_flash_cross_model_validation_on_llama3b`) only exercise FUSED-loss ("ours")
    anchors -- both are satisfied by the OLD (pre-0.5.0-refit)
    `attn_bytes_per_head_token_flash` coefficient (18633.411713264017) too (verified:
    that coefficient gives ratios 1.0785/1.1508 at those two anchors, both inside
    their pinned bands), so neither test would have caught a regression back to it.
    This test closes that gap with the three real STOCK-loss (mlx_lm's own default
    cross-entropy, NOT this project's fused kernel) flash artifacts instead --
    `_artifacts/bench_train_step_flash/..._seq8192_stock.json`,
    `_artifacts/calib_050/flash_n10240/..._stock.json`,
    `_artifacts/calib_050/flash_n12288/..._stock.json` -- and mirrors the SAME
    TrainConfig shape the two contract tests above use (Qwen3-8B-4bit, batch=1,
    lora_rank=8, lora_layers=36, grad_checkpoint=True, impl="kernel",
    attention="flash"; only seq_len varies per anchor).

    RED verification (arithmetic, done against the OLD coefficient directly -- it
    predates this test and is no longer installed, so there is no commit to run
    against): reconstructing `measured_total = estimate_peak(...)[1]["weights"] +
    marginal_peak_gb * 1024**3` exactly the way `scripts/fit_calibration.py`'s
    `_flash_fit_is_one_sided` does, the OLD coefficient (18633.411713264017)
    UNDER-predicts at all three anchors: seq=8192 (marginal_peak_gb=11.5427) by 2.041
    GiB, seq=10240 (marginal_peak_gb=14.3876) by 3.040 GiB, seq=12288
    (marginal_peak_gb=17.2327) by 4.039 GiB -- confirming this is a real regression
    this test would have caught. The CURRENT (committed, envelope-refit) coefficient
    clears all three with margin (own measurement: ~2.12/2.16/2.20 GiB headroom
    respectively)."""
    qwen = ModelShape(vocab=151936, hidden=4096, layers=36, intermediate=12288, heads=32,
                      kv_heads=8, tied=False, quant_bits=4, quant_group=64)
    calib = load_calibration()
    anchors = (
        (8192, 11.5427),    # _artifacts/bench_train_step_flash/..._seq8192_stock.json
        (10240, 14.3876),   # _artifacts/calib_050/flash_n10240/..._stock.json
        (12288, 17.2327),   # _artifacts/calib_050/flash_n12288/..._stock.json
    )
    for seq_len, marginal_peak_gb in anchors:
        cfg = TrainConfig(batch=1, seq_len=seq_len, dtype="bfloat16", lora_rank=8,
                          lora_layers=36, grad_checkpoint=True, impl="kernel",
                          attention="flash")
        predicted_total, components = estimate_peak(qwen, cfg, calib)
        measured_total = components["weights"] + marginal_peak_gb * 1024**3
        assert predicted_total >= measured_total, (
            f"under-predicted the stock-loss anchor at seq_len={seq_len}: "
            f"predicted={predicted_total} measured_total={measured_total}"
        )


def test_flash_never_under_predicts_fused_loss_anchors_past_8192() -> None:
    """Final-gate review fix: the release claims the refit is validated for BOTH loss
    impls to seq 12288, but the fused-loss ("ours") arm's NEW anchors had no
    never-under-predict pin -- only the old seq-8192 ours anchor (in
    `test_predicted_peak_one_sided_and_bounded_qwen3_8b_flash`) was covered, so a
    future coefficient regression hitting specifically the fused arm at long context
    (the mirror of the stock-arm gap the refit fixed) would have passed the suite.
    Anchors: `_artifacts/calib_050/flash_n10240/..._ours.json`
    (marginal_peak_gb=10.4289) and `_artifacts/calib_050/flash_n12288/..._ours.json`
    (marginal_peak_gb=12.4794); same TrainConfig shape as the sibling stock-anchor
    test. The committed envelope coefficient clears both at ratio ~1.416 (own
    measurement, final-gate review)."""
    qwen = ModelShape(vocab=151936, hidden=4096, layers=36, intermediate=12288, heads=32,
                      kv_heads=8, tied=False, quant_bits=4, quant_group=64)
    calib = load_calibration()
    anchors = (
        (10240, 10.4289),   # _artifacts/calib_050/flash_n10240/..._ours.json
        (12288, 12.4794),   # _artifacts/calib_050/flash_n12288/..._ours.json
    )
    for seq_len, marginal_peak_gb in anchors:
        cfg = TrainConfig(batch=1, seq_len=seq_len, dtype="bfloat16", lora_rank=8,
                          lora_layers=36, grad_checkpoint=True, impl="kernel",
                          attention="flash")
        predicted_total, components = estimate_peak(qwen, cfg, calib)
        measured_total = components["weights"] + marginal_peak_gb * 1024**3
        assert predicted_total >= measured_total, (
            f"under-predicted the fused-loss anchor at seq_len={seq_len}: "
            f"predicted={predicted_total} measured_total={measured_total}"
        )
