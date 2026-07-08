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


def test_fit_memory_coeffs_rejects_non_kernel_impl() -> None:
    shape = _shape()
    calib = load_calibration()
    cfg = TrainConfig(batch=1, seq_len=512, dtype="bfloat16", lora_rank=8, lora_layers=2,
                      grad_checkpoint=True, impl="naive")
    points = [FitPoint(shape=shape, cfg=cfg, marginal_peak_bytes=float(i)) for i in (1, 2, 3)]
    with pytest.raises(PlanInputError, match="kernel"):
        fit_memory_coeffs(points, calib=calib)
