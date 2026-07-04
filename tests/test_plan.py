import mlx.core as mx
import pytest

from mlx_train_perf.core.guards import clamped_caps
from mlx_train_perf.errors import PlanInputError
from mlx_train_perf.plan.calibration import load_calibration
from mlx_train_perf.plan.estimate import ModelShape, TrainConfig, estimate_peak, plan_fit


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
    assert comp_n["loss"] == 2 * 512 * 1000 * 4 * 2
    assert comp_k["loss"] < comp_n["loss"] // 100


def test_refuses_and_suggests() -> None:
    s = _shape()
    cfg = TrainConfig(batch=4096, seq_len=8192, dtype="bfloat16", lora_rank=8,
                      lora_layers=2, grad_checkpoint=True, impl="kernel")
    report = plan_fit(s, cfg, budget_bytes=1 * 1024**3)
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


def test_weights_component_prices_only_head_at_quantized_rate() -> None:
    # Same micro shape as _shape(), but quantized (int4/gs64) -- only the head matrix
    # (V*D) is priced at the quantized rate; the rest of param_count() stays bf16.
    s = ModelShape(vocab=1000, hidden=64, layers=2, intermediate=128, heads=4,
                   kv_heads=2, tied=False, quant_bits=4, quant_group=64)
    calib = load_calibration()
    cfg = TrainConfig(batch=1, seq_len=1, dtype="bfloat16", lora_rank=8, lora_layers=2,
                      grad_checkpoint=True, impl="kernel")
    _, comp = estimate_peak(s, cfg, calib)
    p_head = s.vocab * s.hidden
    p_rest = s.param_count() - 2 * p_head  # untied: embed AND head both excluded
    expected = int(p_rest * 2 + p_head * (4 / 8 + 4 / 64))
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
    report = plan_fit(s, cfg, budget_bytes=6_000_000)
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
