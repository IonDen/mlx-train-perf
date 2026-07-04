"""Analytic RAM-fit planner core -- pure arithmetic over ModelShape/TrainConfig/Calibration.

Formulas encode the standard Llama/Qwen3 parameter arithmetic plus this project's own
per-impl loss-layer memory shape (naive materializes the full `(N,V)` logits+softmax
pair; chunked is bounded by its fixed vocab tile, not `V`; the kernel never materializes
`(N,V)` at all -- its `custom_function` returns exactly three N-length fp32 arrays,
`nll_rows, lse, tgt`, see `core/loss.py`). Every constant that is NOT derivable from the
model's own shape (activation footprint per token/layer under grad-checkpointing, AdamW
optimizer bytes/param, a fixed overhead margin for unmodeled fragmentation) comes from
`Calibration`, loaded from a versioned, provenance-carrying data file -- never hardcoded
here.

Sanity-anchored (not tolerance-pinned) against the mlx-train-perf-spike gate
measurements at production shape (n=8192, V=151936, D=4096, bf16, M1 Max 32 GB): the
kernel loss term is negligible next to naive's, and the naive loss term's own
coefficient is fit to a persisted gate artifact at that shape (see
`Calibration.naive_loss_bytes_per_nv`'s docstring for the derivation and its known
limits -- it is an empirical fit to one anchor, not a validated buffer decomposition,
and does not extrapolate linearly to smaller n). The remaining calibration constants
are honest analytic placeholders a later, real-measurement task is expected to replace.

Known limits (not modeled in 0.1.0): full fine-tuning (`lora_rank == 0`) only adds the
loss layer's own `d_w` term to the loss component -- the BASE MODEL's own weight
gradient and optimizer state (the dominant cost for full-FT, since every parameter is
trainable, not just a low-rank adapter) are not modeled anywhere in this planner.
Estimates for `lora_rank == 0` configs are therefore optimistic; `lora_rank > 0`
(LoRA/QLoRA) is the case this planner actually models end to end.
"""
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any

import mlx.core as mx

from mlx_train_perf.core.guards import clamped_caps
from mlx_train_perf.errors import PlanInputError
from mlx_train_perf.plan.calibration import Calibration, load_calibration

_DTYPE_BYTES: dict[str, int] = {"float32": 4, "bfloat16": 2, "float16": 2}

# Mirrors core.loss._KERNEL_TILE / linear_cross_entropy's chunk_size=None default -- the
# chunked impl's memory is bounded by this fixed tile, not by V.
_CHUNK_TILE = 8192

_LORA_MODULES = 2   # q_proj, v_proj -- see _lora_bytes's docstring for what this assumes
_LORA_MATRICES = 2  # A, B factor matrices per adapted module


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelShape:
    vocab: int
    hidden: int
    layers: int
    intermediate: int
    heads: int
    kv_heads: int
    tied: bool
    quant_bits: int | None
    quant_group: int | None

    def param_count(self) -> int:
        """Standard Llama/Qwen3 param arithmetic: embed `V·D` (+ untied head `V·D`), per
        layer attn `D·D·(2 + 2·kv/heads)` + mlp `3·D·I` + norms `2·D`, final norm `D`."""
        d, v, i, layers = self.hidden, self.vocab, self.intermediate, self.layers
        embed = v * d
        head = 0 if self.tied else v * d
        # 2*D*D*(heads+kv_heads)/heads == D*D*(2 + 2*kv_heads/heads), reordered to one
        # floor division so an integer GQA ratio never round-trips through float.
        attn = 2 * d * d * (self.heads + self.kv_heads) // self.heads
        mlp = 3 * d * i
        norms = 2 * d
        return embed + head + layers * (attn + mlp + norms) + d

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ModelShape":
        """Reads a HF `config.json` dict (Llama/Qwen3 keys). `num_key_value_heads`
        defaults to `num_attention_heads` (HF's own plain-MHA fallback);
        `tie_word_embeddings` defaults to False (HF's own default). Quantization
        metadata comes from the `quantization` block `mlx_lm.convert` writes
        (`group_size`/`bits`) -- the same schema `adapters/mlx_lm.py`'s verified
        quantized-module notes read from a converted model's config."""
        heads = int(config["num_attention_heads"])
        kv_heads = int(config.get("num_key_value_heads", heads))
        quant = config.get("quantization")
        quant_bits = int(quant["bits"]) if quant else None
        quant_group = int(quant["group_size"]) if quant else None
        return cls(
            vocab=int(config["vocab_size"]),
            hidden=int(config["hidden_size"]),
            layers=int(config["num_hidden_layers"]),
            intermediate=int(config["intermediate_size"]),
            heads=heads,
            kv_heads=kv_heads,
            tied=bool(config.get("tie_word_embeddings", False)),
            quant_bits=quant_bits,
            quant_group=quant_group,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainConfig:
    batch: int
    seq_len: int
    dtype: str
    lora_rank: int
    lora_layers: int
    grad_checkpoint: bool
    impl: str


@dataclass(frozen=True, slots=True, kw_only=True)
class FitReport:
    fits: bool
    predicted_peak_bytes: int
    budget_bytes: int
    headroom_bytes: int
    components: dict[str, int]
    suggestion: TrainConfig | None
    is_estimate: bool
    provenance: dict[str, str]


def _dtype_bytes(dtype: str) -> int:
    try:
        return _DTYPE_BYTES[dtype]
    except KeyError:
        raise PlanInputError(
            f"unknown dtype {dtype!r}; expected one of {sorted(_DTYPE_BYTES)}"
        ) from None


def _weights_bytes(shape: ModelShape, dtype_size: int) -> int:
    """`P_total x dtype_size` for a dense model. For a quantized model (`quant_bits`/
    `quant_group` set), prices the WHOLE model at the quantized rate (`bits/8 +
    4/group`; the `4` is one bf16 scale + one bf16 bias per group) -- not just the head.

    Real mlx-community 4-bit checkpoints quantize the whole model via `mlx_lm.convert`'s
    uniform `nn.quantize(model, group_size, bits)`, not just the output projection --
    pricing only the head at the quantized rate would leave the (dominant) body layers
    priced at `dtype_size`, a phantom cost that, at flagship scale (~8B params, int4),
    adds roughly 11 GB the checkpoint never actually carries, and would cause the
    planner to refuse configs that genuinely fit. This is an approximation: a handful of
    per-parameter exemptions (norms, biases) are real but negligible next to a
    multi-billion-parameter body, so they're ignored rather than separately modeled. A
    later measured-vs-predicted gate validates this against a real quantized
    checkpoint."""
    p_total = shape.param_count()
    if shape.quant_bits is None or shape.quant_group is None:
        return p_total * dtype_size
    return int(p_total * (shape.quant_bits / 8 + 4 / shape.quant_group))


def _lora_param_count(cfg: TrainConfig, shape: ModelShape) -> int:
    return cfg.lora_layers * _LORA_MODULES * _LORA_MATRICES * cfg.lora_rank * shape.hidden


def _lora_bytes(cfg: TrainConfig, shape: ModelShape) -> int:
    """`lora_layers x 2 modules(q,v) x 2 (A,B) x rank x D x 2 bytes` (bf16). 0.1.0
    assumes the conventional LoRA target set -- query + value projections, the original
    LoRA paper's default and the common default across LoRA tooling. Note: mlx-lm's OWN
    zero-config `linear_to_lora_layers` default (no explicit `lora_parameters.keys`)
    instead targets every Linear/Embedding submodule per block (verified against the
    installed mlx-lm==0.31.3 `tuner/utils.py`) -- this planner does not model that
    broader default; a caller training with a non-default `keys` config gets a less
    accurate estimate from this term."""
    return _lora_param_count(cfg, shape) * 2


def _optimizer_bytes(cfg: TrainConfig, shape: ModelShape, calib: Calibration) -> int:
    return int(calib.optimizer_bytes_per_param * _lora_param_count(cfg, shape))


def _activation_bytes(cfg: TrainConfig, shape: ModelShape, calib: Calibration) -> int:
    return int(calib.act_bytes_per_token_layer * cfg.batch * cfg.seq_len * shape.layers)


def _head_trainable(cfg: TrainConfig) -> bool:
    """Planner rule: `lora_rank == 0` means full fine-tuning (head trainable, the `d_w`
    term below applies); `lora_rank > 0` -- LoRA/QLoRA, the planner's normal case --
    freezes the head, so the term is off."""
    return cfg.lora_rank == 0


def _loss_bytes(cfg: TrainConfig, shape: ModelShape, calib: Calibration) -> int:
    n = cfg.batch * cfg.seq_len
    if cfg.impl == "naive":
        # naive_loss_bytes_per_nv is an empirical fit to a single production-shape
        # anchor (see Calibration.naive_loss_bytes_per_nv's docstring) -- accurate at
        # that shape but known not to extrapolate linearly to smaller n, where it
        # over-predicts (the conservative, safe direction for this planner).
        base = int(calib.naive_loss_bytes_per_nv * n * shape.vocab)
    elif cfg.impl == "chunked":
        base = n * _CHUNK_TILE * 4 * 3
    elif cfg.impl == "kernel":
        base = 3 * n * 4
    else:
        raise PlanInputError(
            f"unknown impl {cfg.impl!r}; expected 'naive', 'chunked', or 'kernel'"
        )
    if _head_trainable(cfg):
        # The d_w term: one (V,D) fp32 gradient buffer, concatenate double-up.
        base += shape.vocab * shape.hidden * 4 * 2
    return base


def estimate_peak(
    shape: ModelShape, cfg: TrainConfig, calib: Calibration
) -> tuple[int, dict[str, int]]:
    """Pure: predicted peak bytes plus a component breakdown. No I/O, no device query --
    `calib` is passed in rather than loaded here so this function has no hidden state."""
    dtype_size = _dtype_bytes(cfg.dtype)
    components = {
        "weights": _weights_bytes(shape, dtype_size),
        "lora": _lora_bytes(cfg, shape),
        "optimizer": _optimizer_bytes(cfg, shape, calib),
        "activations": _activation_bytes(cfg, shape, calib),
        "loss": _loss_bytes(cfg, shape, calib),
    }
    subtotal = sum(components.values())
    predicted_peak = int(subtotal * (1 + calib.overhead_frac))
    return predicted_peak, components


def _suggestion_candidates(cfg: TrainConfig) -> Iterator[TrainConfig]:
    """Deterministic search order: halve batch (floor 1), then step seq_len down by
    1024 (floor 1024). The caller returns the first candidate whose predicted peak
    fits the budget."""
    batch = cfg.batch
    while batch > 1:
        batch = max(1, batch // 2)
        yield replace(cfg, batch=batch)
    seq_len = cfg.seq_len
    while seq_len > 1024:
        seq_len = max(1024, seq_len - 1024)
        yield replace(cfg, batch=1, seq_len=seq_len)


def _find_fit(
    shape: ModelShape, cfg: TrainConfig, calib: Calibration, budget_bytes: int
) -> TrainConfig | None:
    for candidate in _suggestion_candidates(cfg):
        peak, _ = estimate_peak(shape, candidate, calib)
        if peak <= budget_bytes:
            return candidate
    return None


def plan_fit(
    shape: ModelShape, cfg: TrainConfig, *, budget_bytes: int | None = None
) -> FitReport:
    """`budget_bytes` defaults to THIS project's own guarded wired cap
    (`core.guards.clamped_caps`, device-clamped) -- the conservative budget our own
    benches run under. This is deliberately NOT what stock `mlx_lm.tuner.trainer.train()`
    enforces: it calls `mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])`
    at entry (verified against the installed mlx-lm==0.31.3 source,
    `tuner/trainer.py:229-230`), overriding any stricter cap to the raw device max. A
    caller planning specifically for the stock trainer's own path should pass
    `budget_bytes=int(mx.device_info()["max_recommended_working_set_size"])` explicitly
    for an honest stock-trainer budget rather than relying on this stricter default.
    """
    calib = load_calibration()
    if budget_bytes is None:
        dev_max = int(mx.device_info()["max_recommended_working_set_size"])
        budget_bytes, _ = clamped_caps(dev_max)
    peak, components = estimate_peak(shape, cfg, calib)
    fits = peak <= budget_bytes
    suggestion = None if fits else _find_fit(shape, cfg, calib, budget_bytes)
    return FitReport(
        fits=fits,
        predicted_peak_bytes=peak,
        budget_bytes=budget_bytes,
        headroom_bytes=budget_bytes - peak,
        components=components,
        suggestion=suggestion,
        is_estimate=True,
        provenance=calib.provenance,
    )
