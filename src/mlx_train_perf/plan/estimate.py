"""Analytic RAM-fit planner core -- pure arithmetic over ModelShape/TrainConfig/Calibration.

Formulas encode the standard Llama/Qwen3 parameter arithmetic plus this project's own
per-impl loss-layer memory shape (naive materializes the full `(N,V)` logits+softmax
pair; chunked is bounded by its fixed vocab tile, not `V`; the kernel never materializes
`(N,V)` at all -- its `custom_function` returns exactly three N-length fp32 arrays,
`nll_rows, lse, tgt`, see `core/loss.py`). Activation memory has two measured terms: a
LINEAR one (checkpoint-boundary residuals, per token*hidden*layer, gc-aware) and a
QUADRATIC one (the O(N^2) attention backward -- mlx's SDPA backward materializes one
`(N,N)` score matrix at a time -- per head*seq^2, one layer). The quadratic term dominates
peak training memory at long context and is what caps trainable context on MLX regardless
of the loss layer. Every constant that is NOT derivable from the model's own shape (the two
activation coefficients + a fixed base transient, AdamW optimizer bytes/param, a fragmentation
margin) comes from `Calibration`, loaded from a versioned, provenance-carrying data file --
never hardcoded here.

Sanity-anchored (not tolerance-pinned) against the measured gate at production shape
(n=8192, V=151936, D=4096, bf16, M1 Max 32 GB): the
kernel loss term is negligible next to naive's, and the naive loss term's own
coefficient is fit to a persisted gate artifact at that shape (see
`Calibration.naive_loss_bytes_per_nv`'s docstring for the derivation and its known
limits -- it is an empirical fit to one anchor, not a validated buffer decomposition,
and does not extrapolate linearly to smaller n). The base transient and the two
activation coefficients plus the attention coefficient are MEASURED (OLS-fit from real
train-step peaks -- see `fit_memory_coeffs` and the calibration file's provenance);
only `optimizer_bytes_per_param` (analytic AdamW) and `overhead_frac` (a fixed safety
margin) are non-measured constants.

Known limits (not modeled in 0.1.0): full fine-tuning (`lora_rank == 0`) only adds the
loss layer's own `d_w` term to the loss component -- the BASE MODEL's own weight
gradient and optimizer state (the dominant cost for full-FT, since every parameter is
trainable, not just a low-rank adapter) are not modeled anywhere in this planner.
Estimates for `lora_rank == 0` configs are therefore optimistic; `lora_rank > 0`
(LoRA/QLoRA) is the case this planner actually models end to end.
"""
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from typing import Any, Literal

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
    # Attention-backward memory model: "stock" (mlx's O(N^2) SDPA backward, the 0.1.0
    # default that still binds the trainable-context ceiling) or "flash" (the 0.2.0
    # opt-in O(N.D) flash-attention path). Typed as `str` for consistency with the
    # existing `impl`/`dtype` fields and the argparse-`choices` CLI boundary; the runtime
    # refusal in `_attention_bytes` is the guard against an out-of-set value (no silent
    # fallback). Defaults to "stock" so every pre-0.2.0 caller is unchanged.
    attention: str = "stock"


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


@dataclass(frozen=True, slots=True, kw_only=True)
class FitPoint:
    """One real measurement feeding `fit_memory_coeffs`: the model
    shape + training config that produced it, plus the MARGINAL peak this config's
    `train_step` bench condition measured (`bench/worker.py::run_train_step`'s own
    `marginal_peak_gb` field, converted to bytes -- the training LOOP's own
    incremental memory above whatever was already resident before it started, NOT
    `total_peak_gb`, which also counts the already-resident weights)."""
    shape: ModelShape
    cfg: TrainConfig
    marginal_peak_bytes: float


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
    """Linear (checkpoint) activation memory: bytes per (token * hidden * layer), gc-aware.
    grad_checkpoint=True stores only layer-boundary residuals; =False stores every layer's
    forward activations (~45x more). The O(N^2) attention-backward term is SEPARATE -- see
    `_attention_bytes`."""
    a_lin = (calib.act_bytes_per_token_hidden_layer_ckpt if cfg.grad_checkpoint
             else calib.act_bytes_per_token_hidden_layer_full)
    return int(a_lin * cfg.batch * cfg.seq_len * shape.hidden * shape.layers)


def _flash_saved_state_bytes(cfg: TrainConfig, shape: ModelShape) -> int:
    """Analytic O(N.D) flash-attention saved backward state: the attention output `O`
    (batch*seq*hidden in the compute dtype) plus the logsumexp `L` (batch*heads*seq, always
    fp32). This is what a recompute-based flash backward keeps instead of the O(N^2) score
    matrix, and it is computed exactly from shape -- the FITTED `a_flash` coefficient
    captures only the residual live backward transient on top of it. Shared by
    `_attention_bytes` (the flash branch) and `fit_memory_coeffs` (the residual it
    subtracts) so both use one definition."""
    o_saved = cfg.batch * cfg.seq_len * shape.hidden * _dtype_bytes(cfg.dtype)
    l_saved = cfg.batch * shape.heads * cfg.seq_len * 4
    return o_saved + l_saved


def _attention_bytes(cfg: TrainConfig, shape: ModelShape, calib: Calibration) -> int:
    """Attention-backward memory, branching on `cfg.attention`.

    STOCK (default): quadratic, bytes per (head * seq^2), ONE layer for BOTH gc settings --
    mlx's O(N^2) SDPA backward materializes one (N,N) score matrix at a time as the backward
    walks the layer stack. This term dominates peak training memory at long context and caps
    the trainable context length regardless of the loss layer.

    FLASH (0.2.0 opt-in): the analytic O(N.D) saved state (`_flash_saved_state_bytes`) plus
    a fitted LINEAR live-transient term `a_flash * batch * heads * seq`. Driver form settled
    from the T13 single-op scaling (linear growth; the split-regime steepening folded into
    the coefficient, over-predict-safe). bf16-calibrated (dtype folded into a_flash).
    Validated against measured Qwen3-8B-4bit anchors up to seq 12288 (0.5.0 refit, both
    loss impls, envelope fit -- one coefficient covers the worst measured arm, so the
    fused-loss arm reads deliberately conservative); beyond 12288 the fit extrapolates."""
    if cfg.attention == "stock":
        return int(calib.attn_bytes_per_head_token2 * cfg.batch * shape.heads * cfg.seq_len**2)
    if cfg.attention == "flash":
        return int(_flash_saved_state_bytes(cfg, shape)
                   + calib.attn_bytes_per_head_token_flash * cfg.batch * shape.heads * cfg.seq_len)
    raise PlanInputError(
        f"unknown attention {cfg.attention!r}; expected 'stock' or 'flash'"
    )


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
        "base": int(calib.base_transient_bytes),
        "lora": _lora_bytes(cfg, shape),
        "optimizer": _optimizer_bytes(cfg, shape, calib),
        "activations": _activation_bytes(cfg, shape, calib),
        "attention": _attention_bytes(cfg, shape, calib),
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


def _guard_envelope_x_flash_positive(
    flash_points: list[FitPoint], x_flash: Callable[[FitPoint], float]
) -> None:
    """`fit_memory_coeffs`'s envelope flash fit divides by `x_flash(p)` per point --
    each point must individually have `x_flash(p) > 0` (e.g. `batch=0` breaks this).
    The aggregate `sum(x_flash(p)**2) > 0` guard in the caller only bounds the SUM of
    squares, so a single degenerate point mixed with well-formed ones still passes
    that check but would raise a raw, unnamed `ZeroDivisionError` deep inside the
    envelope `max()`; this raises `PlanInputError` naming the offending point instead."""
    for p in flash_points:
        if x_flash(p) <= 0:
            raise PlanInputError(
                "fit_memory_coeffs cannot compute the envelope flash fit: "
                f"FitPoint (seq_len={p.cfg.seq_len}, batch={p.cfg.batch}) has "
                f"x_flash=batch*heads*seq_len={x_flash(p)!r} <= 0"
            )


def fit_memory_coeffs(
    points: list[FitPoint], *, calib: Calibration,
    flash_fit: Literal["ols", "envelope"] = "ols",
) -> dict[str, float]:
    """Fit the calibration memory coefficients from kernel-impl `train_step` FitPoints.

    Returns a dict with keys `base_transient_bytes`,
    `act_bytes_per_token_hidden_layer_ckpt`, `act_bytes_per_token_hidden_layer_full`,
    `attn_bytes_per_head_token2`.

    Each point's MARGINAL peak (kernel impl) is modeled as:

        marginal ~= base
                  + a_lin[gc] * (batch * seq_len * hidden * layers)     # linear activations
                  + a_quad    * (batch * heads * seq_len^2)             # one-layer attention bwd
                  + analytic(lora + optimizer + loss)

    The analytic small terms (`_lora_bytes` + `_optimizer_bytes` + `_loss_bytes`) are
    KNOWN and subtracted from each measured marginal BEFORE the fit, so only
    base + the activation/attention coefficients are solved for. `a_quad` is one-layer
    and gc-INDEPENDENT (mlx's O(N^2) SDPA backward materializes one (N,N) at a time);
    `grad_checkpoint` selects the linear coefficient (`_ckpt` vs `_full`).

    Points are partitioned by `cfg.attention`. STOCK points fit the (base, a_lin, a_quad)
    stock model; FLASH points fit `a_flash` alone.

    STOCK, grad_checkpoint=True points fit (base, a_lin_ckpt, a_quad) by ordinary least
    squares -- needs a FULL-RANK (1, x_lin, x_quad) design to separate the constant, linear
    and quadratic terms: with batch held fixed (the standard calibration regime) that means
    >= 3 distinct seq_len; batch variation can restore rank at 2 distinct seq_len. A
    rank-deficient design raises PlanInputError (lstsq would otherwise silently return
    minimum-norm garbage). STOCK grad_checkpoint=False points then fit a_lin_full alone
    (base and a_quad held from the gc=True fit); with no gc=False points the existing
    `calib` value is kept. With NO stock points at all (a flash-only calibration manifest),
    every stock coefficient is kept unchanged from `calib` -- a flash refit never disturbs
    the stock model.

    FLASH points fit `a_flash` (`attn_bytes_per_head_token_flash`) by a 1-variable
    through-origin residual OLS: `x_flash = batch*heads*seq` is an EXACT scalar multiple of
    `x_lin` at fixed model shape (ratio heads/(hidden*layers)), so a joint (base, a_lin,
    a_flash) fit is rank-deficient at ANY number of same-model points -- instead base and
    a_lin are held FIXED from the passed-in stock `calib`, the analytic small terms and the
    O(N.D) saved state (`_flash_saved_state_bytes`) are subtracted, and a_flash absorbs the
    residual live transient (`num/den`, den = sum x_flash^2 > 0 needs >= 1 flash point).
    With no flash points the existing `calib` value is kept. The Llama-3.2-3B flash point
    (a different heads/(hidden*layers) ratio) is used for cross-model VALIDATION, not
    identification.

    `flash_fit` selects HOW the flash coefficient above is computed from the same
    `flash_residual`/`x_flash` closures: `"ols"` (default) is the shipped 1-variable
    through-origin least-squares fit described above -- minimizing the summed squared
    residual across all flash points, which can UNDER-predict an individual anchor
    whenever the true per-point ratio (`flash_residual(p) / x_flash(p)`) varies across
    points (e.g. a short/long-context split-regime steepening). `"envelope"` instead
    takes `a_flash = max(flash_residual(p) / x_flash(p) for p in flash_points)` -- the
    single largest per-point ratio, reusing the SAME closures (no duplicated residual
    model). Envelope is a strictly conservative (over-predict-safe) fallback for when
    OLS's least-squares average violates the planner's own never-under-predict
    invariant at one or more anchors; the two fits are identical whenever every flash
    point implies the same ratio. Choosing between them is the calling script's job
    (`scripts/fit_calibration.py`), not this function's -- `fit_memory_coeffs` only
    computes whichever one it's asked for.

    Every point must be `impl="kernel"` -- `"chunked"`/`"naive"` measure a materially
    different loss-layer memory shape that would contaminate the fit.
    """
    import numpy as np  # noqa: PLC0415 -- transitive dep, calibration-time only

    for p in points:
        if p.cfg.impl != "kernel":
            raise PlanInputError(
                f"fit_memory_coeffs only accepts impl='kernel' FitPoints (got "
                f"impl={p.cfg.impl!r}) -- 'chunked'/'naive' measure a different loss-layer "
                "memory shape that would contaminate the fit"
            )

    def residual(p: FitPoint) -> float:
        analytic = (_lora_bytes(p.cfg, p.shape) + _optimizer_bytes(p.cfg, p.shape, calib)
                    + _loss_bytes(p.cfg, p.shape, calib))
        return p.marginal_peak_bytes - analytic

    def x_lin(p: FitPoint) -> float:
        return float(p.cfg.batch * p.cfg.seq_len * p.shape.hidden * p.shape.layers)

    def x_quad(p: FitPoint) -> float:
        return float(p.cfg.batch * p.shape.heads * p.cfg.seq_len ** 2)

    def x_flash(p: FitPoint) -> float:
        return float(p.cfg.batch * p.shape.heads * p.cfg.seq_len)

    stock_points = [p for p in points if p.cfg.attention == "stock"]
    flash_points = [p for p in points if p.cfg.attention == "flash"]

    if stock_points:
        ckpt = [p for p in stock_points if p.cfg.grad_checkpoint]
        full = [p for p in stock_points if not p.cfg.grad_checkpoint]
        a_mat = np.array([[1.0, x_lin(p), x_quad(p)] for p in ckpt],
                         dtype=float).reshape(-1, 3)
        # Rank check, not a point/seq_len head-count: `np.linalg.lstsq` never raises on a
        # rank-deficient design -- it returns the minimum-norm solution, i.e. silently-wrong
        # coefficients (review reproduced a NEGATIVE a_quad from 3 points at 2 distinct
        # seq_len, batch fixed). Full rank of the actual (1, x_lin, x_quad) matrix is the
        # exact identifiability condition: batch-fixed runs need >= 3 distinct seq_len,
        # while batch variation can restore rank at 2 (both cases tested).
        if np.linalg.matrix_rank(a_mat) < 3:
            raise PlanInputError(
                "fit_memory_coeffs cannot separate (base, linear, quadratic): the gc=True "
                "design matrix is rank-deficient. Supply at least 3 grad_checkpoint=True "
                "FitPoints whose (1, x_lin, x_quad) rows are independent -- with batch held "
                "fixed that means at least 3 distinct seq_len values (got "
                f"{len(ckpt)} gc=True stock point(s), "
                f"{len({p.cfg.seq_len for p in ckpt})} distinct seq_len)"
            )
        b_vec = np.array([residual(p) for p in ckpt], dtype=float)
        solution = np.linalg.lstsq(a_mat, b_vec, rcond=None)[0]
        base = float(solution[0])
        a_lin_ckpt = float(solution[1])
        a_quad = float(solution[2])
        if full:
            # a_lin_full from: residual - base - a_quad*x_quad = a_lin_full * x_lin (1-var OLS)
            num = sum((residual(p) - base - a_quad * x_quad(p)) * x_lin(p) for p in full)
            den = sum(x_lin(p) ** 2 for p in full)
            a_lin_full = num / den if den else calib.act_bytes_per_token_hidden_layer_full
        else:
            a_lin_full = calib.act_bytes_per_token_hidden_layer_full
    else:
        base = float(calib.base_transient_bytes)
        a_lin_ckpt = calib.act_bytes_per_token_hidden_layer_ckpt
        a_lin_full = calib.act_bytes_per_token_hidden_layer_full
        a_quad = calib.attn_bytes_per_head_token2

    if flash_points:
        # a_flash by 1-variable through-origin residual OLS, holding base/a_lin FIXED from
        # the stock `calib` (the x_flash driver is collinear with x_lin at fixed shape, so
        # they cannot be jointly identified). The analytic small terms and the O(N.D) saved
        # state are subtracted; a_flash captures the residual live backward transient.
        def flash_residual(p: FitPoint) -> float:
            a_lin = (calib.act_bytes_per_token_hidden_layer_ckpt if p.cfg.grad_checkpoint
                     else calib.act_bytes_per_token_hidden_layer_full)
            return (residual(p) - calib.base_transient_bytes - a_lin * x_lin(p)
                    - _flash_saved_state_bytes(p.cfg, p.shape))

        den = sum(x_flash(p) ** 2 for p in flash_points)
        if den <= 0:
            raise PlanInputError(
                "fit_memory_coeffs cannot identify a_flash: the flash design has "
                "sum(x_flash^2) == 0 (need >= 1 flash FitPoint with batch*heads*seq > 0)"
            )
        if flash_fit == "envelope":
            # The single largest per-point ratio -- reuses flash_residual/x_flash
            # rather than re-deriving the residual model (drift hazard). The aggregate
            # `den > 0` check above only bounds the SUM of squares, so a manifest with
            # one degenerate point (batch=0) mixed with other, well-formed points still
            # passes that check but would raise a raw ZeroDivisionError here --
            # `_guard_envelope_x_flash_positive` names the offending point instead.
            _guard_envelope_x_flash_positive(flash_points, x_flash)
            a_flash = max(flash_residual(p) / x_flash(p) for p in flash_points)
        else:
            num = sum(flash_residual(p) * x_flash(p) for p in flash_points)
            a_flash = num / den
    else:
        a_flash = calib.attn_bytes_per_head_token_flash

    return {
        "base_transient_bytes": base,
        "act_bytes_per_token_hidden_layer_ckpt": a_lin_ckpt,
        "act_bytes_per_token_hidden_layer_full": float(a_lin_full),
        "attn_bytes_per_head_token2": a_quad,
        "attn_bytes_per_head_token_flash": float(a_flash),
    }
