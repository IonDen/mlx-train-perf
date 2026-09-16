"""Masked ragged-batch parity: B=4 right-padded rows at the representative
0.8B geometry (Hk=Hv=16, Dk=Dv=128), forward and backward, fp32 and bf16.

Real fine-tuning batches are ragged: rows share a max sequence length but
carry different real content lengths, right-padded to the batch max and
masked. This file pins that the chunk-parallel op's ``mask=`` handling
produces, per row, the same forward output and the same restricted-to-row
backward gradient as running the sequential oracle on that row's own valid
(unpadded) slice -- and that padded positions never leak gradient signal.
"""

import mlx.core as mx
import pytest
from conftest import needs_long_context_room

pytest.importorskip("mlx_lm")
from recurrent_parity_cases import QWEN35_08B_GEOM, build_case, metrics
from test_recurrent_fwd_parity import assert_under_pin

from mlx_train_perf.recurrent.ops import chunked_gated_delta
from mlx_train_perf.recurrent.reference import sequential_gated_delta

SEED = 0
CHUNK_SIZE = 64
T_MAX = 512
LENGTHS = (512, 384, 256, 128)  # T, 3T/4, T/2, T/4 -- right-padded to T_MAX
DTYPES = (("fp32", mx.float32), ("bf16", mx.bfloat16))

# T_MAX=512 is an exact multiple of CHUNK_SIZE=64, so the driver's own structural
# pad (pad_len = (-T) % C) never fires above -- the mask-substitution and
# pad-substitution identity steps are never exercised TOGETHER. T=500 is not a
# multiple of 64 (pad_len = (-500) % 64 = 12), so it exercises that interaction;
# pad_len depends on T, not on the per-row mask lengths below.
T_500 = 500
LENGTHS_500 = (500, 375, 250, 125)  # T, 3T/4, T/2, T/4 -- right-padded to T_500


def _ragged_mask(t_max: int = T_MAX, lengths: tuple[int, ...] = LENGTHS) -> mx.array:
    lengths_arr = mx.array(lengths)
    positions = mx.arange(t_max)
    mask = positions[None, :] < lengths_arr[:, None]
    mx.eval(mask)
    return mask


def _broadcast_row_mask(x: mx.array, base_mask: mx.array) -> mx.array:
    """Broadcast a [B, T] bool mask onto x's leading [B, T, ...] shape."""
    m = base_mask
    for _ in range(x.ndim - base_mask.ndim):
        m = m[..., None]
    return m


def _masked_batch_loss(mask: mx.array, chunk_size: int):
    def loss(q, k, v, g, beta):
        y, s = chunked_gated_delta(q, k, v, g, beta, mask=mask, chunk_size=chunk_size)
        y_valid = y * _broadcast_row_mask(y, mask).astype(mx.float32)
        return (y_valid.astype(mx.float32) ** 2).sum() + (s.astype(mx.float32) ** 2).sum()

    return loss


def _row_oracle_loss():
    def loss(q, k, v, g, beta):
        y, s = sequential_gated_delta()(q, k, v, g, beta, None)
        return (y.astype(mx.float32) ** 2).sum() + (s.astype(mx.float32) ** 2).sum()

    return loss


# Measured in this tree (2026-08-30) on the B=4 ragged batch (lengths 512/
# 384/256/128, chunk_size=64, QWEN35_08B_GEOM), comparing the masked-batch
# forward per row against the sequential oracle on that row's own valid
# slice, and taking the per-tensor (y, state) worst INDEPENDENTLY per metric
# across the 4 rows -- the row with the worst max_abs is not always the row
# with the worst rel_fro (fp32 y: max_abs worst at row0, rel_fro worst at
# row1; bf16 state: max_abs worst at row1, rel_fro also row1 but a
# different row than the y worst). Pinned at measured-worst x ~2.
FWD_MASKED_PINS: dict[str, dict[str, tuple[float, float]]] = {
    # y: max_abs=7.629395e-06 (row0), rel_fro=7.361760e-07 (row1). fp32
    #   accumulation floor ~1e-6 -- matches this tree's single-row
    #   "representative" FWD_PINS entry (7.152557e-06/7.788288e-07) almost
    #   exactly, as expected for the same T=512 geometry.
    # state: max_abs=9.238720e-07 (row2), rel_fro=9.752273e-07 (row0) --
    #   same fp32 floor; note the worst-row differs between y and state too.
    "fp32": {"y": (1.5e-5, 1.5e-6), "state": (2e-6, 2e-6)},
    # y: max_abs=3.906250e-03 (all 4 rows, one bf16 ULP at this magnitude --
    #   matches the tree's single-row "bf16_representative" FWD_PINS y
    #   max_abs identically), rel_fro=4.466737e-05 (row2). bf16 resolution
    #   2^-8 ~= 3.9e-3 -- max_abs sits AT the ceiling by construction, not a
    #   surprise (same mechanism as the single-row bf16_repr_T512 case).
    # state: max_abs=8.344650e-07 (row1), rel_fro=1.095829e-06 (row1) --
    #   state stays fp32 across chunks even in the bf16 arm, so it tracks
    #   the fp32 floor, matching the single-row bf16_representative pin.
    "bf16": {"y": (8e-3, 1e-4), "state": (2e-6, 2.5e-6)},
}

# Separate pin for the STRUCTURAL-PADDING regime (T=500, pad_len=(-500)%64=12) --
# NOT a substitute for FWD_MASKED_PINS["fp32"] above, which stays unmodified and
# keeps governing every T_MAX=512 chunk-aligned (pad_len=0) case.
#
# A controller-run three-arm probe (2026-08-30), same geometry (QWEN35_08B_GEOM),
# chunk_size=64, isolated per-row STATE rel_fro across:
#   T=512 masked, pad_len=0 (chunk-aligned):        8.6e-7 - 9.8e-7
#     (n=512:9.75e-7 n=384:9.58e-7 n=256:9.20e-7 n=128:8.59e-7)
#   T=500 all-valid, pad_len=12, NO masking (padded-only): 1.21e-6 - 2.42e-6
#     (n=500:1.945e-6 n=375:1.988e-6 n=250:1.452e-6 n=125:2.423e-6)
#   T=500 masked, pad_len=12 (padded+masked):        1.85e-6 - 2.09e-6
#     (n=500:1.945e-6 n=375:1.851e-6 n=250:2.091e-6 n=125:1.209e-6)
# The masked arm's worst (2.091e-6) sits INSIDE the unmasked padded-only arm's
# spread (up to 2.423e-6) -- masking contributes NOTHING on top of the
# structural-padding effect. The ~2x elevation over the chunk-aligned floor is a
# property of padding itself (T not a multiple of chunk_size), not of the
# mask/pad interaction the T=500 case was added to probe -- so the two identity-
# step mechanisms compose cleanly, and the elevation gets its own honest pin
# instead of being forced under the chunk-aligned one. fp32 accumulation floor
# ~1e-6 -- the padded worst (2.423e-6) is a real, if small, excursion above that
# floor (~2.4x), still fp32-floor-class, but a genuinely different regime.
# Pinned at measured-worst (2.423e-6) x ~2 = 5e-6. y and state max_abs are
# untouched by padding (measured T=500 masked: y max_abs=6.914139e-06, y
# rel_fro=7.446606e-07, state max_abs=1.788139e-06 -- all comfortably inside the
# existing chunk-aligned FWD_MASKED_PINS["fp32"] values), so only the elevated
# state rel_fro axis gets a new number; y and state max_abs are reused as-is.
FWD_MASKED_PADDED_PINS: dict[str, tuple[float, float]] = {
    "y": FWD_MASKED_PINS["fp32"]["y"],
    "state": (FWD_MASKED_PINS["fp32"]["state"][0], 5e-6),
}

# Measured in this tree (2026-08-30) by running mx.grad of the masked-batch
# loss (loss over the full ragged batch, y masked to valid positions before
# squaring) and, per row, mx.grad of the oracle's own loss on that row's
# unpadded valid slice -- then comparing the batch gradient restricted to
# the row (grads_batch[b, :n]) against the row-oracle gradient. Per
# component, worst tracked INDEPENDENTLY per metric across the 4 rows (same
# convention as FWD_MASKED_PINS and this tree's existing BWD_PINS). dg is
# again not the flat-worst driver here the way it is in BWD_PINS (dk's
# max_abs runs comparably or larger) -- with T fixed at 512 for every row's
# comparison (row length only shrinks the oracle's own T, not the batch's),
# the amplification pattern differs from the T-varying BWD_CASES regimes.
BWD_MASKED_PINS: dict[str, dict[str, tuple[float, float]]] = {
    # dq=1.430511e-04/9.634055e-07 (row0/row1), dk=9.460449e-04/9.670462e-07
    #   (row0/row1), dv=2.670288e-05/1.060221e-06 (row0/row1),
    #   dg=8.773804e-04/2.831093e-06 (row0/row1), dbeta=7.781982e-04/
    #   9.254613e-07 (row0/row1). fp32 accumulation floor ~1e-6 for every
    #   rel_fro -- matches this tree's single-row "representative" BWD_PINS
    #   block (dq=1.487732e-04/1.034223e-06 etc.) closely, as expected for
    #   the same T=512 geometry; row0 here (the one truly-unpadded row) is
    #   the closest analog to that single-row case.
    "fp32": {
        "dq": (3e-4, 2e-6), "dk": (2e-3, 2e-6), "dv": (6e-5, 2.5e-6),
        "dg": (2e-3, 6e-6), "dbeta": (1.6e-3, 2e-6),
    },
    # dq=6.250000e-02/7.328660e-05 (row1/row2), dk=1.000000e+00/2.794690e-03
    #   (row0/row3), dv=7.812500e-03/6.567342e-05 (row0/row2),
    #   dg=6.250000e-02/6.567109e-05 (row2/row2), dbeta=6.250000e-02/
    #   3.125316e-05 (row1/row1). bf16 resolution 2^-8 ~= 3.9e-3 -- dk's
    #   rel_fro (2.8e-3) sits just under that ceiling, matching this tree's
    #   single-row "bf16_representative" BWD_PINS entry (dk rel_fro
    #   2.758466e-03) almost exactly; every other component's rel_fro is an
    #   order of magnitude under the ceiling.
    "bf16": {
        "dq": (0.13, 1.5e-4), "dk": (2.0, 6e-3), "dv": (1.6e-2, 1.4e-4),
        "dg": (0.13, 1.4e-4), "dbeta": (0.13, 6.5e-5),
    },
}


@needs_long_context_room
@pytest.mark.parametrize(("dtype_name", "dtype"), DTYPES)
def test_masked_batch_fwd_matches_per_row_oracle(dtype_name, dtype):
    # Catches: mask=None-shaped defaults leaking padded content into a row's
    # valid-position output, or a wrong per-row state after a masked
    # (identity-step) tail.
    d = build_case(T=T_MAX, chunk_size=CHUNK_SIZE, B=len(LENGTHS), dtype=dtype,
                    seed=SEED, **QWEN35_08B_GEOM)
    mask = _ragged_mask()
    y_batch, st_batch = chunked_gated_delta(
        d["q"], d["k"], d["v"], d["g"], d["beta"], mask=mask, chunk_size=CHUNK_SIZE)
    mx.eval(y_batch, st_batch)

    pin = FWD_MASKED_PINS[dtype_name]
    for b, n in enumerate(LENGTHS):
        y_ref, st_ref = sequential_gated_delta()(
            d["q"][b:b + 1, :n], d["k"][b:b + 1, :n], d["v"][b:b + 1, :n],
            d["g"][b:b + 1, :n], d["beta"][b:b + 1, :n], None)
        mx.eval(y_ref, st_ref)
        assert_under_pin(f"row{b}(n={n})/y",
                          metrics(y_batch[b:b + 1, :n], y_ref), pin["y"])
        assert_under_pin(f"row{b}(n={n})/state",
                          metrics(st_batch[b:b + 1], st_ref), pin["state"])


@needs_long_context_room
@pytest.mark.parametrize(("dtype_name", "dtype"), DTYPES)
def test_masked_batch_bwd_restricted_to_row_matches_row_oracle(dtype_name, dtype):
    # Catches: a masked-batch backward that mixes gradient across rows (no
    # true row independence), or diverges from the oracle's own backward on
    # the equivalent unpadded row once restricted to that row's valid slice.
    d = build_case(T=T_MAX, chunk_size=CHUNK_SIZE, B=len(LENGTHS), dtype=dtype,
                    seed=SEED, **QWEN35_08B_GEOM)
    mask = _ragged_mask()
    loss_batch = _masked_batch_loss(mask, CHUNK_SIZE)
    grads_batch = mx.grad(loss_batch, argnums=(0, 1, 2, 3, 4))(
        d["q"], d["k"], d["v"], d["g"], d["beta"])
    mx.eval(*grads_batch)

    pin = BWD_MASKED_PINS[dtype_name]
    loss_row = _row_oracle_loss()
    for b, n in enumerate(LENGTHS):
        q_r, k_r, v_r, g_r, beta_r = (
            d[name][b:b + 1, :n] for name in ("q", "k", "v", "g", "beta"))
        grads_row = mx.grad(loss_row, argnums=(0, 1, 2, 3, 4))(
            q_r, k_r, v_r, g_r, beta_r)
        mx.eval(*grads_row)
        for name, g_full, g_row in zip(
            ("dq", "dk", "dv", "dg", "dbeta"), grads_batch, grads_row, strict=True
        ):
            assert_under_pin(f"row{b}(n={n})/{name}",
                              metrics(g_full[b:b + 1, :n], g_row), pin[name])


@needs_long_context_room
@pytest.mark.parametrize(("dtype_name", "dtype"), DTYPES)
def test_masked_batch_bwd_padded_positions_get_zero_gradient(dtype_name, dtype):  # noqa: ARG001 -- kept for a readable [fp32]/[bf16] test id
    # Catches: a masked-batch loss that squares the raw (unmasked) y instead
    # of excluding padded positions first -- verified by mutation: dropping
    # the y-mask in _masked_batch_loss leaks a nonzero dq (~33 in this
    # regime) into padded positions while dk/dv/dg/dbeta stay exactly zero
    # (they only reach padded steps through beta=0/g=1 identity-step paths,
    # which q's diagonal q@state term bypasses).
    d = build_case(T=T_MAX, chunk_size=CHUNK_SIZE, B=len(LENGTHS), dtype=dtype,
                    seed=SEED, **QWEN35_08B_GEOM)
    mask = _ragged_mask()
    loss_batch = _masked_batch_loss(mask, CHUNK_SIZE)
    grads = mx.grad(loss_batch, argnums=(0, 1, 2, 3, 4))(
        d["q"], d["k"], d["v"], d["g"], d["beta"])
    mx.eval(*grads)

    pad_mask = mx.logical_not(mask)
    for name, gr in zip(("dq", "dk", "dv", "dg", "dbeta"), grads, strict=True):
        m = _broadcast_row_mask(gr, pad_mask)
        gr_pad = mx.where(m, gr, mx.zeros_like(gr))
        assert float(mx.abs(gr_pad).max()) == 0.0, \
            f"{name}: padded positions received nonzero gradient"


@needs_long_context_room
def test_masked_batch_fwd_matches_per_row_oracle_at_seqlen_not_chunk_multiple() -> None:
    # Catches: the mask-substitution and structural-pad-substitution identity
    # steps composing incorrectly when BOTH fire together in the same forward
    # pass -- T_MAX=512 above is an exact multiple of CHUNK_SIZE=64 and never
    # exercises this interaction; T_500=500 forces pad_len=12 structural pad
    # steps to stack on top of the per-row mask's own identity steps. Graded
    # against FWD_MASKED_PADDED_PINS (see its comment): a controller-run
    # three-arm probe isolated the elevated state-rel_fro reading to the
    # STRUCTURAL PADDING regime itself (not the mask/pad interaction -- the
    # masked arm's worst sits inside the unmasked padded-only arm's spread),
    # so this test is graded against that regime's own pin rather than the
    # chunk-aligned FWD_MASKED_PINS["fp32"], which stays unmodified.
    d = build_case(T=T_500, chunk_size=CHUNK_SIZE, B=len(LENGTHS_500), dtype=mx.float32,
                    seed=SEED, **QWEN35_08B_GEOM)
    mask = _ragged_mask(t_max=T_500, lengths=LENGTHS_500)
    y_batch, st_batch = chunked_gated_delta(
        d["q"], d["k"], d["v"], d["g"], d["beta"], mask=mask, chunk_size=CHUNK_SIZE)
    mx.eval(y_batch, st_batch)

    pin = FWD_MASKED_PADDED_PINS
    for b, n in enumerate(LENGTHS_500):
        y_ref, st_ref = sequential_gated_delta()(
            d["q"][b:b + 1, :n], d["k"][b:b + 1, :n], d["v"][b:b + 1, :n],
            d["g"][b:b + 1, :n], d["beta"][b:b + 1, :n], None)
        mx.eval(y_ref, st_ref)
        assert_under_pin(f"row{b}(n={n})/y",
                          metrics(y_batch[b:b + 1, :n], y_ref), pin["y"])
        assert_under_pin(f"row{b}(n={n})/state",
                          metrics(st_batch[b:b + 1], st_ref), pin["state"])


@needs_long_context_room
def test_masked_batch_bwd_restricted_to_row_matches_row_oracle_at_seqlen_not_chunk_multiple() \
        -> None:
    # Catches: the same pad-x-mask interaction as the forward case above, but
    # for the backward pass -- a structural pad step at the tail of the last
    # chunk (pad_len=12 at T_500=500) could leak gradient into a row's own
    # valid positions, or diverge from the per-row oracle, if pad-substitution
    # and mask-substitution do not compose cleanly in the vjp. Reuses the
    # file's EXISTING fp32 pin (BWD_MASKED_PINS["fp32"]) unmodified.
    d = build_case(T=T_500, chunk_size=CHUNK_SIZE, B=len(LENGTHS_500), dtype=mx.float32,
                    seed=SEED, **QWEN35_08B_GEOM)
    mask = _ragged_mask(t_max=T_500, lengths=LENGTHS_500)
    loss_batch = _masked_batch_loss(mask, CHUNK_SIZE)
    grads_batch = mx.grad(loss_batch, argnums=(0, 1, 2, 3, 4))(
        d["q"], d["k"], d["v"], d["g"], d["beta"])
    mx.eval(*grads_batch)

    pin = BWD_MASKED_PINS["fp32"]
    loss_row = _row_oracle_loss()
    for b, n in enumerate(LENGTHS_500):
        q_r, k_r, v_r, g_r, beta_r = (
            d[name][b:b + 1, :n] for name in ("q", "k", "v", "g", "beta"))
        grads_row = mx.grad(loss_row, argnums=(0, 1, 2, 3, 4))(
            q_r, k_r, v_r, g_r, beta_r)
        mx.eval(*grads_row)
        for name, g_full, g_row in zip(
            ("dq", "dk", "dv", "dg", "dbeta"), grads_batch, grads_row, strict=True
        ):
            assert_under_pin(f"row{b}(n={n})/{name}",
                              metrics(g_full[b:b + 1, :n], g_row), pin[name])
