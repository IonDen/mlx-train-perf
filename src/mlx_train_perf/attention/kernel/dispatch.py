"""Forward-kernel tile/variant dispatch by (n, head_dim) -- encodes the persisted T6 ladder
artifacts under `_artifacts/attention_fwd_rungs/`, not prose memory (the `core/kernel/
dispatch.py` convention: pick the table's winner from a committed artifact, never re-derive
it from recollection).

MEASURED is a SINGLE saturation-bucket data point: the flagship shape (b=1, Hq=32, Hkv=8,
N=8192, D=128, bf16) is the only shape the T6 ladder ran the register-resident P@V MMA body
(`source.build_fwd_mma_source`) at occupancy-saturated N. Four D-slab widths were compared at
THAT one shape:

    scalar          31.6 G MAC/s   rung0_scalar_v0.json      (the v0 zero-reuse baseline)
    mma slab16     513.95 G MAC/s  rung2b_dslab16.json
    mma slab32     931.67 G MAC/s  rung2b_dslab32.json
    mma slab64    1360.96 G MAC/s  rung2b_dslab64.json
    mma slab128   1462.74 G MAC/s  rung2b_dslab128.json      <- the winner

with parity IDENTICAL across every slab width (bf16-ULP class, worst 1.953e-3 at n=256, see
each rung2b_*.json's `parity_worst_n256`) -- the slab choice is a pure throughput lever here,
never a correctness one.

JUDGE AT SATURATION ONLY (`user-metal-kernels` / this project's own gotcha #5): this ladder
was run ONCE, at ONE occupancy-saturated shape. `select_fwd_tile` therefore treats any (n,
head_dim) outside that direct measurement as PROVISIONAL:

- head_dim=128, n inside the measured occupancy regime [8192, 16384): `mma` / `d_slab=128`,
  NOT provisional -- the direct rung2b_dslab128 measurement above.
- head_dim=128, any other n (including well above 16384 -- the ladder never measured whether
  the ordering holds past one saturated bucket): `mma` / `d_slab=128`, PROVISIONAL. Same MSL
  body, same D-slab-independent physics (the KV loop's occupancy is what saturates, not the
  D-slab restructure itself), just not directly measured at that bucket.
- head_dim in {64, 96}: `mma` with the source builder's own default slab (`d_slab=None`),
  PROVISIONAL -- this ladder never ran either head dim. `mma`'s CORRECTNESS at small N is
  independently parity-proven across the whole `test_attention_kernel_fwd.py` grid (every
  case there runs both variants); what is genuinely unmeasured is its RATE off the one
  saturated bucket -- exactly what `provisional` flags.

`scalar` is never SELECTED by this table (it is the ladder's baseline, ~16x slower even at
its own measured shape) -- it stays reachable as a manually-constructed
`TileShape(variant="scalar")` for tests/regression, the only MEASURED-anywhere choice this
ladder never had reason to displace as the correctness fallback.
"""
import math

from mlx_train_perf.attention.kernel.launch import TileShape

_KERNEL_HEAD_DIMS = (64, 96, 128)

# head_dim -> {n-bucket -> (variant, d_slab, measured G MAC/s)}. Only head_dim=128 (the
# flagship's own head dim) has an entry -- the only head dim this ladder ran.
MEASURED: dict[int, dict[int, tuple[str, int, float]]] = {
    128: {8192: ("mma", 128, 1462.74)},  # _artifacts/attention_fwd_rungs/rung2b_dslab128.json
}

# head_dim in {64, 96}: mma with the source builder's own default slab, always provisional --
# see the module docstring for why mma (not scalar) is still the right unmeasured default.
_UNMEASURED_HEAD_DIM_CHOICE: tuple[str, int | None] = ("mma", None)


def select_fwd_tile(n: int, head_dim: int) -> TileShape:
    """Measured/provisional forward tile+variant for shape (n, head_dim) -- see the module
    docstring for the ladder this encodes and the occupancy-window provisional rule (mirrors
    `core/kernel/dispatch.py::select_variant`'s nearest-log2-bucket + `[bucket, 2*bucket)`
    same-occupancy-regime window exactly).

    Raises `ValueError` for a head_dim outside the kernel's supported set (mirrors
    `attention/kernel/source.py`'s own head_dim gate) -- defensive, not the normal failure
    mode: `flash_attention`'s `resolve_attention_impl` already gates head_dim before the
    kernel path (and this function) is ever reached.
    """
    if head_dim not in _KERNEL_HEAD_DIMS:
        raise ValueError(f"head_dim must be one of {_KERNEL_HEAD_DIMS}, got {head_dim}")
    buckets = MEASURED.get(head_dim)
    if not buckets:
        variant, d_slab = _UNMEASURED_HEAD_DIM_CHOICE
        return TileShape(variant=variant, d_slab=d_slab, provisional=True)
    bucket = min(buckets, key=lambda b: (abs(math.log2(n) - math.log2(b)), b))  # tie -> lower
    variant, d_slab, _rate = buckets[bucket]
    provisional = not (bucket <= n < 2 * bucket)  # inside == same occupancy regime
    return TileShape(variant=variant, d_slab=d_slab, provisional=provisional)
