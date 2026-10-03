"""mlx-dfloat's FLUX.1 phase model, expressed on memfit with no private imports.

Constants copied from mlx-dfloat `src/mlx_dfloat/mflux/flux1/memory.py` (measured
2026-09-28, schnell 1024², M1 Max 32 GB); the component sizes are illustrative literals.
The consumer's code is NOT imported. Expected totals are computed by hand below."""
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from mlx_train_perf.memfit import MemoryModel, Phase, estimate, fits, max_int_within_budget

ALLOWANCE_AT_REFERENCE = 1_500_000_000
REFERENCE_TOKENS = 4096 + 256
ALLOWANCE_FLOOR = 500_000_000
MEASURED_LIMIT_AT_1024 = 2_500_000_000
OVERHEAD = 247_712_510
VAE_TRANSIENT_AT_1024 = 8_392_982_528
DENOISE_ACTIVATION_AT_REFERENCE = 1_675_000_000


@dataclass(frozen=True)
class FluxParams:
    height: int
    width: int
    text_tokens: int = 256
    compressed: int = 16_000_000_000
    extras: int = 500_000_000
    encoders: int = 9_000_000_000
    vae: int = 170_000_000
    largest: Mapping[str, int] = field(default_factory=lambda: {"double": 300_000_000,
                                                                "single": 200_000_000})
    vae_with_set: bool = True


def _tokens(p: FluxParams) -> int:
    return p.height * p.width // 256 + p.text_tokens


def _allowance(p: FluxParams) -> int:
    return max(ALLOWANCE_FLOOR, int(ALLOWANCE_AT_REFERENCE * _tokens(p) / REFERENCE_TOKENS))


def _cache(p: FluxParams) -> int:
    limit = p.largest["double"] + p.largest["single"] + _allowance(p)
    return max(limit, MEASURED_LIMIT_AT_1024) if p.height * p.width >= 1024 * 1024 else limit


FLUX: MemoryModel[FluxParams] = MemoryModel(phases=(
    Phase(name="encode", terms={"encoders": lambda p: p.encoders, "activations": _allowance,
                                "overhead": lambda p: OVERHEAD}),  # noqa: ARG005
    Phase(name="denoise", terms={
        "compressed": lambda p: p.compressed, "extras": lambda p: p.extras,
        "decoded": lambda p: max(p.largest.values()), "cache": _cache,
        "activations": lambda p: int(
            DENOISE_ACTIVATION_AT_REFERENCE * _tokens(p) / REFERENCE_TOKENS
        ),
        "overhead": lambda p: OVERHEAD}),  # noqa: ARG005
    Phase(name="vae", terms={
        "compressed": lambda p: p.compressed if p.vae_with_set else 0,
        "extras": lambda p: p.extras if p.vae_with_set else 0,
        "vae": lambda p: p.vae,
        "transient": lambda p: int(VAE_TRANSIENT_AT_1024 * max(1.0, p.height * p.width / 1024**2)),
        "overhead": lambda p: OVERHEAD}),  # noqa: ARG005
))


def test_flux_1024_phases_match_the_hand_computation() -> None:
    # encode  = 9.0e9 + 1.5e9 + 247_712_510                                = 10_747_712_510
    # denoise = 16e9 + 0.5e9 + 0.3e9 + 2.5e9 (cache floor) + 1.675e9 + OH  = 21_222_712_510
    # vae     = 16e9 + 0.5e9 + 0.17e9 + 8_392_982_528 + OH                  = 25_310_695_038
    est = estimate(FLUX, FluxParams(height=1024, width=1024))
    assert est.phase_totals == {"encode": 10_747_712_510, "denoise": 21_222_712_510,
                                "vae": 25_310_695_038}
    assert (est.peak_phase, est.peak_bytes) == ("vae", 25_310_695_038)


def test_dropping_the_set_before_decode_moves_the_peak_to_denoise() -> None:
    est = estimate(FLUX, FluxParams(height=1024, width=1024, vae_with_set=False))
    assert est.phase_totals["vae"] == 8_810_695_038
    assert est.peak_phase == "denoise"


def test_size_dependent_terms_follow_the_varied_resolution() -> None:
    """Spec review I2: allowance and cache must be computed from the varied size, not
    frozen at a base resolution. 1536²: tokens 9472 -> allowance 3_264_705_882."""
    small = estimate(FLUX, FluxParams(height=1024, width=1024)).components["encode"]
    large = estimate(FLUX, FluxParams(height=1536, width=1536)).components["encode"]
    assert small["activations"] == 1_500_000_000
    assert large["activations"] == 3_264_705_882


def test_largest_square_side_in_steps_of_64_that_fits() -> None:
    base = FluxParams(height=64, width=64, vae_with_set=False)
    budget = 23 * 1024**3
    k = max_int_within_budget(FLUX, lambda k: replace(base, height=64 * k, width=64 * k),
                              lo=1, hi=64, budget_bytes=budget)
    at = estimate(FLUX, replace(base, height=64 * k, width=64 * k))
    above = estimate(FLUX, replace(base, height=64 * (k + 1), width=64 * (k + 1)))
    assert fits(at, budget_bytes=budget)
    assert not fits(above, budget_bytes=budget)
    assert k >= 16                     # 1024² fits with the set dropped (21.2e9 < 24.7e9)
