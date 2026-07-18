"""Gate A for the 0.4.0 sequence-packing cycle (spec §9 a/b/d): kernel feasibility
measurements for the PACKED flash-attention variants.

Phases (each writes `_artifacts/gate_packed/<phase>.json` the moment it finishes and is
SKIPPED on re-run if its artifact exists — resume by re-invoking; `--force` re-measures):

- `rope`     (§9b, light):  RoPE offset drift — a 512-token segment's attention outputs at
             pack offset 3584 vs offset 0, fp32 + bf16. The bf16 number pins
             `PACKED_ROPE_TOL` (consumed ONLY by the Task-10 packed-vs-unpacked test).
- `parity8k` (§9a, ~4-8 GiB transient): packed fwd + bwd kernel parity vs the block-diagonal
             oracle at the 8k bucket (b=1, hq=4, hkv=2, n=8192, d=128, bf16, 4 segments
             with non-32-aligned boundaries). Pins the 8k row of `PACKED_KERNEL_TOL`.
- `walls`    (§9d): same-session, same-tile median walls at n=8192 for fwd/dQ/dK+dV —
             causal vs packed(1 segment) vs packed(4 segments). NO-GO if packed regresses
             >15% vs causal on any kernel (register-pressure telltale at saturation).

Run: `uv run python scripts/gate_packed_attention.py [--phase rope|parity8k|walls|all]`
Heavy-run rules: main session, AC power, `memory_pressure` pre-flight, serialized.
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx import nn

from mlx_train_perf.attention.kernel.dispatch import select_bwd_tiles, select_fwd_tile
from mlx_train_perf.attention.kernel.launch import (
    calibrated_bwd_dkv_rate,
    calibrated_bwd_dq_rate,
    calibrated_fwd_rate,
    launch_bwd_D,
    launch_bwd_dkv,
    launch_bwd_dq,
    launch_flash_fwd,
)
from mlx_train_perf.attention.reference import flash_attention_reference, math_attention
from mlx_train_perf.attention.segments import PackedMask
from mlx_train_perf.core.guards import (
    install_guardrails,
    install_memory_watchdog,
    memory_ceiling_bytes,
)

ART_DIR = Path(__file__).resolve().parent.parent / "_artifacts" / "gate_packed"

# 4 segments with deliberately non-32-aligned interior boundaries (mid-block).
_BOUNDS_8K = (0, 2000, 3500, 6100, 8192)
_SCALE = 0.088388  # 1/sqrt(128)


def _segments_from_bounds(bounds: tuple[int, ...], b: int) -> PackedMask:
    n = bounds[-1]
    seg_id = [0] * n
    seg_start = [0] * n
    for s in range(len(bounds) - 1):
        for t in range(bounds[s], bounds[s + 1]):
            seg_id[t] = s
            seg_start[t] = bounds[s]
    sid = mx.broadcast_to(mx.array(seg_id, dtype=mx.int32)[None], (b, n))
    sst = mx.broadcast_to(mx.array(seg_start, dtype=mx.int32)[None], (b, n))
    return PackedMask(seg_id=mx.contiguous(sid), seg_start=mx.contiguous(sst))


def _qkv(b: int, hq: int, hkv: int, n: int, d: int, dtype: mx.Dtype, seed: int
         ) -> tuple[mx.array, mx.array, mx.array]:
    kq, kk, kv = (mx.random.key(seed + i) for i in range(3))
    q = mx.random.normal((b, hq, n, d), key=kq).astype(dtype)
    k = mx.random.normal((b, hkv, n, d), key=kk).astype(dtype)
    v = mx.random.normal((b, hkv, n, d), key=kv).astype(dtype)
    mx.eval(q, k, v)
    return q, k, v


def _maxdiff(a: mx.array, ref: mx.array) -> float:
    return float(mx.abs(a.astype(mx.float32) - ref.astype(mx.float32)).max().item())


def phase_rope() -> dict[str, Any]:
    """§9b: same 512-token segment at pack offset 3584 vs 0 (4096 pack), per dtype."""
    out: dict[str, Any] = {"offset": 3584, "seg_len": 512, "rope_base": 10000.0}
    rope = nn.RoPE(128, traditional=False, base=10000.0)
    for dtype_name, dtype in (("float32", mx.float32), ("bfloat16", mx.bfloat16)):
        q, k, v = _qkv(1, 4, 2, 512, 128, dtype, seed=11)
        q0, k0 = rope(q, offset=0), rope(k, offset=0)
        qp, kp = rope(q, offset=3584), rope(k, offset=3584)
        o0, l0 = flash_attention_reference(q0, k0, v, scale=_SCALE)
        op, lp = flash_attention_reference(qp, kp, v, scale=_SCALE)
        mx.eval(o0, l0, op, lp)
        out[dtype_name] = {
            "max_dO": _maxdiff(op, o0),
            "max_dL": _maxdiff(lp, l0),
        }
    return out


def phase_parity8k() -> dict[str, Any]:
    """§9a: packed kernel fwd+bwd vs the block-diagonal oracle at the 8k bucket, bf16."""
    b, hq, hkv, n, d = 1, 4, 2, 8192, 128
    dtype = mx.bfloat16
    pm = _segments_from_bounds(_BOUNDS_8K, b)
    q, k, v = _qkv(b, hq, hkv, n, d, dtype, seed=42)
    out: dict[str, Any] = {"shape": [b, hq, hkv, n, d], "dtype": "bfloat16",
                           "bounds": list(_BOUNDS_8K)}

    # Oracle forward (fp32 internals).
    o_ref, l_ref = flash_attention_reference(q, k, v, scale=_SCALE, segments=pm)
    mx.eval(o_ref, l_ref)

    # Kernel forward (causal-keyed rates are valid here: rates size the dispatch split,
    # never the math; the packed rate identity lands in Task 5).
    tile = select_fwd_tile(n, d)
    rate = calibrated_fwd_rate(head_dim=d, dtype=dtype, b=b, hq=hq, hkv=hkv, n=n,
                               causal=True, tile=tile)
    o_k, l_k = launch_flash_fwd(q, k, v, scale=_SCALE, causal=True, tile=tile,
                                rate_macs_per_s=rate, seg_id=pm.seg_id,
                                seg_start=pm.seg_start)
    mx.eval(o_k, l_k)
    out["fwd"] = {"max_dO": _maxdiff(o_k, o_ref), "max_dL": _maxdiff(l_k, l_ref)}

    # Oracle backward via autodiff through the packed oracle (the O(N^2) arm).
    d_o = mx.random.normal((b, hq, n, d), key=mx.random.key(7)).astype(dtype)
    mx.eval(d_o)

    def loss(q_: mx.array, k_: mx.array, v_: mx.array) -> mx.array:
        o = math_attention(q_, k_, v_, scale=_SCALE, segments=pm)
        return (o.astype(mx.float32) * d_o.astype(mx.float32)).sum()

    dq_ref, dk_ref, dv_ref = mx.grad(loss, argnums=(0, 1, 2))(q, k, v)
    mx.eval(dq_ref, dk_ref, dv_ref)
    oracle_peak = mx.get_peak_memory()
    mx.synchronize()
    mx.clear_cache()

    # Kernel backward.
    d_arr = launch_bwd_D(d_o, o_k)
    dq_tile, dkv_tile = select_bwd_tiles(n, d)
    dq_rate = calibrated_bwd_dq_rate(head_dim=d, dtype=dtype, b=b, hq=hq, hkv=hkv, n=n,
                                     causal=True, tile=dq_tile)
    dkv_rate = calibrated_bwd_dkv_rate(head_dim=d, dtype=dtype, b=b, hq=hq, hkv=hkv, n=n,
                                       causal=True, tile=dkv_tile)
    dq_k = launch_bwd_dq(q, k, v, d_o, l_k, d_arr, scale=_SCALE, causal=True,
                         rate_macs_per_s=dq_rate, variant=dq_tile.variant,
                         d_slab=dq_tile.d_slab, seg_id=pm.seg_id, seg_start=pm.seg_start)
    dk_k, dv_k = launch_bwd_dkv(q, k, v, d_o, l_k, d_arr, scale=_SCALE, causal=True,
                                rate_macs_per_s=dkv_rate, variant=dkv_tile.variant,
                                d_slab=dkv_tile.d_slab, seg_id=pm.seg_id,
                                seg_start=pm.seg_start)
    mx.eval(dq_k, dk_k, dv_k)
    out["bwd"] = {
        "max_dQ": _maxdiff(dq_k, dq_ref),
        "max_dK": _maxdiff(dk_k, dk_ref),
        "max_dV": _maxdiff(dv_k, dv_ref),
    }
    out["oracle_peak_gb"] = round(oracle_peak / 1024**3, 3)
    out["total_peak_gb"] = round(mx.get_peak_memory() / 1024**3, 3)
    return out


def _median_wall(fn: Any, *, warmup: int = 2, reps: int = 5) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    walls = []
    for _ in range(reps):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        walls.append(time.perf_counter() - t0)
    return statistics.median(walls)


def phase_walls() -> dict[str, Any]:
    """§9d: causal vs packed(1seg) vs packed(4seg) walls at n=8192, same session/tiles."""
    b, hq, hkv, n, d = 1, 4, 2, 8192, 128
    dtype = mx.bfloat16
    q, k, v = _qkv(b, hq, hkv, n, d, dtype, seed=42)
    d_o = mx.random.normal((b, hq, n, d), key=mx.random.key(7)).astype(dtype)
    mx.eval(d_o)

    one_seg = PackedMask(
        seg_id=mx.zeros((b, n), dtype=mx.int32),
        seg_start=mx.zeros((b, n), dtype=mx.int32),
    )
    four_seg = _segments_from_bounds(_BOUNDS_8K, b)
    mx.eval(one_seg.seg_id, one_seg.seg_start, four_seg.seg_id, four_seg.seg_start)

    tile = select_fwd_tile(n, d)
    dq_tile, dkv_tile = select_bwd_tiles(n, d)
    rate = calibrated_fwd_rate(head_dim=d, dtype=dtype, b=b, hq=hq, hkv=hkv, n=n,
                               causal=True, tile=tile)
    dq_rate = calibrated_bwd_dq_rate(head_dim=d, dtype=dtype, b=b, hq=hq, hkv=hkv, n=n,
                                     causal=True, tile=dq_tile)
    dkv_rate = calibrated_bwd_dkv_rate(head_dim=d, dtype=dtype, b=b, hq=hq, hkv=hkv, n=n,
                                       causal=True, tile=dkv_tile)

    # Residuals for the backward launches, computed once from the causal forward.
    o_c, l_c = launch_flash_fwd(q, k, v, scale=_SCALE, causal=True, tile=tile,
                                rate_macs_per_s=rate)
    mx.eval(o_c, l_c)
    d_arr = launch_bwd_D(d_o, o_c)
    mx.eval(d_arr)

    def arms(pm: PackedMask | None) -> dict[str, float]:
        sid = pm.seg_id if pm is not None else None
        sst = pm.seg_start if pm is not None else None
        fwd = _median_wall(lambda: launch_flash_fwd(
            q, k, v, scale=_SCALE, causal=True, tile=tile, rate_macs_per_s=rate,
            seg_id=sid, seg_start=sst)[0])
        dq = _median_wall(lambda: launch_bwd_dq(
            q, k, v, d_o, l_c, d_arr, scale=_SCALE, causal=True,
            rate_macs_per_s=dq_rate, variant=dq_tile.variant, d_slab=dq_tile.d_slab,
            seg_id=sid, seg_start=sst))
        dkv = _median_wall(lambda: launch_bwd_dkv(
            q, k, v, d_o, l_c, d_arr, scale=_SCALE, causal=True,
            rate_macs_per_s=dkv_rate, variant=dkv_tile.variant, d_slab=dkv_tile.d_slab,
            seg_id=sid, seg_start=sst)[0])
        return {"fwd_s": fwd, "dq_s": dq, "dkv_s": dkv}

    causal = arms(None)
    packed1 = arms(one_seg)
    packed4 = arms(four_seg)
    ratios = {
        key: {"packed1_over_causal": packed1[key] / causal[key],
              "packed4_over_causal": packed4[key] / causal[key]}
        for key in causal
    }
    worst = max(
        max(r["packed1_over_causal"], r["packed4_over_causal"]) for r in ratios.values()
    )
    return {
        "shape": [b, hq, hkv, n, d], "dtype": "bfloat16", "reps": 5,
        "causal": causal, "packed_1seg": packed1, "packed_4seg": packed4,
        "ratios": ratios, "worst_packed_over_causal": worst,
        "no_go_threshold": 1.15,
        "note": ("packed_4seg does LESS work than causal (block-diagonal subset); "
                 "its ratio <1 is expected and is not the register-pressure signal — "
                 "packed_1seg (identical work + predicate overhead) is."),
    }


def _run_phase(name: str, fn: Any, force: bool) -> None:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    path = ART_DIR / f"{name}.json"
    if path.exists() and not force:
        print(f"[gate] {name}: artifact exists, skipping ({path})")
        return
    print(f"[gate] {name}: running…", flush=True)
    t0 = time.perf_counter()
    result = fn()
    result["wall_s"] = round(time.perf_counter() - t0, 2)
    result["mlx_version"] = mx.__version__
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[gate] {name}: done in {result['wall_s']}s -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["rope", "parity8k", "walls", "all"],
                        default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    install_guardrails()

    def on_breach(reason: str, details: dict[str, object]) -> None:
        ART_DIR.mkdir(parents=True, exist_ok=True)
        (ART_DIR / "BREACH.json").write_text(
            json.dumps({"reason": reason, "details": {k: str(v) for k, v in
                                                      details.items()}}) + "\n")
        print(f"[gate] MEMORY BREACH: {reason} — aborting", flush=True)
        os._exit(70)

    handle = install_memory_watchdog(
        ceiling_bytes=memory_ceiling_bytes(int(mx.device_info()["memory_size"])),
        wall_budget_s=1800.0,
        on_breach=on_breach,
    )
    try:
        if args.phase in ("rope", "all"):
            _run_phase("rope", phase_rope, args.force)
        if args.phase in ("parity8k", "all"):
            _run_phase("parity8k", phase_parity8k, args.force)
        if args.phase in ("walls", "all"):
            _run_phase("walls", phase_walls, args.force)
    finally:
        handle.stop()


if __name__ == "__main__":
    main()
