"""Golden outputs of `estimate_peak` over a grid, recorded on the pre-memfit code
(dev/v0.9.0 @ d4f08e9) so the memfit rewrite is checked for byte identity.
Regenerate ONLY if a deliberate planner change is reviewed:
`uv run python tests/_plan_golden.py --write`."""
import itertools
import json
import sys
from pathlib import Path

from mlx_train_perf.plan.calibration import load_calibration
from mlx_train_perf.plan.estimate import ModelShape, TrainConfig, estimate_peak

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "plan_golden.json"

_SHAPES = {
    "qwen3_8b": {"vocab": 151936, "hidden": 4096, "layers": 36, "intermediate": 12288,
                 "heads": 32, "kv_heads": 8, "tied": False},
    "llama_3b": {"vocab": 128256, "hidden": 3072, "layers": 28, "intermediate": 8192,
                 "heads": 24, "kv_heads": 8, "tied": True},
}
_QUANT = {"dense": (None, None), "q4": (4, 64)}


def record() -> dict[str, object]:
    calib = load_calibration()
    out: dict[str, object] = {}
    for (sname, sargs), (qname, (bits, group)), attention, impl, gc, dtype, seq, layers in (
        itertools.product(_SHAPES.items(), _QUANT.items(), ("stock", "flash"),
                          ("kernel", "chunked", "naive"), (True, False),
                          ("bfloat16", "float16", "float32"), (512, 2048, 8192), (0, 16))
    ):
        shape = ModelShape(**sargs, quant_bits=bits, quant_group=group)  # type: ignore[arg-type]
        cfg = TrainConfig(batch=1, seq_len=seq, dtype=dtype, lora_rank=8, lora_layers=layers,
                          grad_checkpoint=gc, impl=impl, attention=attention)
        key = f"{sname}/{qname}/{attention}/{impl}/gc={gc}/{dtype}/seq={seq}/lora={layers}"
        try:
            peak, comps = estimate_peak(shape, cfg, calib)
            out[key] = {"peak": peak, "components": comps}
        except Exception as exc:  # the error contract is part of the golden
            out[key] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


if __name__ == "__main__" and "--write" in sys.argv:
    FIXTURE.parent.mkdir(exist_ok=True)
    FIXTURE.write_text(json.dumps(record(), indent=1, sort_keys=True) + "\n")
    print(f"wrote {FIXTURE}")
