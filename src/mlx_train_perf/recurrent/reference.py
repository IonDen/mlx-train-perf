"""Sequential GatedDelta oracle access + source pins for the mirrored surfaces.

mlx-lm is an optional extra: every import is lazy. The source pins turn an
in-range mlx-lm patch that mutates the oracle or any mirrored surface into a
red test instead of a silently shifted parity target.
"""

import hashlib
import inspect
from collections.abc import Callable
from typing import Any, cast

from mlx_train_perf.errors import MissingDependencyError

_MIRRORED: tuple[tuple[str, str, str], ...] = (
    ("gated_delta_ops", "mlx_lm.models.gated_delta", "gated_delta_ops"),
    ("gated_delta_update", "mlx_lm.models.gated_delta", "gated_delta_update"),
    ("compute_g", "mlx_lm.models.gated_delta", "compute_g"),
    ("GatedDeltaNet.__init__", "mlx_lm.models.qwen3_5", "GatedDeltaNet.__init__"),
    ("GatedDeltaNet.__call__", "mlx_lm.models.qwen3_5", "GatedDeltaNet.__call__"),
    ("DecoderLayer.__call__", "mlx_lm.models.qwen3_5", "DecoderLayer.__call__"),
    ("Qwen3_5TextModel.__call__", "mlx_lm.models.qwen3_5", "Qwen3_5TextModel.__call__"),
    ("TextModel.cast_predicate", "mlx_lm.models.qwen3_5", "TextModel.cast_predicate"),
    ("RMSNormGated.__call__", "mlx_lm.models.qwen3_next",
     "Qwen3NextRMSNormGated.__call__"),
    ("create_ssm_mask", "mlx_lm.models.base", "create_ssm_mask"),
)

# sha256(inspect.getsource) on installed mlx-lm 0.31.3, recorded 2026-08-30
PINNED_SOURCE_HASHES: dict[str, str] = {
    "gated_delta_ops": "9751b1354ae88830f815b23c3a2ed4ec3a5154ccbbb2bbaa400f407efdd0540a",
    "gated_delta_update": "ffe520f93b13109db81e9a0e4e27446fcf3e6f6f73ea93f93bd35a0b50bf02ce",
    "compute_g": "2557432693f6ac3904a34a185766e986d5f5e5c45b336b718b379fe2a8f0a1c9",
    "GatedDeltaNet.__init__": "0bd9597daecd5aaa677498a15441b71c8ac428344ba3346268ab504b88b57612",
    "GatedDeltaNet.__call__": "9f27bdc613912fa9b3b0509d8e3f684ac672810bf45e29831f5815a1c3b43091",
    "DecoderLayer.__call__": "f10022c1d06a81721132714ec04781b200d494a26971d89ebe2627cee7b8828b",
    "Qwen3_5TextModel.__call__": "45fd379f8885e93b4ba685c3d6c2aa1b891d535bb00fd042c6b1e56613345b6d",
    "TextModel.cast_predicate": "5c83d9ad9259cf484573a6259743bd382feda95dde1cc5ef1f4d94cd8fc40294",
    "RMSNormGated.__call__": "117b7200e4e185555005330369b66a52d796a8f946efcfe149b85de0ea8c5fb8",
    "create_ssm_mask": "b929fae0c2ee287a6648168f1f269e1326a3ef4cf382e93d4470244d0f3a3557",
}


def _require_mlx_lm(module: str) -> Any:
    import importlib  # noqa: PLC0415 -- lazy: mlx-lm is an optional extra

    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise MissingDependencyError(
            "the qwen3_5 recurrent path requires the 'mlx-lm' extra: "
            "pip install 'mlx-train-perf[mlx-lm]'"
        ) from exc


def sequential_gated_delta() -> Callable[..., Any]:
    """The parity oracle: mlx-lm's sequential ``gated_delta_ops``."""
    return cast(
        "Callable[..., Any]",
        _require_mlx_lm("mlx_lm.models.gated_delta").gated_delta_ops,
    )


def _resolve(module: str, qualname: str) -> Any:
    obj: Any = _require_mlx_lm(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    if isinstance(obj, property):
        # `TextModel.cast_predicate` is a @property (the A_log fp32 contract); a plain
        # class-attribute walk returns the descriptor itself, which `inspect.getsource`
        # rejects outright. `.fget` is the underlying getter function and still carries
        # the full decorated source (including the `@property` line).
        obj = obj.fget
    return obj


def mirrored_source_hashes() -> dict[str, str]:
    return {
        key: hashlib.sha256(
            inspect.getsource(_resolve(module, qualname)).encode()
        ).hexdigest()
        for key, module, qualname in _MIRRORED
    }
