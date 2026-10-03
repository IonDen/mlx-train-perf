"""mlx-train-perf. The loss API below loads lazily, on first attribute access, so that
importing a submodule (`mlx_train_perf.memfit`, `mlx_train_perf.machine`) does not load
mlx."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx_train_perf.core.loss import (
        DenseHead,
        HeadRef,
        QuantizedHead,
        Resolution,
        linear_cross_entropy,
        resolve_impl,
        tied_head,
    )

__all__ = ["DenseHead", "HeadRef", "QuantizedHead", "Resolution", "linear_cross_entropy",
           "resolve_impl", "tied_head"]


def __getattr__(name: str) -> object:
    if name in __all__:
        from mlx_train_perf.core import loss  # noqa: PLC0415

        value = getattr(loss, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
