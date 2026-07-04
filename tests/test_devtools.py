import builtins

import pytest

from mlx_train_perf.core.kernel.source import build_dense_source
from mlx_train_perf.devtools.regpressure import (
    _prepare_msl,
    _strip_banner_and_fences,
    compiled_ceiling,
)
from mlx_train_perf.errors import MissingDependencyError

# ---------------------------------------------------------------------------
# Pure text-transform helpers: GPU-free, pyobjc-free -- always run in the default lane.
# ---------------------------------------------------------------------------


def test_strip_banner_and_fences_removes_banner_and_markdown_fences() -> None:
    raw = "Generated source code for `mtp_probe`:\n```\nline one\nline two\n```\n"
    assert _strip_banner_and_fences(raw) == "line one\nline two"


def test_strip_banner_and_fences_is_noop_without_a_banner_line() -> None:
    raw = "```\nno banner here\n```"
    assert _strip_banner_and_fences(raw) == "no banner here"


def test_strip_banner_and_fences_only_strips_bare_fence_lines() -> None:
    # A ``` occurring mid-line (not its own line) must NOT be stripped -- only lines whose
    # stripped content is exactly the fence marker are capture noise.
    raw = "Generated source code for `mtp_probe`:\n```\nuint x = 1; // ```not a fence```\n```\n"
    assert _strip_banner_and_fences(raw) == "uint x = 1; // ```not a fence```"


def test_prepare_msl_prepends_the_minimal_jit_prelude() -> None:
    raw = "Generated source code for `mtp_probe`:\n```\nbody text\n```\n"
    result = _prepare_msl(raw)
    assert result.startswith("#include <metal_stdlib>\n")
    assert "using namespace metal;" in result
    assert "typedef bfloat bfloat16_t;" in result
    assert result.endswith("body text")


# ---------------------------------------------------------------------------
# compiled_ceiling: the public probe entry point
# ---------------------------------------------------------------------------


def test_missing_pyobjc_is_precise_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real = builtins.__import__

    def fake(name: str, *a: object, **k: object) -> object:
        if name == "Metal":
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(MissingDependencyError) as ei:
        compiled_ceiling("out[0] = 1.0f;")
    assert "probe" in str(ei.value)


@pytest.mark.metal
def test_dense_v2e_ceiling_matches_spike_measurement() -> None:
    # spike-measured compiled ceilings: v2e (RT=4) -> 448, v2d (RT=2) -> 640
    # NOTE: compiled_ceiling wraps a bare function shell around the body; reconcile the
    # wrapper so shapes/template refs resolve (port the spike script's shell verbatim).
    assert compiled_ceiling(build_dense_source(4)) == 448
    assert compiled_ceiling(build_dense_source(2)) == 640
