"""--run-smoke: qwen3_5 gated-delta training end-to-end through mlx-lm's REAL
compiled `train()`, real checkpoint.

Subprocess-isolated (`tests/_qwen35_smoke_child.py`): `train()` overrides the
process's wired-memory cap at entry (gotcha 8), so running it in-process would
corrupt that cap for every test that follows in the same session -- the same
reasoning `tests/test_packed_smoke.py` and `tests/test_attention_wrapper.py`'s
gc=True case apply, made a hard requirement here because this smoke test also
exercises the gated-delta chunked training path for the first time against a
real checkpoint. Model: mlx-community/Qwen3.5-0.8B-4bit, expected
pre-downloaded -- this test never fetches it, and skips cleanly when it is not
already in the local Hugging Face cache.
"""
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import pytest

_MODEL_REPO = "mlx-community/Qwen3.5-0.8B-4bit"
_CHILD = Path(__file__).parent / "_qwen35_smoke_child.py"


def _model_is_cached(repo_id: str) -> bool:
    """True iff `repo_id` is already fully present in the local Hugging Face
    cache. Checked with `local_files_only=True`, which never touches the
    network -- unlike a bare `snapshot_download`/`mlx_lm.load()` call, which
    would fetch a missing model instead of letting this test skip."""
    from huggingface_hub import snapshot_download  # noqa: PLC0415 -- lazy: smoke-gated

    try:
        snapshot_download(repo_id, local_files_only=True)
    except Exception:
        return False
    return True


@pytest.mark.smoke
def test_qwen35_trainer_smoke(tmp_path: Path) -> None:
    """One real mlx_lm `train()` run (LoRA rank 8 on the GatedDelta
    projections, 4 iters, padded batches of at least two distinct lengths so
    the compiled step retraces) against the real checkpoint, in a CHILD
    process. Verdict on the child's exit code, its `::SMOKE_OK::` sentinel,
    and whether the LoRA adapter weights it snapshots before/after training
    actually differ -- an exit code and a print alone cannot tell a real
    training loop from a silent no-op. No assertion touches a loss value
    (loss values come from plain forwards, never from an assertion on
    `value_and_grad`'s output)."""
    if not _model_is_cached(_MODEL_REPO):
        pytest.skip(f"{_MODEL_REPO} is not in the local Hugging Face cache")

    before_path = tmp_path / "adapters_before.safetensors"
    after_path = tmp_path / "adapters_after.safetensors"
    proc = subprocess.run(
        [sys.executable, str(_CHILD), str(tmp_path)],
        capture_output=True, text=True, timeout=600, check=False,
    )
    assert proc.returncode == 0, f"child failed:\n{proc.stdout}\n{proc.stderr}"
    assert "::SMOKE_OK::" in proc.stdout, proc.stdout

    before = mx.load(str(before_path))
    after = mx.load(str(after_path))
    assert before.keys() == after.keys(), (sorted(before.keys()), sorted(after.keys()))
    assert before, "no LoRA-trainable parameters were saved -- did the keys config match?"
    changed = any(bool(mx.any(before[k] != after[k]).item()) for k in before)
    assert changed, f"adapter weights are bit-identical before/after training\n{proc.stdout}"
