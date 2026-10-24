"""The planner warns when a STOCK attention estimate runs on an mlx outside the versions
its quadratic attention term was calibrated on, and changes nothing else.

The installed mlx version is faked at its source, `importlib.metadata.version("mlx")`,
so these run in the default lane with no GPU. Each test names the one-line bug it catches.
"""
import json
import re
import subprocess
import sys
import warnings
from collections.abc import Callable
from importlib import metadata
from pathlib import Path

import pytest

from mlx_train_perf import cli
from mlx_train_perf.cli import main
from mlx_train_perf.errors import CalibrationWarning, PlanInputError
from mlx_train_perf.plan.calibration import load_calibration
from mlx_train_perf.plan.estimate import ModelShape, TrainConfig, estimate_peak, plan_fit
from mlx_train_perf.plan.inverse import max_batch_for_budget, max_seq_len_for_budget

_REAL_DIST_VERSION = metadata.version


def _shape() -> ModelShape:  # the micro "llama" the other planner tests use
    return ModelShape(vocab=1000, hidden=64, layers=2, intermediate=128, heads=4,
                      kv_heads=2, tied=False, quant_bits=None, quant_group=None)


def _cfg(*, attention: str = "stock", batch: int = 1, seq_len: int = 512) -> TrainConfig:
    return TrainConfig(batch=batch, seq_len=seq_len, dtype="bfloat16", lora_rank=8,
                       lora_layers=2, grad_checkpoint=True, impl="kernel",
                       attention=attention)


@pytest.fixture
def mlx_version(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], None]:
    """Fakes the installed mlx distribution's version; every other distribution reads
    through to the real metadata."""
    def _set(version: str) -> None:
        def _version(name: str) -> str:
            return version if name == "mlx" else _REAL_DIST_VERSION(name)
        monkeypatch.setattr(metadata, "version", _version)
    return _set


def _caught(fn: Callable[[], object]) -> list[warnings.WarningMessage]:
    """Runs `fn` with every warning recorded (no once-per-location dedupe), and returns
    the CalibrationWarnings it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return [w for w in caught if issubclass(w.category, CalibrationWarning)]


def _calibration_warnings(fn: Callable[[], object]) -> list[str]:
    return [str(w.message) for w in _caught(fn)]


# --- (1) stock on an uncalibrated mlx warns, once -------------------------------------


@pytest.mark.parametrize("version", ["0.32.4", "0.32.5", "0.32.10", "0.33.0", "0.33.0.dev1"])
def test_stock_estimate_on_mlx_from_0_32_4_names_the_over_statement(
    mlx_version: Callable[[str], None], version: str,
) -> None:
    """Bug caught: the warning call deleted; the membership test inverted (`if version in
    CALIBRATED`); the >= 0.32.4 check done on strings ("0.32.10" < "0.32.4"), or off by
    one at the boundary (`>` instead of `>=`), or crashing on a ".dev1" suffix."""
    mlx_version(version)
    messages = _calibration_warnings(
        lambda: estimate_peak(_shape(), _cfg(), load_calibration()))
    assert len(messages) == 1
    msg = messages[0]
    assert "stock attention" in msg
    assert version in msg
    assert "0.32.0" in msg
    assert "0.32.4" in msg
    assert "over-state" in msg
    assert "causal attention" in msg


@pytest.mark.parametrize("version", ["0.31.1", "0.32.1", "0.32.3"])
def test_stock_estimate_on_older_uncalibrated_mlx_warns_neutrally(
    mlx_version: Callable[[str], None], version: str,
) -> None:
    """Bug caught: the text naming mlx 0.32.4's behavior (and an over-statement) for every
    uncalibrated version, so a user on 0.31.1 is told the estimate errs high when nothing
    supports that."""
    mlx_version(version)
    messages = _calibration_warnings(
        lambda: estimate_peak(_shape(), _cfg(), load_calibration()))
    assert len(messages) == 1
    msg = messages[0]
    assert "stock attention" in msg
    assert version in msg
    assert "not calibrated" in msg
    assert "over-state" not in msg
    assert "0.32.4" not in msg


def test_calibration_warning_is_a_user_warning() -> None:
    """Bug caught: CalibrationWarning derived from a category the default filters hide
    (e.g. DeprecationWarning), so API users would never see it."""
    assert issubclass(CalibrationWarning, UserWarning)


# --- (2) stock on a calibrated mlx, or with no mlx distribution, is silent -------------


@pytest.mark.parametrize("version", ["0.32.0", "0.31.2"])
def test_stock_estimate_on_calibrated_mlx_is_silent(
    mlx_version: Callable[[str], None], version: str,
) -> None:
    """Bug caught: a calibrated version missing from the set (e.g. the set holding only
    "0.31.2"), or the warning firing unconditionally."""
    mlx_version(version)
    assert _calibration_warnings(
        lambda: estimate_peak(_shape(), _cfg(), load_calibration())) == []


def test_stock_estimate_without_an_mlx_distribution_is_silent_and_unchanged(
    monkeypatch: pytest.MonkeyPatch, mlx_version: Callable[[str], None],
) -> None:
    """Bug caught: the `PackageNotFoundError` from the metadata lookup not caught, so a
    pure estimate on a machine without mlx raises instead of returning its numbers."""
    calib = load_calibration()
    mlx_version("0.32.0")
    with_mlx = estimate_peak(_shape(), _cfg(), calib)

    def _no_mlx(name: str) -> str:
        if name == "mlx":
            raise metadata.PackageNotFoundError(name)
        return _REAL_DIST_VERSION(name)

    monkeypatch.setattr(metadata, "version", _no_mlx)
    results: list[object] = []
    messages = _calibration_warnings(
        lambda: results.append(estimate_peak(_shape(), _cfg(), calib)))
    assert messages == []
    assert results == [with_mlx]


def test_stock_estimate_does_not_import_mlx() -> None:
    """Bug caught: the planner reading the version through `mlx.core.__version__`, which
    loads mlx (and Metal) into a process that only asked for arithmetic."""
    code = """
import sys
from mlx_train_perf.plan.calibration import load_calibration
from mlx_train_perf.plan.estimate import ModelShape, TrainConfig, estimate_peak
shape = ModelShape(vocab=1000, hidden=64, layers=2, intermediate=128, heads=4, kv_heads=2,
                   tied=False, quant_bits=None, quant_group=None)
cfg = TrainConfig(batch=1, seq_len=256, dtype="bfloat16", lora_rank=8, lora_layers=2,
                  grad_checkpoint=True, impl="kernel", attention="stock")
peak, _ = estimate_peak(shape, cfg, load_calibration())
print(peak > 0, "mlx" in sys.modules, "mlx.core" in sys.modules)
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       timeout=120, check=False)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "True False False"


# --- (3) flash on an uncalibrated mlx is silent ---------------------------------------


def test_flash_estimate_on_uncalibrated_mlx_is_silent(
    mlx_version: Callable[[str], None],
) -> None:
    """Bug caught: the `attention == "stock"` condition dropped, so a flash-only
    estimate (which never uses the quadratic term) warns too."""
    mlx_version("0.32.4")
    assert _calibration_warnings(
        lambda: estimate_peak(_shape(), _cfg(attention="flash"), load_calibration())) == []


# --- (4) one warning per user call, not per searched candidate ------------------------


def test_plan_fit_warns_once_across_its_suggestion_search(
    mlx_version: Callable[[str], None],
) -> None:
    """Bug caught: `_find_fit`'s candidate loop going through the warning entry point,
    so one `plan_fit` call warns once per candidate (13 candidates here)."""
    mlx_version("0.32.4")
    reports = []
    messages = _calibration_warnings(lambda: reports.append(
        plan_fit(_shape(), _cfg(batch=4096, seq_len=8192), budget_bytes=3 * 1024**3)))
    assert reports[0].fits is False
    assert reports[0].suggestion is not None
    assert len(messages) == 1


# A budget the micro model fits well above the floor and well below the ceiling, so the
# bisection must probe several values on either side.
_SEARCH_BUDGET = 2 * 1024**3


def test_max_seq_len_search_warns_once(mlx_version: Callable[[str], None]) -> None:
    """Bug caught: the floor check or the bisection warning per probed seq_len."""
    mlx_version("0.32.4")
    found: list[int] = []
    messages = _calibration_warnings(lambda: found.append(max_seq_len_for_budget(
        _shape(), _cfg(), budget_bytes=_SEARCH_BUDGET)))
    assert len(messages) == 1
    assert 1 < found[0] < 65536  # the search went past the floor: many probes, one warning


def test_max_batch_search_warns_once(mlx_version: Callable[[str], None]) -> None:
    """Bug caught: the inverse batch query missing the warning, or warning per probe."""
    mlx_version("0.32.4")
    found: list[int] = []
    messages = _calibration_warnings(lambda: found.append(max_batch_for_budget(
        _shape(), _cfg(), budget_bytes=_SEARCH_BUDGET)))
    assert len(messages) == 1
    assert 1 < found[0] < 4096  # the search went past the floor: many probes, one warning


# --- the warning points at the caller's line ------------------------------------------


@pytest.mark.parametrize("entry", ["estimate_peak", "plan_fit", "max_seq", "max_batch"])
def test_warning_points_at_the_callers_line(
    mlx_version: Callable[[str], None], entry: str,
) -> None:
    """Bug caught: a fixed `stacklevel` that is right for `estimate_peak` but points into
    `plan/inverse.py` for the two inverse searches (one more frame deep)."""
    mlx_version("0.32.4")
    calls: dict[str, Callable[[], object]] = {
        "estimate_peak": lambda: estimate_peak(_shape(), _cfg(), load_calibration()),
        "plan_fit": lambda: plan_fit(_shape(), _cfg(), budget_bytes=_SEARCH_BUDGET),
        "max_seq": lambda: max_seq_len_for_budget(_shape(), _cfg(),
                                                  budget_bytes=_SEARCH_BUDGET),
        "max_batch": lambda: max_batch_for_budget(_shape(), _cfg(),
                                                  budget_bytes=_SEARCH_BUDGET),
    }
    caught = _caught(calls[entry])
    assert len(caught) == 1
    assert caught[0].filename == __file__


# --- an invalid config raises before any warning --------------------------------------


@pytest.mark.parametrize("entry", ["estimate_peak", "plan_fit", "max_seq", "max_batch"])
def test_invalid_config_raises_without_a_warning_first(
    mlx_version: Callable[[str], None], entry: str,
) -> None:
    """Bug caught: the calibration warning raised before `_validate_cfg`, so a config
    the planner refuses (here `lora_rank=-1`) also prints a note about its estimate."""
    mlx_version("0.32.4")
    bad = TrainConfig(batch=1, seq_len=512, dtype="bfloat16", lora_rank=-1,
                      lora_layers=2, grad_checkpoint=True, impl="kernel", attention="stock")
    calls: dict[str, Callable[[], object]] = {
        "estimate_peak": lambda: estimate_peak(_shape(), bad, load_calibration()),
        "plan_fit": lambda: plan_fit(_shape(), bad, budget_bytes=_SEARCH_BUDGET),
        "max_seq": lambda: max_seq_len_for_budget(_shape(), bad,
                                                  budget_bytes=_SEARCH_BUDGET),
        "max_batch": lambda: max_batch_for_budget(_shape(), bad,
                                                  budget_bytes=_SEARCH_BUDGET),
    }

    def _call() -> None:
        with pytest.raises(PlanInputError, match="lora_rank"):
            calls[entry]()

    assert _calibration_warnings(_call) == []


# --- no number changes ----------------------------------------------------------------


def test_warning_does_not_change_the_estimate(mlx_version: Callable[[str], None]) -> None:
    """Bug caught: the uncalibrated branch also rescaling the attention term (this
    release warns; it does not re-fit)."""
    calib = load_calibration()
    cfg = _cfg(seq_len=8192)
    mlx_version("0.32.0")
    calibrated = estimate_peak(_shape(), cfg, calib)
    mlx_version("0.32.4")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CalibrationWarning)
        uncalibrated = estimate_peak(_shape(), cfg, calib)
    assert uncalibrated == calibrated


# --- (5) the CLI prints it on stderr and keeps its exit code --------------------------


def _config(tmp_path: Path) -> Path:
    cfg = {"vocab_size": 1000, "hidden_size": 64, "num_hidden_layers": 2,
           "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
           "tie_word_embeddings": False, "model_type": "llama"}
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return p


def _plan_argv(tmp_path: Path, *extra: str) -> list[str]:
    return ["plan", "--config", str(_config(tmp_path)), "--lora-rank", "8", *extra]


@pytest.mark.parametrize(("extra", "expected_rc"), [
    (("--batch", "1", "--seq-len", "512", "--budget-gb", "8", "--json"), 0),
    (("--batch", "4096", "--seq-len", "8192", "--budget-gb", "3", "--json"), 1),
])
def test_cli_plan_prints_warning_on_stderr_and_keeps_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
    mlx_version: Callable[[str], None], extra: tuple[str, ...], expected_rc: int,
) -> None:
    """Bug caught: the CLI swallowing the warning (or sending it to stdout, which would
    also break `--json`), or the warning path changing the fits/does-not-fit exit code."""
    mlx_version("0.32.0")
    rc_calibrated = main(_plan_argv(tmp_path, *extra))
    calibrated = capsys.readouterr()
    mlx_version("0.32.4")
    rc = main(_plan_argv(tmp_path, *extra))
    captured = capsys.readouterr()
    assert rc == rc_calibrated == expected_rc
    assert json.loads(captured.out) == json.loads(calibrated.out)
    assert "stock attention" not in calibrated.err
    warning_lines = [line for line in captured.err.splitlines() if "stock attention" in line]
    assert len(warning_lines) == 1
    assert warning_lines[0].startswith("warning: ")
    assert "0.32.4" in warning_lines[0]


def test_cli_plan_search_prints_the_warning_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], mlx_version: Callable[[str], None],
) -> None:
    """Bug caught: `--max-seq` makes two planner calls (the search, then the forward
    report); printing each call's warning shows it twice for one command."""
    mlx_version("0.32.4")
    rc = main(_plan_argv(tmp_path, "--batch", "1", "--budget-gb", "8", "--max-seq"))
    err = capsys.readouterr().err
    assert rc == 0
    assert len([line for line in err.splitlines() if "stock attention" in line]) == 1


def test_cli_plan_flash_prints_no_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], mlx_version: Callable[[str], None],
) -> None:
    """Bug caught: the CLI printing a calibration note for every plan regardless of the
    attention model."""
    mlx_version("0.32.4")
    rc = main(_plan_argv(tmp_path, "--batch", "1", "--seq-len", "512", "--budget-gb", "8",
                         "--attention", "flash"))
    assert rc == 0
    assert "stock attention" not in capsys.readouterr().err


def test_cli_plan_shows_another_warning_its_module_filter_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mlx_version: Callable[[str], None],
) -> None:
    """Bug caught: the CLI re-issuing a non-calibration warning through the filters a
    second time (`warn_explicit`). The re-issue has no module name, so a filter that
    allowed the warning by its module no longer matches, the catch-all "ignore" behind
    it wins, and the user never sees a warning the filters had already let through."""
    mlx_version("0.32.0")
    message = "an unrelated planner note"
    real_plan_fit = cli.plan_fit

    def _noisy_plan_fit(*args: object, **kwargs: object) -> object:
        warnings.warn(message, UserWarning, stacklevel=1)
        return real_plan_fit(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(cli, "plan_fit", _noisy_plan_fit)
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter("ignore")
        warnings.filterwarnings("always", category=UserWarning,
                                module=re.escape(__name__) + "$")
        rc = main(_plan_argv(tmp_path, "--batch", "1", "--seq-len", "512",
                             "--budget-gb", "8"))
    assert rc == 0
    assert [str(w.message) for w in shown] == [message]
