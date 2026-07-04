"""Register-pressure probe: a direct, compile-time observable for kernel developers.

`compiled_ceiling` recompiles a JIT-generated kernel body standalone through the Metal
framework (PyObjC) and reads back `MTLComputePipelineState.maxTotalThreadsPerThreadgroup`
-- the device schedules fewer threads per threadgroup as a compiled kernel's per-thread
register footprint grows, so this value shrinks below the device maximum exactly when the
compiler allocated more registers per lane.

SEMANTICS -- read this before drawing any conclusion from a ceiling number:

This is a register-pressure TELLTALE, not a performance verdict. A drop in the compiled
ceiling after an edit means the compiler allocated more registers per lane than before,
which is worth investigating -- it is never, by itself, proof that a kernel got slower.
Two kernel variants compiled at the exact SAME ceiling can still differ substantially in
measured throughput (their issue streams differ in ways this observable can't see), and a
variant compiled at a LOWER ceiling than another can still be the faster one in practice.
Judge kernel variants only by measured rate at a saturating shape (n large enough that
occupancy stops being the limiter); use `compiled_ceiling` to help explain *why* a rate
changed, never to predict *whether* one variant will beat another.

How the probe works: the given body is launched once, at a deliberately tiny shape, as a
real kernel with `verbose=True` -- this makes the JIT print the exact MSL text it compiled
for that dispatch (buffer bindings, the auto-generated `<name>_shape` arrays, the template
instantiation) to stdout. That text is captured at the file-descriptor level (the dump
happens below Python's own stdout redirection), the banner line and markdown code fences
the dump is wrapped in are stripped, and the minimal prelude the JIT's own preprocessing
supplies at real compile time is prepended. The result is recompiled standalone via
PyObjC's Metal bindings, and `maxTotalThreadsPerThreadgroup` is read off the resulting
compute pipeline state.
"""
import contextlib
import os
import re
import tempfile
from collections.abc import Callable, Iterator
from typing import IO, cast

import mlx.core as mx

from mlx_train_perf.errors import MissingDependencyError, RegisterProbeError

# The installed mlx 0.31.2 stub types mx.fast.metal_kernel's return as `object` (nanobind
# gives it no more specific type); it is documented + actually a callable kernel invoker
# (same cast convention as mlx_train_perf.core.kernel.launch._MetalKernel).
_MetalKernel = Callable[..., list[mx.array]]

_BANNER_RE = re.compile(r"Generated source code for `[^`]+`:\s*\n")
_PRELUDE = "#include <metal_stdlib>\nusing namespace metal;\ntypedef bfloat bfloat16_t;\n"

# Tiny probe shape: large enough to compile and dispatch validly, small enough that a
# probe launch never approaches the launch-budget guard. Matches the shape this capture
# technique was validated against before landing here.
_PROBE_N = 8
_PROBE_D = 32
_PROBE_V = 16

# The dense forward kernel's fixed input/output contract
# (mlx_train_perf.core.kernel.launch._dense_kernel) -- the body shape `compiled_ceiling`
# is ground-truthed against.
_INPUT_NAMES = ["hidden", "w", "targets", "offs", "lse_in", "tgt_in"]
_OUTPUT_NAMES = ["lse_out", "tgt_out"]


def _strip_banner_and_fences(raw: str) -> str:
    """Pure text transform: drop the `verbose=True` banner line and the markdown code
    fences the dump is wrapped in. No MLX/pyobjc dependency -- safe to unit-test directly
    against a synthetic string."""
    text = _BANNER_RE.sub("", raw, count=1)
    return "\n".join(line for line in text.splitlines() if line.strip() != "```")


def _prepare_msl(raw: str) -> str:
    """Pure assembly: strip the capture noise, then prepend the minimal prelude a real
    compile step supplies automatically but a standalone recompile needs spelled out."""
    return _PRELUDE + _strip_banner_and_fences(raw)


@contextlib.contextmanager
def _capture_fd_stdout() -> Iterator[IO[str]]:
    """`verbose=True` prints from MLX's C++ layer, which writes straight to fd 1 -- a
    Python-level stdout redirect would miss it, so capture at the file-descriptor level."""
    with tempfile.TemporaryFile(mode="w+") as buf:
        saved = os.dup(1)
        os.dup2(buf.fileno(), 1)
        try:
            yield buf
        finally:
            os.dup2(saved, 1)
            os.close(saved)


def _capture_generated_msl(msl_body: str, *, header: str) -> str:
    """Launch `msl_body` once, at the tiny probe shape, as a real kernel with
    `verbose=True`, and return the cleaned MSL text the JIT actually compiled."""
    mx.random.seed(0)
    hidden = mx.random.normal((_PROBE_N, _PROBE_D)).astype(mx.bfloat16)
    w = mx.random.normal((_PROBE_V, _PROBE_D)).astype(mx.bfloat16)
    targets = mx.random.randint(0, _PROBE_V, (_PROBE_N,))
    offs = mx.array([0, _PROBE_V], dtype=mx.uint32)
    lse = mx.full((_PROBE_N,), float("-inf"), dtype=mx.float32)
    tgt = mx.zeros((_PROBE_N,), dtype=mx.float32)
    mx.eval(hidden, w, targets, offs, lse, tgt)
    kernel = cast(
        _MetalKernel,
        mx.fast.metal_kernel(
            name="mtp_regpressure_probe",
            input_names=_INPUT_NAMES,
            output_names=_OUTPUT_NAMES,
            source=msl_body,
            header=header,
        ),
    )
    with _capture_fd_stdout() as buf:
        out = kernel(
            inputs=[hidden, w, targets, offs, lse, tgt],
            template=[("T", mx.bfloat16)],
            grid=(32, 8, 1),
            threadgroup=(32, 8, 1),
            output_shapes=[(_PROBE_N,), (_PROBE_N,)],
            output_dtypes=[mx.float32, mx.float32],
            verbose=True,
        )
        mx.eval(out)
        os.fsync(1)
        buf.seek(0)
        raw = buf.read()
    if "[[kernel]]" not in raw and "kernel void" not in raw:  # pragma: no cover -- tripwire;
        # only reachable if MLX's verbose dump format changes underneath this probe.
        raise RegisterProbeError(f"no MSL captured for the probe launch; got:\n{raw[:500]}")
    return _prepare_msl(raw)


def compiled_ceiling(msl_body: str, *, header: str = "") -> int:
    """Compile-time register-pressure telltale for a kernel body (see the module
    docstring for exact semantics -- diagnostic only, never a performance verdict).

    Launches `msl_body` once at a tiny shape to capture the real JIT-generated MSL,
    recompiles it standalone through PyObjC's Metal bindings, and returns
    `MTLComputePipelineState.maxTotalThreadsPerThreadgroup` for the resulting pipeline.
    `header` is forwarded to the probe launch's `mx.fast.metal_kernel(header=...)` for
    bodies that need helper functions declared outside the kernel (as the quantized
    kernel's dequantize helper does).
    """
    try:
        import Metal  # noqa: PLC0415
    except ImportError as exc:
        raise MissingDependencyError(
            "pyobjc-framework-Metal is required for the register-pressure probe; "
            'install the optional "probe" extra (pip install "mlx-train-perf[probe]")'
        ) from exc
    msl = _capture_generated_msl(msl_body, header=header)
    device = Metal.MTLCreateSystemDefaultDevice()
    options = Metal.MTLCompileOptions.new()
    library, compile_error = device.newLibraryWithSource_options_error_(msl, options, None)
    if library is None:  # pragma: no cover -- tripwire; only reachable if the capture+
        # prelude assembly produces MSL the standalone compiler rejects.
        raise RegisterProbeError(f"standalone MSL recompile failed: {compile_error}")
    names = list(library.functionNames())
    if len(names) != 1:  # pragma: no cover -- tripwire; the probe always launches exactly
        # one kernel function per call.
        raise RegisterProbeError(f"expected exactly one compiled kernel function, got {names}")
    function = library.newFunctionWithName_(names[0])
    pipeline_state, pso_error = device.newComputePipelineStateWithFunction_error_(function, None)
    if pipeline_state is None:  # pragma: no cover -- tripwire; a function that compiled
        # into a library always yields a valid pipeline state.
        raise RegisterProbeError(f"failed to create a compute pipeline state: {pso_error}")
    return int(pipeline_state.maxTotalThreadsPerThreadgroup())
