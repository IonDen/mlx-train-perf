from mlx_train_perf.errors import UnverifiedMlxError

VERIFIED_MLX_VERSIONS: frozenset[str] = frozenset({"0.31.2"})


def _installed_mlx_version() -> str:
    import mlx.core as mx  # noqa: PLC0415

    return str(mx.__version__)  # type: ignore[attr-defined]  # mlx stubs omit __version__


def check_mlx_verified(*, allow_unverified: bool) -> None:
    """Kernel impl gate: JIT codegen is version-sensitive (register cliffs are sharp)."""
    ver = _installed_mlx_version()
    if ver in VERIFIED_MLX_VERSIONS or allow_unverified:
        return
    raise UnverifiedMlxError(
        f"mlx {ver} is not kernel-verified (verified: {sorted(VERIFIED_MLX_VERSIONS)}). "
        f"Pass allow_unverified_mlx=True to proceed at your own risk (run the parity suite "
        f"first: pytest --run-metal), or use impl='chunked'."
    )
