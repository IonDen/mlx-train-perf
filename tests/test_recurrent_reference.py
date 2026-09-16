import pytest

pytest.importorskip("mlx_lm")

from mlx_lm.models.gated_delta import gated_delta_ops

from mlx_train_perf.recurrent.reference import (
    PINNED_SOURCE_HASHES,
    mirrored_source_hashes,
    sequential_gated_delta,
)


def test_sequential_oracle_is_mlx_lm_ops():
    assert sequential_gated_delta() is gated_delta_ops


def test_mirrored_sources_match_pins():
    # Catches: an mlx-lm patch inside >=0.31.3,<0.32 changing any surface the
    # proxy mirrors or the parity oracle -- silent semantic drift becomes red.
    assert mirrored_source_hashes() == PINNED_SOURCE_HASHES
