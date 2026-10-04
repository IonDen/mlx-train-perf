import json

from _plan_golden import FIXTURE, record  # tests/ is on sys.path via conftest rootdir


def test_estimate_peak_matches_the_pre_memfit_golden_grid() -> None:
    """Bug caught: any change in a component value, the peak rounding, the component
    order, or which configs raise (and with what message) after the memfit rewrite."""
    assert record() == json.loads(FIXTURE.read_text())


def test_component_order_is_the_documented_order() -> None:
    """Bug caught: reordering the terms in `_train_step_model` (dict equality in the grid
    test ignores key order, but the order is the public component order and the order in
    which input errors surface)."""
    rec = record()
    for value in rec.values():
        assert isinstance(value, dict)
        if "components" in value:
            assert list(value["components"]) == [
                "weights", "base", "lora", "optimizer", "activations", "attention", "loss"
            ]
