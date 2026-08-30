from qwen35_tiny import tiny_qwen35


def test_tiny_model_has_both_flavors():
    # Catches: a tiny config whose every layer is one flavor -- the wrapper and
    # refusal tests would silently cover half the surface.
    layers = tiny_qwen35().language_model.model.layers
    assert any(layer.is_linear for layer in layers)
    assert any(not layer.is_linear for layer in layers)
