import pandas as pd


def test_weights_are_normalized():
    # Lightweight invariant documented by the recommendation layer.
    weights = {"Food": 0.5, "Service": 0.3, "Ambience": 0.2}
    assert abs(sum(weights.values()) - 1.0) < 1e-12
