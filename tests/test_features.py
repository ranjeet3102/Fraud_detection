def test_scaled_output_shape(split_data, scaled_data):
    X, _ = split_data
    assert scaled_data.shape == X.shape

def test_scaled_mean_std(scaled_data):
    import numpy as np
    assert np.allclose(scaled_data.mean(), 0, atol=1e-7)
    assert np.allclose(scaled_data.std(), 1, atol=1e-7)
