"""Parameter slicing tests; no optimizer is run."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from param_loader import load_metanet_params, param_slice


def test_static_parameters_are_not_sliced_or_padded():
    params = load_metanet_params(num_segments=3)
    actual = param_slice(params, 1, 2, desired_length=4)

    assert actual.keys() == params.keys()
    for key in params:
        assert actual[key].shape == (3,)
        np.testing.assert_array_equal(actual[key], params[key])


@pytest.mark.parametrize("stored_timesteps", [5, 9])
def test_dynamic_parameters_are_sliced_regardless_of_stored_duration(stored_timesteps):
    params = load_metanet_params(num_segments=2)
    dynamic = np.arange(stored_timesteps * 2, dtype=float).reshape(stored_timesteps, 2)
    params["tau"] = dynamic

    actual = param_slice(params, 1, 4)

    np.testing.assert_array_equal(actual["tau"], dynamic[1:4])
    assert not np.shares_memory(actual["tau"], dynamic)
    for key in params:
        if key != "tau":
            np.testing.assert_array_equal(actual[key], params[key])


@pytest.mark.parametrize("start, end, desired_length, expected_rows", [
    (3, 5, 4, [3, 4, 4, 4]),
    (3, 10, 4, [3, 4, 4, 4]),
    (5, 7, 2, [4, 4]),
    (1, 4, 3, [1, 2, 3]),
])
def test_dynamic_padding_repeats_final_available_row(start, end, desired_length, expected_rows):
    params = load_metanet_params(num_segments=2)
    dynamic = np.arange(10, dtype=float).reshape(5, 2)
    params["tau"] = dynamic.copy()

    actual = param_slice(params, start, end, desired_length=desired_length)

    assert actual["tau"].shape == (desired_length, 2)
    np.testing.assert_array_equal(actual["tau"], dynamic[expected_rows])
    np.testing.assert_array_equal(params["tau"], dynamic)
