"""Parameter-loading contracts; all input files live in tmp_path."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from param_loader import load_metanet_params


@pytest.fixture
def parameter_files(tmp_path):
    expected = {
        "tau": np.array([18., 20.]) / 3600,
        "K": np.array([40., 42.]),
        "eta_high": np.array([30., 35.]),
        "p_crit": np.array([37.45, 40.]),
        "v_free": np.array([120., 115.]),
        "a": np.array([1.4, 1.6]),
    }
    for key, values in expected.items():
        filename = "rho_crit" if key == "p_crit" else key
        np.save(tmp_path / f"{filename}.npy", values)
    return tmp_path, expected


@pytest.mark.parametrize("with_optional_files", [False, True])
@pytest.mark.parametrize("string_path", [False, True])
def test_static_parameter_loading(parameter_files, with_optional_files, string_path):
    path, expected = parameter_files
    optional = {
        "r": ("r_inflow_array.npy", [100., 200.], 0),
        "beta": ("beta_array.npy", [0.1, 0.2], 0),
        "gamma": ("gamma_array.npy", [0.8, 0.9], 1),
    }
    for key, (filename, values, default) in optional.items():
        if with_optional_files:
            np.save(path / filename, values)
        expected[key] = np.array(values) if with_optional_files else np.full(2, default)
    expected["q_capacity"] = np.full(2, 2400)

    actual = load_metanet_params(
        path=str(path) if string_path else path, num_segments=2
    )

    assert actual.keys() == expected.keys()
    for key, values in expected.items():
        assert actual[key].shape == values.shape
        np.testing.assert_array_equal(actual[key], values, err_msg=key)


@pytest.mark.parametrize("filename", [
    "tau.npy", "K.npy", "eta_high.npy", "rho_crit.npy", "v_free.npy", "a.npy",
])
def test_missing_required_parameter_raises(parameter_files, filename):
    path, _ = parameter_files
    (path / filename).unlink()
    with pytest.raises(FileNotFoundError):
        load_metanet_params(path=path, num_segments=2)


@pytest.mark.parametrize("filename", [
    "r_inflow_array.npy", "beta_array.npy", "gamma_array.npy",
])
def test_corrupt_optional_parameter_raises(parameter_files, filename):
    path, _ = parameter_files
    (path / filename).write_bytes(b"not a numpy file")
    with pytest.raises(ValueError):
        load_metanet_params(path=path, num_segments=2)


@pytest.fixture
def dynamic_parameter_files(parameter_files):
    path, required = parameter_files
    expected_blocks = []
    for i in (1, 2):
        folder = path / "control_h_3" / f"params_{i}"
        folder.mkdir(parents=True)
        expected = {key: values * i for key, values in required.items()}
        for key, values in expected.items():
            filename = "rho_crit" if key == "p_crit" else key
            np.save(folder / f"{filename}.npy", values)
        expected.update(q_capacity=np.full(2, 2400), r=np.zeros(2),
                        beta=np.zeros(2), gamma=np.ones(2))
        # First block uses defaults; second has distinct optional values.
        if i == 2:
            for key, filename, values in (
                ("r", "r_inflow_array.npy", [100., 200.]),
                ("beta", "beta_array.npy", [0.1, 0.2]),
                ("gamma", "gamma_array.npy", [0.8, 0.9]),
            ):
                expected[key] = np.array(values)
                np.save(folder / filename, expected[key])
        expected_blocks.append(expected)
    return path, expected_blocks


@pytest.mark.parametrize("num_timesteps", [2, 3, 5, 6])
def test_dynamic_block_order_and_duration(dynamic_parameter_files, num_timesteps):
    path, expected_blocks = dynamic_parameter_files
    actual = load_metanet_params(
        path=path, control_h=3, num_timesteps=num_timesteps, num_segments=2
    )

    assert actual.keys() == expected_blocks[0].keys()
    for key, values in actual.items():
        assert values.shape == (num_timesteps, 2)
        for t in range(num_timesteps):
            np.testing.assert_array_equal(
                values[t], expected_blocks[t // 3][key], err_msg=f"{key} at t={t}"
            )


def test_dynamic_missing_block_raises(dynamic_parameter_files):
    path, _ = dynamic_parameter_files
    # Seven steps require a third folder, which does not exist.
    with pytest.raises(FileNotFoundError):
        load_metanet_params(path=path, control_h=3, num_timesteps=7, num_segments=2)


@pytest.mark.parametrize("filename, key", [
    ("tau.npy", "tau"), ("r_inflow_array.npy", "r"),
])
def test_dynamic_wrong_segment_count_raises(dynamic_parameter_files, filename, key):
    path, _ = dynamic_parameter_files
    np.save(path / "control_h_3" / "params_2" / filename, np.zeros(3))
    with pytest.raises(ValueError, match=f"params_2/{key}: expected 2 values, got 3"):
        load_metanet_params(path=path, control_h=3, num_timesteps=6, num_segments=2)


@pytest.mark.parametrize("control_h, num_timesteps", [(0, 6), (-1, 6), (3, 0), (3, -1)])
def test_dynamic_nonpositive_duration_raises(tmp_path, control_h, num_timesteps):
    with pytest.raises(ValueError, match="must be positive"):
        load_metanet_params(
            path=tmp_path, control_h=control_h,
            num_timesteps=num_timesteps, num_segments=2,
        )
