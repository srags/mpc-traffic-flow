"""Three persistence checks: complete round-trip, safe writes, and run lookup."""

import sys
from copy import deepcopy
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traffic_flow import load_result, save_result
from traffic_flow.config import CalRef, InitMode, MPCConfig, ScenarioConfig, Study
from traffic_flow.inputs.scenario import Scenario, prepare_traffic_data
from traffic_flow.model.parameters import default_metanet_params
from traffic_flow.types import OptimizationResult, RunResult, SimulationResult
import traffic_flow.results.policies as policies


@pytest.fixture
def run_result():
    spec = ScenarioConfig("test", "day", 0.4, 2, 10 / 3600, 6)
    traffic = prepare_traffic_data(
        np.full((6, 4), 50.), np.full((6, 4), 3000.), np.array([4., 3.]),
    )
    params = default_metanet_params(2)
    params["r"] = np.arange(12, dtype=np.float32).reshape(6, 2)
    history = np.arange(14, dtype=float).reshape(7, 2)
    result = OptimizationResult(
        np.arange(12, dtype=float).reshape(6, 2) + 200.,
        SimulationResult(history + 30., history + 80., np.ones((7, 1)), 12.25),
        SimulationResult(history + 25., history + 85., np.zeros((7, 1)), 9.75),
    )
    config = MPCConfig(
        3, 2, control_zone=(1,), safety_temporal=0.7, safety_spatial=25,
        initialize_vsl=np.full((6, 2), 100., dtype=np.float32),
        init_fixed=(InitMode.ADAPTIVE, 120.),
    )
    return RunResult(Scenario(spec, traffic), params, config, result)


def test_complete_run_round_trip(tmp_path, monkeypatch, run_result):
    original = deepcopy(run_result)
    directory = tmp_path / "new" / "run"
    path = save_result(run_result, directory)
    assert path == directory / "run.npz"
    before = path.read_bytes()
    real_load = np.load

    def read_bundle_only(file, *args, **kwargs):
        assert Path(file) == path, "Loading must not reread observations or calibration"
        assert kwargs.get("allow_pickle") is False
        return real_load(file, *args, **kwargs)

    monkeypatch.setattr(policies.np, "load", read_bundle_only)
    restored = load_result(directory)
    # Compare both the restored result and the input, which saving must not mutate.
    for actual in (restored, run_result):
        assert actual.scenario.spec == original.scenario.spec
        for field in fields(original.config):
            expected, value = getattr(original.config, field.name), getattr(actual.config, field.name)
            if isinstance(expected, np.ndarray):
                assert value.dtype == expected.dtype
                np.testing.assert_array_equal(value, expected)
            else:
                assert value == expected
        groups = (
            (actual.scenario.traffic._asdict(), original.scenario.traffic._asdict()),
            (actual.params, original.params),
            (actual.optimization.baseline._asdict(), original.optimization.baseline._asdict()),
            (actual.optimization.controlled._asdict(), original.optimization.controlled._asdict()),
            ({"vsl": actual.optimization.vsl}, {"vsl": original.optimization.vsl}),
        )
        for values, expected_values in groups:
            assert values.keys() == expected_values.keys()
            for key, expected in expected_values.items():
                value, expected = np.asarray(values[key]), np.asarray(expected)
                assert value.shape == expected.shape and value.dtype == expected.dtype, key
                np.testing.assert_array_equal(value, expected, err_msg=key)
    assert path.read_bytes() == before
    assert list(directory.iterdir()) == [path]


def test_failed_save_keeps_previous_bundle(tmp_path, monkeypatch, run_result):
    path = save_result(run_result, tmp_path)
    before = path.read_bytes()

    def fail_write(stream, **arrays):
        stream.write(b"partial archive")
        raise OSError("simulated write failure")

    monkeypatch.setattr(policies.np, "savez_compressed", fail_write)
    with pytest.raises(OSError, match="simulated write failure"):
        save_result(run_result, tmp_path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_settings_separate_runs_and_lookup_finds_the_right_one(tmp_path, monkeypatch, run_result):
    monkeypatch.setattr(policies, "REPO_DIR", tmp_path)
    calibration = CalRef("custom_fit", interval=3)
    study = Study.SAFETY_SWEEP
    spec, config = run_result.scenario.spec, run_result.config
    first = policies.run_dir(spec, calibration=calibration, config=config, study=study)
    assert first.parent == tmp_path / "results/test/test_day/custom_fit/control_h_3/safety_sweep/runs"
    assert policies.run_dir(
        spec, calibration=calibration, study=study,
        config=replace(config, verbose=True, tee=True),
    ) == first
    # Same visible bounds, different horizon or seed must not overwrite each other.
    for settings in (
        replace(config, pred_horizon=4),
        replace(config, initialize_vsl=config.initialize_vsl + 1),
    ):
        assert policies.run_dir(spec, calibration=calibration, config=settings, study=study) != first
    assert list(tmp_path.iterdir()) == []  # Resolving paths does not create anything.

    other = replace(run_result, config=replace(config, speed_lb=60.))
    second = policies.run_dir(spec, calibration=calibration, config=other.config, study=study)
    assert second != first
    save_result(run_result, first)
    save_result(other, second)
    # An identical run in another study must not leak into this lookup.
    save_result(run_result, policies.run_dir(spec, calibration=calibration, config=config))
    selected = policies.load_runs(
        spec.freeway, spec.date, calibration=calibration, study=study,
        where=lambda settings: settings.speed_lb == config.speed_lb,
    )
    assert len(selected) == 1
    np.testing.assert_array_equal(selected[0].optimization.vsl, run_result.optimization.vsl)
    assert selected[0].config.pred_horizon == config.pred_horizon
    assert len(policies.load_runs(spec.freeway, spec.date, calibration=calibration, study=study)) == 2
    assert policies.load_runs(spec.freeway, "missing", calibration=calibration, study=study) == []
