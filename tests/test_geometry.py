"""Road geometry belongs to the scenario, not observations or calibration options."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traffic_flow.config import CalibrationConfig, RobustOptConfig, ScenarioSpec
from traffic_flow.inputs import scenario as scenario_module
from traffic_flow.inputs.scenario import Scenario, prepare_traffic_data
from traffic_flow.pipeline import calibrate
from traffic_flow.solvers import calibration


@pytest.mark.parametrize("missing_mask", [None, "on", "off"], ids=["ramp-files", "missing-on", "missing-off"])
def test_load_scenario_separates_geometry_from_observations(tmp_path, monkeypatch, missing_mask):
    # All fixture files are temporary; repository data/ remains read-only.
    directory = tmp_path / "data" / "i24" / "i24_test"
    directory.mkdir(parents=True)
    raw_density = np.tile([10., 25., 40., 12.], (3, 1))
    raw_flow = raw_density * 80.
    np.save(directory / "rho_hat.npy", raw_density)
    np.save(directory / "q_hat.npy", raw_flow)
    np.save(directory / "lane_mapping.npy", np.array([99., 2.5, 4., 98.]))
    if missing_mask != "on":
        np.save(directory / "on_ramp_mapping.npy", np.array([1, 1, 0, 1], dtype=bool))
    if missing_mask != "off":
        np.save(directory / "off_ramp_mapping.npy", np.array([1, 0, 1, 1], dtype=bool))
    monkeypatch.setattr(scenario_module, "REPO_DIR", tmp_path)

    if missing_mask is not None:
        with pytest.raises(FileNotFoundError, match=f"{missing_mask}_ramp_mapping"):
            scenario_module.load_scenario("i24", "test", smoothing_window=None)
        return

    scenario = scenario_module.load_scenario("i24", "test", smoothing_window=None)
    spec, traffic = scenario.spec, scenario.traffic
    assert (spec.time_steps, spec.num_segments) == (3, 2)
    np.testing.assert_array_equal(spec.lanes, [2.5, 4.])
    np.testing.assert_array_equal(spec.on_ramps, [True, False])
    np.testing.assert_array_equal(spec.off_ramps, [False, True])
    assert traffic._fields == (
        "density", "flow", "velocity", "inflow", "downstream_density",
        "initial_density", "initial_velocity",
    )
    np.testing.assert_array_equal(traffic.density, raw_density[:, 1:-1] / spec.lanes)
    np.testing.assert_array_equal(traffic.flow, raw_flow[:, 1:-1])
    np.testing.assert_array_equal(traffic.velocity, np.full((3, 2), 80.))
    np.testing.assert_array_equal(traffic.downstream_density, raw_density[:, -1] / spec.lanes[-1])


@pytest.mark.parametrize("mode", ["nominal", "time-varying", "robust"])
def test_calibration_uses_scenario_geometry(monkeypatch, mode):
    """Build real models, checking handoff, units and masks without solving."""
    spec = ScenarioSpec("test", "day", 0.4, 2, 10 / 3600, 3,
                          lanes=np.array([2.5, 4.]),
                          on_ramps=np.array([True, False]),
                          off_ramps=np.array([False, True]))
    density = np.tile([10., 25., 40., 12.], (3, 1))
    traffic = prepare_traffic_data(density, density * 80., spec.lanes, smoothing_window=None)
    scenario = Scenario(spec, traffic)
    status = SimpleNamespace(solver=SimpleNamespace(
        termination_condition=calibration.pyo.TerminationCondition.optimal,
    ))
    solver = Mock()
    solver.options = {}
    solver.solve.return_value = status
    monkeypatch.setattr(calibration.pyo, "SolverFactory", Mock(return_value=solver))
    robust = mode == "robust"
    fitter = Mock(wraps=calibration.metanet_param_fit_robust if robust else calibration.metanet_param_fit)
    unused_fitter = Mock(side_effect=AssertionError("Wrong calibration mode"))
    monkeypatch.setattr(calibration, "metanet_param_fit", unused_fitter if robust else fitter)
    monkeypatch.setattr(calibration, "metanet_param_fit_robust", fitter if robust else unused_fitter)
    config = CalibrationConfig(
        robust_opt=RobustOptConfig(S=2, bc_noise_percent=0.) if robust else None,
        time_varying_ramps=mode == "time-varying",
        tee=False,
    )

    params = calibrate(scenario, config)
    fitter.assert_called_once()
    unused_fitter.assert_not_called()
    solver.solve.assert_called_once()
    model = solver.solve.call_args.args[0]
    args, kwargs = fitter.call_args
    np.testing.assert_array_equal(args[0], traffic.velocity)
    np.testing.assert_array_equal(args[1], traffic.density * spec.lanes)
    assert args[2:4] == (spec.time_step, spec.L)
    np.testing.assert_array_equal(args[4], spec.lanes)
    np.testing.assert_array_equal(args[5], traffic.inflow)
    np.testing.assert_array_equal(args[6], traffic.downstream_density * spec.lanes[-1])
    np.testing.assert_array_equal(kwargs["on_ramps"], spec.on_ramps)
    np.testing.assert_array_equal(kwargs["off_ramps"], spec.off_ramps)
    assert "ramp_mapping" not in kwargs

    np.testing.assert_array_equal([calibration.pyo.value(model.n_lanes[i]) for i in model.i], spec.lanes)
    for t in model.t:
        for i in model.i:
            assert calibration.pyo.value(model.v_hat[t, i]) == traffic.velocity[t, i]
            assert calibration.pyo.value(model.rho_hat[t, i]) == traffic.density[t, i] * spec.lanes[i]
    assert model.beta[0].fixed and not model.beta[1].fixed
    assert calibration.pyo.value(model.beta[0]) == 1e-3
    inflows = ([model.r_inflow[t, 0], model.r_inflow[t, 1]] for t in model.t) if mode == "time-varying" else (
        [model.r_inflow[0], model.r_inflow[1]],
    )
    for allowed, absent in inflows:
        assert not allowed.fixed and absent.fixed
        assert calibration.pyo.value(absent) == (0. if robust else 1e-3)
    assert params["r"].shape == ((spec.time_steps, spec.num_segments) if mode == "time-varying" else (spec.num_segments,))
    np.testing.assert_array_equal(params["beta"], [1e-3, 1e-3])
