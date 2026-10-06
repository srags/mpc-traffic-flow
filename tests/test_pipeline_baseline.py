"""Representative numerical regressions; no solver or plot runs."""

import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_flow import load_params, load_result, load_scenario
from traffic_flow.config import CalRef
from traffic_flow.pipeline import simulate_scenario

REFERENCE = "4f82654b1d6c9812f7577d21e33b1df406099edd"
RTOL, ATOL = 1e-10, 1e-10
DATA = ROOT / "data/i24/i24_11_28"
PARAMS = DATA / "calibration_static/fixed_ramping"
# Historical policies stay in the backup; newly generated bundles stay in results.
VSL = ROOT / "results_bu/i24/i24_11_28/calibration_static/fixed_ramping/optimal_vsl.npy"
RUNS = ROOT / "results/i24/i24_11_30/calibration_static/fixed_ramping/runs"


@pytest.fixture(scope="module")
def baseline():
    # Load the old dependency set independently, without checkout/reset or any
    # writes to source/data. Restore current imports before yielding to tests.
    names = ("sim_types", "paths", "param_loader", "data_loader", "traffic_sim")
    saved = {name: sys.modules.get(name) for name in names}
    loaded = {}
    try:
        for name in names:
            source = subprocess.check_output(
                ["git", "show", f"{REFERENCE}:src/{name}.py"], cwd=ROOT, text=True
            )
            module = ModuleType(name)
            module.__file__ = str(ROOT / "src" / f"{name}.py")
            sys.modules[name] = module
            exec(compile(source, f"{REFERENCE}:src/{name}.py", "exec"), module.__dict__)
            loaded[name] = module
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return SimpleNamespace(**loaded)


def load_observations(module, initial_step=0):
    if not (DATA / "rho_hat.npy").exists():
        pytest.skip("I-24 11/28 measurements are not installed")
    rows, columns = np.load(DATA / "rho_hat.npy", mmap_mode="r").shape
    freeway = module.Freeway("i24", "11_28", L=0.4, num_segments=columns - 2,
                             time_step=10 / 3600, time_steps=rows,
                             start_time=initial_step * 10 / 3600)
    # Avoid floating-point time-to-index truncation in this unrelated test.
    freeway.start_time_step = initial_step
    return freeway.load_real_data()


@pytest.mark.parametrize("real_data,controlled", [
    pytest.param(True, False, id="measured-inflow-baseline"),
    pytest.param(False, True, id="modeled-origin-policy"),
])
def test_preparation_and_simulation_match_pre_refactor(baseline, real_data, controlled):
    old = load_observations(baseline.data_loader)
    scenario = load_scenario("i24", "11_28")
    spec, traffic = scenario.spec, scenario.traffic
    assert (spec.L, spec.time_step) == (0.4, 10 / 3600)
    assert (spec.time_steps, spec.num_segments) == old[0].shape
    for name, actual, expected in zip(traffic._fields, traffic, old):
        assert actual.shape == expected.shape, name
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=False)

    if not PARAMS.exists():
        pytest.skip("I-24 11/28 saved calibration is not installed")
    if controlled and not VSL.is_file():
        pytest.fail(f"Historical policy is missing: {VSL}")
    controls = (np.load(VSL, allow_pickle=False) if controlled
                else np.full((spec.time_steps, spec.num_segments), 150.))
    actual = simulate_scenario(
        traffic, load_params(scenario, calibration=CalRef()),
        T=spec.time_step, l=spec.L, vsl=controls, real_data=real_data,
    )
    _, _, _, lanes, inflow, downstream, density, velocity = old
    simulator = baseline.traffic_sim.METANET_Simulator(
        T=spec.time_step, l=spec.L,
        params=baseline.param_loader.load_metanet_params(path=PARAMS, num_segments=spec.num_segments),
        lanes=dict(enumerate(lanes)), real_data=real_data,
    )
    expected = simulator.run_with_history(
        np.pad(inflow, (0, 1), mode="edge"), downstream,
        baseline.sim_types.MetanetState(density.copy(), velocity.copy(), float(inflow[0]), 0.),
        controls.copy(),
    )
    for name, new, previous in zip(actual._fields, actual, expected):
        assert np.shape(new) == np.shape(previous), name
        assert np.isfinite(new).all() and np.isfinite(previous).all(), name
        np.testing.assert_allclose(new, previous, rtol=RTOL, atol=ATOL,
                                   equal_nan=False, err_msg=name)


def test_latest_saved_cc_run_matches_pre_refactor(baseline):
    # One representative new run keeps routine test time independent of sweep size.
    paths = list(RUNS.glob("*/run.npz"))
    if not paths:
        pytest.skip("Run cc_run.py once to check a newly saved bundle")
    path = max(paths, key=lambda candidate: (candidate.stat().st_mtime_ns, str(candidate)))
    before = path.read_bytes()
    run = load_result(path.parent)
    spec, traffic = run.scenario.spec, run.scenario.traffic
    assert (spec.freeway, spec.date) == ("i24", "11_30")
    assert run.optimization.vsl.shape == (spec.time_steps, spec.num_segments)
    for phase, controls in (
        ("baseline", np.full_like(run.optimization.vsl, 150.)),
        ("controlled", run.optimization.vsl.copy()),
    ):
        simulator = baseline.traffic_sim.METANET_Simulator(
            T=spec.time_step, l=spec.L,
            params={key: value.copy() for key, value in run.params.items()},
            lanes=dict(enumerate(traffic.lanes)), real_data=False,
        )
        expected = simulator.run_with_history(
            np.pad(traffic.inflow, (0, 1), mode="edge"),
            traffic.downstream_density.copy(),
            baseline.sim_types.MetanetState(
                traffic.initial_density.copy(), traffic.initial_velocity.copy(),
                float(traffic.inflow[0]), 0.,
            ),
            controls,
        )
        actual = getattr(run.optimization, phase)
        for name, saved, previous in zip(actual._fields, actual, expected):
            assert np.shape(saved) == np.shape(previous), f"{phase}.{name}"
            assert np.isfinite(saved).all() and np.isfinite(previous).all()
            np.testing.assert_allclose(saved, previous, rtol=RTOL, atol=ATOL,
                                       equal_nan=False, err_msg=f"{phase}.{name}")
    assert path.read_bytes() == before


def test_synthetic_reports_preserve_original_replay():
    """Protect the two headline peaks and the old unpadded-demand convention."""
    from traffic_flow.config import SyntheticConfig
    from traffic_flow.model.parameters import default_metanet_params
    from traffic_flow.model.simulation import METANET_Simulator
    from traffic_flow.results.synthetic import SyntheticResult, load_synthetic_results
    from traffic_flow.types import MetanetState

    policy_dir = ROOT / "results_bu/synthetic_10km/demand"
    peaks = (5500, 6250)
    paths = [policy_dir / f"demand{float(peak)}_duration0.5.csv" for peak in peaks]
    if not all(path.is_file() for path in paths):
        pytest.skip("Historical synthetic demand policies are not installed")
    original_files = [path.read_bytes() for path in paths]
    config = SyntheticConfig()
    assert (config.time_steps, config.num_segments, config.total_distance) == (720, 25, 10.0)
    params = default_metanet_params(25)
    results = load_synthetic_results(peaks, policy_dir)
    assert [result.peak for result in results] == list(peaks)
    assert all(isinstance(result, SyntheticResult) for result in results)

    for peak, path, result in zip(peaks, paths, results):
        # Independent transcription of the original experiment, not the new
        # demand/config/metric helpers. Keep its redundant padding as reference.
        T = 10 / 3600
        demand = np.array([
            float(peak) if int(0.055 / T) <= t < int((0.055 + 0.5) / T) else 4000.0
            for t in range(721)
        ])
        lanes = {i: 4.0 if i < 20 else 2.0 for i in range(25)}
        state = MetanetState(np.full(25, demand[0] / (4 * 90)), np.full(25, 90.0), demand[0], 0)
        sim = METANET_Simulator(T=T, l=0.4, params=params, lanes=lanes, real_data=False)
        baseline = sim.run_with_history(demand, np.zeros(720), state)
        vsl = np.loadtxt(path, delimiter=",")
        padded = np.pad(demand, (0, 756 - len(demand)), mode="edge")
        controlled = sim.run_with_history(padded[:720], np.zeros(756)[:720], state, vsl)
        np.testing.assert_array_equal(result.optimization.vsl, vsl)
        for actual, expected in ((result.optimization.baseline, baseline),
                                 (result.optimization.controlled, controlled)):
            for new, old in zip(actual, expected):
                np.testing.assert_allclose(new, old, rtol=RTOL, atol=ATOL, equal_nan=False)

        num_veh = float(np.sum(demand)) * T
        ff = num_veh * (10.0 / float(params["v_free"][0]))
        delay, delay_c = baseline[-1] - ff, controlled[-1] - ff
        np.testing.assert_allclose(
            (result.num_veh, result.free_flow_ttt, result.delay, result.controlled_delay, result.cc),
            (num_veh, ff, delay, delay_c, (delay - delay_c) / delay * 100),
            rtol=RTOL, atol=ATOL, equal_nan=False,
        )
    assert [path.read_bytes() for path in paths] == original_files


def test_synthetic_missing_or_malformed_policy(tmp_path):
    from traffic_flow.results.synthetic import load_synthetic_results

    with pytest.raises(FileNotFoundError, match="No saved VSL policy"):
        load_synthetic_results([5500], tmp_path)
    with pytest.raises(FileNotFoundError, match="No requested synthetic policies"):
        load_synthetic_results([5500], tmp_path, skip_missing=True)
    policy = tmp_path / "demand5500.0_duration0.5.csv"
    np.savetxt(policy, np.full((2, 25), 150.0), delimiter=",")
    with pytest.raises(ValueError, match="expected VSL shape"):
        load_synthetic_results([5500], tmp_path, skip_missing=True)


def test_cc_report_keeps_metrics_and_printed_summary(monkeypatch, capsys):
    """Moving cc.py must preserve report values, without plotting or solving."""
    from unittest.mock import Mock

    from traffic_flow.config import ScenarioConfig
    from traffic_flow.inputs.scenario import Scenario
    from traffic_flow.model.parameters import default_metanet_params
    from traffic_flow.results.analysis import cc_report
    from traffic_flow.types import OptimizationResult, SimulationResult, TrafficData

    spec = ScenarioConfig("test", "day", 1., 2, 0.01, 2)
    density = np.array([[10., 20.], [10., 20.]])
    velocity = np.array([[50., 99.], [50., 99.]])
    lanes = np.array([2., 3.])
    traffic = TrafficData(
        density, density * velocity * lanes, velocity, lanes,
        np.full(2, 5000.), np.full(2, 20.), density[0].copy(), velocity[0].copy(),
    )
    scenario = Scenario(spec, traffic)
    params = default_metanet_params(2)
    params["v_free"] = np.full(2, 100.)
    # The terminal row and diagnostic TTS deliberately differ: neither belongs
    # in the fit comparison or the saved policy's modeled-origin CC calculation.
    diagnostic = SimulationResult(
        np.vstack((density, [999., 999.])),
        np.vstack((velocity, [999., 999.])), np.zeros((3, 1)), 999.,
    )
    result = OptimizationResult(
        np.array([[120., 100.], [80., 90.]]),
        diagnostic._replace(total_travel_time=12.),
        diagnostic._replace(velocity=np.array([[80., 100.], [80., 100.], [999., 999.]]),
                            total_travel_time=7.),
    )
    # Protect supplied observations, parameters, policy and saved histories.
    for array in (*traffic, *params.values(), result.vsl, *result.baseline,
                  *result.controlled, *diagnostic):
        if isinstance(array, np.ndarray):
            array.flags.writeable = False
    simulate = Mock(return_value=diagnostic)
    monkeypatch.setattr("traffic_flow.pipeline.simulate_scenario", simulate)

    report = cc_report(scenario, params, result)

    simulate.assert_called_once_with(traffic, params, T=0.01, l=1., steps=2, real_data=True)
    assert report.result is result
    np.testing.assert_allclose((report.free_flow_ttt, report.delay, report.controlled_delay),
                               (2., 10., 5.), rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(report.display_vsl, [[150., 100.], [80., 90.]])
    np.testing.assert_array_equal(result.vsl, [[120., 100.], [80., 90.]])
    np.testing.assert_allclose(report.observed_delay, [[0.6, 0.6], [1 / 165, 1 / 165]],
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(report.pct_decrease, [[75., 75.], [0., 0.]], rtol=RTOL, atol=ATOL)
    assert report.fit_rows == {
        "Velocity": "MAPE 0.00%, RMSE 0.00",
        "Density": "MAPE 0.00%, RMSE 0.00",
        "Flow": "MAPE 0.00%, RMSE 0.00",
        "Travel Time": "1.60 veh-hr vs 1.60 veh-hr (MAPE 0.00%)",
    }
    assert report.cc_rows == {
        "Total free flow travel time": "2.00 veh-hrs",
        "Controllable congestion": "50.00%",
        "Delay (ground truth)": "0.01 min - 0.60 min",
        "Pct decrease": "0.00% - 75.00%",
    }

    from traffic_flow.results.analysis import print_cc_report

    print_cc_report(report)
    output = capsys.readouterr().out
    assert output.index("No Control") < output.index("Optimized VSLs")
    for label, value in (*report.fit_rows.items(), *report.cc_rows.items()):
        assert label in output and value in output


def test_virtual_trajectory_analysis_preserves_existing_values():
    """Freeze the current positive-speed integration before moving its module."""
    from traffic_flow.results.analysis import get_virtual_trajectory, vt_travel_time_stats

    speed = np.array([
        [30., 60., 90., 30., 60., 90., 30., 60.],
        [60., 90., 30., 60., 90., 30., 60., 90.],
        [90., 30., 60., 90., 30., 60., 90., 30.],
    ])
    speed.flags.writeable = False
    # Captured from trajectories.py before consolidation; preserve the repeated
    # starting point and boundary behavior rather than changing the algorithm.
    for direction, times, positions in (
        (1, [2., 3., 3.25, 3.25, 4., 5.], [0., 1., 1.25, 1.25, 2., 2.5]),
        (-1, [2., 2.5, 3., 3.25, 3.25, 3.5, 4., 4.5],
         [2.5, 2., 1.5, 1.25, 1.25, 1., 0.5, 0.]),
    ):
        actual = get_virtual_trajectory(speed, 3.25, 1.25, 30., 0.5, direction=direction)
        np.testing.assert_allclose(actual, (times, positions), rtol=RTOL, atol=ATOL)

    extended = np.tile(speed, (1, 4))
    extended.flags.writeable = False
    times = vt_travel_time_stats(extended, time_step=10 / 3600, num_samples=4)
    assert times is not None, "vt_travel_time_stats must return vt_times outside the plotting branch"
    # The legacy integrator uses d_time=4 and d_space=0.5 internally, despite
    # converting travel time with the caller's time_step. This move keeps that.
    np.testing.assert_allclose(times, [3.666583333333333, 3.444444444444444,
                                      1.722222222222222, 0.], rtol=RTOL, atol=ATOL)
