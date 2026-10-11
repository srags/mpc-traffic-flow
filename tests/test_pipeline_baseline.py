"""Representative numerical regressions; no solver or plot runs."""

import ast
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_flow import load_params, load_scenario
from traffic_flow.results.io import load_npz
from traffic_flow.config import CalRef
from traffic_flow.pipeline import simulate

REFERENCE = "4f82654b1d6c9812f7577d21e33b1df406099edd"
MAIN_REFERENCE = "b332162f1f08acd366778a597e21c6ade026311b"
RTOL, ATOL = 1e-10, 1e-10
DATA = ROOT / "data/i24/i24_11_28"
PARAMS = DATA / "calibration_static/fixed_ramping"
# Historical policies stay in the backup; newly generated bundles stay in results.
VSL = ROOT / "results_bu/i24/i24_11_28/calibration_static/fixed_ramping/optimal_vsl.npy"
RUNS = ROOT / "results/i24/i24_11_30/calibration_static/fixed_ramping"


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
    actual = simulate(
        scenario, load_params(scenario, calibration=CalRef()),
        vsl=controls, real_data=real_data,
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
    # The current history API excludes terminal road occupancy from TTT, unlike
    # 4f82654. Keep all trajectories and the full queue contribution unchanged.
    terminal_road_ttt = spec.time_step * spec.L * (expected[0][-1] @ lanes)
    expected = (*expected[:-1], expected[-1] - terminal_road_ttt)
    for name, new, previous in zip(actual._fields, actual, expected):
        assert np.shape(new) == np.shape(previous), name
        assert np.isfinite(new).all() and np.isfinite(previous).all(), name
        np.testing.assert_allclose(new, previous, rtol=RTOL, atol=ATOL,
                                   equal_nan=False, err_msg=name)


def test_latest_saved_cc_run_matches_pre_refactor(baseline):
    # One representative new run keeps routine test time independent of sweep size.
    paths = [path for path in RUNS.glob("*.npz") if path.is_file()]
    if not paths:
        pytest.skip("Run cc_run.py once to check a newly saved bundle")
    path = max(paths, key=lambda candidate: (candidate.stat().st_mtime_ns, str(candidate)))
    before = path.read_bytes()
    run = load_npz(path)
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
        # Saved current runs use the same terminal-density exclusion as the
        # history API. Do not relax tolerances or accept arbitrary TTT changes.
        terminal_road_ttt = spec.time_step * spec.L * (expected[0][-1] @ traffic.lanes)
        expected = (*expected[:-1], expected[-1] - terminal_road_ttt)
        actual = getattr(run.optimization, phase)
        for name, saved, previous in zip(actual._fields, actual, expected):
            assert np.shape(saved) == np.shape(previous), f"{phase}.{name}"
            assert np.isfinite(saved).all() and np.isfinite(previous).all()
            np.testing.assert_allclose(saved, previous, rtol=RTOL, atol=ATOL,
                                       equal_nan=False, err_msg=f"{phase}.{name}")
    assert path.read_bytes() == before


@pytest.mark.parametrize("method,until_ff", [
    pytest.param("run_with_history", False, id="fixed-horizon"),
    pytest.param("run_with_history", True, id="drained"),
    pytest.param("run_with_opt", False, id="optimization-history"),
    pytest.param("run_with_opt", True, id="optimization-history-drained"),
])
def test_history_ttt_excludes_terminal_density_but_keeps_full_queue(monkeypatch, method, until_ff):
    """Pin the new accounting with a nonzero terminal queue and unequal lanes."""
    from unittest.mock import Mock

    from traffic_flow.model.parameters import default_metanet_params
    from traffic_flow.model.simulation import METANET_Simulator
    from traffic_flow.types import MetanetState

    terminal_density = [0.1, 0.2] if until_ff else [500, 600]
    states = [
        MetanetState(np.array(rho, dtype=float), np.full(2, 90.), 100., queue)
        for rho, queue in (([10, 20], 1.), ([30, 40], 2.), (terminal_density, 7.))
    ]
    simulator = METANET_Simulator(
        T=0.1, l=0.5, params=default_metanet_params(2), lanes={0: 2., 1: 3.},
    )
    simulator.until_ff = until_ff
    simulator.drain_tol = 1.0
    simulator.max_drain_steps = 2
    step = Mock(side_effect=states[1:])
    monkeypatch.setattr(simulator, "_step", step)
    steps = 1 if until_ff else 2
    result = getattr(simulator, method)(
        np.full(steps, 100.), np.zeros(steps), states[0],
    )
    density, velocity, queue = result[:3]
    ttt = result[-1]

    assert step.call_count == 2
    np.testing.assert_array_equal(density, [[10, 20], [30, 40], terminal_density])
    np.testing.assert_array_equal(velocity, np.full((3, 2), 90.))
    np.testing.assert_array_equal(queue, [[1], [2], [7]])
    # 0.1 * (0.5 * (10*2 + 20*3 + 30*2 + 40*3) + 1 + 2 + 7)
    np.testing.assert_allclose(ttt, 14., rtol=RTOL, atol=ATOL)


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


@pytest.fixture(scope="module")
def latest_main():
    """Read the pinned main implementation without checkout, imports or file writes."""
    source = subprocess.check_output(
        ["git", "show", f"{MAIN_REFERENCE}:src/traffic_sim.py"], cwd=ROOT, text=True,
    )
    namespace = {}
    exec(compile(source, f"{MAIN_REFERENCE}:src/traffic_sim.py", "exec"), namespace)
    # Load just the numerical metrics, not cc_analysis's plotting/data imports.
    source = subprocess.check_output(
        ["git", "show", f"{MAIN_REFERENCE}:src/cc_analysis.py"], cwd=ROOT, text=True,
    )
    functions = [node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef)
                 and node.name in {"ff_tts_vkm", "vehicles_served"}]
    assert len(functions) == 2
    exec(compile(ast.Module(body=functions, type_ignores=[]),
                 f"{MAIN_REFERENCE}:src/cc_analysis.py", "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.fixture(scope="module", params=[False, True], ids=["static", "time-varying"])
def drain_case(request):
    """Small corridor with unequal lanes, both ramp types and a nontrivial drain."""
    from traffic_flow.config import ScenarioConfig
    from traffic_flow.inputs.scenario import Scenario
    from traffic_flow.model.parameters import default_metanet_params
    from traffic_flow.types import TrafficData

    n, segments = 30, 3
    spec = ScenarioConfig("test", "day", 0.4, segments, 5 / 3600, n)
    density = np.tile([15., 18., 20.], (n, 1))
    velocity = np.full((n, segments), 90.)
    lanes = np.array([3., 4., 2.])
    traffic = TrafficData(
        density, density * velocity * lanes, velocity, lanes,
        np.full(n, 2000.), np.full(n, 10.), density[0].copy(), velocity[0].copy(),
    )
    params = default_metanet_params(segments)
    # Make baseline choices observable: modeled admission cannot meet demand,
    # and a 150 km/h cap is not equivalent to unrestricted free-flow speed.
    params["q_capacity"] = np.full(segments, 300.)
    params["v_free"] = np.array([170., 120., 110.])
    params["r"] = np.array([0., 150., 0.])
    params["beta"] = np.array([0., 0.05, 0.1])
    if request.param:
        params = {key: np.tile(value.astype(float), (n, 1)) for key, value in params.items()}
        params["v_free"] *= np.linspace(1., 0.9, n)[:, None]
        params["r"] *= np.linspace(0.8, 1.2, n)[:, None]
    vsl = np.full((n, segments), 70.)
    # Production calls must not alter observations, calibration or the saved policy.
    for array in (*traffic, *params.values(), vsl):
        array.flags.writeable = False
    return Scenario(spec, traffic), params, vsl


def main_replay(reference, scenario, params, vsl, *, real_data, until_ff):
    from traffic_flow.types import SimulationResult

    spec, traffic = scenario.spec, scenario.traffic
    result = reference.run_metanet_sim(
        T=spec.time_step, l=spec.L,
        init_traffic_state=(traffic.initial_density.copy(), traffic.initial_velocity.copy(),
                            float(traffic.inflow[0]), 0.),
        demand=traffic.inflow.copy(), downstream_density=traffic.downstream_density.copy(),
        params=deepcopy(params), lanes=dict(enumerate(traffic.lanes)),
        vsl_speeds=None if vsl is None else vsl.copy(),
        plotting=True, real_data=real_data, until_ff=until_ff,
    )
    # The agreed convention differs from main ONLY by terminal road occupancy.
    # Never subtract terminal queue or loosen tolerances to obtain parity.
    terminal = spec.time_step * spec.L * (result[0][-1] @ traffic.lanes)
    return SimulationResult(*result[:3], float(result[3] - terminal))


def assert_simulation_close(actual, expected):
    for name, new, old in zip(actual._fields, actual, expected):
        assert np.shape(new) == np.shape(old), name
        assert np.isfinite(new).all() and np.isfinite(old).all(), name
        np.testing.assert_allclose(new, old, rtol=RTOL, atol=ATOL, equal_nan=False,
                                   err_msg=name)


def test_simulation_matches_latest_main_with_agreed_ttt(latest_main, drain_case):
    scenario, params, vsl = drain_case
    for real_data, controls in ((True, None), (False, vsl)):
        for until_ff in (False, True):
            expected = main_replay(latest_main, scenario, params, controls,
                                   real_data=real_data, until_ff=until_ff)
            actual = simulate(scenario, params, controls, real_data=real_data, until_ff=until_ff)
            assert_simulation_close(actual, expected)
            if until_ff:
                # Catch ignored flags, a missing early stop, and untrimmed allocation.
                assert scenario.spec.time_steps + 1 < len(actual.density) < 11 * scenario.spec.time_steps + 1


@pytest.fixture(scope="module")
def drained_run(latest_main, drain_case):
    from traffic_flow.config import MPCConfig
    from traffic_flow.types import OptimizationResult, RunResult

    scenario, params, vsl = drain_case
    baseline = main_replay(latest_main, scenario, params, None, real_data=True, until_ff=True)
    controlled = main_replay(latest_main, scenario, params, vsl, real_data=False, until_ff=True)
    return RunResult(scenario, params, MPCConfig(pred_horizon=10, control_horizon=2),
                     OptimizationResult(vsl, baseline, controlled))


def test_evaluate_uses_main_baseline_and_drains_both_runs(drained_run):
    from traffic_flow.pipeline import evaluate

    run = drained_run
    actual = evaluate(run.scenario, run.params, run.optimization.vsl, until_ff=True)
    np.testing.assert_array_equal(actual.vsl, run.optimization.vsl)
    for phase in ("baseline", "controlled"):
        assert_simulation_close(getattr(actual, phase), getattr(run.optimization, phase))


def test_drained_report_matches_main_with_agreed_ttt(latest_main, drained_run):
    from traffic_flow.results.analysis import run_analysis

    run = drained_run
    spec, traffic = run.scenario.spec, run.scenario.traffic
    baseline, controlled = run.optimization.baseline, run.optimization.controlled
    rho, velocity = baseline.density[:-1], baseline.velocity[:-1]
    # Main's metric accepts aligned beta; use its own extension for dynamic inputs.
    extended = latest_main._extend_params_for_drain(
        run.params, spec.time_steps, len(rho), spec.num_segments,
    )
    ff = latest_main.ff_tts_vkm(rho, velocity, traffic.lanes, spec.time_step, spec.L, run.params["v_free"])
    served = latest_main.vehicles_served(rho, velocity, traffic.lanes, extended["beta"], spec.time_step)
    gt_tt = spec.time_step * spec.L * (traffic.density.sum(axis=0) @ traffic.lanes)
    window_tt = spec.time_step * spec.L * (rho[:spec.time_steps].sum(axis=0) @ traffic.lanes)
    sim_tt, opt_tt = baseline.total_travel_time, controlled.total_travel_time
    expected = {
        "gt_tt": gt_tt, "sim_tt": sim_tt, "opt_tt": opt_tt, "ff_tt": ff, "num_veh": served,
        "sim_error": np.mean(np.abs((traffic.velocity - velocity[:spec.time_steps]) / traffic.velocity)) * 100,
        "tts_error": abs(gt_tt - window_tt) / gt_tt * 100,
        "cc": np.clip((sim_tt - opt_tt) / (sim_tt - ff) * 100, 0, 100),
        "avg_tt_reduced_per_veh": (sim_tt - opt_tt) / served * 60,
        "uncontrolled_avg_tts": (sim_tt - ff) / served * 60,
        "controlled_avg_tts": (opt_tt - ff) / served * 60,
    }
    actual = run_analysis(run)
    for name, value in expected.items():
        np.testing.assert_allclose(getattr(actual, name), value, rtol=RTOL, atol=ATOL,
                                   equal_nan=False, err_msg=name)
