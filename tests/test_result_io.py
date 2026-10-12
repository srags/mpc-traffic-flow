"""Persistence and viewer checks for explicit NPZ files."""

import importlib.util
import json
import sys
import warnings
from copy import deepcopy
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traffic_flow import save_result
from traffic_flow.results.io import load_npz
from traffic_flow.config import CalRef, InitMode, MPCConfig, ScenarioConfig, Study
from traffic_flow.inputs.scenario import Scenario, prepare_traffic_data
from traffic_flow.model.parameters import default_metanet_params
from traffic_flow.types import OptimizationResult, RunResult, SimulationResult
import traffic_flow.paths as paths
import traffic_flow.results.analysis as analysis
import traffic_flow.results.io as result_io


@pytest.fixture
def run_result(tmp_path, monkeypatch):
    # Both the path builder and I/O's display root stay inside pytest's sandbox.
    monkeypatch.setattr(paths, "REPO_DIR", tmp_path)
    monkeypatch.setattr(result_io, "REPO_DIR", tmp_path)
    spec = ScenarioConfig("test", "day", 0.4, 2, 10 / 3600, 6)
    traffic = prepare_traffic_data(
        np.full((6, 4), 50.), np.full((6, 4), 3000.), np.array([4., 3.]),
    )
    params = default_metanet_params(2)
    params["r"] = np.arange(12, dtype=np.float32).reshape(6, 2)
    history = np.arange(14, dtype=float).reshape(7, 2)
    result = OptimizationResult(
        np.arange(12, dtype=float).reshape(6, 2) + 200.,
        SimulationResult(history + 30., history + 80., np.ones((7, 1)),
                         spec.time_step * (spec.L * ((history[:-1] + 30.) * traffic.lanes).sum() + 7.)),
        SimulationResult(history + 25., history + 85., np.zeros((7, 1)),
                         spec.time_step * spec.L * ((history[:-1] + 25.) * traffic.lanes).sum()),
    )
    config = MPCConfig(
        3, 2, control_zone=(1,), safety_temporal=0.7, safety_spatial=25,
        initialize_vsl=np.full((6, 2), 100., dtype=np.float32),
        init_fixed=(InitMode.ADAPTIVE, 120.),
    )
    return RunResult(Scenario(spec, traffic), params, config, result)


def test_complete_run_round_trip(tmp_path, monkeypatch, run_result):
    original = deepcopy(run_result)
    path = save_result(run_result, calibration=CalRef("custom_fit"), study=None, file_name="trial")
    directory = tmp_path / "results/test/test_day/custom_fit"
    assert path == directory / "trial.npz"
    before = path.read_bytes()
    real_load = np.load

    def read_bundle_only(file, *args, **kwargs):
        assert Path(file) == path, "Loading must not reread observations or calibration"
        assert kwargs.get("allow_pickle") is False
        return real_load(file, *args, **kwargs)

    monkeypatch.setattr(result_io.np, "load", read_bundle_only)
    restored = load_npz(path)
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
    path = save_result(run_result, calibration=CalRef(), study=None)
    before = path.read_bytes()

    def fail_write(stream, **arrays):
        stream.write(b"partial archive")
        raise OSError("simulated write failure")

    monkeypatch.setattr(result_io.np, "savez_compressed", fail_write)
    with pytest.raises(OSError, match="simulated write failure"):
        save_result(run_result, calibration=CalRef(), study=None)
    assert path.read_bytes() == before
    assert list(path.parent.iterdir()) == [path]


def test_named_bundles_and_lookup_find_the_right_run(tmp_path, monkeypatch, run_result):
    calibration = CalRef("custom_fit", interval=3)
    study = Study.SAFETY_SWEEP
    spec, config = run_result.scenario.spec, run_result.config
    directory = paths.collection_dir(spec.freeway, spec.date, calibration=calibration, study=study)
    assert directory == tmp_path / "results/test/test_day/custom_fit/control_h_3/safety_sweep"
    assert paths.find_npzs(spec.freeway, spec.date, calibration=calibration, study=study) == []
    assert list(tmp_path.iterdir()) == []  # Resolving paths does not create anything.

    other = replace(run_result, config=replace(config, speed_lb=60.))
    first = save_result(run_result, calibration=calibration, study=study, file_name="run")
    second = save_result(other, calibration=calibration, study=study, file_name="run2")
    assert (first, second) == (directory / "run.npz", directory / "run2.npz")
    # An identical run in another study must not leak into this lookup.
    ordinary = save_result(run_result, calibration=calibration, study=None)
    (directory / "unfinished").mkdir()

    def no_array_loading(*args, **kwargs):
        pytest.fail("Finding run directories must not load their arrays")

    with monkeypatch.context() as discovery_only:
        discovery_only.setattr(np, "load", no_array_loading)
        assert paths.find_npzs(
            spec.freeway, spec.date, calibration=calibration, study=study,
        ) == sorted([first, second])
        assert paths.find_npzs(
            spec.freeway, "missing", calibration=calibration, study=study,
        ) == []
        # Default discovery stays in the selected directory: nested studies
        # must not leak into the ordinary-run lookup.
        assert paths.find_npzs(spec.freeway, spec.date, calibration=calibration) == [ordinary]

    selected = result_io.load_results(
        spec.freeway, spec.date, calibration=calibration, study=study,
        where=lambda settings: settings.speed_lb == config.speed_lb,
    )
    assert len(selected) == 1
    assert selected[0].config.speed_lb == config.speed_lb
    np.testing.assert_array_equal(selected[0].optimization.vsl, run_result.optimization.vsl)
    assert selected[0].config.pred_horizon == config.pred_horizon
    assert len(result_io.load_results(spec.freeway, spec.date, calibration=calibration, study=study)) == 2
    assert len(result_io.load_results(spec.freeway, spec.date, calibration=calibration)) == 1
    assert result_io.load_results(spec.freeway, "missing", calibration=calibration, study=study) == []

    # A misplaced bundle is an input error, even if the filter would exclude it.
    wrong_scenario = replace(run_result.scenario, spec=replace(spec, date="wrong_day"))
    wrong_path = save_result(replace(run_result, scenario=wrong_scenario),
                             calibration=calibration, study=study)
    wrong_path.replace(first)
    with pytest.raises(ValueError, match="Scenario metadata disagrees with directory"):
        result_io.load_results(
            spec.freeway, spec.date, calibration=calibration, study=study,
            where=lambda settings: False,
        )


@pytest.fixture
def viewer(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "html/server.py"
    module_spec = importlib.util.spec_from_file_location("traffic_viewer_server", source)
    viewer = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(viewer)
    monkeypatch.setattr(viewer, "RESULTS_ROOT", tmp_path / "results")
    return viewer


def test_viewer_discovers_named_npzs_and_passes_full_paths(tmp_path, monkeypatch, run_result, viewer):
    root = viewer.RESULTS_ROOT
    calibration = CalRef("custom_fit", interval=3)
    first = save_result(run_result, calibration=calibration, study=Study.SAFETY_SWEEP)
    second = save_result(run_result, calibration=calibration, study=Study.SAFETY_SWEEP,
                         file_name="run2")
    (second.parent / "broken.npz").write_bytes(b"not an archive")

    loaded = []

    def load_file(path):
        assert path in (first, second, second.parent / "broken.npz")
        assert path.is_file() and path.suffix == ".npz"
        loaded.append(path)
        return load_npz(path)

    monkeypatch.setattr(viewer, "load_npz", load_file)
    choices = viewer.discover_scenarios()
    assert {choice["id"] for choice in choices} == {
        first.relative_to(root).as_posix(), second.relative_to(root).as_posix(),
    }
    assert len({choice["label"] for choice in choices}) == 2
    assert all(choice["calibration"] == "custom_fit/control_h_3/safety_sweep" for choice in choices)
    assert {choice["run"] for choice in choices} == {first.stem, second.stem}
    for choice in choices:
        assert choice["dataset"] == "test"
        assert choice["dateKey"] == "day"
        assert choice["calibrationSource"] == "custom_fit"
        assert choice["calibrationInterval"] == 3
        assert choice["study"] == "safety_sweep"
        assert choice["parameters"] == {
            "hold_length": 1, "speed_lb": 40.,
            "safety_temporal": 0.7, "safety_spatial": 25,
            "pred_horizon": 3, "control_horizon": 2,
        }
    assert set(loaded) == {first, second, second.parent / "broken.npz"}
    payload = viewer.load_scenario(second.relative_to(root).as_posix())
    assert payload["scenario"]["label"].endswith(" · run2")
    assert payload["metadata"]["timeSteps"] == run_result.scenario.spec.time_steps
    np.testing.assert_allclose(payload["delay"]["byCutoff"][0], payload["delay"]["baseline"])
    np.testing.assert_allclose(payload["delay"]["byCutoff"][-1], payload["delay"]["controlled"])

    for invalid in (str(second), "../outside.npz", "test/day/not-npz.txt"):
        with pytest.raises(ValueError, match="Invalid scenario path"):
            viewer.load_scenario(invalid)


@pytest.mark.parametrize("dataset,date,source,interval,study", [
    ("i24", "11_30", "calibration_static/fixed_ramping", None, ""),
    ("i24", "11_29", "calibration_dynamic", 6, "hold_length"),
    ("i24", "11_28", "calibration_static/time_varying_ramping", None, "speed_lb"),
    ("test", "day", "custom/speed_lb/nested", 3, "safety_sweep"),
])
def test_viewer_summary_separates_filters_and_uses_saved_settings(
    run_result, viewer, dataset, date, source, interval, study,
):
    spec = replace(run_result.scenario.spec, freeway=dataset, date=date)
    config = replace(run_result.config, hold_length=5, speed_lb=57.,
                     safety_temporal=None, safety_spatial=0.25,
                     pred_horizon=17, control_horizon=4)
    run = replace(run_result, scenario=replace(run_result.scenario, spec=spec), config=config)
    context = Path(source)
    if interval is not None:
        context /= f"control_h_{interval}"
    if study:
        context /= study
    # Filenames are run identities, not the source of the numerical settings.
    relative = Path(dataset) / f"{dataset}_{date}" / context / "hold_999_temp_999_spat_999.npz"
    summary = viewer._scenario_summary(run, relative)
    assert summary["dataset"] == dataset
    assert summary["dateKey"] == date
    assert summary["date"] == date.replace("_", "/")
    assert summary["calibrationSource"] == source
    assert summary["calibrationInterval"] == interval
    assert summary["study"] == study
    assert summary["calibration"] == context.as_posix()
    assert summary["run"] == relative.stem
    assert summary["parameters"] == {
        "hold_length": 5, "speed_lb": 57.,
        "safety_temporal": None, "safety_spatial": 0.25,
        "pred_horizon": 17, "control_horizon": 4,
    }
    json.dumps(summary, allow_nan=False)


def extended_run(run, baseline_steps, controlled_steps, *, legacy=False):
    """Tiny unequal histories with nonzero terminal costs to expose accounting errors."""
    spec, traffic = run.scenario.spec, run.scenario.traffic
    simulations = []
    for steps, offset in ((baseline_steps, 30.), (controlled_steps, 25.)):
        density = np.arange((steps + 1) * spec.num_segments, dtype=float).reshape(-1, spec.num_segments) + offset
        velocity = density + 40.
        queue = np.arange(steps + 1, dtype=float).reshape(-1, 1) + 1.
        counted_density = density if legacy else density[:-1]
        total = spec.time_step * (spec.L * (counted_density @ traffic.lanes).sum() + queue.sum())
        simulations.append(SimulationResult(density, velocity, queue, float(total)))
    return replace(run, optimization=OptimizationResult(run.optimization.vsl, *simulations))


@pytest.mark.parametrize("baseline_steps,controlled_steps", [(8, 10), (10, 8)])
@pytest.mark.parametrize("legacy", [False, True])
def test_viewer_unequal_histories_preserve_costs_and_reporting(
    tmp_path, monkeypatch, run_result, viewer, baseline_steps, controlled_steps, legacy,
):
    monkeypatch.setattr(analysis, "LATEST_UPDATE", not legacy)
    run = extended_run(run_result, baseline_steps, controlled_steps, legacy=legacy)
    original = deepcopy(run)
    path = save_result(run, calibration=CalRef(), study=None)
    # Inspect the same arrays after loading to catch accidental in-place padding.
    monkeypatch.setattr(viewer, "load_npz", lambda path: run)
    monkeypatch.setattr(viewer, "replay", lambda *args, **kwargs: pytest.fail("Saved drainage needs no replay"))
    payload = viewer.load_scenario(path.relative_to(viewer.RESULTS_ROOT).as_posix())
    steps = max(baseline_steps, controlled_steps)
    observed = run.scenario.spec.time_steps
    expected_metadata = {
        "timeSteps": steps, "observedSteps": observed,
        "baselineSteps": baseline_steps, "controlledSteps": controlled_steps,
        "hasDrainage": True, "drainageRequested": False,
    }
    assert {key: payload["metadata"][key] for key in expected_metadata} == expected_metadata
    assert payload["vsl"][:observed] == run.optimization.vsl.tolist()
    assert payload["vsl"][observed:] == [None] * (steps - observed)
    free_flow = analysis.run_analysis(run).ff_tt
    assert payload["delay"]["freeFlow"] == pytest.approx(free_flow)

    def elapsed(simulation, cutoff):
        own_steps = len(simulation.density) - 1
        used = min(cutoff, own_steps)
        road = run.scenario.spec.L * (simulation.density[:used] @ run.scenario.traffic.lanes).sum()
        queue = simulation.queue[:used].sum()
        if cutoff >= own_steps:
            queue += simulation.queue[-1].sum()
            if legacy:
                road += run.scenario.spec.L * (simulation.density[-1] @ run.scenario.traffic.lanes)
        return run.scenario.spec.time_step * (road + queue)

    baseline, controlled = run.optimization.baseline, run.optimization.controlled
    expected = [baseline.total_travel_time - elapsed(baseline, cutoff)
                + elapsed(controlled, cutoff) - free_flow for cutoff in range(steps + 1)]
    np.testing.assert_allclose(payload["delay"]["byCutoff"], expected)
    assert payload["delay"]["byCutoff"][0] == pytest.approx(baseline.total_travel_time - free_flow)
    assert payload["delay"]["byCutoff"][-1] == pytest.approx(controlled.total_travel_time - free_flow)
    for shown, simulation, before in (
        (payload["baseline"], baseline, original.optimization.baseline),
        (payload, controlled, original.optimization.controlled),
    ):
        own_steps = len(simulation.density) - 1
        for name in ("velocity", "density", "queue"):
            values = getattr(simulation, name)
            displayed = np.asarray(shown[name])
            assert displayed.shape == ((steps,) if name == "queue" else (steps, run.scenario.spec.num_segments))
            expected_values = values.reshape(-1) if name == "queue" else values
            np.testing.assert_allclose(displayed[:own_steps], expected_values[:-1])
            if own_steps < steps:
                np.testing.assert_allclose(displayed[own_steps:], np.broadcast_to(expected_values[-1], displayed[own_steps:].shape))
            np.testing.assert_array_equal(values, getattr(before, name))
    json.dumps(payload, allow_nan=False)


def test_viewer_drainage_is_opt_in_cached_and_does_not_write(tmp_path, monkeypatch, run_result, viewer):
    path = save_result(run_result, calibration=CalRef(), study=None)
    before = path.read_bytes()
    scenario_id = path.relative_to(viewer.RESULTS_ROOT).as_posix()
    drained = extended_run(run_result, 8, 10)
    calls = []

    def replay(run, *, until_ff):
        assert until_ff is True
        np.testing.assert_array_equal(run.optimization.vsl, run_result.optimization.vsl)
        calls.append(run)
        warnings.warn("Drainage stopped at its step limit", RuntimeWarning)
        return drained

    monkeypatch.setattr(viewer, "replay", replay)
    assert [choice["id"] for choice in viewer.discover_scenarios()] == [scenario_id]
    saved = viewer.load_scenario(scenario_id)
    assert calls == []
    assert saved["metadata"]["timeSteps"] == run_result.scenario.spec.time_steps
    assert saved["metadata"]["hasDrainage"] is False
    assert saved["metadata"]["drainageRequested"] is False
    expanded = viewer.load_scenario(scenario_id, drain=True)
    assert expanded["metadata"]["timeSteps"] == 10
    assert expanded["metadata"]["drainageRequested"] is True
    assert expanded["metadata"]["hasDrainage"] is True
    assert expanded["metadata"]["warnings"] == ["Drainage stopped at its step limit"]
    assert viewer.load_scenario(scenario_id, drain=True) is expanded
    assert viewer.load_scenario(scenario_id) is saved
    assert len(calls) == 1
    assert path.read_bytes() == before
    assert list(path.parent.iterdir()) == [path]
