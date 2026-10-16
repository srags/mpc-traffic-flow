"""Small, temporary-tree checks for importing main's legacy result collection."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "experiments/migrate_main_results.py"
SPEC = importlib.util.spec_from_file_location("main_result_migration", SOURCE)
migration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(migration)

from traffic_flow.model.parameters import default_metanet_params
from traffic_flow.types import SimulationResult
from traffic_flow.results import io as result_io


BASE = Path("i24/i24_test/calibration_static/fixed_ramping")


@pytest.mark.parametrize("source,target", [
    ("optimal_vsl.npy", "run.npz"),
    ("optimal_vsl_test.npy", "variants/test.npz"),
    ("optimal_vsl_tiled.npy", "variants/tiled.npz"),
    ("hold_length/optimal_vsl_12.npy", "hold_length/hl_12.npz"),
    ("hold_length/optimal_vsl_test.npy", "hold_length/variants/test.npz"),
    ("speed_lb/optimal_vsl_80.npy", "speed_lb/slb_80.npz"),
    ("safety_sweep/optimal_vsl_temp0.70_spat25.0.npy", "safety_sweep/temp_0.7_spat_25.npz"),
])
def test_policy_names_preserve_variants_outside_canonical_lookups(source, target):
    assert migration.destination_path(BASE / source) == BASE / target
    dynamic = Path("i24/i24_test/calibration_dynamic/control_h_90/optimal_vsl.npy")
    assert migration.destination_path(dynamic) == dynamic.with_name("run.npz")


def test_truncated_sidecar_preserves_only_complete_fields(tmp_path):
    path = tmp_path / "optimal_vsl_80_config.json"
    original = '{"speed_lb": 80, "control_zone": [], "initialize_vsl": '
    path.write_text(original)
    raw, notes = migration.read_config(path)
    assert raw == {"speed_lb": 80, "control_zone": []}
    assert len(notes) == 1 and "initialization unknown" in notes[0]
    config, details = migration.convert_config(raw, tmp_path)
    assert config.speed_lb == 80 and config.control_zone == ()
    assert config.initialize_vsl is None and config.init_fixed is None
    assert details["defaulted_fields"]["pred_horizon"] == 45
    assert details["defaulted_fields"]["control_horizon"] == 5
    assert details["defaulted_fields"]["initialize_vsl"] is None
    assert path.read_text() == original


def test_initializers_use_selected_source_tree_and_keep_empty_zone(tmp_path):
    results = tmp_path / "results"
    initializer = results / BASE / "optimal_vsl.npy"
    initializer.parent.mkdir(parents=True)
    expected = np.array([[110., 120.], [100., 90.]])
    np.save(initializer, expected)
    old_path = f"loaded from file /unavailable/author/results/{BASE}/optimal_vsl.npy"
    raw = {"initialize_vsl": old_path, "control_zone": [], "init_fixed": ["adaptive", 120]}
    config, details = migration.convert_config(raw, results)
    np.testing.assert_array_equal(config.initialize_vsl, expected)
    assert config.control_zone == () and config.init_fixed == ("adaptive", 120)
    assert details["initialization_source"] == f"{BASE}/optimal_vsl.npy"
    assert raw["initialize_vsl"] == old_path and raw["control_zone"] == []

    unknown, details = migration.convert_config(
        {"initialize_vsl": "loaded from file [[87.1 ... 125.5]]"}, results,
    )
    assert unknown.initialize_vsl is None
    assert "Unavailable historical initial VSL" in details["notes"][0]


@pytest.fixture
def miniature_main(tmp_path, monkeypatch):
    """Exercise real persistence/installation; no solver or long simulation runs."""
    source = tmp_path / "main"
    results = source / "results"
    policies = {
        BASE / "optimal_vsl.npy": 100.,
        BASE / "optimal_vsl_test.npy": 110.,
        BASE / "optimal_vsl_tiled.npy": 120.,
        BASE / "hold_length/optimal_vsl_1.npy": 130.,
        BASE / "hold_length/optimal_vsl_test.npy": 150.,
        BASE / "safety_sweep/optimal_vsl_temp0.5_spat25.npy": 90.,
        BASE / "speed_lb/optimal_vsl_80.npy": 80.,
    }
    for relative, speed in policies.items():
        policy = results / relative
        policy.parent.mkdir(parents=True, exist_ok=True)
        np.save(policy, np.full((2, 2), speed))
        config = {"pred_horizon": 2, "control_horizon": 1, "control_zone": []}
        sidecar = policy.with_name(policy.stem + "_config.json")
        sidecar.write_text(json.dumps(config) if speed != 80. else
                           '{"speed_lb": 80, "initialize_vsl": ')
    orphan = BASE / "safety_sweep/optimal_vsl_temp0.8_spat25_config.json"
    (results / orphan).write_text('{"safety_temporal": 0.8}\n')
    csv = Path("synthetic_10km/demand/demand5500.0_duration0.5.csv")
    (results / csv).parent.mkdir(parents=True)
    (results / csv).write_bytes(b"0,1\r\n100.0,120.0\r\n")

    data = source / "data/i24/i24_test"
    data.mkdir(parents=True)
    np.save(data / "on_ramp_mapping.npy", np.array([False, True, False, False]))
    np.save(data / "off_ramp_mapping.npy", np.array([False, False, True, False]))
    (source / "src").mkdir()
    for name in ("cc_analysis.py", "traffic_sim.py", "param_loader.py"):
        (source / "src" / name).write_text("# Mocked source module for this temporary fixture.\n")
    day = dict(
        num_segments=2, lane_counts=np.array([4., 2.]),
        flow_data=np.full((2, 4), 1000.), d_trimmed=np.full((2, 2), 20.),
        v_trimmed=np.full((2, 2), 90.), data_inflow=np.full(2, 1000.),
        ds_density_norm=np.zeros(2),
        init_state=(np.full(2, 20.), np.full(2, 90.), 1000., 0.),
        static_params=default_metanet_params(2),
    )
    calls = []

    def checked_simulation(reference, scenario, params, policy, *, real_data):
        calls.append(real_data)
        assert real_data == (policy is None)
        return SimulationResult(np.full((3, 2), 20.), np.full((3, 2), 90.),
                                np.zeros((3, 1)), 10. if real_data else 8.)

    monkeypatch.setattr(migration, "source_modules", lambda _: (lambda date: day, None, None))
    monkeypatch.setattr(migration, "checked_simulation", checked_simulation)
    monkeypatch.setattr(migration.subprocess, "check_output", lambda *args, **kwargs: "fixture-commit\n")
    monkeypatch.setattr(result_io, "collection_dir", lambda *args, **kwargs:
                        pytest.fail("Migration must save only to its explicit staging directory"))
    destination = tmp_path / "refactor/results"
    destination.mkdir(parents=True)
    (destination / "old-result.bin").write_bytes(b"original collection must survive")
    return SimpleNamespace(
        source=source, destination=destination, backup=destination.with_name("results_bu"),
        policies=policies, csv=csv, orphan=orphan, calls=calls,
        source_before=migration.inventory(source), old=migration.inventory(destination),
    )


def test_preview_then_install_preserves_every_policy_and_original_file(miniature_main):
    case = miniature_main
    migration.migrate(case.source, case.destination, case.backup)
    assert case.calls == [] and not case.backup.exists()
    assert migration.inventory(case.destination) == case.old
    assert not list(case.destination.parent.glob(".results-main-*"))

    migration.migrate(case.source, case.destination, case.backup, apply=True)
    assert migration.inventory(case.source) == case.source_before
    assert migration.inventory(case.backup) == case.old
    targets = {migration.destination_path(path) for path in case.policies}
    assert {path.relative_to(case.destination) for path in case.destination.rglob("*.npz")} == targets
    for relative, speed in case.policies.items():
        run = migration.load_npz(case.destination / migration.destination_path(relative))
        np.testing.assert_array_equal(run.optimization.vsl, np.full((2, 2), speed))
        assert run.scenario.spec.start_hour == 8.
        np.testing.assert_array_equal(run.scenario.spec.on_ramps, [True, False])
        np.testing.assert_array_equal(run.scenario.spec.off_ramps, [False, True])
    # Also verifies save_result's output_dir override never writes to the repository.
    assert len(case.calls) == len(case.policies) + 1  # One shared static baseline.
    assert (case.destination / case.csv).read_bytes() == (case.source / "results" / case.csv).read_bytes()
    for original in (case.source / "results").rglob("*.json"):
        relative = original.relative_to(case.source / "results")
        copied = case.destination / "_migration/source_configs" / relative
        assert copied.read_bytes() == original.read_bytes()
    manifest = json.loads((case.destination / "migration_manifest.json").read_text())
    assert manifest["orphan_configs"] == [case.orphan.as_posix()]
    assert {run["source"] for run in manifest["runs"]} == {p.as_posix() for p in case.policies}
    assert sum(run["all_150_policy"] for run in manifest["runs"]) == 1
    assert manifest["source_files"] == migration.inventory(case.source / "results")
    assert not list(case.destination.parent.glob(".results-main-*"))


def test_existing_backup_refuses_before_building_or_writing(miniature_main, monkeypatch):
    case = miniature_main
    case.backup.mkdir()
    (case.backup / "important.bin").write_bytes(b"previous backup")
    before = migration.inventory(case.destination.parent)
    monkeypatch.setattr(migration, "source_modules", lambda _: pytest.fail("Must refuse before building"))
    with pytest.raises(FileExistsError, match="Backup already exists"):
        migration.migrate(case.source, case.destination, case.backup, apply=True)
    assert migration.inventory(case.destination.parent) == before
    assert migration.inventory(case.source) == case.source_before
    assert case.calls == []


@pytest.mark.parametrize("failure", ["replay", "install"])
def test_failed_migration_keeps_or_restores_original_results(miniature_main, monkeypatch, failure):
    case = miniature_main
    if failure == "replay":
        def fail_replay(*args, **kwargs):
            raise RuntimeError("simulated replay mismatch")
        monkeypatch.setattr(migration, "checked_simulation", fail_replay)
    else:
        rename = Path.rename

        def fail_install(path, target):
            if path.name.startswith(".results-main-") and target == case.destination:
                raise RuntimeError("simulated install failure")
            return rename(path, target)

        monkeypatch.setattr(Path, "rename", fail_install)
    with pytest.raises(RuntimeError, match="simulated"):
        migration.migrate(case.source, case.destination, case.backup, apply=True)
    assert migration.inventory(case.destination) == case.old
    assert migration.inventory(case.source) == case.source_before
    assert not case.backup.exists()
    assert len(list(case.destination.parent.glob(".results-main-*"))) == 1
