"""Opt-in, read-only audit of I-24 figure/table data against pinned Git snapshots.

Run with the project Python: python -B tests/audit_main_parity.py [--phase post]
No solver, plotting, checkout, data writes, or saved-result updates are performed.
This is deliberately separate from the fast pytest suite. All available current
NPZ bundles are checked, including sweep points excluded from displayed figures.
The reference is the full I-24 corridor used by main's reports, not a subset.
POST metrics repeat the final dynamic beta row during drainage; main's literal
dynamic report otherwise has incompatible array lengths. Raw TTT includes the
terminal road state; the agreed comparison excludes it and retains queue terms.
"""
import argparse
import ast
import io
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_flow.inputs.scenario import Scenario
from traffic_flow.pipeline import simulate
from traffic_flow.results import analysis as current_report
from traffic_flow.results.analysis import run_analysis
from traffic_flow.results.io import load_npz
from traffic_flow.types import SimulationResult, TrafficData

REFERENCES = {
    "pre": "150bc4f51227d41bea0c0b3d5c7af5e6d19457ad",
    "post": "b332162f1f08acd366778a597e21c6ade026311b",
}
RTOL = ATOL = 1e-10


class Snapshot:
    """Read reference code AND arrays from Git, not the mutable backup folder."""
    def __init__(self, commit):
        self.commit = commit

    @lru_cache(maxsize=None)
    def blob(self, relative):
        result = subprocess.run(["git", "show", f"{self.commit}:{relative}"],
                                cwd=ROOT, capture_output=True, check=False)
        if result.returncode:
            raise FileNotFoundError(f"{self.commit}:{relative}")
        return result.stdout

    def load(self, path, *args, **kwargs):
        relative = Path(path).relative_to(ROOT).as_posix()
        return np.load(io.BytesIO(self.blob(relative)), allow_pickle=False)

    @lru_cache(maxsize=None)
    def exists(self, path):
        relative = Path(path).relative_to(ROOT).as_posix()
        return subprocess.run(
            ["git", "cat-file", "-e", f"{self.commit}:{relative}"],
            cwd=ROOT, capture_output=True, check=False,
        ).returncode == 0

    def __getattr__(self, name):
        # A NumPy proxy for reference loaders; current NumPy is never monkeypatched.
        return getattr(np, name)

    def definitions(self, relative, namespace, names=None):
        tree = ast.parse(self.blob(relative).decode())
        tree.body = [node for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                     and (names is None or node.name in names)]
        exec(compile(tree, f"{self.commit}:{relative}", "exec"), namespace)

    def modules(self):
        simulator = {}
        exec(compile(self.blob("src/traffic_sim.py"),
                     f"{self.commit}:src/traffic_sim.py", "exec"), simulator)
        loader = {"np": self, "os": SimpleNamespace(path=SimpleNamespace(exists=self.exists))}
        self.definitions("src/param_loader.py", loader)
        report = {"np": self, "DATA_ROOT": str(ROOT / "data/i24"),
                  "METANET_Params": loader["METANET_Params"]}
        self.definitions("src/cc_analysis.py", report,
                         {"load_day_data", "smooth_inflow", "get_ff_tts",
                          "ff_tts_vkm", "vehicles_served"})
        return simulator, loader, report


def policy_path(path, run):
    parent = path.parent
    config = run.config
    if parent.name == "hold_length":
        name = f"optimal_vsl_{config.hold_length:g}.npy"
    elif parent.name == "speed_lb":
        name = f"optimal_vsl_{config.speed_lb:g}.npy"
    elif parent.name == "safety_sweep":
        name = f"optimal_vsl_temp{config.safety_temporal:g}_spat{config.safety_spatial:g}.npy"
    else:
        name = "optimal_vsl.npy"
    return parent / name


def reference_simulation(module, scenario, params, vsl, real_data, drain):
    spec, traffic = scenario.spec, scenario.traffic
    kwargs = {"until_ff": True} if drain else {}
    result = module["run_metanet_sim"](
        spec.time_step, spec.L,
        (traffic.initial_density, traffic.initial_velocity, float(traffic.inflow[0]), 0.),
        traffic.inflow, traffic.downstream_density, params,
        lanes=dict(enumerate(traffic.lanes)), vsl_speeds=vsl,
        plotting=True, real_data=real_data, **kwargs,
    )
    terminal = spec.time_step * spec.L * (result[0][-1] @ traffic.lanes)
    return SimulationResult(*result[:3], float(result[3] - terminal)), float(result[3])


def report_metrics(module, scenario, params, baseline, controlled, drain):
    spec, traffic = scenario.spec, scenario.traffic
    T, L, lanes = spec.time_step, spec.L, traffic.lanes
    rho, velocity = baseline.density[:-1], baseline.velocity[:-1]
    if drain:
        ff = module["ff_tts_vkm"](rho, velocity, lanes, T, L, params["v_free"])
        beta = params["beta"]
        if beta.ndim == 2:
            beta = beta[np.minimum(np.arange(len(rho)), len(beta) - 1)]
        served = module["vehicles_served"](rho, velocity, lanes, beta, T)
    else:
        vf = params["v_free"]
        ff = module["get_ff_tts"](traffic.inflow, T, L, vf.max(axis=0) if vf.ndim == 2 else vf)
        served = float(sum(traffic.inflow) * T)
    observed_tt = T * L * (traffic.density.sum(axis=0) @ lanes)
    window_tt = T * L * (rho[:len(traffic.velocity)].sum(axis=0) @ lanes)
    b, c = baseline.total_travel_time, controlled.total_travel_time
    return dict(sim_tt=b, opt_tt=c, ff_tt=float(ff), num_veh=served,
                sim_error=float(np.mean(np.abs((traffic.velocity - velocity[:len(traffic.velocity)]) / traffic.velocity)) * 100),
                tts_error=float(abs(observed_tt - window_tt) / observed_tt * 100),
                cc=float(np.clip((b - c) / (b - ff) * 100, 0, 100)),
                avg_tt_reduced_per_veh=float((b - c) / served * 60),
                uncontrolled_avg_tts=float((b - ff) / served * 60),
                controlled_avg_tts=float((c - ff) / served * 60))


def audit(phase):
    reference = Snapshot(REFERENCES[phase])
    module, loader, report = reference.modules()
    drain = phase == "post"
    groups = defaultdict(lambda: defaultdict(lambda: {
        "checked": 0, "failed": 0, "shape_failures": 0, "max_abs": 0.0, "worst": None,
    }))
    ordinary = []
    missing_references = []
    analysis_errors = []
    scenarios = {}

    def check(group, name, actual, expected, path):
        actual, expected = np.asarray(actual), np.asarray(expected)
        entry = groups[group][name]
        entry["checked"] += 1
        same_shape = actual.shape == expected.shape
        if not same_shape:
            entry["shape_failures"] += 1
            entry["example_shapes"] = {"actual": actual.shape, "reference": expected.shape}
        finite = np.isfinite(actual).all() and np.isfinite(expected).all()
        close = same_shape and finite and np.allclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=False)
        entry["failed"] += int(not close)
        error = float(np.max(np.abs(actual - expected))) if same_shape and finite else None
        if error is None or error > entry["max_abs"]:
            entry["max_abs"] = error if error is not None else entry["max_abs"]
            entry["worst"] = str(path.relative_to(ROOT))
        return close

    paths = sorted((ROOT / "results/i24").rglob("*.npz"))
    for index, path in enumerate(paths, 1):
        run = load_npz(path)
        spec = run.scenario.spec
        calibration = "dynamic" if "calibration_dynamic" in path.parts else "static"
        group = path.parent.name if path.parent.name in {"hold_length", "speed_lb", "safety_sweep"} else "ordinary"
        interval = (int(next(part[10:] for part in path.parts if part.startswith("control_h_")))
                    if calibration == "dynamic" else None)
        key = (spec.date, calibration, interval)
        if key not in scenarios:
            day = report["load_day_data"](spec.date)
            traffic = TrafficData(day["d_trimmed"], day["flow_data"][:, 1:-1], day["v_trimmed"],
                                  day["lane_counts"], day["data_inflow"], day["ds_density_norm"],
                                  day["init_state"][0], day["init_state"][1])
            reference_spec = replace(spec, L=0.4, time_step=10 / 3600,
                                     time_steps=len(traffic.inflow), num_segments=day["num_segments"])
            scenario = Scenario(reference_spec, traffic)
            params = day["static_params"]
            if calibration == "dynamic":
                params = loader["METANET_Params"](
                    path=str(ROOT / f"data/i24/i24_{spec.date}/calibration_dynamic"),
                    control_h=interval, num_timesteps=reference_spec.time_steps,
                    num_segments=day["flow_data"].shape[1],
                ).get_params()
            expected_b, raw_b = reference_simulation(module, scenario, params, None, True, drain)
            actual_b = simulate(scenario, params, real_data=True, until_ff=drain)
            scenarios[key] = scenario, params, expected_b, raw_b, actual_b
        scenario, params, expected_b, raw_b, actual_b = scenarios[key]
        try:
            controls = reference.load(policy_path(path, run))
        except FileNotFoundError as error:
            missing_references.append({"bundle": str(path.relative_to(ROOT)), "reference": str(error)})
            continue
        expected_c, raw_c = reference_simulation(module, scenario, params, controls, False, drain)
        actual_c = simulate(scenario, params, controls, real_data=False, until_ff=drain)

        for name in ("L", "time_step", "time_steps", "num_segments"):
            check("saved_spec", name, getattr(spec, name), getattr(scenario.spec, name), path)
        for name, actual, expected in zip(scenario.traffic._fields, run.scenario.traffic, scenario.traffic):
            check("saved_inputs", name, actual, expected, path)
        for name, expected in params.items():
            # Main constructs some unused default entries for the two boundary cells.
            check("saved_params", name, run.params[name], expected[..., :scenario.spec.num_segments], path)
        check(group, "policy", run.optimization.vsl, controls, path)
        for phase_name, actual, expected in (("baseline", actual_b, expected_b), ("controlled", actual_c, expected_c)):
            for field, a, e in zip(actual._fields, actual, expected):
                check("same_input_core", phase_name + "." + field, a, e, path)
                check(group, "saved_" + phase_name + "." + field,
                      getattr(getattr(run.optimization, phase_name), field), e, path)
        metrics = report_metrics(report, scenario, params, expected_b, expected_c, drain)
        # Check fresh replay + current metric helpers separately from the public
        # report on stored histories. This does not bypass a saved-report failure.
        actual_core_metrics = report_metrics(vars(current_report), scenario, params, actual_b, actual_c, drain)
        for name, expected in metrics.items():
            check("same_input_metrics", name, actual_core_metrics[name], expected, path)
        raw_metrics = report_metrics(report, scenario, params,
                                     expected_b._replace(total_travel_time=raw_b),
                                     expected_c._replace(total_travel_time=raw_c), drain)
        try:
            actual_metrics = run_analysis(run)
            for name, expected in metrics.items():
                check(group, "metric." + name, getattr(actual_metrics, name), expected, path)
            current = {name: float(getattr(actual_metrics, name)) for name in metrics}
        except (ValueError, TypeError) as error:
            current = {"error": str(error)}
            analysis_errors.append({"bundle": str(path.relative_to(ROOT)), "error": str(error)})
        if group == "ordinary":
            ordinary.append(dict(path=str(path.relative_to(ROOT)), current=current,
                                 main_raw=raw_metrics, main_agreed_ttt=metrics))
        if index % 25 == 0 or index == len(paths):
            print(f"Audited {index}/{len(paths)} bundles ({phase})", file=sys.stderr, flush=True)
    return dict(reference=reference.commit, phase=phase, rtol=RTOL, atol=ATOL,
                bundles=len(paths), scenarios=len(scenarios), groups=groups, ordinary=ordinary,
                missing_references=missing_references, analysis_errors=analysis_errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=REFERENCES, default="pre")
    options = parser.parse_args()
    result = audit(options.phase)
    print(json.dumps(result, indent=2, allow_nan=False))
    sys.exit(int(bool(result["missing_references"] or result["analysis_errors"]) or
                 any(entry["failed"] for group in result["groups"].values() for entry in group.values())))
