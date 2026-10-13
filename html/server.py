#!/usr/bin/env python3
"""View saved runs, with optional in-memory drainage replay; never run MPC or write results."""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
import warnings
from dataclasses import fields
from functools import lru_cache
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Timer
from urllib.parse import parse_qs, urlparse
from zipfile import BadZipFile

import numpy as np

SITE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SITE_ROOT.parent
RESULTS_ROOT = (REPO_ROOT / "results").resolve()
sys.path.insert(0, str(REPO_ROOT / "src"))

from traffic_flow.results.io import load_npz  # noqa: E402
from traffic_flow.results.analysis import run_analysis  # noqa: E402
from traffic_flow.pipeline import replay  # noqa: E402
from traffic_flow.config import Study  # noqa: E402
from traffic_flow.types import RunResult  # noqa: E402

def _safe_scenario_path(scenario_id: str) -> Path:
    path = (RESULTS_ROOT / scenario_id).resolve()
    if Path(scenario_id).is_absolute() or RESULTS_ROOT not in path.parents or path.suffix != ".npz":
        raise ValueError("Invalid scenario path")
    if not path.is_file():
        raise FileNotFoundError("Scenario does not exist")
    return path


def _scenario_summary(run: RunResult, relative: Path) -> dict[str, object]:
    spec, config = run.scenario.spec, run.config
    # Calibration/study identity is encoded by collection_dir, not in the NPZ
    # schema. Read these labels from folders, but all parameter values from the
    # saved config (a filename such as slb_95 is not authoritative).
    collection = list(relative.parts[2:-1])
    context = "/".join(collection) or "saved run"
    study = collection.pop() if collection and collection[-1] in {item.value for item in Study} else ""
    interval = None
    if collection and collection[-1].startswith("control_h_"):
        suffix = collection[-1].removeprefix("control_h_")
        if suffix.isdigit():
            interval = int(suffix)
            collection.pop()
    calibration_source = "/".join(collection) or "saved run"
    run_id = relative.stem
    options = [f"min {config.speed_lb:g}", f"hold {config.hold_length}",
               f"predict {config.pred_horizon}", f"apply {config.control_horizon}"]
    if config.safety_temporal is not None:
        options.append(f"temporal {config.safety_temporal:g}")
    if config.safety_spatial is not None:
        options.append(f"spatial {config.safety_spatial:g}")
    if config.initialize_vsl is not None:
        options.append("warm start")
    initialization = config.init_fixed
    if initialization is not None:
        choices = initialization if isinstance(initialization, (tuple, list)) else (initialization,)
        options.append("init " + "/".join(str(value) for value in choices))
    variant = " · ".join(options)
    date = spec.date.replace("_", "/")
    network = spec.freeway.upper().replace("I24", "I-24")
    return {
        "id": relative.as_posix(),
        "label": f"{network} · {date} · {context} · {variant} · {run_id}",
        "date": date,
        "calibration": context,
        "variant": variant,
        "network": network,
        "dataset": spec.freeway,
        "dateKey": spec.date,
        "calibrationSource": calibration_source,
        "calibrationInterval": interval,
        "study": study,
        "run": run_id,
        "parameters": {name: getattr(config, name) for name in (
            "hold_length", "speed_lb", "safety_temporal", "safety_spatial",
            "pred_horizon", "control_horizon",
        )},
    }


def _delay_curve(run: RunResult) -> dict[str, object]:
    """TTT of the displayed mix minus one fixed free-flow TTT.

    Use full-precision saved states and preserve the saved TTT convention.
    Each trajectory's terminal contribution belongs to its own final interval;
    intervals after that trajectory ends contribute zero. This aligns unequal
    drainage lengths without inventing travel time. The curve describes the
    mixed display, not a new partial-policy simulation.
    """
    spec, traffic = run.scenario.spec, run.scenario.traffic
    baseline, controlled = run.optimization.baseline, run.optimization.controlled
    steps = max(len(baseline.density), len(controlled.density)) - 1
    costs = []
    for simulation in (baseline, controlled):
        road_cost = spec.time_step * spec.L * (simulation.density @ traffic.lanes)
        cost = road_cost + spec.time_step * simulation.queue.reshape(-1)
        if not np.isclose(cost.sum(), simulation.total_travel_time, rtol=1e-9, atol=1e-8):
            # Current run_with_history omits terminal road occupancy, but keeps
            # the full queue sum. Older saved runs included both terminal terms.
            cost[-1] -= road_cost[-1]
        if not np.isclose(cost.sum(), simulation.total_travel_time, rtol=1e-9, atol=1e-8):
            raise ValueError("Saved travel time disagrees with the saved density/queue histories")
        intervals = cost[:-1].copy()
        intervals[-1] += cost[-1]
        costs.append(np.pad(intervals, (0, steps - len(intervals))))
    difference = costs[1] - costs[0]
    curve = baseline.total_travel_time + np.r_[0.0, np.cumsum(difference)]
    curve[-1] = controlled.total_travel_time  # Remove floating-point summation drift only.
    # Match the current Python reports, including their dynamic-speed convention.
    v_free = np.asarray(run.params["v_free"])
    if v_free.ndim == 2:
        v_free = v_free.max(axis=0)
    if v_free.shape != (spec.num_segments,) or not np.isfinite(v_free).all() or np.any(v_free <= 0):
        raise ValueError("Saved free-flow speeds must be positive and finite, one per segment")
    free_flow = float(run_analysis(run).ff_tt)
    if not np.isfinite(free_flow):
        raise ValueError("Saved inputs produce a non-finite free-flow travel time")
    return {
        "baseline": baseline.total_travel_time - free_flow,
        "controlled": controlled.total_travel_time - free_flow,
        "byCutoff": (curve - free_flow).tolist(),
        "freeFlow": free_flow,
    }


def load_scenario(scenario_id: str, *, drain: bool = False) -> dict[str, object]:
    path = _safe_scenario_path(scenario_id)
    stamp = path.stat()
    return _load_scenario(scenario_id, stamp.st_mtime_ns, stamp.st_size, stamp.st_ino, drain)


@lru_cache(maxsize=12)
def _load_scenario(scenario_id: str, modified_ns: int, size: int, inode: int,
                   drain: bool = False) -> dict[str, object]:
    # Fingerprinting invalidates cached data when save_result replaces a bundle.
    path = _safe_scenario_path(scenario_id)
    run = load_npz(path)
    notices = []
    if drain:
        # Reuse the simulator with this bundle's inputs and raw policy. There is
        # no optimizer, data-folder lookup, or save_result call on this path.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            run = replay(run, until_ff=True)
        notices = list(dict.fromkeys(str(warning.message) for warning in caught))
    spec, traffic = run.scenario.spec, run.scenario.traffic
    result = run.optimization
    observed_steps, segments = spec.time_steps, spec.num_segments
    if observed_steps < 1 or segments < 1 or not all(
        np.isfinite(value) and value > 0 for value in (spec.time_step, spec.L)
    ) or not np.isfinite(spec.start_hour):
        raise ValueError("Invalid scenario geometry or timing")
    if traffic.lanes.shape != (segments,) or not np.isfinite(traffic.lanes).all() or np.any(traffic.lanes <= 0):
        raise ValueError("Invalid saved lane counts")
    if result.vsl.shape != (observed_steps, segments) or not np.isfinite(result.vsl).all():
        raise ValueError("Saved VSL dimensions/values do not match the scenario")

    for simulation in (result.baseline, result.controlled):
        rows = len(simulation.density)
        if rows < observed_steps + 1 or simulation.density.shape != (rows, segments) or simulation.velocity.shape != (rows, segments):
            raise ValueError("Saved histories must cover the observation window and include a terminal row")
        if simulation.queue.shape not in ((rows,), (rows, 1)):
            raise ValueError("Saved queue dimensions do not match the scenario")
        if not all(np.isfinite(value).all() for value in simulation):
            raise ValueError("Saved results contain non-finite values")

    baseline_steps = len(result.baseline.density) - 1
    controlled_steps = len(result.controlled.density) - 1
    steps = max(baseline_steps, controlled_steps)

    def display(simulation):
        def aligned(array):
            # Only for transport/display: the frontend masks intervals beyond
            # each recorded history. The cost curve never charges padded states.
            tail = np.repeat(array[-1:], steps - len(array) + 1, axis=0)
            return np.round(np.concatenate((array[:-1], tail)), 3).tolist()

        return {
            "velocity": aligned(simulation.velocity),
            "density": aligned(simulation.density),
            "queue": aligned(simulation.queue.reshape(-1)),
        }

    config = {field.name: getattr(run.config, field.name)
              for field in fields(run.config) if field.name != "initialize_vsl"}
    config["warm_start"] = run.config.initialize_vsl is not None
    velocity_max = max(120.0, float(np.ceil(max(
        result.controlled.velocity.max(), result.baseline.velocity.max()
    ) / 10) * 10))
    density_max = max(80.0, float(np.ceil(max(
        result.controlled.density.max(), result.baseline.density.max()
    ) / 10) * 10))
    return {
        "scenario": _scenario_summary(run, path.relative_to(RESULTS_ROOT)),
        **display(result.controlled),
        "vsl": np.round(result.vsl, 3).tolist() + [None] * (steps - observed_steps),
        "baseline": display(result.baseline),
        "delay": _delay_curve(run),
        "config": config,
        "metadata": {
            "timeSteps": steps,
            "observedSteps": observed_steps,
            "baselineSteps": baseline_steps,
            "controlledSteps": controlled_steps,
            "drainageRequested": drain,
            "hasDrainage": steps > observed_steps,
            "warnings": notices,
            "timeStepSeconds": spec.time_step * 3600,
            "segments": segments,
            "segmentLengthKm": spec.L,
            "startHour": spec.start_hour,
            "lanes": traffic.lanes.tolist(),
        },
        "scales": {"velocity": [0.0, velocity_max], "density": [0.0, density_max]},
    }


def discover_scenarios() -> list[dict[str, object]]:
    scenarios = []
    for path in sorted(RESULTS_ROOT.rglob("*.npz")):
        if not path.is_file():
            continue
        try:
            payload = load_scenario(path.relative_to(RESULTS_ROOT).as_posix())
            scenarios.append(payload["scenario"])
        except (ValueError, OSError, KeyError, TypeError, BadZipFile, EOFError) as error:
            print(f"[traffic-explorer] Skipping {path}: {error}")
    return sorted(scenarios, key=lambda item: (str(item["date"]), str(item["label"])))


class ScenarioHandler(SimpleHTTPRequestHandler):
    STATIC_FILES = {
        "/": "index.html",
        "/index.html": "index.html",
        "/styles.css": "styles.css",
        "/app.js": "app.js",
    }

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, directory=str(SITE_ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/scenarios":
                self._json({"scenarios": discover_scenarios()})
                return
            if parsed.path == "/api/scenario":
                query = parse_qs(parsed.query)
                scenario_id = query.get("id", [""])[0]
                if not scenario_id:
                    self._json({"error": "Missing scenario id"}, HTTPStatus.BAD_REQUEST)
                    return
                drain = query.get("drain", ["0"])[0]
                if drain not in {"0", "1"}:
                    raise ValueError("drain must be 0 or 1")
                self._json(load_scenario(scenario_id, drain=drain == "1"))
                return
            static_file = self.STATIC_FILES.get(parsed.path)
            if static_file:
                self.path = f"/{static_file}"
                super().do_GET()
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except (ValueError, FileNotFoundError, KeyError, StopIteration) as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:  # keep the local UI informative
            self._json({"error": f"Scenario processing failed: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[traffic-explorer] {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local traffic scenario explorer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="Open the explorer in a browser")
    args = parser.parse_args()
    try:
        server = ThreadingHTTPServer((args.host, args.port), ScenarioHandler)
    except OSError as error:
        parser.error(f"{error}. Try another port, e.g. --port 8001")
    browser_host = "localhost" if args.host in {"127.0.0.1", "0.0.0.0"} else args.host
    url = f"http://{browser_host}:{server.server_port}"
    print(f"Traffic explorer: {url}", flush=True)
    if args.open:
        Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
