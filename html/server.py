#!/usr/bin/env python3
"""Serve the local viewer from saved run bundles; no simulations or data reads."""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
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

from traffic_flow.results.policies import load_result  # noqa: E402
from traffic_flow.types import RunResult  # noqa: E402

# Wall-clock origin is not saved in run.npz. Other datasets use elapsed time.
START_HOURS = {"i24": 7.5}


def _safe_scenario_path(scenario_id: str) -> Path:
    path = (RESULTS_ROOT / scenario_id).resolve()
    if Path(scenario_id).is_absolute() or RESULTS_ROOT not in path.parents or path.name != "run.npz":
        raise ValueError("Invalid scenario path")
    if not path.is_file():
        raise FileNotFoundError("Scenario does not exist")
    return path


def _scenario_summary(run: RunResult, relative: Path) -> dict[str, str]:
    spec, config = run.scenario.spec, run.config
    # The file supplies geometry/settings; folders are only display labels.
    parts = relative.parts
    context = "/".join(parts[2:-3]) if len(parts) >= 6 else "saved run"
    run_id = relative.parent.name.rsplit("__", 1)[-1][:8]
    options = [f"min {config.speed_lb:g}", f"hold {config.hold_length}"]
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
    }


def _travel_time_curve(run: RunResult) -> dict[str, object]:
    """TTT of the revealed controlled prefix plus the remaining baseline.

    Use full-precision saved states, including queue and the terminal row.
    The terminal contribution switches with the final displayed interval, so
    cutoff 0/T exactly matches the saved baseline/controlled total. This is
    an accounting of the mixed display, not a new partial-policy simulation.
    """
    spec, traffic = run.scenario.spec, run.scenario.traffic
    baseline, controlled = run.optimization.baseline, run.optimization.controlled
    costs = []
    for simulation in (baseline, controlled):
        cost = spec.time_step * (
            spec.L * (simulation.density @ traffic.lanes) + simulation.queue.reshape(-1)
        )
        if not np.isclose(cost.sum(), simulation.total_travel_time, rtol=1e-9, atol=1e-8):
            raise ValueError("Saved travel time disagrees with the saved density/queue histories")
        costs.append(cost)
    difference = costs[1] - costs[0]
    increments = difference[:-1].copy()
    increments[-1] += difference[-1]
    curve = baseline.total_travel_time + np.r_[0.0, np.cumsum(increments)]
    curve[-1] = controlled.total_travel_time  # Remove floating-point summation drift only.
    return {
        "baseline": baseline.total_travel_time,
        "controlled": controlled.total_travel_time,
        "byCutoff": curve.tolist(),
    }


def load_scenario(scenario_id: str) -> dict[str, object]:
    path = _safe_scenario_path(scenario_id)
    stamp = path.stat()
    return _load_scenario(scenario_id, stamp.st_mtime_ns, stamp.st_size, stamp.st_ino)


@lru_cache(maxsize=12)
def _load_scenario(scenario_id: str, modified_ns: int, size: int, inode: int) -> dict[str, object]:
    # Fingerprinting invalidates cached data when save_result replaces a bundle.
    path = _safe_scenario_path(scenario_id)
    run = load_result(path.parent)
    spec, traffic = run.scenario.spec, run.scenario.traffic
    result = run.optimization
    steps, segments = spec.time_steps, spec.num_segments
    if steps < 1 or segments < 1 or not all(
        np.isfinite(value) and value > 0 for value in (spec.time_step, spec.L)
    ) or not np.isfinite(spec.start_time):
        raise ValueError("Invalid scenario geometry or timing")
    if traffic.lanes.shape != (segments,) or not np.isfinite(traffic.lanes).all() or np.any(traffic.lanes <= 0):
        raise ValueError("Invalid saved lane counts")
    if result.vsl.shape != (steps, segments) or not np.isfinite(result.vsl).all():
        raise ValueError("Saved VSL dimensions/values do not match the scenario")

    for simulation in (result.baseline, result.controlled):
        if simulation.density.shape != (steps + 1, segments) or simulation.velocity.shape != (steps + 1, segments):
            raise ValueError("Saved histories must include one terminal row")
        if simulation.queue.shape not in ((steps + 1,), (steps + 1, 1)):
            raise ValueError("Saved queue dimensions do not match the scenario")
        if not all(np.isfinite(value).all() for value in simulation):
            raise ValueError("Saved results contain non-finite values")

    def display(simulation):
        return {
            "velocity": np.round(simulation.velocity[:-1], 3).tolist(),
            "density": np.round(simulation.density[:-1], 3).tolist(),
            "queue": np.round(simulation.queue.reshape(-1)[:-1], 3).tolist(),
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
    start_hour = START_HOURS.get(spec.freeway)
    return {
        "scenario": _scenario_summary(run, path.relative_to(RESULTS_ROOT)),
        **display(result.controlled),
        "vsl": np.round(result.vsl, 3).tolist(),
        "baseline": display(result.baseline),
        "travelTime": _travel_time_curve(run),
        "config": config,
        "metadata": {
            "timeSteps": steps,
            "timeStepSeconds": spec.time_step * 3600,
            "segments": segments,
            "segmentLengthKm": spec.L,
            "startHour": None if start_hour is None else start_hour + spec.start_time,
            "lanes": traffic.lanes.tolist(),
        },
        "scales": {"velocity": [0.0, velocity_max], "density": [0.0, density_max]},
    }


def discover_scenarios() -> list[dict[str, str]]:
    scenarios = []
    for path in sorted(RESULTS_ROOT.rglob("run.npz")):
        try:
            payload = load_scenario(path.relative_to(RESULTS_ROOT).as_posix())
            scenarios.append(payload["scenario"])
        except (ValueError, OSError, KeyError, TypeError, BadZipFile, EOFError) as error:
            print(f"[traffic-explorer] Skipping {path}: {error}")
    return sorted(scenarios, key=lambda item: (item["date"], item["label"]))


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
                scenario_id = parse_qs(parsed.query).get("id", [""])[0]
                if not scenario_id:
                    self._json({"error": "Missing scenario id"}, HTTPStatus.BAD_REQUEST)
                    return
                self._json(load_scenario(scenario_id))
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
