#!/usr/bin/env python3
"""Serve the local traffic explorer and its read-only scenario API."""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from functools import lru_cache
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Timer
from urllib.parse import parse_qs, urlparse

import numpy as np

SITE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SITE_ROOT.parent
RESULTS_ROOT = (REPO_ROOT / "results").resolve()
DATA_ROOT = (REPO_ROOT / "data").resolve()
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from sim_types import MetanetState  # noqa: E402
from traffic_sim import METANET_Simulator  # noqa: E402

NETWORKS = {
    "i24": {"segment_length_km": 0.4, "time_step_hours": 10 / 3600, "start_hour": 7.5},
}


def _smooth_two(values: np.ndarray) -> np.ndarray:
    """Equivalent to uniform_filter1d(values, 2, mode='nearest')."""
    return (np.concatenate(([values[0]], values[:-1])) + values) / 2


def _optional_array(folder: Path, name: str, fallback: np.ndarray) -> np.ndarray:
    path = folder / name
    return np.load(path, allow_pickle=False) if path.exists() else fallback.copy()


def _static_params(folder: Path, segments: int) -> dict[str, np.ndarray]:
    zeros = np.zeros(segments, dtype=float)
    ones = np.ones(segments, dtype=float)
    return {
        "tau": np.load(folder / "tau.npy", allow_pickle=False),
        "K": np.load(folder / "K.npy", allow_pickle=False),
        "eta_high": np.load(folder / "eta_high.npy", allow_pickle=False),
        "p_crit": np.load(folder / "rho_crit.npy", allow_pickle=False),
        "v_free": np.load(folder / "v_free.npy", allow_pickle=False),
        "a": np.load(folder / "a.npy", allow_pickle=False),
        "q_capacity": np.full(segments, 2400.0),
        "r": _optional_array(folder, "r_inflow_array.npy", zeros),
        "beta": _optional_array(folder, "beta_array.npy", zeros),
        "gamma": _optional_array(folder, "gamma_array.npy", ones),
    }


def _dynamic_params(folder: Path, steps: int, segments: int, interval: int) -> dict[str, np.ndarray]:
    parameter_folders = sorted(
        (path for path in folder.glob("params_*") if path.is_dir()),
        key=lambda path: int(path.name.split("_")[-1]),
    )
    if not parameter_folders:
        raise FileNotFoundError(f"No dynamic parameter folders found in {folder}")

    names = {
        "tau": "tau.npy",
        "K": "K.npy",
        "eta_high": "eta_high.npy",
        "p_crit": "rho_crit.npy",
        "v_free": "v_free.npy",
        "a": "a.npy",
        "r": "r_inflow_array.npy",
        "beta": "beta_array.npy",
        "gamma": "gamma_array.npy",
    }
    defaults = {"r": 0.0, "beta": 0.0, "gamma": 1.0}
    params: dict[str, np.ndarray] = {}
    for key, filename in names.items():
        chunks = []
        for parameter_folder in parameter_folders:
            path = parameter_folder / filename
            values = (
                np.load(path, allow_pickle=False).reshape(-1)
                if path.exists()
                else np.full(segments, defaults[key])
            )
            chunks.append(np.tile(values, (interval, 1)))
        joined = np.vstack(chunks)
        if joined.shape[0] < steps:
            joined = np.pad(joined, ((0, steps - joined.shape[0]), (0, 0)), mode="edge")
        params[key] = joined[:steps]
    params["q_capacity"] = np.full((steps, segments), 2400.0)
    return params


def _scenario_parts(relative: Path) -> tuple[str, str, str, int | None, str]:
    parts = relative.parts
    if len(parts) < 5:
        raise ValueError("Unexpected result path")
    network, date_folder = parts[0], parts[1]
    calibration_family, calibration_detail = parts[2], parts[3]
    if calibration_family == "calibration_dynamic" and calibration_detail.startswith("control_h_"):
        interval = int(calibration_detail.removeprefix("control_h_"))
        calibration_id = calibration_family
    else:
        interval = None
        calibration_id = f"{calibration_family}/{calibration_detail}"
    variant = "/".join(parts[4:-1])
    return network, date_folder, calibration_id, interval, variant


def _pretty_run_name(stem: str) -> str:
    suffix = stem.removeprefix("optimal_vsl").strip("_")
    if not suffix:
        return "Default MPC"
    return (
        suffix.replace("temp", "temporal ")
        .replace("spat", " · spatial ")
        .replace("_", " ")
        .strip()
        .title()
    )


@lru_cache(maxsize=1)
def discover_scenarios() -> list[dict[str, str]]:
    scenarios: list[dict[str, str]] = []
    for path in RESULTS_ROOT.rglob("optimal_vsl*.npy"):
        relative = path.relative_to(RESULTS_ROOT)
        try:
            network, date_folder, calibration_id, interval, variant = _scenario_parts(relative)
        except (ValueError, IndexError):
            continue
        data_folder = DATA_ROOT / network / date_folder
        if network not in NETWORKS or not (data_folder / "rho_hat.npy").exists():
            continue
        calibration = calibration_id if interval is None else f"{calibration_id}/control_h_{interval}"
        calibration_label = "Static" if interval is None else f"Dynamic {interval}"
        date = date_folder.removeprefix(f"{network}_").replace("_", "/")
        run_name = _pretty_run_name(path.stem)
        scenario_variant = " / ".join(value for value in (variant, run_name) if value)
        scenarios.append({
            "id": relative.as_posix(),
            "label": f"{network.upper().replace('I24', 'I-24')} · {date} · {calibration_label} · {scenario_variant}",
            "date": date,
            "calibration": calibration,
            "variant": scenario_variant,
        })
    return sorted(
        scenarios,
        key=lambda item: (
            item["date"] != "11/30",
            "safety_sweep" in item["variant"],
            item["date"],
            item["label"],
        ),
    )


def _safe_scenario_path(scenario_id: str) -> Path:
    path = (RESULTS_ROOT / scenario_id).resolve()
    if RESULTS_ROOT not in path.parents or path.suffix != ".npy" or not path.name.startswith("optimal_vsl"):
        raise ValueError("Invalid scenario path")
    if not path.is_file():
        raise FileNotFoundError("Scenario does not exist")
    return path


@lru_cache(maxsize=12)
def load_scenario(scenario_id: str) -> dict[str, object]:
    result_path = _safe_scenario_path(scenario_id)
    relative = result_path.relative_to(RESULTS_ROOT)
    network, date_folder, calibration_id, interval, _ = _scenario_parts(relative)
    network_info = NETWORKS[network]
    data_folder = DATA_ROOT / network / date_folder

    density_raw = np.clip(np.load(data_folder / "rho_hat.npy", allow_pickle=False), 1e-3, None)
    flow_raw = np.clip(np.load(data_folder / "q_hat.npy", allow_pickle=False), 1e-3, None)
    lanes = np.load(data_folder / "lane_mapping.npy", allow_pickle=False)[1:-1]
    velocity_raw = flow_raw / density_raw

    vsl = np.asarray(np.load(result_path, allow_pickle=False), dtype=float)
    if vsl.ndim != 2:
        raise ValueError("VSL result must be a two-dimensional time-space array")
    if vsl.shape[1] != len(lanes) and vsl.shape[0] == len(lanes):
        vsl = vsl.T
    if vsl.shape[1] != len(lanes):
        raise ValueError(f"VSL has {vsl.shape[1]} segments; data has {len(lanes)}")

    steps = min(vsl.shape[0], density_raw.shape[0])
    segments = len(lanes)
    vsl = vsl[:steps]
    initial_density = density_raw[0, 1:-1] / lanes
    initial_velocity = velocity_raw[0, 1:-1]
    demand = _smooth_two(flow_raw[:, 0])[:steps]
    downstream_density = (_smooth_two(density_raw[:, -1]) / lanes[-1])[:steps]

    if interval is None:
        params = _static_params(data_folder / calibration_id, segments)
    else:
        params = _dynamic_params(data_folder / calibration_id / f"control_h_{interval}", steps, segments, interval)

    simulator = METANET_Simulator(
        T=float(network_info["time_step_hours"]),
        l=float(network_info["segment_length_km"]),
        params=params,
        lanes={index: float(count) for index, count in enumerate(lanes)},
        real_data=False,
    )
    baseline_density, baseline_velocity, baseline_queue, _ = simulator.run_with_history(
        demand=demand,
        downstream_density=downstream_density,
        init_traffic_state=MetanetState(initial_density, initial_velocity, float(demand[0]), 0.0),
        vsl_speeds=None,
    )
    density, velocity, queue, travel_time = simulator.run_with_history(
        demand=demand,
        downstream_density=downstream_density,
        init_traffic_state=MetanetState(initial_density, initial_velocity, float(demand[0]), 0.0),
        vsl_speeds=vsl,
    )
    density = density[:-1]
    velocity = velocity[:-1]
    queue_values = queue[:-1, 0]
    baseline_density = baseline_density[:-1]
    baseline_velocity = baseline_velocity[:-1]
    baseline_queue_values = baseline_queue[:-1, 0]

    config_path = result_path.with_name(f"{result_path.stem}_config.json")
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    summary = next(item for item in discover_scenarios() if item["id"] == scenario_id)
    velocity_max = max(120.0, float(np.ceil(max(np.max(velocity), np.max(baseline_velocity)) / 10) * 10))
    density_max = max(80.0, float(np.ceil(max(np.max(density), np.max(baseline_density)) / 10) * 10))

    return {
        "scenario": summary,
        "velocity": np.round(velocity, 3).tolist(),
        "density": np.round(density, 3).tolist(),
        "vsl": np.round(vsl, 3).tolist(),
        "queue": np.round(queue_values, 3).tolist(),
        "baseline": {
            "velocity": np.round(baseline_velocity, 3).tolist(),
            "density": np.round(baseline_density, 3).tolist(),
            "queue": np.round(baseline_queue_values, 3).tolist(),
        },
        "config": config,
        "metadata": {
            "timeSteps": steps,
            "timeStepSeconds": float(network_info["time_step_hours"]) * 3600,
            "segments": segments,
            "segmentLengthKm": float(network_info["segment_length_km"]),
            "startHour": float(network_info["start_hour"]),
            "lanes": np.asarray(lanes, dtype=float).tolist(),
        },
        "stats": {
            "travelTime": round(float(travel_time), 4),
            "meanVelocity": round(float(np.mean(velocity)), 4),
            "peakDensity": round(float(np.max(density)), 4),
            "peakQueue": round(float(np.max(queue_values)), 4),
        },
        "scales": {
            "velocity": [0.0, velocity_max],
            "density": [0.0, density_max],
        },
    }


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
                discover_scenarios.cache_clear()
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
    server = ThreadingHTTPServer((args.host, args.port), ScenarioHandler)
    browser_host = "localhost" if args.host in {"127.0.0.1", "0.0.0.0"} else args.host
    url = f"http://{browser_host}:{args.port}"
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
