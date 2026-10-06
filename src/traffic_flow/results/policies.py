import json
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

from ..config import StudyChoice, MPCConfig, ScenarioConfig, CalRef
from ..paths import REPO_DIR
from ..types import (
    MetanetParams, OptimizationResult, RunResult,
    SimulationResult, TrafficData
)
from typing import Callable, cast
from .console import colored
from ..inputs.scenario import Scenario

from tempfile import NamedTemporaryFile
from hashlib import sha256

def save_result(result: RunResult, output_dir: Path) -> Path:
    """Save one complete run, replacing an existing bundle only after success."""
    settings = {field.name: getattr(result.config, field.name) for field in fields(result.config)}
    initialization = settings.pop("initialize_vsl")

    metadata = json.dumps({
        "schema_version": 1,
        "scenario": asdict(result.scenario.spec),
        "config": settings,
        # JSON otherwise loses the distinction between tuples and lists.
        "init_fixed_is_tuple": isinstance(result.config.init_fixed, tuple),
    }, allow_nan=False)

    arrays: dict[str, np.ndarray] = {
        "metadata": np.asarray(metadata),
        "vsl": np.asarray(result.optimization.vsl),
    }
    groups = {
        "traffic": result.scenario.traffic._asdict(),
        "params": result.params,
        "baseline": result.optimization.baseline._asdict(),
        "controlled": result.optimization.controlled._asdict(),
    }
    for prefix, values in groups.items():
        arrays.update({
            f"{prefix}.{key}": np.asarray(value)
            for key, value in values.items()
        })

    if initialization is not None:
        arrays["config.initialize_vsl"] = np.asarray(initialization)

    if any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError("Run bundles cannot contain object arrays.")

    output_dir = Path(output_dir)
    path = output_dir / "run.npz"
    output_dir.mkdir(parents=True, exist_ok=True)

    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=output_dir, suffix=".npz", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    display = path.relative_to(REPO_DIR) if path.is_relative_to(REPO_DIR) else path
    print(f"Saved to {colored(display, 'green')}")
    return path

def load_result(output_dir: Path) -> RunResult:
    """Restore a complete run without reading source data or recomputing results."""
    with np.load(Path(output_dir) / "run.npz", allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata"].item())
        version = metadata.get("schema_version")
        if version != 1:
            raise ValueError(f"Unsupported run schema version: {version}")

        spec = metadata["scenario"]
        if set(spec) != {field.name for field in fields(ScenarioConfig)}:
            raise ValueError("Incomplete or unsupported scenario metadata.")

        settings = metadata["config"]
        expected = {
            field.name for field in fields(MPCConfig)
        } - {"initialize_vsl"}
        if set(settings) != expected:
            raise ValueError("Incomplete or unsupported MPC settings.")

        # Restore tuple-valued settings converted to lists by JSON.
        for name in ("control_zone", "control_changepoints"):
            if settings[name] is not None:
                settings[name] = tuple(settings[name])
        if metadata["init_fixed_is_tuple"]:
            settings["init_fixed"] = tuple(settings["init_fixed"])

        settings["initialize_vsl"] = (
            archive["config.initialize_vsl"]
            if "config.initialize_vsl" in archive.files else None
        )

        scenario = Scenario(
            ScenarioConfig(**spec),
            TrafficData(**{
                name: archive[f"traffic.{name}"]
                for name in TrafficData._fields
            }),
        )
        params = cast(MetanetParams, {
            name: archive[f"params.{name}"]
            for name in MetanetParams.__required_keys__
        })

        def simulation(prefix: str) -> SimulationResult:
            return SimulationResult(
                density=archive[f"{prefix}.density"],
                velocity=archive[f"{prefix}.velocity"],
                queue=archive[f"{prefix}.queue"],
                total_travel_time=float(
                    archive[f"{prefix}.total_travel_time"].item()
                ),
            )

        return RunResult(
            scenario, params, MPCConfig(**settings),
            OptimizationResult(
                archive["vsl"],
                simulation("baseline"),
                simulation("controlled"),
            ),
        )


def _runs_dir(dataset: str, date: str, *, calibration: CalRef, study: StudyChoice = None) -> Path:
    directory = REPO_DIR / "results" / dataset / f"{dataset}_{date}" / calibration.source
    if calibration.interval is not None: directory /= f"control_h_{calibration.interval}"
    return directory / (study or "") / "runs"


def run_dir(
    spec: ScenarioConfig,
    calibration: CalRef,
    config: MPCConfig,
    study: StudyChoice = None,
) -> Path:
    """Resolve a settings-specific destination without creating it."""
    identity = _run_id(spec, config)

    labels = {
        "lb": config.speed_lb,
        "hold": config.hold_length,
        "temp": config.safety_temporal,
        "spat": config.safety_spatial,
    }
    label = "_".join(
        f"{name}_{float(value or 0):g}"
        for name, value in labels.items()
        if value is not None
    )
    return _runs_dir(spec.freeway, spec.date, calibration=calibration, study=study) / f"{label}__{identity}"


def load_runs(
    dataset: str, date: str,
    calibration: CalRef,
    study: StudyChoice = None,
    where: Callable[[MPCConfig], bool] | None = None,
) -> list[RunResult]:
    """Load matching bundles from either the old or new directory naming."""
    directory = _runs_dir(dataset, date, calibration=calibration, study=study)
    results = []

    for path in sorted(directory.glob("*/run.npz")):
        result = load_result(path.parent)
        spec = result.scenario.spec
        if (spec.freeway, spec.date) != (dataset, date):
            raise ValueError(f"Scenario metadata disagrees with directory: {path}")
        if where is None or where(result.config): results.append(result)

    return results



def _run_id(spec: ScenarioConfig, config: MPCConfig) -> str:
    """Identify numerical settings independently of logging and array layout."""
    def normalize(value):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject: raise ValueError("Run settings cannot contain object arrays.")
            return {"shape": list(value.shape), "values": normalize(value.tolist())}
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            if not np.isfinite(value): raise ValueError("Run settings must be finite.")
            # Treat equivalent numbers consistently: 0, 0.0 and -0.0.
            return int(value) if value.is_integer() else value
        if isinstance(value, (tuple, list)):
            return [normalize(item) for item in value]
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        return value

    settings = {field.name: getattr(config, field.name)
        for field in fields(config) if field.name not in {"verbose", "tee"}
    }
    identity = {"version": 1, "scenario": asdict(spec), "config": settings}
    encoded = json.dumps(
        normalize(identity),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()[:16]
