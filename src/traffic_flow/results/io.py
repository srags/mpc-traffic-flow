import json
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

from ..config import StudyChoice, MPCConfig, ScenarioConfig, CalRef
from ..paths import REPO_DIR, find_run_dirs
from ..types import (
    MetanetParams, OptimizationResult, RunResult,
    SimulationResult, TrafficData
)
from typing import Callable, cast
from .console import colored
from ..inputs.scenario import Scenario

from tempfile import NamedTemporaryFile

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
        with NamedTemporaryFile(dir=output_dir, suffix=".npz", delete=False) as stream:
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
        expected = {field.name for field in fields(MPCConfig)} - {"initialize_vsl"}
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


def load_runs(dataset: str, date: str, calibration: CalRef,
    study: StudyChoice = None, where: Callable[[MPCConfig], bool] | None = None
) -> list[RunResult]:
    """Load saved runs matching the requested configuration."""
    results = []

    for directory in find_run_dirs(dataset, date, calibration=calibration, study=study):
        result = load_result(directory)
        spec = result.scenario.spec
        if (spec.freeway, spec.date) != (dataset, date):
            raise ValueError(f"Scenario metadata disagrees with directory: {directory / 'run.npz'}")

        if where is None or where(result.config):
            results.append(result)

    return results

