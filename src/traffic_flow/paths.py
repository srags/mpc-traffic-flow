from __future__ import annotations

import json
from dataclasses import asdict, fields
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import CalRef, MPCConfig, ScenarioConfig, StudyChoice

from os.path import abspath
import os


REPO_DIR = Path(abspath(__file__)).parents[2]

#: The calibration every figure in the paper is built from.
DEFAULT_CALIBRATION = "calibration_static/fixed_ramping"

# ── Helpers ──────────────────────────────────────────────────────────────────

def ensure_dir(path: Path):
    """Create ``path`` if missing and return it, so it can be used inline."""
    os.makedirs(path, exist_ok=True)
    return path

def cut_repo(path: Path):
    """Cut the path to the repository root."""
    return path.relative_to(REPO_DIR)

def _collection_dir(dataset: str, date: str, *, calibration: CalRef, study: StudyChoice = None) -> Path:
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
    return _collection_dir(spec.freeway, spec.date, calibration=calibration, study=study) / f"{label}__{identity}"


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


def find_run_dirs(
    dataset: str,
    date: str,
    calibration: CalRef,
    study: StudyChoice = None,
) -> list[Path]:
    """Find saved-run directories without loading arrays or creating anything."""
    directory = _collection_dir(dataset, date, calibration=calibration, study=study,)
    return [path.parent for path in sorted(directory.glob("*/run.npz")) if path.is_file()]