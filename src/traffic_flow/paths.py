import json
from dataclasses import asdict, fields
from hashlib import sha256
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np

from .config import CalRef, MPCConfig, ScenarioConfig, StudyChoice, Study

from os.path import abspath
import os

REPO_DIR = Path(abspath(__file__)).parents[2]

# ── Helpers ──────────────────────────────────────────────────────────────────

def ensure_dir(path: Path):
  """Create ``path`` if missing and return it, so it can be used inline."""
  os.makedirs(path, exist_ok=True)
  return path

def cut_repo(path: Path):
  """Cut the path to the repository root."""
  return path.relative_to(REPO_DIR)

# def prepend_every_row(paragraph: str, prefix: str) -> str:
#   """Prepend ``prefix`` to every line of ``paragraph``."""
#   return "\n".join(f"{prefix}{line}" for line in paragraph.split("\n"))

# import re
# from collections.abc import Iterator
# from dataclasses import dataclass, field

# K = TypeVar("K")
# V = TypeVar("V")

# @dataclass(repr=False)
# class _Tree(Generic[K, V]):
#   directory: Path; name: str; children: dict[K, V] = field(default_factory=dict)
#   def __getitem__(self, key: K) -> V: return self.children[key]
#   def __iter__(self) -> Iterator[V]: return iter(self.children.values())
#   def __len__(self) -> int: return len(self.children)
#   def __repr__(self) -> str:
#     lines = [self.name]
#     for i, (key, child) in enumerate(self.children.items()):
#       last = i == len(self.children) - 1
#       branch = "└── " if last else "├── "
#       prefix = "    " if last else "│   "
#       first, *rest = (repr(child) if isinstance(child, _Tree) else str(key)).splitlines()
#       lines.append(branch + first)
#       lines.extend(prefix + line for line in rest)
#     return "\n".join(lines)
# class StudyResultPath(_Tree[str, Path]): pass
# class CalibrationResultPath(_Tree[StudyChoice, StudyResultPath]):
#   def __getitem__(self, study: StudyChoice) -> StudyResultPath: return super().__getitem__(study or None)
# class DateResultPath(_Tree[CalRef, CalibrationResultPath]): pass
# class FreewayResultPath(_Tree[str, DateResultPath]): pass
# class ResultPath(_Tree[str, FreewayResultPath]):
#   def __init__(self, directory: Path | None = None):
#     super().__init__(REPO_DIR / "results" if directory is None else Path(directory), "results")
#     self.refresh()
#   def refresh(self) -> "ResultPath":
#     freeways: dict[str, FreewayResultPath] = {}
#     for bundle in sorted(self.directory.rglob("run.npz")):
#       parts = bundle.relative_to(self.directory).parts
#       if not bundle.is_file() or len(parts) < 6 or parts[-3] != "runs": continue
#       freeway, date_folder = parts[:2]
#       prefix = f"{freeway}_"
#       if not date_folder.startswith(prefix) or date_folder == prefix: continue
#       date = date_folder.removeprefix(prefix)
#       # Decode calibration source, optional interval, and optional study.
#       source = list(parts[2:-3])
#       study = next((s for s in Study if s == source[-1]), None)
#       if study is not None: source.pop()
#       interval = None
#       if source and (match := re.fullmatch(r"control_h_(\d+)", source[-1])):
#         interval = int(match[1])
#         source.pop()
#       if not source: continue
#       calibration = CalRef("/".join(source), interval=interval)
#       collection = bundle.parent.parent
#       calibration_dir = (collection.parent if study is None else collection.parent.parent)
#       road = freeways.setdefault(freeway, FreewayResultPath(self.directory / freeway, freeway))
#       day = road.children.setdefault(date, DateResultPath(road.directory / date_folder, date))
#       fitted = day.children.setdefault(calibration,
#         CalibrationResultPath(calibration_dir, calibration_dir.relative_to(day.directory).as_posix()))
#       runs = fitted.children.setdefault(study, 
#         StudyResultPath(collection, str(study) if study else "(default)"))
#       runs.children[bundle.parent.name] = bundle.parent
#     self.children = freeways
#     return self

def _collection_dir(dataset: str, date: str, calibration: CalRef, study: StudyChoice = None) -> Path:
  directory = REPO_DIR / "results" / dataset / f"{dataset}_{date}" / calibration.source
  if calibration.interval is not None: directory /= f"control_h_{calibration.interval}"
  return directory / (study or "") / "runs"

def run_dir(spec: ScenarioConfig, calibration: CalRef, config: MPCConfig, study: StudyChoice = None) -> Path:
  """Resolve a settings-specific destination without creating it."""
  return _collection_dir(spec.freeway, spec.date, calibration=calibration, study=study) / run_name(spec, config)

def run_name(spec: ScenarioConfig, config: MPCConfig) -> str:
  """Generate a human-readable run name from the MPC configuration."""
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
  return f"{label}__{run_hash(spec, config)}"

def run_hash(spec: ScenarioConfig, config: MPCConfig) -> str:
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

def find_run_dirs(dataset: str, date: str, calibration: CalRef, study: StudyChoice = None) -> list[Path]:
  """Find saved-run directories without loading arrays or creating anything."""
  directory = _collection_dir(dataset, date, calibration=calibration, study=study)
  return [path.parent for path in sorted(directory.glob("*/run.npz")) if path.is_file()]