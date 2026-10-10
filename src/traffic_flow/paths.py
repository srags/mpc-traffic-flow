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

def collection_dir(dataset: str, date: str, calibration: CalRef, study: StudyChoice = None) -> Path:
  directory = REPO_DIR / "results" / dataset / f"{dataset}_{date}" / calibration.source
  if calibration.interval is not None: directory /= f"control_h_{calibration.interval}"
  return directory / (study or "")

def find_npzs(dataset: str, date: str, calibration: CalRef, study: StudyChoice = None, file_name: str = "*") -> list[Path]:
  """Find saved-run directories without loading arrays or creating anything."""
  directory = collection_dir(dataset, date, calibration=calibration, study=study)
  return [path for path in sorted(directory.glob(f"**/{file_name}.npz")) if path.is_file()]