from os.path import abspath
import os, pathlib

REPO_DIR = pathlib.Path(abspath(__file__)).parents[2]

#: The calibration every figure in the paper is built from.
DEFAULT_CALIBRATION = "calibration_static/fixed_ramping"

# ── Helpers ──────────────────────────────────────────────────────────────────

def ensure_dir(path: pathlib.Path):
    """Create ``path`` if missing and return it, so it can be used inline."""
    os.makedirs(path, exist_ok=True)
    return path

def cut_repo(path: pathlib.Path):
    """Cut the path to the repository root."""
    return path.relative_to(REPO_DIR)