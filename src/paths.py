"""Single source of truth for every path in this repository.

Nothing here is machine-specific. ``REPO_ROOT`` is derived from the location of
this file, so a fresh clone works with no edits by anyone -- that is the entire
point of the module. Do not hardcode an absolute path anywhere else in the repo.

Environment variables are an escape hatch, not the normal path. Set one only
when a tree genuinely lives outside the repo (results on an external drive, data
on cluster scratch):

    MPC_ROOT           repo root           default: the parent of src/
    MPC_DATA_ROOT      the data/ tree      default: $MPC_ROOT/data
    MPC_RESULTS_ROOT   the results/ tree   default: $MPC_ROOT/results
    MPC_FIGS_ROOT      the figs/ tree      default: $MPC_ROOT/figs

Every constant and helper returns ``str``, never ``pathlib.Path``, so they drop
straight into the f-string and ``+`` path building used throughout the codebase.

From a script in experiments/::

    import os, sys
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    from paths import i24_data, i24_results, fig

From a notebook in experiments/::

    import sys; sys.path.append("../src")
    from paths import i24_data, i24_results, fig

Run ``python src/paths.py`` to print every resolved path and whether it exists.
"""

import os

__all__ = [
    "REPO_ROOT", "DATA_ROOT", "RESULTS_ROOT", "FIGS_ROOT", "SRC_ROOT",
    "I24_DATA", "I24_RESULTS", "SYNTHETIC_RESULTS", "DEFAULT_CALIBRATION",
    "i24_data", "i24_calibration", "i24_results", "synthetic_results",
    "fig", "ensure_dir", "i24_dates", "describe",
]


def _resolve(env_var, default):
    """Absolute path from ``$env_var`` if it is set and non-empty, else default."""
    return os.path.abspath(os.environ.get(env_var) or default)


# ── Roots ────────────────────────────────────────────────────────────────────

REPO_ROOT = _resolve(
    "MPC_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA_ROOT    = _resolve("MPC_DATA_ROOT",    os.path.join(REPO_ROOT, "data"))
RESULTS_ROOT = _resolve("MPC_RESULTS_ROOT", os.path.join(REPO_ROOT, "results"))
FIGS_ROOT    = _resolve("MPC_FIGS_ROOT",    os.path.join(REPO_ROOT, "figs"))
SRC_ROOT     = os.path.join(REPO_ROOT, "src")

I24_DATA          = os.path.join(DATA_ROOT, "i24")
I24_RESULTS       = os.path.join(RESULTS_ROOT, "i24")
SYNTHETIC_RESULTS = os.path.join(RESULTS_ROOT, "synthetic_10km")

#: The calibration every figure in the paper is built from.
DEFAULT_CALIBRATION = "calibration_static/fixed_ramping"


# ── Helpers ──────────────────────────────────────────────────────────────────

def ensure_dir(path):
    """Create ``path`` if missing and return it, so it can be used inline."""
    os.makedirs(path, exist_ok=True)
    return path


def _date_dir(date):
    """Accept ``"11_30"`` or ``"i24_11_30"``; always return ``"i24_11_30"``."""
    date = str(date)
    return date if date.startswith("i24_") else f"i24_{date}"


def i24_data(date, *parts):
    """One date's I-24 MOTION data directory, plus an optional sub-path.

        i24_data("11_30")               -> <data>/i24/i24_11_30
        i24_data("11_30", "q_hat.npy")  -> <data>/i24/i24_11_30/q_hat.npy
    """
    return os.path.join(I24_DATA, _date_dir(date), *parts)


def i24_calibration(date, calibration_id=DEFAULT_CALIBRATION, interval=None):
    """Calibrated METANET parameters for one date.

    ``interval`` selects a dynamic calibration (``control_h_<interval>``); leave
    it ``None`` for the static calibration used throughout the paper. Mirrors the
    ``cal_path`` expression the notebooks build by hand.
    """
    path = i24_data(date, calibration_id)
    if interval is not None:
        path = os.path.join(path, f"control_h_{interval}")
    return path


def i24_results(date, *parts, calibration_id=DEFAULT_CALIBRATION, interval=None):
    """Where solved VSL policies for one date live.

        i24_results("11_30")                       -> .../i24_11_30/calibration_static/fixed_ramping
        i24_results("11_30", "speed_lb")           -> .../fixed_ramping/speed_lb
        i24_results("11_30", "optimal_vsl.npy")    -> .../fixed_ramping/optimal_vsl.npy

    ``interval`` inserts ``control_h_<interval>`` directly after the calibration
    id, matching what is on disk (``calibration_dynamic/control_h_90/...``) and
    what ``param_loader.METANET_Params`` expects on the data side. Note this is a
    deliberate change: the old notebook line appended the suffix *after* the
    constraint subdirectory instead. Nothing on disk uses that combination -- every
    committed sweep is a static calibration -- so no existing result moves.
    """
    path = os.path.join(I24_RESULTS, _date_dir(date), calibration_id)
    if interval is not None:
        path = os.path.join(path, f"control_h_{interval}")
    return os.path.join(path, *parts)


def synthetic_results(*parts):
    """Solved policies for the synthetic 10 km bottleneck.

        synthetic_results("demand")              -> <results>/synthetic_10km/demand
        synthetic_results("holdlength", "x.csv") -> <results>/synthetic_10km/holdlength/x.csv
    """
    return os.path.join(SYNTHETIC_RESULTS, *parts)


def fig(name):
    """Absolute path for a figure, creating ``figs/`` if it does not exist.

    The directory is created here so ``plt.savefig(fig("x.png"))`` cannot fail on
    a fresh clone that has no ``figs/`` yet.
    """
    ensure_dir(FIGS_ROOT)
    return os.path.join(FIGS_ROOT, name)


def i24_dates():
    """Sorted ``["11_21", ...]`` for every date directory actually present."""
    if not os.path.isdir(I24_DATA):
        return []
    return sorted(
        d[len("i24_"):] for d in os.listdir(I24_DATA)
        if d.startswith("i24_") and os.path.isdir(os.path.join(I24_DATA, d))
    )


def describe():
    """Print every resolved root and whether it exists. Run this first when
    something cannot find its data."""
    overrides = {v: os.environ[v] for v in
                 ("MPC_ROOT", "MPC_DATA_ROOT", "MPC_RESULTS_ROOT", "MPC_FIGS_ROOT")
                 if os.environ.get(v)}
    print("Resolved paths\n" + "-" * 62)
    for name in ("REPO_ROOT", "DATA_ROOT", "RESULTS_ROOT", "FIGS_ROOT",
                 "I24_DATA", "I24_RESULTS", "SYNTHETIC_RESULTS"):
        path = globals()[name]
        print(f"  {'OK ' if os.path.exists(path) else 'MISSING'}  {name:<18} {path}")
    print("-" * 62)
    print(f"  environment overrides: {overrides or 'none (all derived from this file)'}")
    dates = i24_dates()
    print(f"  I-24 dates available:  {', '.join(dates) if dates else 'none'}")


if __name__ == "__main__":
    describe()
