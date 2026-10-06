"""Optimize a safety-bound grid and save complete runs; no plotting."""
from dataclasses import replace
import logging
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traffic_flow import load_params, load_scenario, optimize, save_result
from traffic_flow.config import CalRef, CalSource, InitMode, Study, cc_mpc
from traffic_flow.paths import REPO_DIR
from traffic_flow.results.console import colored
from traffic_flow.results.io import run_dir

# Select inputs, swept bounds and initialization, as in run_safety_sweep.ipynb.
DATASET = "i24"
DATE = "11_30"
CALIBRATION = CalRef(CalSource.FIXED_RAMPS, interval = None)
STUDY = Study.SAFETY_SWEEP
TEMPORAL_VALUE = 0.7
SPATIAL_VALUE = 25
INIT_FIXED = (InitMode.ADAPTIVE, 120, 100, 80, 60, 40)

# Explicit legacy input, reused for every pair (not chained between solves).
# Set to None to use only INIT_FIXED and the solver's final cold attempt.
WARM_START: Path | None = REPO_DIR / (
    "results_bu/i24/i24_11_30/calibration_static/fixed_ramping/"
    "safety_sweep/optimal_vsl_temp0.9_spat25.npy"
)

def main() -> None:
    logging.getLogger("pyomo.core").setLevel(logging.ERROR)
    print(colored("Safety-bound optimization", "bold", "yellow"))

    scenario = load_scenario(DATASET, DATE)
    params = load_params(scenario, calibration=CALIBRATION)
    mpc = replace(
        cc_mpc(scenario.spec),
        initialize_vsl=np.load(WARM_START, allow_pickle=False) if WARM_START is not None else None,
        init_fixed=INIT_FIXED,
    )
    config = replace(mpc, safety_temporal=TEMPORAL_VALUE, safety_spatial=SPATIAL_VALUE)
    result = optimize(scenario, params, config)
    directory = run_dir(result.scenario.spec, calibration=CALIBRATION, config=result.config, study=STUDY)
    save_result(result, directory)

if __name__ == "__main__":
    main()
