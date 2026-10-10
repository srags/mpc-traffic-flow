"""Optimize the selected I-24 scenario and save its policy + settings."""
import sys; from pathlib import Path
REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

import logging, sys, numpy as np
from traffic_flow.console import announce_title, announce_file
from traffic_flow import (load_params, load_scenario, optimize, save_result, 
                          CalSource, InitMode, Study, CalRef, Study)

def cc_optimize(dataset: str, date: str, calibration: CalRef, study, **kwargs):
  logging.getLogger("pyomo.core").setLevel(logging.ERROR)
  announce_title("VSL Optimization")
  scenario = load_scenario(dataset, date)
  params = load_params(scenario, calibration=calibration)
  config = scenario.spec.to_mpc_config(**kwargs)
  result = optimize(scenario, params, config)
  save_result(result, calibration=calibration, study=study)

if __name__ == "__main__":
  announce_file(Path(__file__))
  # regular run
  if 1: [cc_optimize(
    dataset = "i24", date = x,
    calibration = CalRef(CalSource.DYNAMIC, interval = 90),
    study = None, speed_lb = 0,
    init_fixed = 150.0 # (InitMode.ADAPTIVE, 150.0)
  ) for x in ["11_30"]]
  # safety_sweep
  if 0: cc_optimize(
    dataset = "i24", date = "11_30",
    calibration = CalRef(CalSource.FIXED_RAMPS, interval = None),
    study = Study.SAFETY_SWEEP,
    safety_temporal = 0.7,
    safety_spatial = 25,
    initialize_vsl = np.load(REPO_DIR / "results_bu/i24/i24_11_30/calibration_static/fixed_ramping/safety_sweep/optimal_vsl_temp0.9_spat25.npy", allow_pickle=False),
    init_fixed = (InitMode.ADAPTIVE, 120, 100, 80, 60, 40)
  )