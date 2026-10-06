"""Optimize the selected I-24 scenario and save its policy + settings."""
from dataclasses import replace
import logging, sys, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from traffic_flow.paths import REPO_DIR

from traffic_flow.config import CalSource, InitMode, StudyChoice, CalRef, Study, default_mpc_config
from traffic_flow.results.console import colored
from traffic_flow.results.policies import run_dir
from traffic_flow import load_params, load_scenario, optimize, save_result

def cc_optimize(dataset: str, date: str, calibration: CalRef, study: StudyChoice, **kwargs):
  logging.getLogger("pyomo.core").setLevel(logging.ERROR)
  print(colored("VSL Optimization", "bold", "yellow"))
  scenario = load_scenario(dataset, date)
  params = load_params(scenario, calibration=calibration)
  config = replace(default_mpc_config(scenario.spec), **kwargs)
  result = optimize(scenario, params, config)
  save_result(result, run_dir(result.scenario.spec, calibration=calibration, config=result.config, study=study))

if __name__ == "__main__":
  # regular run
  if 1: cc_optimize(
    dataset = "i24", date = "11_30",
    calibration = CalRef(CalSource.FIXED_RAMPS, interval = None),
    study = None, speed_lb = 0,
    init_fixed = 150.0 # (InitMode.ADAPTIVE, 150.0)
  )
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