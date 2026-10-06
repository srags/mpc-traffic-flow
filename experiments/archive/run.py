"""Optimize the selected I-24 scenario and save its policy + settings."""
from dataclasses import replace
import logging, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traffic_flow.config import StudyChoice, CalRef, default_mpc_config
from traffic_flow.results.console import colored
from traffic_flow.results.io import run_dir
from traffic_flow import load_params, load_scenario, optimize, save_result

def cc_optimize(dataset: str, date: str, calibration: CalRef, study: StudyChoice, **kwargs):
  logging.getLogger("pyomo.core").setLevel(logging.ERROR)
  print(colored("VSL Optimization", "bold", "yellow"))
  scenario = load_scenario(dataset, date)
  params = load_params(scenario, calibration=calibration)
  config = replace(default_mpc_config(scenario.spec), **kwargs)
  result = optimize(scenario, params, config)
  save_result(result, run_dir(result.scenario.spec, calibration=calibration, config=result.config, study=study))
