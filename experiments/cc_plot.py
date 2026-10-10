"""Replay a saved policy and generate the two CC figures; never run MPC."""
import sys; from pathlib import Path
REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

from traffic_flow.results.analysis import mape, rmse, run_analysis
from traffic_flow import CalRef, CalSource, InitMode, Study, load_one_result
from traffic_flow.paths import REPO_DIR
from traffic_flow.results.plots import save_cc_plots, save_reveal_plots
from traffic_flow.console import announce_title, announce_file

from tabulate import tabulate
import numpy as np

FIGURE_DIR = REPO_DIR / "figs"
START_HOUR = 7.5

def main() -> None:
  result = load_one_result("i24", "11_28", calibration=CalRef(CalSource.FIXED_RAMPS, interval = None), study=None)
  stats = run_analysis(result)
  traffic, uncontrolled, controlled = result.scenario.traffic, result.optimization.baseline, result.optimization.controlled
  announce_title("No Control")

  data = [("Velocity", traffic.velocity, uncontrolled.velocity[:-1]), 
          ("Density", traffic.density, uncontrolled.density[:-1]), 
          ("Flow", traffic.flow, uncontrolled.density[:-1] * uncontrolled.velocity[:-1] * traffic.lanes)]
  print(tabulate(headers=("Quantity", "Result"), tablefmt="outline", tabular_data=
                 [[name, f"MAPE {mape(a, b):.2f}%, RMSE {rmse(a, b):.2f}"] for name, a, b in data] + \
                 [["Travel Time", f"{stats.gt_tt:.2f} veh-hr vs {stats.sim_tt:.2f} veh-hr (MAPE {mape(stats.gt_tt, stats.sim_tt):.2f}%)"]]))

  spec, v_free = result.scenario.spec, result.params["v_free"][(slice(None),) + (None,)*(2-result.params["v_free"].ndim)]
  observed_delay = (spec.L / traffic.velocity.T - spec.L / v_free) * 60
  controlled_delay = (spec.L / controlled.velocity[:-1].T - spec.L / v_free) * 60
  pct_decrease = np.where(observed_delay > 0.01, (observed_delay - controlled_delay) / observed_delay * 100, 0)

  announce_title("Optimized VSLs")
  print(tabulate(headers=("Metric", "Value"), tablefmt="outline", tabular_data=[
    ['Total free flow travel time', f'{stats.ff_tt:.2f} veh-hrs'],
    ['Controllable congestion', f'{stats.cc:.2f}%'],
    ['Delay (ground truth)', f'{observed_delay.min():.2f} min - {observed_delay.max():.2f} min'],
    ['Pct decrease', f'{pct_decrease.min():.2f}% - {pct_decrease.max():.2f}%']
  ]))

  save_cc_plots(pct_decrease, result, stats, FIGURE_DIR, start_hour=START_HOUR)
  save_reveal_plots(result, FIGURE_DIR, checkpoints=16, batches=4)

if __name__ == "__main__":
  announce_file(Path(__file__))
  main()
