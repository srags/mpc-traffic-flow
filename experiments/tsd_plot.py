"""Velocity time-space diagrams for the I-24 case study.

One row per date; columns are the observed field, the calibrated model with no
control, and the model under optimal speed limit control. The default pair
(11/28 and 12/02) is nearly identical in total travel time, total delay and
delay per vehicle, yet differs by a factor of more than two in controllable
congestion.

Run from anywhere:

    python experiments/I_24_tsd_plotting.py

Saves to figs/i24_tsd.png and prints the delay / CC figures for each row.
"""
import sys; from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

import numpy as np
from traffic_flow import load_one_result, load_scenario, CalRef, CalSource, LATEST_UPDATE

from traffic_flow.console import print_file, print_table
from traffic_flow.results.plots import Plotter
from traffic_flow.results.analysis import format_date_label, run_analysis
from traffic_flow.pipeline import replay

DATES = ["11_28", "11_29", "11_30", "12_01", "12_02"]

if __name__ == "__main__":
  print_file(Path(__file__))
  scenario = load_scenario("i24", DATES[0])
  CONTROL_ZONE_START_SEGMENT = 2
  corridor_km = scenario.spec.num_segments * scenario.spec.L
  duration_min = scenario.traffic.velocity.shape[0] * scenario.spec.time_step * 60

  extent = (0, duration_min, 0, corridor_km)
  vmax = float(np.ceil(max(load_scenario("i24", date).traffic.velocity.max() for date in DATES) / 10) * 10)

  n_rows, n_cols = len(DATES), 3
  p = Plotter(n_rows, n_cols, figsize=(4.6 * n_cols, 3.6 * n_rows))

  table = []
  for row, date in enumerate(DATES):
    run = load_one_result("i24", date, calibration=CalRef(CalSource.FIXED_RAMPS, interval=None), study=None)
    run = replay(run, until_ff=LATEST_UPDATE)
    n_time = len(run.scenario.traffic.velocity)
    stats = run_analysis(run)
    table.append([format_date_label(date), f"{stats.sim_tt:.2f}", f"{stats.sim_tt-stats.ff_tt:.2f}", f"{stats.cc:.2f}"])
    for col, (title, tsd) in enumerate(zip(
      ["Observed", "No control", "Optimal control"], 
      [run.scenario.traffic.velocity, run.optimization.baseline.velocity[:n_time], run.optimization.controlled.velocity[:n_time]]
      )):
      head = f"({'abcdefghijklmno'[row * n_cols + col]}) {format_date_label(date)}, {title}"
      if title == "No control": 
        head += f"\nDelay = {stats.sim_tt - stats.ff_tt:.0f} veh-hrs"
      elif title == "Optimal control": 
        head += (f"\nDelay = {stats.opt_tt - stats.ff_tt:.0f} veh-hrs, CC = {stats.cc:.0f}\\%".replace("\\%", "%"))
      im = p[row, col].plot_time_space(tsd, run.scenario.spec, vmax, title=head, colorbar=None)
      p[row, col].axhline(CONTROL_ZONE_START_SEGMENT, color="blue", linestyle="--", linewidth=2.5)
      p[row, col].set(ylabel="Distance (km)" if col == 0 else "", xlabel=None)
  p.fig.tight_layout(rect=(0, 0, 0.9, 1), h_pad=1.5, w_pad=0.5)
  cbar_ax = p.fig.add_axes((0.91, 0.05, 0.015, 0.90))
  cbar = p.fig.colorbar(im, cax=cbar_ax)
  cbar.set_label("Velocity (km/hr)")
  print_table(headers=["Date", "TTT (vhr)", "Delay (vhr)", "CC (%)"], data=table)
  p.savefig(REPO_DIR / "figs" / "tsd_plot" / "tsd.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
  