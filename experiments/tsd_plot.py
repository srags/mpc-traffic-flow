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
from tabulate import tabulate
from traffic_flow import load_one_result, load_scenario, CalRef, CalSource

from traffic_flow.console import announce_save, announce_file
from traffic_flow.results.plots import Plotter
from traffic_flow.results.analysis import format_date_label, run_analysis

DATES = ["11_28", "11_29", "11_30", "12_01", "12_02"]
# Segments 0 and 1 are left uncontrolled (see Section 5.2); the control zone
# begins at this distance along the corridor.


if __name__ == "__main__":
  announce_file(Path(__file__))
  scenario = load_scenario("i24", DATES[0])
  CONTROL_ZONE_START_KM = 2 * scenario.spec.L
  corridor_km = scenario.spec.num_segments * scenario.spec.L
  duration_min = scenario.traffic.velocity.shape[0] * scenario.spec.time_step * 60

  extent = (0, duration_min, 0, corridor_km)
  vmax = float(np.ceil(max(load_scenario("i24", date).traffic.velocity.max() for date in DATES) / 10) * 10)

  n_rows, n_cols = len(DATES), 3
  p = Plotter(n_rows, n_cols, figsize=(4.6 * n_cols, 3.6 * n_rows))

  table = []
  for row, date in enumerate(DATES):
    run = load_one_result("i24", date, calibration=CalRef(CalSource.FIXED_RAMPS, interval=None), study=None)
    stats = run_analysis(run)
    table.append([format_date_label(date), f"{stats.sim_tt:.2f}", f"{stats.sim_tt-stats.ff_tt:.2f}", f"{stats.cc:.2f}"])
    for col, (title, tsd) in enumerate(zip(
      ["Observed", "No control", "Optimal control"], 
      [run.scenario.traffic.velocity, run.optimization.baseline.velocity[:-1, :], run.optimization.controlled.velocity[:-1, :]]
      )):
      im = p[row, col].imshow(tsd.T, aspect="auto", origin="lower", cmap="RdYlGn", 
                              interpolation="none", vmin=0, vmax=vmax, extent=extent)
      p[row, col].axhline(CONTROL_ZONE_START_KM, color="blue", linestyle="--", linewidth=2.5)
      head = f"({'abcdefghijklmno'[row * n_cols + col]}) {format_date_label(date)}, {title}"
      if title == "No control": 
        head += f"\nDelay = {stats.sim_tt - stats.ff_tt:.0f} veh-hrs"
      elif title == "Optimal control": 
        head += (f"\nDelay = {stats.opt_tt - stats.ff_tt:.0f} veh-hrs, CC = {stats.cc:.0f}\\%".replace("\\%", "%"))
      p[row, col] = {"title": head, "xticks": np.arange(0, duration_min + 1, 15),
                      "yticks": np.arange(0, corridor_km + 0.1, 1),
                      "ylabel": "Distance (km)" if col == 0 else None,
                      "xlabel": "Time (min)" if row == n_rows - 1 else None}
  print(tabulate(headers=["Date", "TTT (vhr)", "Delay (vhr)", "CC (%)"], tablefmt="outline", tabular_data=table))

  p.fig.subplots_adjust(right=0.89, hspace=0.32, wspace=0.08)
  cbar_ax = p.fig.add_axes((0.91, 0.12, 0.015, 0.76))
  cbar = p.fig.colorbar(im, cax=cbar_ax)
  cbar.set_label("Velocity (km/hr)")

  p.savefig(REPO_DIR / "figs" / "tsd_plot" / "tsd.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
  announce_save(REPO_DIR / "figs" / "tsd_plot" / "tsd.png")
