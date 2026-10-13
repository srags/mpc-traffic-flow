"""Replay a saved policy and generate the two CC figures; never run MPC."""
import sys; from pathlib import Path
REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

from traffic_flow import CalRef, CalSource, InitMode, Study, load_one_result, LATEST_UPDATE
from traffic_flow.pipeline import replay
from traffic_flow.results.analysis import mape, rmse, run_analysis
from traffic_flow.results.plots import Plotter
from traffic_flow.console import print_title, print_file, print_table
import numpy as np

FIGURE_DIR = REPO_DIR / "figs"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

def main() -> None:
  result = load_one_result("i24", "11_30", calibration=CalRef(CalSource.FIXED_RAMPS, interval = None), study = None)
  result = replay(result, until_ff=LATEST_UPDATE)
  stats = run_analysis(result)
  traffic, uncontrolled, controlled = result.scenario.traffic, result.optimization.baseline, result.optimization.controlled
  
  print_title("No Control")
  n = len(traffic.velocity)
  data = [("Velocity", traffic.velocity, uncontrolled.velocity[:n]), 
          ("Density", traffic.density, uncontrolled.density[:n]), 
          ("Flow", traffic.flow, uncontrolled.flow_per_lane[:n] * traffic.lanes)]
  print_table(headers=["Quantity", "Result"], data=[
    [name, f"MAPE {mape(a, b):.2f}%, RMSE {rmse(a, b):.2f}"] for name, a, b in data] + [
    ["Travel Time", f"{stats.gt_tt:.2f} veh-hr vs {stats.sim_window_tt:.2f} veh-hr (MAPE {stats.tts_error:.2f}%)"]
  ])

  spec, v_free = result.scenario.spec, result.params["v_free"]
  if v_free.ndim == 2: v_free = v_free[:n]
  observed_delay = (spec.L / traffic.velocity - spec.L / v_free) * 60
  controlled_delay = (spec.L / controlled.velocity[:n] - spec.L / v_free) * 60
  pct_decrease = np.where(observed_delay > 0.01, (observed_delay - controlled_delay) / observed_delay * 100, 0)

  print_title("Optimized VSLs")
  print_table(headers=["Metric", "Value"], data=[
    ['Total free flow travel time', f'{stats.ff_tt:.2f} veh-hrs'],
    ['Controllable congestion', f'{stats.cc:.2f}%'],
    ['Delay (ground truth)', f'{observed_delay.min():.2f} min - {observed_delay.max():.2f} min'],
    ['Pct decrease', f'{pct_decrease.min():.2f}% - {pct_decrease.max():.2f}%']
  ])

  p = Plotter(1, 1)
  display_vsl = np.where(result.optimization.vsl > v_free, 150, result.optimization.vsl)
  p[0].plot_time_space(display_vsl, spec, spec.start_hour, v_free.max(), "Optimal VSL Speeds", ("v", "VSL speed (km/hr)"), grid=True)
  p.savefig(FIGURE_DIR / "optimal_vsl.png")

  p = Plotter(1, 4)
  uncontrolled, controlled = result.optimization.baseline, result.optimization.controlled
  vmax = max(traffic.velocity.max(), result.optimization.baseline.velocity[:n].max())
  for ax, mat, title in zip(p, (traffic.velocity, uncontrolled.velocity[:n], controlled.velocity[:n]), (
     f'{spec.freeway.upper().replace("I24", "I-24")} Ground Truth',
     f'METANET Simulation\n(TT: {uncontrolled.total_travel_time:.2f} veh-hr, Delay: {np.round(stats.sim_tt - stats.ff_tt, 2)} veh-hr)',
     f'VSL Control\n(TT: {controlled.total_travel_time:.2f} veh-hr, Delay: {np.round(stats.opt_tt - stats.ff_tt, 2)} veh-hr)'
  )):
    ax.plot_time_space(mat, spec, spec.start_hour, vmax, title=title, colorbar=("h", "Velocity (km/hr)"))
  p[3].plot_time_space(pct_decrease, spec, spec.start_hour, vmin=-100, vmax=100, title="% Decrease in Delay", 
                       colorbar=("h", "% Decrease in Delay"), cmap='inferno', smooth=True)
  p.savefig(FIGURE_DIR / "controlled.png")

  p = Plotter(8, 4, figsize=(20, 20))
  vmax = max(result.scenario.traffic.velocity.max(), uncontrolled.velocity.max())
  for k, t in enumerate(np.rint(np.linspace(0, n, 16)).astype(int)):
    batch, col = divmod(k, 4)
    for row, data, label in ((2 * batch, np.concatenate([display_vsl[:t], np.full_like(display_vsl[t:], 150)]), "VSL Speeds"), 
                             (2 * batch + 1, np.concatenate([controlled.velocity[:t], uncontrolled.velocity[t:]]), "Time Space")):
      p[row, col].plot_time_space(data, result.scenario.spec, spec.start_hour, vmax, title=f"{label} {k}")
      if 0 < t < len(display_vsl): p[row, col].axvline(t, color="black", linewidth=3)
  p.fig.tight_layout()
  p.savefig(FIGURE_DIR / "cc_reveal_evaluate.png", dpi=150)

if __name__ == "__main__":
  print_file(Path(__file__))
  main()
