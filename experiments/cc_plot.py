"""Replay a saved policy and generate the two CC figures; never run MPC."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traffic_flow.results.analysis import cc_report, print_cc_report
from traffic_flow.config import CalRef, CalSource, InitMode, Study
from traffic_flow.results.io import load_runs
from traffic_flow.paths import REPO_DIR
from traffic_flow.results.plots import save_cc_plots
from traffic_flow.results.console import colored

from traffic_flow.types import RunResult
from traffic_flow import evaluate

FIGURE_DIR = REPO_DIR / "figs"
START_HOUR = 7.5

def main() -> None:
  matches = load_runs(
    "i24", "11_30", calibration=CalRef(CalSource.FIXED_RAMPS, interval = None), study=Study.SAFETY_SWEEP, 
    where=lambda config: all([
      # config.speed_lb == 0,
      config.initialize_vsl is not None
      ]),
  )
  if len(matches) != 1: raise ValueError(f"Expected one run, found {len(matches)}. Refine the selection.")
  result = matches[0]
  report = cc_report(result.scenario, result.params, result.optimization)
  print_cc_report(report)
  save_cc_plots(report, FIGURE_DIR, start_hour=START_HOUR)
  save_reveal_plot(result, FIGURE_DIR, checkpoints=16, batches=4)

def save_reveal_plot(result: RunResult, output_dir: Path, checkpoints: int = 16, batches: int = 4) -> Path:
  """Save the evaluate-based reveal using modeled inflow/queue and a 150 km/hr baseline."""
  import numpy as np
  from traffic_flow.results.plots import Plotter

  if checkpoints < 2 or batches < 1 or checkpoints % batches:
    raise ValueError("CHECKPOINTS must be at least 2 and divisible by BATCHES")
  scenario, params = result.scenario, result.params
  spec = scenario.spec
  baseline = result.optimization.baseline.velocity[:-1]
  # Keep the notebook's speed substitution to isolate the origin-mode change.
  optimal_vsl = np.where(result.optimization.vsl > params["v_free"], 150, result.optimization.vsl)
  free_vsl = np.full_like(optimal_vsl, 150)

  times = np.rint(np.linspace(0, len(optimal_vsl), checkpoints)).astype(int)
  columns = checkpoints // batches
  p = Plotter(2 * batches, columns, figsize=(5 * columns, 5 * batches))
  vmax = max(scenario.traffic.velocity.max(), baseline.max())
  extent = (0, spec.time_steps * spec.time_step * 60, 0, spec.num_segments * spec.L)

  for k, t in enumerate(times):
    vsl = free_vsl.copy()
    vsl[:t] = optimal_vsl[:t]
    velocity = baseline.copy()
    if t:
      replay = evaluate(scenario, params, vsl)
      velocity[:t] = replay.controlled.velocity[:t]
    batch, column = divmod(k, columns)
    for row, data, label, first, last in (
      (2 * batch, vsl, "VSL Speeds", "Free VSL Speeds", "Optimal VSL Speeds"),
      (2 * batch + 1, velocity, "Time Space", "Baseline Time Space", "Optimized Time Space"),
    ):
      ax = p[row, column]
      ax.imshow(data.T, cmap="RdYlGn", aspect="auto", interpolation="none",
                origin="lower", extent=extent, vmin=0, vmax=vmax)
      if 0 < t < len(optimal_vsl): ax.axvline(t * spec.time_step * 60, color="black", linewidth=3)
      ax.set(title=first if k == 0 else last if k == checkpoints - 1 else f"{label} {k}",
              xlabel="Time (min)", ylabel="Distance (km)")

  p.fig.tight_layout()
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  path = output_dir / "cc_reveal_evaluate.png"
  p.savefig(path, dpi=150)
  display = path.relative_to(REPO_DIR) if path.is_relative_to(REPO_DIR) else path
  print(f"Saved to {colored(display, 'green')}")
  return path

if __name__ == "__main__":
  main()
