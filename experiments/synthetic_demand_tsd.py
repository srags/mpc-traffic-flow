"""Replay two synthetic demand cases and save baseline/controlled heatmaps."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traffic_flow.config import SyntheticConfig
from traffic_flow.paths import REPO_DIR
from traffic_flow.results.synthetic import load_synthetic_results, print_synthetic_report, save_synthetic_tsd

CONFIG = SyntheticConfig()
PEAKS = (5500, 6250)
POLICY_DIR = REPO_DIR / "results_bu" / "synthetic_10km" / "demand"
FIGURE_DIR = REPO_DIR / "figs"
CONTROL_START_KM = 4.0


def main() -> None:
  results = load_synthetic_results(PEAKS, POLICY_DIR, config=CONFIG)
  print_synthetic_report(results)
  save_synthetic_tsd(results, FIGURE_DIR, control_start_km=CONTROL_START_KM)


if __name__ == "__main__":
  main()
