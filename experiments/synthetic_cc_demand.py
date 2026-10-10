"""Replay the synthetic demand sweep and save controllable congestion bars."""
import sys; from pathlib import Path
REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

from traffic_flow.config import SyntheticConfig
from traffic_flow.paths import REPO_DIR
from traffic_flow.results.synthetic import load_synthetic_results, print_synthetic_report, save_synthetic_cc

CONFIG = SyntheticConfig()
PEAKS = range(5100, 6501, 50)
POLICY_DIR = REPO_DIR / "results_bu" / "synthetic_10km" / "demand"
FIGURE_DIR = REPO_DIR / "figs"

def main() -> None:
  results = load_synthetic_results(PEAKS, POLICY_DIR, config=CONFIG, skip_missing=True)
  print_synthetic_report(results)
  save_synthetic_cc(results, FIGURE_DIR, units="min")

if __name__ == "__main__":
  main()
