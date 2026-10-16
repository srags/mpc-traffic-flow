"""Replay the synthetic demand sweep and save controllable congestion bars."""
import sys; from pathlib import Path
REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

from traffic_flow.config import SyntheticConfig
from traffic_flow.paths import REPO_DIR
from traffic_flow.results.synthetic import (
  load_synthetic_results, print_synthetic_report, 
  save_synthetic_cc, save_synthetic_tsd
)

def main() -> None:
  results = load_synthetic_results(peaks=range(5100, 6501, 50), policy_dir=REPO_DIR / "results" / "synthetic_10km" / "demand", 
                                   config=SyntheticConfig(), skip_missing=True)
  print_synthetic_report(results)
  save_synthetic_cc(results, REPO_DIR / "figs" / "synthetic", units="min")

  results = load_synthetic_results(peaks=(5500, 6250), policy_dir=REPO_DIR / "results" / "synthetic_10km" / "demand", 
                                  config=SyntheticConfig(), skip_missing=True)
  print_synthetic_report(results)
  save_synthetic_tsd(results, REPO_DIR / "figs" / "synthetic", control_start_km=4.0)
  
if __name__ == "__main__":
  main()
