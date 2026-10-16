import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

import numpy as np

from traffic_flow import CalRef, LATEST_UPDATE, load_one_result
from traffic_flow.results.plots import Plotter
from traffic_flow.results.sensitivity import one_at_a_time_cc, monte_carlo_cc


DATE = "11_28"
SHOW_MAPE = True
PERTURB_PCT = np.arange(-20, 21) * 0.0005  # -1% through +1%


if __name__ == "__main__":
  assert LATEST_UPDATE, "Use LATEST_UPDATE=True to match current main."

  run = load_one_result("i24", DATE, calibration=CalRef(), study=None)
  result = one_at_a_time_cc(run, PERTURB_PCT)

  zero = result.perturb_pct == 0
  for key in result.cc:
    np.testing.assert_allclose(
      result.cc[key][zero], result.nominal.cc_unclipped,
      rtol=1e-10, atol=1e-10,
    )
    np.testing.assert_allclose(
      result.mape[key][zero], result.nominal.sim_error,
      rtol=1e-10, atol=1e-10,
    )

  p = Plotter(2, 3, figsize=(12, 6))
  p.plot_param_sensitivity(result, show_mape=SHOW_MAPE)

  output = REPO_DIR / "figs" / "param_sensitivity"
  output.mkdir(parents=True, exist_ok=True)
  suffix = "_with_mape" if SHOW_MAPE else ""
  p.savefig(output / f"{DATE}{suffix}.pdf", bbox_inches="tight")
  p.show()



  run = load_one_result("i24", DATE, calibration=CalRef(), study=None)
  mc = monte_carlo_cc(run, n_samples=50, seed=0)

  print(f"Nominal CC: {mc.nominal.cc_unclipped:.2f}%")
  print(f"Accepted: {len(mc.cc)}/{mc.attempted}")
  if mc.cc.size:
      print(f"CC mean ± std: {mc.cc.mean():.2f} ± {mc.cc.std():.2f}%")