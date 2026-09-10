"""Controllable congestion vs. peak demand for the synthetic bottleneck
(Figure 3a of the manuscript).

Red bar is the average delay per vehicle with no control; the green overlay is
the portion of that delay removed by optimal speed limit control, so the green
fraction of each bar is visually equal to that scenario's controllable
congestion, printed above the bar.

Regenerates the plot formerly produced by the __main__ block of
src/generate_demand_synthetic.py, with two corrections:

  1. The free-flow reference is computed over the unpadded two-hour demand
     profile rather than the profile padded out to the MPC horizon. The
     original mixed the two conventions -- delay used the padded profile while
     the vehicle count used the unpadded one -- which inflated T_FF and shifted
     every controllable-congestion value by roughly three percentage points.
  2. Model parameters come from METANET_Params(path=None); the DefaultParams
     class the original imported no longer exists in src/. The two differ only
     in q_capacity (2400 vs 2200 veh/hr/lane), which has no effect here because
     the origin merge never binds at these demand levels.

Run from anywhere:

    python experiments/synthetic_cc_demand.py
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from synthetic_demand_tsd import (        # noqa: E402
    NUM_SEGMENTS,
    METANET_Params,
    run_scenario,
    REPO,
)

from paths import fig                         # noqa: E402

SAVE_PATH = fig("synthetic_cc_demand.png")

PEAK_MIN, PEAK_MAX, NUM_SCENARIOS = 5100, 6500, 29
TEXT_FONTSIZE = 18

# "min" matches the units of the I-24 bar chart in src/cc_analysis.py;
# "hr" reproduces the units of the original delay_reduction.png.
UNITS = "min"


def collect(peaks, params):
    results = []
    for peak in peaks:
        try:
            res = run_scenario(peak, params)
        except FileNotFoundError:
            print(f"  [skip] no saved policy for peak demand {peak:.0f}")
            continue
        scale = 60.0 if UNITS == "min" else 1.0
        res["avg_delay_uc"] = res["delay_uc"] / res["num_veh"] * scale
        res["avg_delay_c"] = res["delay_c"] / res["num_veh"] * scale
        results.append(res)
        print(f"  {res['peak']:6.0f}  delay {res['delay_uc']:7.2f} -> "
              f"{res['delay_c']:7.2f} veh-hrs   CC {res['cc']:6.2f}%")
    return results


def plot(results, save_path=SAVE_PATH):
    original_font = mpl.rcParams["font.family"]
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman"]

    peaks = np.array([r["peak"] for r in results])
    uncontrolled = np.array([r["avg_delay_uc"] for r in results])
    controlled = np.array([r["avg_delay_c"] for r in results])
    cc = np.array([r["cc"] for r in results])
    reduced = uncontrolled - controlled

    scenarios = np.arange(1, len(results) + 1)
    width = 0.85

    fig, ax = plt.subplots(figsize=(15, 6))
    plt.grid()
    ax.set_axisbelow(True)

    ax.bar(scenarios, uncontrolled, label="Delay without control",
           color="#d33b19", width=width)
    ax.bar(scenarios, reduced, label="Delay reduced by control",
           color="#4e9858", width=width)

    for i in range(len(scenarios)):
        ax.text(scenarios[i], uncontrolled[i] + 0.01 * np.max(uncontrolled),
                f"{cc[i]:.0f}", ha="center", va="bottom",
                fontsize=TEXT_FONTSIZE - 6, fontname="Times New Roman",
                fontweight="bold")

    unit_label = "min" if UNITS == "min" else "hrs"
    ax.set_xlabel("Peak demand of scenario (veh/hr)",
                  fontsize=TEXT_FONTSIZE, fontname="Times New Roman")
    ax.set_ylabel(f"Average delay per vehicle ({unit_label})",
                  fontsize=TEXT_FONTSIZE, fontname="Times New Roman")
    ax.set_xticks(scenarios)
    ax.set_xticklabels(peaks.astype(int), rotation=45, ha="center")
    ax.legend(prop={"family": "Times New Roman", "size": TEXT_FONTSIZE})

    ax.tick_params(labelsize=TEXT_FONTSIZE - 4)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("Times New Roman")

    ax.set_ylim(0, np.max(uncontrolled) * 1.12)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"\nFigure saved to: {save_path}")
    mpl.rcParams["font.family"] = original_font
    return fig


if __name__ == "__main__":
    params = METANET_Params(path=None, num_segments=NUM_SEGMENTS).get_params()
    peaks = np.linspace(PEAK_MIN, PEAK_MAX, NUM_SCENARIOS, endpoint=True)

    print("── CC vs peak demand ─────────────────────────────────────")
    results = collect(peaks, params)
    plot(results)

    cc = np.array([r["cc"] for r in results])
    peaks_run = np.array([r["peak"] for r in results])
    i_max = int(np.argmax(cc))
    drops = np.diff(cc)
    i_drop = int(np.argmin(drops))
    print(f"\n  CC peaks at {peaks_run[i_max]:.0f} veh/hr with CC = {cc[i_max]:.2f}%")
    print(f"  Largest single-step drop: {peaks_run[i_drop]:.0f} veh/hr "
          f"({cc[i_drop]:.2f}%) -> {peaks_run[i_drop + 1]:.0f} veh/hr "
          f"({cc[i_drop + 1]:.2f}%)")
