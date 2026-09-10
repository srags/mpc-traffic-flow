"""2x2 velocity time-space diagrams for the synthetic bottleneck case study.

Rows are peak demand levels, columns are uncontrolled vs. optimally controlled.
The point of the figure is the contrast between the two rows: at 5500 veh/hr
speed control largely dissipates the jam, while at 6250 veh/hr -- the first
scenario past the controllable-congestion cliff -- it no longer can.

Run from anywhere:

    python experiments/synthetic_demand_tsd.py

Saves to figs/synthetic_demand_tsd.png and prints the delay / CC figures for
each panel so the caption can be written directly from the output.
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from paths import fig, synthetic_results         # noqa: E402
from param_loader import METANET_Params          # noqa: E402
from traffic_sim import run_metanet_sim          # noqa: E402

DEMAND_DIR = synthetic_results("demand")
SAVE_PATH = fig("synthetic_demand_tsd.png")

# ── Scenario definition (mirrors generate_demand_synthetic.__main__) ─────────
SIM_TIME = 2.0                                    # hours
TIME_STEP = 10 / 3600                             # hours
SEG_LENGTH = 0.4                                  # km
TOTAL_DISTANCE = 10.0                             # km
NUM_SEGMENTS = int(TOTAL_DISTANCE / SEG_LENGTH)   # 25
FLOW_STANDARD = 4000                              # veh/hr
PEAK_START = 0.055                                # hours
PEAK_DURATION = 0.5                               # hours
LANES = {i: 4 if i < NUM_SEGMENTS - 5 else 2 for i in range(NUM_SEGMENTS)}

BOTTLENECK_KM = (NUM_SEGMENTS - 5) * SEG_LENGTH   # 8.0 km, where 4 lanes -> 2
CONTROL_ZONE_START_KM = 10 * SEG_LENGTH           # 4.0 km, first controlled segment

DEMANDS = [5500, 6250]
TEXT_FONTSIZE = 18


def generate_demand(peak_demand):
    """Rectangular peak on a constant base, as in generate_demand_synthetic."""
    total_time_steps = int(SIM_TIME / TIME_STEP)
    peak_end = PEAK_START + PEAK_DURATION
    demand = np.empty(total_time_steps + 1)
    for i in range(total_time_steps + 1):
        if int(PEAK_START / TIME_STEP) <= i < int(peak_end / TIME_STEP):
            demand[i] = peak_demand
        else:
            demand[i] = FLOW_STANDARD
    return demand


def get_num_veh(demand):
    """Vehicles served over the (unpadded) demand profile."""
    return float(np.sum(demand)) * TIME_STEP


def free_flow_tts(demand, v_free):
    """Free-flow TTS over the unpadded demand profile -- the convention that
    reproduces the delay figures reported for this case study.

    Note: the original driver computed this from the demand profile *after* it
    was padded out to the MPC horizon, which inflates T_FF and raises every CC
    value by roughly three percentage points. We use the unpadded profile so
    that the free-flow reference spans the same two hours as the simulation.
    """
    return get_num_veh(demand) * (TOTAL_DISTANCE / v_free)


def run_scenario(peak_demand, params):
    """Return velocity fields and delay for the uncontrolled and controlled runs."""
    time_steps = int(SIM_TIME / TIME_STEP)          # 720
    mpc_time_steps = time_steps + 40 - 5            # 755

    demand = generate_demand(peak_demand)
    start_state = (np.full(NUM_SEGMENTS, demand[0] / (LANES[0] * 90)),
                   np.full(NUM_SEGMENTS, 90.0),
                   demand[0],
                   0)

    _, v_uc, _, tts_uc = run_metanet_sim(
        TIME_STEP, SEG_LENGTH, start_state, demand, np.zeros(time_steps),
        params, lanes=LANES, real_data=False, vsl_speeds=None, plotting=True,
    )

    policy = os.path.join(DEMAND_DIR, f"demand{float(peak_demand)}_duration{PEAK_DURATION}.csv")
    if not os.path.exists(policy):
        raise FileNotFoundError(
            f"No saved VSL policy for peak demand {peak_demand}: {policy}"
        )
    optimal_vsl = np.loadtxt(policy, delimiter=",")

    demand_padded = demand.copy()
    while len(demand_padded) < mpc_time_steps + 1:
        demand_padded = np.append(demand_padded, demand_padded[-1])

    _, v_c, _, tts_c = run_metanet_sim(
        TIME_STEP, SEG_LENGTH, start_state,
        demand_padded[0:time_steps], np.zeros(mpc_time_steps + 1)[0:time_steps],
        params, lanes=LANES, vsl_speeds=optimal_vsl,
        real_data=False, plotting=True,
    )

    ff = free_flow_tts(demand, float(params["v_free"][0]))
    delay_uc, delay_c = tts_uc - ff, tts_c - ff

    return {
        "peak": peak_demand,
        "v_uc": v_uc, "v_c": v_c,
        "delay_uc": delay_uc, "delay_c": delay_c,
        "num_veh": get_num_veh(demand),
        "cc": (delay_uc - delay_c) / delay_uc * 100,
    }


def plot(results, save_path=SAVE_PATH):
    original_font = mpl.rcParams["font.family"]
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman"]

    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 4.6 * n_rows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    duration_min = SIM_TIME * 60
    extent = [0, duration_min, 0, TOTAL_DISTANCE]
    vmin, vmax = 0, 120

    panel_labels = "abcdefgh"
    im = None
    for row, res in enumerate(results):
        for col, (key, title) in enumerate(
            [("v_uc", "No control"), ("v_c", "Optimal speed limit control")]
        ):
            ax = axes[row, col]
            im = ax.imshow(res[key].T, aspect="auto", origin="lower",
                           cmap="RdYlGn", interpolation="none",
                           vmin=vmin, vmax=vmax, extent=extent)

            ax.axhline(BOTTLENECK_KM, color="black", linestyle="--", linewidth=2.5)
            ax.axhline(CONTROL_ZONE_START_KM, color="blue", linestyle="--", linewidth=2.5)

            delay = res["delay_uc"] if key == "v_uc" else res["delay_c"]
            label = (f"({panel_labels[row * 2 + col]}) {res['peak']:.0f} veh/hr, "
                     f"{title}\nDelay = {delay:.0f} veh-hrs")
            ax.set_title(label, fontsize=TEXT_FONTSIZE - 4, fontname="Times New Roman")

            if col == 0:
                ax.set_ylabel("Distance (km)", fontsize=TEXT_FONTSIZE - 2,
                              fontname="Times New Roman")
            if row == n_rows - 1:
                ax.set_xlabel("Time (min)", fontsize=TEXT_FONTSIZE - 2,
                              fontname="Times New Roman")

            ax.set_xticks(np.arange(0, duration_min + 1, 20))
            ax.set_yticks(np.arange(0, TOTAL_DISTANCE + 1, 2))
            ax.tick_params(labelsize=TEXT_FONTSIZE - 6)
            for lab in ax.get_xticklabels() + ax.get_yticklabels():
                lab.set_fontname("Times New Roman")

    fig.subplots_adjust(right=0.88, hspace=0.28, wspace=0.08)
    cbar_ax = fig.add_axes([0.90, 0.12, 0.02, 0.76])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Velocity (km/hr)", fontsize=TEXT_FONTSIZE - 2,
                   fontname="Times New Roman")
    cbar.ax.tick_params(labelsize=TEXT_FONTSIZE - 6)
    for lab in cbar.ax.get_yticklabels():
        lab.set_fontname("Times New Roman")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"\nFigure saved to: {save_path}")
    mpl.rcParams["font.family"] = original_font
    return fig


if __name__ == "__main__":
    params = METANET_Params(path=None, num_segments=NUM_SEGMENTS).get_params()

    results = []
    for peak in DEMANDS:
        res = run_scenario(peak, params)
        results.append(res)
        print(f"peak {res['peak']:.0f} veh/hr: "
              f"delay {res['delay_uc']:.2f} -> {res['delay_c']:.2f} veh-hrs, "
              f"CC = {res['cc']:.2f}%")

    plot(results)
