"""Velocity time-space diagrams for the I-24 case study.

One row per date; columns are the observed field, the calibrated model with no
control, and the model under optimal speed limit control. The default pair
(11/28 and 12/02) is nearly identical in total travel time, total delay and
delay per vehicle, yet differs by a factor of more than two in controllable
congestion.

Run from anywhere:

    python experiments/i24_tsd.py

Saves to figs/i24_tsd.png and prints the delay / CC figures for each row.
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from cc_analysis import (                    # noqa: E402
    L, time_step, RESULTS_ROOT,
    load_day_data, get_ff_tts, format_date_label,
)
from traffic_sim import run_metanet_sim      # noqa: E402

SAVE_PATH = os.path.join(REPO, "figs", "i24_tsd.png")

DATES = ["11_28", "12_02"]
# Segments 0 and 1 are left uncontrolled (see Section 5.2); the control zone
# begins at this distance along the corridor.
CONTROL_ZONE_START_KM = 2 * L
SHOW_OBSERVED = True
TEXT_FONTSIZE = 18


def run_day(date):
    """Observed, uncontrolled and controlled velocity fields for one date."""
    day = load_day_data(date)
    params = day["static_params"]

    _, v_sim, _, tts_sim = run_metanet_sim(
        time_step, L, day["init_state"], day["data_inflow"],
        day["ds_density_norm"], params, lanes=day["lane_dict"],
        vsl_speeds=None, plotting=True, real_data=True,
    )

    vsl_path = f"{RESULTS_ROOT}/i24_{date}/calibration_static/fixed_ramping/optimal_vsl.npy"
    vsl = np.load(vsl_path)

    _, v_opt, _, tts_opt = run_metanet_sim(
        time_step, L, day["init_state"], day["data_inflow"],
        day["ds_density_norm"], params, lanes=day["lane_dict"],
        vsl_speeds=vsl, plotting=True, real_data=False,
    )

    v_free = params["v_free"]
    ff_ttt = get_ff_tts(day["data_inflow"], time_step, L,
                        np.max(v_free, axis=0) if v_free.ndim == 2 else v_free)
    delay, opt_delay = tts_sim - ff_ttt, tts_opt - ff_ttt
    cc = float(np.clip((delay - opt_delay) / delay * 100, 0, 100))

    return {
        "date": date,
        "label": format_date_label(date),
        "observed": day["v_trimmed"],
        "no_control": v_sim[:-1, :],
        "controlled": v_opt[:-1, :],
        "tts": tts_sim,
        "delay": delay,
        "opt_delay": opt_delay,
        "cc": cc,
        "num_segments": day["num_segments"],
    }


def plot(results, save_path=SAVE_PATH):
    original_font = mpl.rcParams["font.family"]
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman"]

    panels = ([("observed", "Observed")] if SHOW_OBSERVED else []) + [
        ("no_control", "No control"),
        ("controlled", "Optimal speed limit control"),
    ]
    n_rows, n_cols = len(results), len(panels)

    corridor_km = results[0]["num_segments"] * L
    duration_min = results[0]["observed"].shape[0] * time_step * 60
    extent = [0, duration_min, 0, corridor_km]
    vmax = float(np.ceil(max(r["observed"].max() for r in results) / 10) * 10)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.6 * n_cols, 3.6 * n_rows),
                             sharex=True, sharey=True, squeeze=False)

    panel_labels = "abcdefghi"
    im = None
    for row, res in enumerate(results):
        for col, (key, title) in enumerate(panels):
            ax = axes[row][col]
            im = ax.imshow(res[key].T, aspect="auto", origin="lower",
                           cmap="RdYlGn", interpolation="none",
                           vmin=0, vmax=vmax, extent=extent)
            ax.axhline(CONTROL_ZONE_START_KM, color="blue",
                       linestyle="--", linewidth=2.5)

            head = f"({panel_labels[row * n_cols + col]}) {res['label']}, {title}"
            if key == "no_control":
                head += f"\nDelay = {res['delay']:.0f} veh-hrs"
            elif key == "controlled":
                head += (f"\nDelay = {res['opt_delay']:.0f} veh-hrs, "
                         f"CC = {res['cc']:.0f}\\%".replace("\\%", "%"))
            ax.set_title(head, fontsize=TEXT_FONTSIZE - 5,
                         fontname="Times New Roman")

            if col == 0:
                ax.set_ylabel("Distance (km)", fontsize=TEXT_FONTSIZE - 3,
                              fontname="Times New Roman")
            if row == n_rows - 1:
                ax.set_xlabel("Time (min)", fontsize=TEXT_FONTSIZE - 3,
                              fontname="Times New Roman")

            ax.set_xticks(np.arange(0, duration_min + 1, 15))
            ax.set_yticks(np.arange(0, corridor_km + 0.1, 1))
            ax.tick_params(labelsize=TEXT_FONTSIZE - 7)
            for lab in ax.get_xticklabels() + ax.get_yticklabels():
                lab.set_fontname("Times New Roman")

    fig.subplots_adjust(right=0.89, hspace=0.32, wspace=0.08)
    cbar_ax = fig.add_axes([0.91, 0.12, 0.015, 0.76])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Velocity (km/hr)", fontsize=TEXT_FONTSIZE - 3,
                   fontname="Times New Roman")
    cbar.ax.tick_params(labelsize=TEXT_FONTSIZE - 7)
    for lab in cbar.ax.get_yticklabels():
        lab.set_fontname("Times New Roman")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"\nFigure saved to: {save_path}")
    mpl.rcParams["font.family"] = original_font
    return fig


if __name__ == "__main__":
    results = []
    for date in DATES:
        res = run_day(date)
        results.append(res)
        print(f"{res['label']}: TTT {res['tts']:.2f} veh-hrs, "
              f"delay {res['delay']:.2f} -> {res['opt_delay']:.2f} veh-hrs, "
              f"CC = {res['cc']:.2f}%")

    plot(results)
