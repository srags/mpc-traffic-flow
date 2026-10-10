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
import sys

import numpy as np

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from traffic_flow.inputs.scenario import load_params, load_scenario
from traffic_flow.paths import REPO_DIR, cut_repo

from traffic_flow.results.console import colored
from traffic_flow.results.io import load_results
from traffic_flow.results.plots import Plotter

from traffic_flow.config import CalRef, CalSource
from traffic_flow.model.simulation import METANET_Simulator # noqa: E402
from traffic_flow.pipeline import init_state
from traffic_flow.results.analysis import get_ff_tts, format_date_label

SAVE_PATH = REPO_DIR / "figs" / "tsd_plot" / "tsd.png"

DATES = ["11_28", "12_02"]
# Segments 0 and 1 are left uncontrolled (see Section 5.2); the control zone
# begins at this distance along the corridor.
L, time_step = 0.4, 10/3600
CONTROL_ZONE_START_KM = 2 * L
SHOW_OBSERVED = True
TEXT_FONTSIZE = 18

def run_day(date):
    """Observed, uncontrolled and controlled velocity fields for one date."""
    scenario = load_scenario("i24", date)
    spec, traffic = scenario.spec, scenario.traffic
    params = load_params(scenario, CalRef(CalSource.FIXED_RAMPS, interval=None))
    assert L == spec.L
    sim = METANET_Simulator(T=time_step, l=L, params=params, lanes=dict(enumerate(traffic.lanes)), real_data=True)
    _, v_sim, _, tts_sim = sim.run_with_history(traffic.inflow, traffic.downstream_density, init_state(traffic))

    runs = load_results("i24", date, calibration=CalRef(CalSource.FIXED_RAMPS, interval=None), study=None)
    assert len(runs) == 1, f"Expected 1 run for date {date}, got {len(runs)}"
    vsl = runs[0].optimization.vsl

    sim.real_data = False
    _, v_opt, _, tts_opt = sim.run_with_history(traffic.inflow, traffic.downstream_density, init_state(traffic), vsl_speeds=vsl)

    v_free = params["v_free"]
    ff_ttt = get_ff_tts(traffic.inflow, time_step, L, np.max(v_free, axis=0) if v_free.ndim == 2 else v_free)
    delay, opt_delay = tts_sim - ff_ttt, tts_opt - ff_ttt
    cc = float(np.clip((delay - opt_delay) / delay * 100, 0, 100))

    return {
        "date": date,
        "label": format_date_label(date),
        "observed": traffic.velocity,
        "no_control": v_sim[:-1, :],
        "controlled": v_opt[:-1, :],
        "tts": tts_sim,
        "delay": delay,
        "opt_delay": opt_delay,
        "cc": cc,
        "num_segments": spec.num_segments,
    }


def plot(results, save_path=SAVE_PATH):

    panels = ([("observed", "Observed")] if SHOW_OBSERVED else []) + [
        ("no_control", "No control"),
        ("controlled", "Optimal speed limit control"),
    ]
    n_rows, n_cols = len(results), len(panels)

    corridor_km = results[0]["num_segments"] * L
    duration_min = results[0]["observed"].shape[0] * time_step * 60
    extent = (0, duration_min, 0, corridor_km)
    vmax = float(np.ceil(max(r["observed"].max() for r in results) / 10) * 10)

    p = Plotter(n_rows, n_cols, figsize=(4.6 * n_cols, 3.6 * n_rows))

    panel_labels = "abcdefghi"
    for row, res in enumerate(results):
        for col, (key, title) in enumerate(panels):
            ax = p[row, col]
            im = ax.imshow(res[key].T, aspect="auto", origin="lower",
                           cmap="RdYlGn", interpolation="none",
                           vmin=0, vmax=vmax, extent=extent)
            ax.axhline(CONTROL_ZONE_START_KM, color="blue",
                       linestyle="--", linewidth=2.5)

            head = f"({panel_labels[row * n_cols + col]}) {res['label']}, {title}"
            if key == "no_control": head += f"\nDelay = {res['delay']:.0f} veh-hrs"
            elif key == "controlled": head += (f"\nDelay = {res['opt_delay']:.0f} veh-hrs, "
                                                f"CC = {res['cc']:.0f}\\%".replace("\\%", "%"))
            ax.set_title(head, fontsize=TEXT_FONTSIZE - 5)

            if col == 0:
                ax.set_ylabel("Distance (km)", fontsize=TEXT_FONTSIZE - 3)
            if row == n_rows - 1:
                ax.set_xlabel("Time (min)", fontsize=TEXT_FONTSIZE - 3)

            ax.set_xticks(np.arange(0, duration_min + 1, 15))
            ax.set_yticks(np.arange(0, corridor_km + 0.1, 1))
            ax.tick_params(labelsize=TEXT_FONTSIZE - 7)

    p.fig.subplots_adjust(right=0.89, hspace=0.32, wspace=0.08)
    cbar_ax = p.fig.add_axes((0.91, 0.12, 0.015, 0.76))
    cbar = p.fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Velocity (km/hr)", fontsize=TEXT_FONTSIZE - 3)
    cbar.ax.tick_params(labelsize=TEXT_FONTSIZE - 7)

    p.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"Figure saved to: {colored(cut_repo(save_path), 'bold', 'green')}")
    return p.fig


if __name__ == "__main__":
    results = []
    for date in DATES:
        res = run_day(date)
        results.append(res)
        print(f"{res['label']}: TTT {res['tts']:.2f} veh-hrs, "
              f"delay {res['delay']:.2f} -> {res['opt_delay']:.2f} veh-hrs, "
              f"CC = {res['cc']:.2f}%")

    plot(results)
