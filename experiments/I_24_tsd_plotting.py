"""Velocity time-space diagrams for the I-24 case study.

One row per date; columns are the observed field, the calibrated model with no
control, and the model under optimal speed limit control. The default pair
(11/28 and 12/02) is nearly identical in total travel time, total delay and
delay per vehicle, yet differs by a factor of more than two in controllable
congestion.

Run from anywhere:

    python experiments/I_24_tsd_plotting.py

Saves to figs/i24_tsd.pdf and prints the delay / CC figures for each row. It also
saves figs/i24_tsd_cum_departures.pdf, the same figure with one extra column per row:
that day's cumulative departures, drawn as in capacity_drop_diagnostics.ipynb
(free flow, no control, optimal control; oblique, with the delay regions shaded).
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from paths import fig, i24_results           # noqa: E402
from cc_analysis import (                    # noqa: E402
    L, time_step,
    load_day_data, ff_tts_vkm, format_date_label, free_flow_departures,
)
from traffic_sim import run_metanet_sim      # noqa: E402

SAVE_PATH = fig("i24_tsd.pdf")
SAVE_PATH_CUM = fig("i24_tsd_cum_departures.pdf")

DATES = ["11_28", "12_02"]
# Segments 0 and 1 are left uncontrolled (see Section 5.2); the control zone
# begins at this distance along the corridor.
CONTROL_ZONE_START_KM = 2 * L
SHOW_OBSERVED = True
TEXT_FONTSIZE = 18
LEGEND_FONTSIZE = TEXT_FONTSIZE - 5   # legend under the cumulative-departures column
COLORBAR_HEIGHT_IN = 0.14             # height of the horizontal velocity colorbar

NC_COLOR, CTL_COLOR = "#d33b19", "#4e9858"   # no control / optimal control
# Subtract a background rate from the cumulative curves so the delay regions are
# readable against ~8000 vehicles. It tilts the axes without changing any area.
OBLIQUE = True
DRAIN_TOL = 3     # vehicles left in the corridor at which a drain stops


def cumulative_departures(density, velocity, lane_counts, beta, n_steps):
    """Vehicles that have left the corridor by the end of each step.

    A vehicle departs out of the last segment or at an off-ramp, which takes
    q_i * beta_i / (1 - beta_i). Inputs are (time_steps, num_segments) per-lane
    density and velocity. The series is held flat after the run has drained so it
    shares a time axis of `n_steps` with the other runs.
    """
    q = density * velocity * lane_counts                       # veh/h per segment
    departures = q[:, -1] + (q * beta / (1 - beta)).sum(axis=1)  # veh/h leaving
    c = np.cumsum(departures) * time_step
    out = np.full(n_steps, c[-1])
    out[: len(c)] = c
    return out


def run_day(date):
    """Observed, uncontrolled and controlled velocity fields for one date."""
    day = load_day_data(date)
    params = day["static_params"]

    # Both runs drain after the demand horizon so every vehicle finishes its trip and the
    # two serve the same vehicle-km, which is what delay and CC are scored on. The panels
    # below still show only the 60 min data window; the drain is identical to the fixed
    # horizon up to that point, so the extra steps are simply sliced off.
    p_sim, v_sim, _, tts_sim = run_metanet_sim(
        time_step, L, day["init_state"], day["data_inflow"],
        day["ds_density_norm"], params, lanes=day["lane_dict"],
        vsl_speeds=None, plotting=True, real_data=True, until_ff=True,
    )

    vsl_path = i24_results(date, "optimal_vsl.npy")
    vsl = np.load(vsl_path)

    p_opt, v_opt, _, tts_opt = run_metanet_sim(
        time_step, L, day["init_state"], day["data_inflow"],
        day["ds_density_norm"], params, lanes=day["lane_dict"],
        vsl_speeds=vsl, plotting=True, real_data=False, until_ff=True,
    )

    n_window = day["v_trimmed"].shape[0]

    # One free-flow baseline for both runs, taken from the uncontrolled one.
    ff_ttt = ff_tts_vkm(p_sim[:-1, :], v_sim[:-1, :], day["lane_counts"],
                        time_step, L, params["v_free"])
    delay, opt_delay = tts_sim - ff_ttt, tts_opt - ff_ttt
    cc = float(np.clip((delay - opt_delay) / delay * 100, 0, 100))

    # Cumulative departures, computed as in capacity_drop_diagnostics.ipynb: both runs
    # drain until DRAIN_TOL vehicles are left, so they serve the same vehicles, and the
    # free-flow curve takes the same arrivals through the corridor at v_free.
    def drained(speeds):
        rho, vel, _, _ = run_metanet_sim(
            time_step, L, day["init_state"], day["data_inflow"],
            day["ds_density_norm"], params, lanes=day["lane_dict"],
            vsl_speeds=speeds, plotting=True, real_data=False,
            until_ff=True, drain_tol=DRAIN_TOL,
        )
        return rho[:-1], vel[:-1]

    rho_nc, vel_nc = drained(None)
    rho_c, vel_c = drained(vsl)
    n_steps = max(len(rho_nc), len(rho_c))
    beta = np.asarray(params["beta"], dtype=float)
    lanes = np.asarray(day["lane_counts"], dtype=float)
    arrivals, dep_ff = free_flow_departures(day, params, n_steps)

    return {
        "date": date,
        "label": format_date_label(date),
        "observed": day["v_trimmed"],
        "no_control": v_sim[:n_window, :],
        "controlled": v_opt[:n_window, :],
        "tts": tts_sim,
        "delay": delay,
        "opt_delay": opt_delay,
        "cc": cc,
        "num_segments": day["num_segments"],
        "dep_time_min": np.arange(n_steps) * time_step * 60,
        "dep_no_control": cumulative_departures(rho_nc, vel_nc, lanes, beta, n_steps),
        "dep_controlled": cumulative_departures(rho_c, vel_c, lanes, beta, n_steps),
        "dep_free_flow": dep_ff,
        "arrivals": float(arrivals[-1]),
    }


def plot_cumulative_departures(ax, res, duration_min, letter):
    """One day's cumulative departures against free flow, as in the diagnostics notebook.

    Free-flow, no-control and optimal-control departures, with the delay each run
    adds (the area between it and free flow) shaded. With OBLIQUE the curves are
    shown minus q0 * t, where q0 is the vehicles served per hour of demand. The
    legend is added once, below the column, by `plot`.
    """
    t_min = res["dep_time_min"]
    q0 = res["arrivals"]
    tilt = (lambda c: c - q0 * t_min / 60) if OBLIQUE else (lambda c: c)
    ff, nc, ctl = (tilt(res[k]) for k in ("dep_free_flow", "dep_no_control",
                                          "dep_controlled"))

    ax.fill_between(t_min, ff, nc, color=NC_COLOR, alpha=0.15, linewidth=0,
                    label="Delay, no control")
    ax.fill_between(t_min, ff, ctl, color=CTL_COLOR, alpha=0.30, linewidth=0,
                    label="Delay, optimal control")
    ax.plot(t_min, ff, color="black", linestyle="--", linewidth=1.6,
            label="Free-flow departures")
    ax.plot(t_min, nc, color=NC_COLOR, linewidth=2.2, label="No control")
    ax.plot(t_min, ctl, color=CTL_COLOR, linewidth=2.2, label="Optimal control")
    # End of the 60 min of demand; everything after it is the drain.
    ax.axvline(duration_min, color="0.7", linewidth=1)
    ax.annotate("demand ends", (duration_min, 1), xycoords=("data", "axes fraction"),
                xytext=(-6, -12), textcoords="offset points", ha="right",
                fontsize=TEXT_FONTSIZE - 9, color="0.4",
                fontname="Times New Roman")

    ax.set_xlim(0, t_min[-1])
    ax.grid(True, color="lightgrey", linewidth=0.5)
    ax.set_axisbelow(True)
    second = (rf"oblique at $q_0$ = {q0:,.0f} veh/h" if OBLIQUE
              else f"Vehicles served = {q0:,.0f}")
    ax.set_title(f"({letter}) {res['label']}, Cumulative departures\n{second}",
                 fontsize=TEXT_FONTSIZE - 5, fontname="Times New Roman")
    ax.set_ylabel(r"Num. departures $-\ q_0 t$ (veh)" if OBLIQUE
                  else "Num. departures (veh)",
                  fontsize=TEXT_FONTSIZE - 3, fontname="Times New Roman")
    ax.set_xticks(np.arange(0, t_min[-1] + 1, 15))
    ax.tick_params(labelsize=TEXT_FONTSIZE - 7)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontname("Times New Roman")


def plot(results, save_path=SAVE_PATH, cumulative=False):
    """Time-space diagrams, one row per date.

    With `cumulative=True` each row gets one extra column on the right: that day's
    cumulative departures (free flow, no control, optimal control; see
    `plot_cumulative_departures`), out to the end of the drain. In that layout the
    velocity colorbar is horizontal, below the time-space panels, and the
    departures legend is below its column.
    """
    original_font = mpl.rcParams["font.family"]
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman"]

    panels = ([("observed", "Observed")] if SHOW_OBSERVED else []) + [
        ("no_control", "No control"),
        ("controlled", "Optimal speed limit control"),
    ]
    n_rows, n_tsd = len(results), len(panels)
    n_cols = n_tsd + (1 if cumulative else 0)   # panel letters run across this many

    corridor_km = results[0]["num_segments"] * L
    duration_min = results[0]["observed"].shape[0] * time_step * 60
    extent = [0, duration_min, 0, corridor_km]
    vmax = float(np.ceil(max(r["observed"].max() for r in results) / 10) * 10)

    if not cumulative:
        fig, axes = plt.subplots(n_rows, n_tsd,
                                 figsize=(4.6 * n_tsd, 3.6 * n_rows),
                                 sharex=True, sharey=True, squeeze=False)
    else:
        # Top: [time-space block | room for the y label | cumulative departures].
        # Bottom strip: the velocity colorbar under the time-space block (the
        # departures legend hangs below the cumulative column).
        fig = plt.figure(figsize=(4.6 * n_tsd + 5.4, 3.6 * n_rows + 1.0))
        outer = fig.add_gridspec(
            2, 3, width_ratios=[4.6 * n_tsd, 1.1, 4.2], height_ratios=[1, 0.07],
            wspace=0.0, hspace=0.42,
            left=0.045, right=0.99, bottom=0.06, top=0.90)
        gs_tsd = outer[0, 0].subgridspec(n_rows, n_tsd, wspace=0.08, hspace=0.32)
        gs_cum = outer[0, 2].subgridspec(n_rows, 1, hspace=0.32)
        gs_cbar = outer[1, 0].subgridspec(1, 3, width_ratios=[0.9, 1.2, 0.9], wspace=0.0)

        axes = np.empty((n_rows, n_tsd), dtype=object)
        for r in range(n_rows):
            for c in range(n_tsd):
                ref = axes[0][0] if (r or c) else None
                axes[r][c] = fig.add_subplot(gs_tsd[r, c], sharex=ref, sharey=ref)
                axes[r][c].label_outer()
        cax = fig.add_subplot(gs_cbar[0, 1])

    panel_labels = "abcdefghijkl"
    im = None
    cum_axes = []
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

        if cumulative:
            ref = cum_axes[0] if row else None
            cax_row = fig.add_subplot(gs_cum[row, 0], sharex=ref, sharey=ref)
            cax_row.label_outer()
            cum_axes.append(cax_row)
            plot_cumulative_departures(
                cax_row, res, duration_min,
                panel_labels[row * n_cols + n_tsd])
            if row == n_rows - 1:
                cax_row.set_xlabel("Time (min)", fontsize=TEXT_FONTSIZE - 3,
                                   fontname="Times New Roman")

    if cumulative:
        # One legend for the column, centered below the bottom panel.
        handles, labels = cum_axes[-1].get_legend_handles_labels()
        legend = cum_axes[-1].legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.30),
            ncol=2, prop={"family": "Times New Roman", "size": LEGEND_FONTSIZE})
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    else:
        fig.subplots_adjust(right=0.89, hspace=0.32, wspace=0.08)
        cbar_ax = fig.add_axes([0.91, 0.12, 0.015, 0.76])
        cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Velocity (km/hr)", fontsize=TEXT_FONTSIZE - 3,
                   fontname="Times New Roman")
    cbar.ax.tick_params(labelsize=TEXT_FONTSIZE - 7)
    for lab in cbar.ax.get_xticklabels() + cbar.ax.get_yticklabels():
        lab.set_fontname("Times New Roman")

    if cumulative:
        # Vertically center the whole colorbar (bar, tick labels and label) on the
        # legend so the two read as one band. Positions are only known once the figure
        # has been laid out, so draw first, then move the colorbar.
        fig.canvas.draw()
        to_fig = fig.transFigure.inverted()
        legend_box = legend.get_window_extent().transformed(to_fig)
        pos = cax.get_position()
        height = COLORBAR_HEIGHT_IN / fig.get_figheight()
        cax.set_position([pos.x0, legend_box.y0 + (legend_box.height - height) / 2,
                          pos.width, height])
        fig.canvas.draw()
        unit_box = cax.get_tightbbox().transformed(to_fig)   # incl. ticks and label
        shift = ((legend_box.y0 + legend_box.y1) - (unit_box.y0 + unit_box.y1)) / 2
        pos = cax.get_position()
        cax.set_position([pos.x0, pos.y0 + shift, pos.width, pos.height])

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
    plot(results, SAVE_PATH_CUM, cumulative=True)
