"""Operational-constraint sensitivity of controllable congestion on I-24.

Produces a 2x2 figure for Section 6.3:

    (a) hold length            (b) minimum posted speed limit
    (c) temporal smoothness    (d) spatial smoothness

Panels (c) and (d) are one-dimensional slices through the two-dimensional
safety sweep: the temporal curve holds the spatial bound fixed at
FIXED_SPATIAL, and the spatial curve holds the temporal bound fixed at
FIXED_TEMPORAL, so each shows the effect of one constraint in isolation.

Runs in which the solver exhausted every warm-start tier were saved as the
do-nothing fallback (VSL = 150 km/hr everywhere, see mpc_metanet.mpc_find_vsl).
These are detected and excluded rather than plotted as zeros, which would be
indistinguishable from a genuinely uncontrollable configuration. The number
excluded is reported at run time.

Run from anywhere:

    python experiments/i24_constraints.py
"""

import os
import re
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from cc_analysis import (                    # noqa: E402
    L, time_step, RESULTS_ROOT,
    load_day_data, get_ff_tts,
)
from traffic_sim import run_metanet_sim      # noqa: E402

DATE = "11_30"
SWEEP_ROOT = f"{RESULTS_ROOT}/i24_{DATE}/calibration_static/fixed_ramping"
SAVE_PATH = os.path.join(REPO, "figs", "i24_constraints.png")
HEATMAP_SAVE_PATH = os.path.join(REPO, "figs", "i24_safety_heatmap.png")

# Fixed value of the *other* bound in each smoothness slice, in km/hr. Chosen
# for coverage: these rows/columns of the grid have the most successful runs.
# Set to None to auto-select whichever value has the most successful runs.
FIXED_SPATIAL = 25.0     # held fixed in panel (c), the temporal curve
FIXED_TEMPORAL = 25.0    # held fixed in panel (d), the spatial curve

# Largest value of the swept parameter to plot in each panel, in that panel's
# own units. Points above this are excluded, which also sets the right-hand
# x-axis limit. None plots every available point.
MAX_SPEED_LB = 120      # caps the x axis of panel (b), in km/hr
MAX_TEMP = 10          # caps the x axis of panel (c), the temporal curve
MAX_SPAT = 10          # caps the x axis of panel (d), the spatial curve

TEXT_FONTSIZE = 18
LINE_COLOR = "#2b6b4f"
FILL_COLOR = "#4e9858"
MAXLINE_COLOR = "#cf2e1c"


# ── Baseline ─────────────────────────────────────────────────────────────────

def load_baseline(date=DATE):
    """Uncontrolled delay for `date`, against which every sweep run is scored."""
    day = load_day_data(date)
    params = day["static_params"]

    _, _, _, tts_base = run_metanet_sim(
        time_step, L, day["init_state"], day["data_inflow"],
        day["ds_density_norm"], params, lanes=day["lane_dict"],
        vsl_speeds=None, plotting=True, real_data=True,
    )
    v_free = params["v_free"]
    ff_ttt = get_ff_tts(day["data_inflow"], time_step, L,
                        np.max(v_free, axis=0) if v_free.ndim == 2 else v_free)
    return day, params, tts_base - ff_ttt, ff_ttt


def is_fallback(vsl):
    """True if this run is the do-nothing fallback saved after solver failure."""
    return bool(np.all(vsl == 150.0))


def cc_for(path, day, params, delay_base, ff_ttt):
    """Controllable congestion for one saved VSL run, or None if it failed."""
    vsl = np.load(path)
    if is_fallback(vsl):
        return None
    _, _, _, tts = run_metanet_sim(
        time_step, L, day["init_state"], day["data_inflow"],
        day["ds_density_norm"], params, lanes=day["lane_dict"],
        vsl_speeds=vsl, plotting=True, real_data=False,
    )
    return float(np.clip((delay_base - (tts - ff_ttt)) / delay_base * 100, 0, 100))


# ── Sweep loaders ────────────────────────────────────────────────────────────

def load_scalar_sweep(subdir, ctx, max_value=None):
    """Sweeps keyed by a single integer, i.e. hold_length and speed_lb.

    `max_value` caps the swept parameter in that sweep's own units (time steps
    for hold_length, km/hr for speed_lb). Runs above the cap are skipped
    without being simulated, and the cap becomes the right-hand x-axis limit
    of the corresponding panel.

    Returns (values, ccs, n_failed, n_capped) sorted by value, where both
    counts refer to the plotted range.
    """
    path = f"{SWEEP_ROOT}/{subdir}"
    out, failed, capped = [], 0, 0
    for fname in os.listdir(path):
        m = re.fullmatch(r"optimal_vsl_(\d+)\.npy", fname)
        if not m:
            continue
        value = int(m.group(1))
        if max_value is not None and value > max_value:
            capped += 1
            continue
        cc = cc_for(os.path.join(path, fname), *ctx)
        if cc is None:
            failed += 1
            continue
        out.append((value, cc))
    out.sort()
    values = np.array([v for v, _ in out], dtype=float)
    ccs = np.array([c for _, c in out])
    return values, ccs, failed, capped


def load_safety_grid(ctx):
    """Full (temporal, spatial) -> CC grid, excluding failed runs."""
    path = f"{SWEEP_ROOT}/safety_sweep"
    grid, failed = {}, 0
    for fname in os.listdir(path):
        m = re.fullmatch(r"optimal_vsl_temp([\d.]+)_spat([\d.]+)\.npy", fname)
        if not m:
            continue
        cc = cc_for(os.path.join(path, fname), *ctx)
        if cc is None:
            failed += 1
            continue
        grid[(float(m.group(1)), float(m.group(2)))] = cc
    return grid, failed


def slice_grid(grid, axis, fixed_value, max_value=None):
    """One-dimensional slice through the safety grid.

    axis="temporal" varies the temporal bound at a fixed spatial bound;
    axis="spatial" does the reverse. If `fixed_value` is None, picks whichever
    fixed value yields the most points.

    `max_value` caps the swept bound: only points with bound <= max_value are
    returned, which in turn sets the right-hand x-axis limit of the panel. The
    cap is applied before the automatic choice of `fixed_value`, so that choice
    reflects coverage over the plotted range rather than the whole grid.
    """
    keep = 1 if axis == "temporal" else 0     # index of the held-fixed bound
    vary = 1 - keep

    in_range = {key: cc for key, cc in grid.items()
                if max_value is None or key[vary] <= max_value + 1e-9}

    if fixed_value is None:
        counts = {}
        for key in in_range:
            counts[key[keep]] = counts.get(key[keep], 0) + 1
        fixed_value = max(counts, key=counts.get)

    pts = sorted((key[vary], cc) for key, cc in in_range.items()
                 if np.isclose(key[keep], fixed_value))
    n_available = sum(1 for key in grid if np.isclose(key[keep], fixed_value))
    print([v for v, _ in pts])
    print([c for _, c in pts])
    return (np.array([0] + [v for v, _ in pts]),
            np.array([0] + [c for _, c in pts]),
            fixed_value,
            n_available - len(pts))


def report_coverage(grid):
    """Print how many successful runs each candidate fixed value has, so the
    FIXED_SPATIAL / FIXED_TEMPORAL choices can be checked."""
    for axis, keep, label in (("temporal", 1, "spatial"), ("spatial", 0, "temporal")):
        counts = {}
        for key in grid:
            counts[key[keep]] = counts.get(key[keep], 0) + 1
        summary = "  ".join(f"{v:g}:{n}" for v, n in sorted(counts.items()))
        print(f"  points per fixed {label} bound ({axis} curve): {summary}")


# ── Plotting ─────────────────────────────────────────────────────────────────

def _panel(ax, x, y, xlabel, title, logx=False):
    """One curve with shaded area and a horizontal line at the panel maximum."""
    ax.plot(x, y, color=LINE_COLOR, linewidth=4, marker="o",
            markersize=5, zorder=3)
    ax.fill_between(x, 0, y, color=FILL_COLOR, alpha=0.25, zorder=2)

    y_max = float(np.max(y))
    ax.axhline(y_max, color=MAXLINE_COLOR, linestyle="--", linewidth=3,
               zorder=4)
    # ax.text(0.98, y_max, f" Max CC = {y_max:.1f}%", transform=ax.get_yaxis_transform(),
    #         ha="right", va="bottom", color=MAXLINE_COLOR,
    #         fontsize=TEXT_FONTSIZE - 6, fontname="Times New Roman",
    #         fontweight="bold")

    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=TEXT_FONTSIZE - 2, fontname="Times New Roman")
    ax.set_ylabel("Controllable congestion (%)", fontsize=TEXT_FONTSIZE - 2,
                  fontname="Times New Roman")
    ax.set_title(title, fontsize=TEXT_FONTSIZE - 2, fontname="Times New Roman")
    ax.set_ylim(0, max(60, y_max * 1.2))
    ax.set_xlim(0, float(np.max(x)))
    ax.grid(which="both", linestyle="-", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=TEXT_FONTSIZE - 6)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontname("Times New Roman")


def plot(hold, speed, temporal, spatial, save_path=SAVE_PATH):
    original_font = mpl.rcParams["font.family"]
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    _panel(axes[0][0], hold[0] * 10 / 60, hold[1],
           "Hold length (min)", "(a) Update interval")
    _panel(axes[0][1], speed[0], speed[1],
           "Minimum speed limit (km/hr)", "(b) Minimum posted speed limit")
    _panel(axes[1][0], temporal[0], temporal[1],
           r"Temporal bound $\mathcal{S}_{\mathrm{temp}}$ (km/hr per step)",
           rf"(c) Temporal smoothness ($\mathcal{{S}}_{{\mathrm{{spat}}}}$ = {temporal[2]:g} km/hr)")
    _panel(axes[1][1], spatial[0], spatial[1],
           r"Spatial bound $\mathcal{S}_{\mathrm{spat}}$ (km/hr)",
           rf"(d) Spatial smoothness ($\mathcal{{S}}_{{\mathrm{{temp}}}}$ = {spatial[2]:g} km/hr)")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"\nFigure saved to: {save_path}")
    mpl.rcParams["font.family"] = original_font
    return fig


def plot_heatmap(grid, save_path=HEATMAP_SAVE_PATH, annotate=True, min_points=3):
    """Controllable congestion over the full (temporal, spatial) safety grid.

    Combinations whose run failed are masked and drawn in grey, so that a
    missing result is visually distinct from a genuinely low one.

    `min_points` drops any row or column with fewer than that many successful
    runs, which removes bounds that were only partially swept and would
    otherwise appear as near-empty stripes. Set to 0 to keep everything.
    """
    import copy

    original_font = mpl.rcParams["font.family"]
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman"]

    temps = sorted({t for t, _ in grid})
    spats = sorted({s for _, s in grid})
    if min_points:
        temps = [t for t in temps
                 if sum(1 for k in grid if k[0] == t) >= min_points]
        spats = [s for s in spats
                 if sum(1 for k in grid if k[1] == s) >= min_points]
        grid = {k: v for k, v in grid.items() if k[0] in temps and k[1] in spats}

    values = np.full((len(temps), len(spats)), np.nan)
    for (t, s), cc in grid.items():
        values[temps.index(t), spats.index(s)] = cc
    masked = np.ma.masked_invalid(values)

    cmap = copy.copy(plt.cm.viridis)
    cmap.set_bad(color="lightgray")

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(masked, cmap=cmap, aspect="auto", origin="lower",
                   vmin=0, vmax=float(masked.max()))

    ax.set_xticks(range(len(spats)))
    ax.set_xticklabels([f"{s:g}" for s in spats])
    ax.set_yticks(range(len(temps)))
    ax.set_yticklabels([f"{t:g}" for t in temps])
    ax.set_xlabel(r"Spatial bound $\mathcal{S}_{\mathrm{spat}}$ (km/hr)",
                  fontsize=TEXT_FONTSIZE - 2, fontname="Times New Roman")
    ax.set_ylabel(r"Temporal bound $\mathcal{S}_{\mathrm{temp}}$ (km/hr per step)",
                  fontsize=TEXT_FONTSIZE - 2, fontname="Times New Roman")
    ax.tick_params(labelsize=TEXT_FONTSIZE - 2 )
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontname("Times New Roman")

    if annotate:
        threshold = 0.55 * float(masked.max())
        for i in range(len(temps)):
            for j in range(len(spats)):
                if masked.mask[i, j]:
                    continue
                ax.text(j, i, f"{values[i, j]:.0f}", ha="center", va="center",
                        fontsize=TEXT_FONTSIZE - 4, fontname="Times New Roman",
                        color="white" if values[i, j] < threshold else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Controllable congestion (%)", fontsize=TEXT_FONTSIZE - 2,
                   fontname="Times New Roman")
    cbar.ax.tick_params(labelsize=TEXT_FONTSIZE - 2)
    for lab in cbar.ax.get_yticklabels():
        lab.set_fontname("Times New Roman")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"Heatmap saved to: {save_path}")
    mpl.rcParams["font.family"] = original_font
    return fig


if __name__ == "__main__":
    day, params, delay_base, ff_ttt = load_baseline()
    ctx = (day, params, delay_base, ff_ttt)
    print(f"Baseline {DATE}: uncontrolled delay {delay_base:.2f} veh-hrs\n")

    hold_x, hold_y, hold_failed, _ = load_scalar_sweep("hold_length", ctx)
    print(f"hold_length: {len(hold_x)} runs used, {hold_failed} excluded (solver fallback)")

    speed_x, speed_y, speed_failed, speed_capped = load_scalar_sweep(
        "speed_lb", ctx, MAX_SPEED_LB)
    print(f"speed_lb:    {len(speed_x)} runs used, {speed_failed} excluded (solver fallback)"
          + ("" if MAX_SPEED_LB is None else
             f", {speed_capped} beyond cap U_min <= {MAX_SPEED_LB:g}"))

    grid, grid_failed = load_safety_grid(ctx)
    print(f"safety_sweep: {len(grid)} runs used, {grid_failed} excluded (solver fallback)")
    report_coverage(grid)

    temp_x, temp_y, temp_fixed, temp_capped = slice_grid(
        grid, "temporal", FIXED_SPATIAL, MAX_TEMP)
    spat_x, spat_y, spat_fixed, spat_capped = slice_grid(
        grid, "spatial", FIXED_TEMPORAL, MAX_SPAT)

    print(f"\n(c) temporal curve at S_spat = {temp_fixed:g}: {len(temp_x)} points"
          + ("" if MAX_TEMP is None else
             f", capped at S_temp <= {MAX_TEMP:g} ({temp_capped} excluded)"))
    print(f"(d) spatial  curve at S_temp = {spat_fixed:g}: {len(spat_x)} points"
          + ("" if MAX_SPAT is None else
             f", capped at S_spat <= {MAX_SPAT:g} ({spat_capped} excluded)"))

    plot((hold_x, hold_y), (speed_x, speed_y),
         (temp_x, temp_y, temp_fixed), (spat_x, spat_y, spat_fixed))
    plot_heatmap(grid)
