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

    python experiments/I_24_constraint_plotting.py
"""

import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from traffic_flow.results.console import colored
from traffic_flow.pipeline import init_state
from traffic_flow.results.plots import Plotter

from traffic_flow.inputs.scenario import load_params, load_scenario
from traffic_flow.config import CalSource, CalRef
from traffic_flow.paths import REPO_DIR, cut_repo
from traffic_flow.results.i24 import L, time_step, get_ff_tts
from traffic_flow.model.simulation import METANET_Simulator  # noqa: E402

DATE = "11_30"
SWEEP_ROOT = REPO_DIR / "results_bu" / "i24" / f"i24_{DATE}" / CalSource.FIXED_RAMPS
SAVE_PATH = REPO_DIR / "figs" / "i24_constraints.png"
HEATMAP_SAVE_PATH = REPO_DIR / "figs" / "i24_safety_heatmap.png"

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

LINE_COLOR = "#2b6b4f"
FILL_COLOR = "#4e9858"
MAXLINE_COLOR = "#cf2e1c"

# ── Baseline ─────────────────────────────────────────────────────────────────

def load_baseline(date=DATE):
    """Uncontrolled delay for `date`, against which every sweep run is scored."""
    scenario = load_scenario("i24", date)
    traffic = scenario.traffic
    params = load_params(scenario, CalRef(CalSource.FIXED_RAMPS, interval=None))
    sim = METANET_Simulator(T=time_step, l=L, params=params, lanes=dict(enumerate(traffic.lanes)), real_data=True)
    _, tts_base = sim.run(traffic.inflow, traffic.downstream_density, init_state(traffic))
    v_free = params["v_free"]
    ff_ttt = get_ff_tts(traffic.inflow, time_step, L, np.max(v_free, axis=0) if v_free.ndim == 2 else v_free)
    return scenario, params, tts_base - ff_ttt, ff_ttt

def cc_for(path, scenario, params, delay_base, ff_ttt):
    """Controllable congestion for one saved VSL run, or None if it failed."""
    traffic = scenario.traffic
    vsl = np.load(path)
    if (vsl == 150.0).all(): return None
    sim = METANET_Simulator(T=time_step, l=L, params=params, lanes=dict(enumerate(traffic.lanes)), real_data=False)
    _, tts = sim.run(traffic.inflow, traffic.downstream_density, init_state(traffic), vsl_speeds = vsl)
    return float(np.clip((delay_base - (tts - ff_ttt)) / delay_base * 100, 0, 100))

# ── Sweep loaders ────────────────────────────────────────────────────────────

def load_scalar_sweep(subdir: str, ctx, max_value=None):
    """Sweeps keyed by a single integer, i.e. hold_length and speed_lb.

    `max_value` caps the swept parameter in that sweep's own units (time steps
    for hold_length, km/hr for speed_lb). Runs above the cap are skipped
    without being simulated, and the cap becomes the right-hand x-axis limit
    of the corresponding panel.

    Returns (values, ccs, n_failed, n_capped) sorted by value, where both
    counts refer to the plotted range.
    """
    path = SWEEP_ROOT / subdir
    out, failed, capped = [], 0, 0
    for fname in os.listdir(path):
        m = re.fullmatch(r"optimal_vsl_(\d+)\.npy", fname)
        if not m: continue
        value = int(m.group(1))
        if max_value is not None and value > max_value:
            capped += 1
            continue
        cc = cc_for(path / fname, *ctx)
        if cc is None:
            failed += 1
            continue
        out.append((value, cc))
    out.sort()
    values = np.array([v for v, _ in out], dtype=float)
    ccs = np.array([c for _, c in out])
    return values, ccs, failed, capped

def load_safety_grid(ctx) -> tuple[dict[tuple[float, float], float], int]:
    """Full (temporal, spatial) -> CC grid, excluding failed runs."""
    path = SWEEP_ROOT / "safety_sweep"
    grid, failed = {}, 0
    for fname in os.listdir(path):
        m = re.fullmatch(r"optimal_vsl_temp([\d.]+)_spat([\d.]+)\.npy", fname)
        if not m: continue
        cc = cc_for(path / fname, *ctx)
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
        fixed_value = max(counts, key=counts.__getitem__)

    pts = sorted((key[vary], cc) for key, cc in in_range.items()
                 if np.isclose(key[keep], fixed_value))
    n_available = sum(1 for key in grid if np.isclose(key[keep], fixed_value))
    return (np.array([0] + [v for v, _ in pts]),
            np.array([0] + [c for _, c in pts]),
            fixed_value, n_available - len(pts))


def report_coverage(grid):
    """Print how many successful runs each candidate fixed value has, so the
    FIXED_SPATIAL / FIXED_TEMPORAL choices can be checked."""
    for axis, keep, label in (("temporal", 1, "spatial"), ("spatial", 0, "temporal")):
        counts = {}
        for key in grid: counts[key[keep]] = counts.get(key[keep], 0) + 1
        summary = "  ".join(f"{v:g}:{n}" for v, n in sorted(counts.items()))
        print(f"  points per fixed {label} bound ({axis} curve): {summary}")


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot(hold, speed, temporal, spatial, save_path=SAVE_PATH):
    p = Plotter(2, 2, figsize=(13, 9))
    for i, param in enumerate([hold, speed, temporal, spatial]):
        p[i].plot(*param, color=LINE_COLOR, linewidth=4, marker="o", markersize=5, zorder=3)
        p[i].fill_between(param[0], 0, param[1], color=FILL_COLOR, alpha=0.25, zorder=2)
        p[i].axhline(param[1].max(), color=MAXLINE_COLOR, linestyle="--", linewidth=3, zorder=4)
        p[i] = {'xlim': (0, float(np.max(param[0]))), 'ylim': (0, max(60, param[1].max() * 1.2))}
        p[i].grid(which="both", linestyle="-", linewidth=0.5, alpha=0.6)
        p[i].set_axisbelow(True)
    for i, (xtitle, title) in enumerate([
        ("Hold length (min)", "(a) Update interval"), 
        ("Minimum speed limit (km/hr)", "(b) Minimum posted speed limit"), 
        (r"Temporal bound $\mathcal{S}_{\mathrm{temp}}$ (km/hr per step)",
            f"(c) Temporal smoothness ($\\mathcal{{S}}_{{\\mathrm{{spat}}}}$ = {temporal[2]:g} km/hr)"),
        (r"Spatial bound $\mathcal{S}_{\mathrm{spat}}$ (km/hr)",
            f"(d) Spatial smoothness ($\\mathcal{{S}}_{{\\mathrm{{temp}}}}$ = {spatial[2]:g} km/hr)")
    ]): p[i] = {'xlabel': xtitle, 'ylabel': "Controllable congestion (%)", 'title': title}
    p.fig.tight_layout()
    p.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"\nFigure saved to: {colored(cut_repo(save_path), 'green', 'bold')}")


def plot_heatmap(grid, save_path=HEATMAP_SAVE_PATH, annotate=True, min_points=3):
    """Controllable congestion over the full (temporal, spatial) safety grid.

    Combinations whose run failed are masked and drawn in grey, so that a
    missing result is visually distinct from a genuinely low one.

    `min_points` drops any row or column with fewer than that many successful
    runs, which removes bounds that were only partially swept and would
    otherwise appear as near-empty stripes. Set to 0 to keep everything.
    """
    import copy

    temps = sorted({t for t, _ in grid})
    spats = sorted({s for _, s in grid})
    if min_points:
        temps = [t for t in temps if sum(1 for k in grid if k[0] == t) >= min_points]
        spats = [s for s in spats if sum(1 for k in grid if k[1] == s) >= min_points]
        grid = {k: v for k, v in grid.items() if k[0] in temps and k[1] in spats}

    values = np.full((len(temps), len(spats)), np.nan)
    for (t, s), cc in grid.items(): values[temps.index(t), spats.index(s)] = cc
    masked = np.ma.masked_invalid(values)

    cmap = copy.copy(plt.get_cmap("viridis"))
    cmap.set_bad(color="lightgray")

    p = Plotter(1, 1)

    im = p[0].imshow(masked, cmap=cmap, aspect="auto", origin="lower",
                   vmin=0, vmax=float(masked.max()))
    p[0] = {'xticks': range(len(spats)), 'yticks': range(len(temps)),
            'xticklabels': [f"{s:g}" for s in spats],
            'yticklabels': [f"{t:g}" for t in temps],
            'xlabel': r"Spatial bound $\mathcal{S}_{\mathrm{spat}}$ (km/hr)",
            'ylabel': r"Temporal bound $\mathcal{S}_{\mathrm{temp}}$ (km/hr per step)"}

    if annotate:
        threshold = 0.55 * float(masked.max())
        for i in range(len(temps)):
            for j in range(len(spats)):
                if masked.mask[i, j]:
                    continue
                p[0].text(j, i, f"{values[i, j]:.0f}", ha="center", va="center",
                        color="white" if values[i, j] < threshold else "black")
    p.fig.colorbar(im, ax=p[0], fraction=0.046, pad=0.02, label="Controllable congestion (%)")
    p.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"Heatmap saved to: {colored(cut_repo(save_path), 'bold', 'green')}")
    return p.fig


if __name__ == "__main__":
    scenario, params, delay_base, ff_ttt = ctx = load_baseline()
    print(f"Baseline {DATE}: uncontrolled delay {delay_base:.2f} veh-hrs")

    hold_x, hold_y, hold_failed, _ = load_scalar_sweep("hold_length", ctx)
    speed_x, speed_y, speed_failed, speed_capped = load_scalar_sweep("speed_lb", ctx, MAX_SPEED_LB)
    grid, grid_failed = load_safety_grid(ctx)

    report_coverage(grid)

    temp_x, temp_y, temp_fixed, temp_capped = slice_grid(grid, "temporal", FIXED_SPATIAL, MAX_TEMP)
    spat_x, spat_y, spat_fixed, spat_capped = slice_grid(grid, "spatial", FIXED_TEMPORAL, MAX_SPAT)

    print(tabulate(headers=["Constraint", "Runs Used", "Excluded", f"Capped at"],
            tablefmt="grid", tabular_data=[
                ["hold_length", len(hold_x), hold_failed, None],
                ["speed_lb", len(speed_x), speed_failed, f"(U_min <= {MAX_SPEED_LB:g}) {speed_capped}"],
                ["safety_sweep", len(grid), grid_failed, None],
                [f"temporal (S_spat = {temp_fixed})", len(temp_x), None, f"(S_temp <= {MAX_TEMP:g}) {temp_capped}"],
                [f"spatial (S_temp = {spat_fixed})", len(spat_x), None, f"(S_spat <= {MAX_SPAT:g}) {spat_capped}"]
            ]))

    plot((hold_x * 10 / 60, hold_y), (speed_x, speed_y),
         (temp_x, temp_y, temp_fixed), (spat_x, spat_y, spat_fixed))
    plot_heatmap(grid)
