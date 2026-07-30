"""Plotting utilities for the safety_temporal / safety_spatial sweep results
produced by safety_sweep.ipynb.

These work directly off the saved `optimal_vsl_temp{T}_spat{S}.npy` files in
`results_path` — combinations that haven't been run yet (or that failed) are
simply skipped / shown as gaps, so these functions can be called at any point
while the sweep notebook is still running.
"""

import copy
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_THIS_DIR, "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

from traffic_sim import run_metanet_sim


def _vsl_path(results_path, safety_temporal, safety_spatial):
    return os.path.join(
        results_path, f"optimal_vsl_temp{safety_temporal}_spat{safety_spatial}.npy"
    )


def load_sweep_results(
    results_path, safety_temporal_values, safety_spatial_values,
    time_step, L, init_state, demand, downstream_density,
    model_params, lane_dict, control_zone=None, start_time=0,
):
    """Load whichever (safety_temporal, safety_spatial) VSL results exist on disk
    and compute TTS + VSL "roughness" (temporal/spatial) for each.

    Combinations with no saved file (not yet run, or failed) are simply omitted
    from the returned dict — every plotting function below treats a missing key
    as a gap rather than an error.

    Returns
    -------
    dict mapping (safety_temporal, safety_spatial) -> {
        "vsl": np.ndarray,
        "tts": float,
        "temporal_roughness": float,  # mean |vsl[t+1,m]-vsl[t,m]|, controlled segs only
        "spatial_roughness": float,   # mean |vsl[t,m]-vsl[t,m-1]|, controlled segs only
    }
    """
    results = {}
    for st in safety_temporal_values:
        for ss in safety_spatial_values:
            path = _vsl_path(results_path, st, ss)
            if not os.path.exists(path):
                continue
            try:
                vsl = np.load(path)
            except Exception as exc:
                print(f"Warning: failed to load {path}: {exc}")
                continue

            _, _, _, tts = run_metanet_sim(
                time_step, L, init_state,
                demand[start_time:], downstream_density[start_time:],
                model_params, lanes=lane_dict, vsl_speeds=vsl,
                plotting=True, real_data=False,
            )

            vsl_for_roughness = vsl[:, control_zone] if control_zone is not None else vsl
            temporal_roughness = np.mean(np.abs(np.diff(vsl_for_roughness, axis=0)))
            spatial_roughness = np.mean(np.abs(np.diff(vsl_for_roughness, axis=1)))

            results[(st, ss)] = {
                "vsl": vsl,
                "tts": tts,
                "temporal_roughness": temporal_roughness,
                "spatial_roughness": spatial_roughness,
                # mpc_find_vsl saves a flat VSL=150 when it abandons a run, so a
                # perfectly uniform 150 trajectory means "no optimizer result",
                # not "the optimizer found ~zero benefit".
                "abandoned": bool(np.all(vsl == 150.0)),
            }
    return results


def _cc_value(entry, tts_baseline, ff_ttt):
    """Controllable congestion (%): the fraction of the *baseline delay*
    (TTS above free-flow travel time) eliminated by control — matches the
    `cc = (delay_baseline - opt_delay) / delay_baseline * 100` calculation in
    I_24_constraint_plotting.py, NOT a raw TTS-reduction percentage (delay is
    always <= TTS, so dividing by TTS instead of delay understates the effect).
    Falls back to raw TTS if tts_baseline/ff_ttt aren't both given.
    """
    if tts_baseline is not None and ff_ttt is not None:
        delay_baseline = tts_baseline - ff_ttt
        opt_delay = entry["tts"] - ff_ttt
        return 100.0 * (delay_baseline - opt_delay) / delay_baseline
    return entry["tts"]


def _cc_label(tts_baseline, ff_ttt):
    if tts_baseline is not None and ff_ttt is not None:
        return "Controllable congestion (%)"
    return "TTS (veh-hr)"


def _save_fig(ax, save_path, dpi=300):
    """Save the figure `ax` belongs to (not just whatever the current figure is)."""
    if save_path is None:
        return
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    ax.get_figure().savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved figure to {save_path}")


def plot_tts_heatmap(
    results, safety_temporal_values, safety_spatial_values,
    tts_baseline=None, ff_ttt=None, ax=None, save_path=None,
    mark_abandoned=True,
):
    """Option 1: 2D heatmap of controllable congestion (%) — or raw TTS if
    tts_baseline/ff_ttt aren't given — across the safety_temporal x
    safety_spatial grid. Missing combinations are left blank.

    Runs that `mpc_find_vsl` abandoned (VSL saved as a flat 150 everywhere, i.e.
    the do-nothing fallback rather than a real optimizer result) are shown
    hatched with an 'x' label when `mark_abandoned` is True, so they aren't
    misread as a genuine ~0% controllable-congestion result.
    """
    grid = np.full((len(safety_temporal_values), len(safety_spatial_values)), np.nan)
    abandoned = np.zeros_like(grid, dtype=bool)
    for i, st in enumerate(safety_temporal_values):
        for j, ss in enumerate(safety_spatial_values):
            entry = results.get((st, ss))
            if entry is None:
                continue
            grid[i, j] = _cc_value(entry, tts_baseline, ff_ttt)
            abandoned[i, j] = bool(entry.get("abandoned", False))

    if mark_abandoned:
        grid = np.where(abandoned, np.nan, grid)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    cmap = copy.copy(plt.cm.viridis)
    cmap.set_bad(color="lightgray")
    masked = np.ma.masked_invalid(grid)

    im = ax.imshow(masked, cmap=cmap, aspect="auto", origin="lower")
    ax.set_xticks(range(len(safety_spatial_values)))
    ax.set_xticklabels(safety_spatial_values)
    ax.set_yticks(range(len(safety_temporal_values)))
    ax.set_yticklabels(safety_temporal_values)
    ax.set_xlabel("safety_spatial")
    ax.set_ylabel("safety_temporal")

    label = _cc_label(tts_baseline, ff_ttt)
    ax.set_title(label)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(label)

    for i in range(len(safety_temporal_values)):
        for j in range(len(safety_spatial_values)):
            if not np.isnan(grid[i, j]):
                ax.text(
                    j, i, f"{grid[i, j]:.1f}",
                    ha="center", va="center", color="white", fontsize=9,
                )
            elif mark_abandoned and abandoned[i, j]:
                ax.text(
                    j, i, "x", ha="center", va="center",
                    color="dimgray", fontsize=11, fontweight="bold",
                )

    n_ab = int(abandoned.sum())
    if mark_abandoned and n_ab:
        print(f"Note: {n_ab} combination(s) were abandoned runs (flat VSL=150 fallback), "
              f"shown as 'x' rather than a controllable-congestion value.")

    plt.tight_layout()
    _save_fig(ax, save_path)
    return ax


def plot_tts_lines(
    results, safety_temporal_values, safety_spatial_values,
    tts_baseline=None, ff_ttt=None, vary="safety_temporal", ax=None,
    save_path=None, skip_abandoned=True,
):
    """Option 2: line plot of controllable congestion (%) — or raw TTS if
    tts_baseline/ff_ttt aren't given — vs. one swept parameter, with one line
    per value of the other parameter. Missing points become gaps (NaN) in the
    line rather than raising an error.
    """
    if vary == "safety_temporal":
        x_values, series_values = safety_temporal_values, safety_spatial_values
        get_entry = lambda x, s: results.get((x, s))
        xlabel, series_label = "safety_temporal", "safety_spatial"
    elif vary == "safety_spatial":
        x_values, series_values = safety_spatial_values, safety_temporal_values
        get_entry = lambda x, s: results.get((s, x))
        xlabel, series_label = "safety_spatial", "safety_temporal"
    else:
        raise ValueError("vary must be 'safety_temporal' or 'safety_spatial'")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    for s in series_values:
        y = []
        for x in x_values:
            entry = get_entry(x, s)
            if entry is None or (skip_abandoned and entry.get("abandoned")):
                y.append(np.nan)
            else:
                y.append(_cc_value(entry, tts_baseline, ff_ttt))
        ax.plot(x_values, y, marker="o", label=f"{series_label}={s}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(_cc_label(tts_baseline, ff_ttt))
    ax.legend(title=series_label)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_fig(ax, save_path)
    return ax


def plot_tts_vs_smoothness(
    results, safety_temporal_values, safety_spatial_values,
    tts_baseline=None, ff_ttt=None, metric="combined", annotate=True, ax=None,
    save_path=None, skip_abandoned=True,
):
    """Option 3: scatter of controllable congestion (%) — or raw TTS if
    tts_baseline/ff_ttt aren't given — vs. a VSL "roughness" metric derived
    from the saved trajectories themselves (mean |delta VSL| across time
    and/or segments), regardless of which safety_temporal/safety_spatial
    combination produced it — visualizes the smoothness/performance trade-off
    directly rather than per-parameter.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    xs, ys, labels = [], [], []
    for st in safety_temporal_values:
        for ss in safety_spatial_values:
            entry = results.get((st, ss))
            if entry is None or (skip_abandoned and entry.get("abandoned")):
                continue
            if metric == "temporal":
                x = entry["temporal_roughness"]
            elif metric == "spatial":
                x = entry["spatial_roughness"]
            elif metric == "combined":
                x = entry["temporal_roughness"] + entry["spatial_roughness"]
            else:
                raise ValueError("metric must be 'temporal', 'spatial', or 'combined'")

            xs.append(x)
            ys.append(_cc_value(entry, tts_baseline, ff_ttt))
            labels.append(f"({st},{ss})")

    ax.scatter(xs, ys)
    if annotate:
        for x, y, label in zip(xs, ys, labels):
            ax.annotate(label, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel(f"VSL roughness ({metric})")
    ax.set_ylabel(_cc_label(tts_baseline, ff_ttt))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_fig(ax, save_path)
    return ax
