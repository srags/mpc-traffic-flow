"""Standalone data-loading + plotting for the safety_temporal / safety_spatial
sweep, runnable independently of safety_sweep.ipynb.

    python safety_sweep.py

or, from another script:

    from safety_sweep import load_data, main
    sweep_results = main()
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_THIS_DIR, "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

from traffic_sim import run_metanet_sim
from param_loader import METANET_Params
from generate_demand_synthetic import get_ff_tts

from safety_sweep_plots import (
    load_sweep_results,
    plot_tts_heatmap,
    plot_tts_lines,
    plot_tts_vs_smoothness,
)


def smooth_inflow(inflow, window_size=2):
    kernel = np.ones(window_size) / window_size
    pad_left = window_size // 2
    pad_right = window_size - pad_left - 1

    if inflow.ndim == 1:
        padded = np.pad(inflow, (pad_left, pad_right), mode="edge")
    else:
        padded = np.pad(inflow, ((pad_left, pad_right), (0, 0)), mode="edge")

    smoothed = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode="valid"), axis=0, arr=padded
    )
    return smoothed


def load_data(
    data_path="/Users/shreyaar/Desktop/PhD/research/MPC/data/i24/i24_11_30",
    calibration_id="calibration_static/fixed_ramping",
    calibration_interval=None,
    L=0.4,
    time_step=10 / 3600,
    start_time=0,
):
    """Load the I-24 data/calibration, build the initial state, and run the
    "before control" baseline simulation — everything the sweep-plotting code
    needs, mirroring the data-loading cells in safety_sweep.ipynb.
    """
    start_time_step = int(start_time / time_step)

    results_path = (
        f"/Users/shreyaar/Desktop/PhD/research/MPC/results/i24/i24_11_30/"
        f"{calibration_id}/safety_sweep"
    )
    cal_path = (
        f"{data_path}/{calibration_id}"
        if calibration_interval is None
        else f"{data_path}/{calibration_id}/control_h_{calibration_interval}"
    )
    if calibration_interval is not None:
        results_path += f"/control_h_{calibration_interval}"

    assert os.path.exists(cal_path), f"Calibration parameters not found at {cal_path}"
    assert os.path.exists(f"{data_path}/v_hat.npy") and os.path.exists(
        f"{data_path}/rho_hat.npy"
    ), f"Data not found at {data_path}"

    density_data = np.load(f"{data_path}/rho_hat.npy")
    flow_data = np.load(f"{data_path}/q_hat.npy")
    density_data = np.where(density_data == 0.0, 1e-3, density_data)
    flow_data = np.where(flow_data == 0.0, 1e-3, flow_data)
    velocity_data = flow_data / density_data

    true_density_initial = density_data[start_time_step, 1:-1].reshape(-1)
    true_velocity_initial = velocity_data[start_time_step, 1:-1].reshape(-1)

    downstream_density = density_data[0:, -1].reshape(-1)
    data_inflow = (velocity_data[0:, 0] * density_data[0:, 0]).reshape(-1)
    data_inflow = smooth_inflow(data_inflow, window_size=2)
    downstream_density = smooth_inflow(downstream_density, window_size=2)

    time_steps = downstream_density.shape[0]
    num_segments = density_data.shape[1] - 2

    lane_counts = (
        np.load(f"{cal_path}/num_lanes.npy").reshape(-1)
        if calibration_interval is None
        else np.load(f"{cal_path}/params_1/num_lanes.npy").reshape(-1)
    )
    lane_dict = {i: lane_counts[i] for i in range(num_segments)}

    velocity_data = velocity_data[:, 1:-1]  # remove boundary conditions
    density_data = density_data[:, 1:-1]

    downstream_density = downstream_density / lane_dict[num_segments - 1]
    true_density_initial = true_density_initial / lane_counts

    subfolders = [
        f for f in os.listdir(cal_path)
        if os.path.isdir(os.path.join(cal_path, f)) and f.startswith("params_")
    ]
    if subfolders:
        control_h = time_steps // len(subfolders)
        model_params = METANET_Params(
            path=f"{data_path}/{calibration_id}", control_h=control_h,
            num_timesteps=time_steps, num_segments=num_segments,
        ).get_params()
    else:
        model_params = METANET_Params(
            path=cal_path, num_timesteps=time_steps, num_segments=num_segments,
        ).get_params()

    init_state = (
        true_density_initial,
        true_velocity_initial,
        data_inflow[start_time_step],
        0,
    )

    vsl_baseline = np.ones((downstream_density.shape[0], num_segments)) * 150
    _, v_baseline, _, tts_baseline = run_metanet_sim(
        time_step, L, init_state,
        data_inflow[start_time:], downstream_density[start_time:],
        model_params, lanes=lane_dict, vsl_speeds=vsl_baseline,
        plotting=True, real_data=False,
    )

    control_zone = [i for i in range(2, num_segments)]

    # Free-flow travel time — the "uncontrollable" floor of TTS that can never be
    # eliminated by VSL. Controllable congestion (%) is expressed relative to
    # delay = TTS - ff_ttt, NOT relative to raw TTS (see I_24_constraint_plotting.py).
    ff_ttt = get_ff_tts(data_inflow, time_step, L, model_params["v_free"])

    return {
        "results_path": results_path,
        "time_step": time_step,
        "L": L,
        "start_time": start_time,
        "init_state": init_state,
        "data_inflow": data_inflow,
        "downstream_density": downstream_density,
        "model_params": model_params,
        "lane_dict": lane_dict,
        "control_zone": control_zone,
        "velocity_data": velocity_data,
        "num_segments": num_segments,
        "tts_baseline": tts_baseline,
        "v_baseline": v_baseline,
        "ff_ttt": ff_ttt,
    }


# Keep these in sync with the grid actually being swept in safety_sweep.ipynb —
# they're only used to know which (safety_temporal, safety_spatial) combinations
# to look for, so a mismatch just means some real results get missed, not an error.
DEFAULT_SAFETY_TEMPORAL_VALUES = (0.5, 1, 2, 3, 4, 5, 7.5, 10, 15, 20, 25)
DEFAULT_SAFETY_SPATIAL_VALUES = (0.5, 1, 2, 3, 4, 5, 7.5, 10, 15, 20, 25)

FIGS_DIR = "/Users/shreyaar/Desktop/PhD/research/MPC/figs"


def main(
    safety_temporal_values=DEFAULT_SAFETY_TEMPORAL_VALUES,
    safety_spatial_values=DEFAULT_SAFETY_SPATIAL_VALUES,
    show=True,
    figs_dir=FIGS_DIR,
    save_all=False,
):
    """Load data, load whichever sweep results exist on disk, and produce the
    three summary plots (heatmap, line plot, smoothness trade-off). Safe to call
    at any point while the sweep notebook is still running — combinations that
    haven't finished yet (or failed) just show up as gaps.

    The heatmap is saved to `figs_dir` as i24_safety_sweep_heatmap.png. Pass
    `save_all=True` to also save the line and smoothness plots, or
    `figs_dir=None` to skip saving entirely.
    """
    data = load_data()

    sweep_results = load_sweep_results(
        data["results_path"], safety_temporal_values, safety_spatial_values,
        data["time_step"], data["L"], data["init_state"],
        data["data_inflow"], data["downstream_density"],
        data["model_params"], data["lane_dict"],
        control_zone=data["control_zone"],
    )

    def _path(name):
        return None if figs_dir is None else os.path.join(figs_dir, name)

    plot_tts_heatmap(
        sweep_results, safety_temporal_values, safety_spatial_values,
        tts_baseline=data["tts_baseline"], ff_ttt=data["ff_ttt"],
        save_path=_path("i24_safety_sweep_heatmap.png"),
    )
    plot_tts_lines(
        sweep_results, safety_temporal_values, safety_spatial_values,
        tts_baseline=data["tts_baseline"], ff_ttt=data["ff_ttt"], vary="safety_spatial",
        save_path=_path("i24_safety_sweep_lines.png") if save_all else None,
    )
    plot_tts_vs_smoothness(
        sweep_results, safety_temporal_values, safety_spatial_values,
        tts_baseline=data["tts_baseline"], ff_ttt=data["ff_ttt"], metric="combined",
        save_path=_path("i24_safety_sweep_smoothness.png") if save_all else None,
    )

    if show:
        plt.show()

    return sweep_results


if __name__ == "__main__":
    main()
