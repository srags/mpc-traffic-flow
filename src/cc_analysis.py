import numpy as np
import warnings
import time
import pandas as pd
import random
import numpy.ma as ma
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.param_loader import METANET_Params
from src.traffic_sim import run_metanet_sim

# ── Simulation parameters ────────────────────────────────────────────────────
L          = 0.4
time_step  = 10 / 3600
total_time = 1   # hours

DATES = ["11_30"]
# DATES = [
#     "11_21", "11_22", "11_23", "11_24", "11_25",
#     "11_28", "11_29", "11_30", "12_01", "12_02"
# ]

def mape(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def get_num_veh(demand_profile, time_step):
    return sum(demand_profile) * time_step

def get_ff_tts(demand_profile, time_step, length, v_free):
    num_vehicles   = sum(demand_profile) * time_step
    ff_travel_time = sum(length / v for v in v_free)
    return num_vehicles * ff_travel_time

def smooth_inflow(inflow, window_size=2):
    kernel    = np.ones(window_size) / window_size
    pad_left  = window_size // 2
    pad_right = window_size - pad_left - 1
    if inflow.ndim == 1:
        padded = np.pad(inflow, (pad_left, pad_right), mode='edge')
    else:
        padded = np.pad(inflow, ((pad_left, pad_right), (0, 0)), mode='edge')
    return np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode="valid"), axis=0, arr=padded
    )

def plot_results_table(results, save_path=None):
    import matplotlib as mpl

    original_font = mpl.rcParams['font.family']
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']

    dates            = [r['date'].replace('_', '/') for r in results]
    static_sim_mapes = [f"{r['static_sim_mape']:.2f}%" for r in results]
    static_tts_mapes = [f"{r['static_tts_mape']:.2f}%" for r in results]
    static_ccs       = [f"{r['static_cc']:.2f}%" for r in results]
    dyn_sim_mapes    = [f"{r['dyn_sim_mape']:.2f}%" for r in results]
    dyn_tts_mapes    = [f"{r['dyn_tts_mape']:.2f}%" for r in results]
    dyn_ccs          = [f"{r['dyn_cc']:.2f}%" for r in results]

    columns = [
        'Date',
        'Static Sim MAPE', 'Static TTS MAPE', 'Static CC (%)',
        'Dynamic Sim MAPE', 'Dynamic TTS MAPE', 'Dynamic CC (%)',
    ]
    rows = list(zip(dates, static_sim_mapes, static_tts_mapes, static_ccs,
                    dyn_sim_mapes, dyn_tts_mapes, dyn_ccs))
    n_rows = len(rows)

    row_height  = 0.5
    header_height = 0.6
    fig_height  = header_height + n_rows * row_height + 0.8
    fig, ax = plt.subplots(figsize=(13, fig_height))
    ax.axis('off')

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc='center',
        cellLoc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, row_height / 0.22)

    HEADER_BG = '#1a2744'
    HEADER_FG = 'white'
    for col_idx in range(len(columns)):
        cell = table[0, col_idx]
        cell.set_facecolor(HEADER_BG)
        cell.set_text_props(color=HEADER_FG, fontweight='bold',
                            fontfamily='serif', fontname='Times New Roman')
        cell.set_edgecolor(HEADER_BG)

    ROW_ODD  = '#f5f6fa'
    ROW_EVEN = 'white'
    EDGE     = '#d0d4e0'
    for row_idx in range(1, n_rows + 1):
        bg = ROW_ODD if row_idx % 2 != 0 else ROW_EVEN
        for col_idx in range(len(columns)):
            cell = table[row_idx, col_idx]
            cell.set_facecolor(bg)
            cell.set_edgecolor(EDGE)
            cell.set_text_props(color='#1a1a2e',
                                fontfamily='serif', fontname='Times New Roman')

    col_widths = [0.10, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
    for col_idx, w in enumerate(col_widths):
        for row_idx in range(n_rows + 1):
            table[row_idx, col_idx].set_width(w)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Table saved to: {save_path}")

    plt.show()
    mpl.rcParams['font.family'] = original_font


def run_one_day(init_state, data_inflow, ds_density_norm, model_params,
                lane_dict, lane_counts, v_trimmed, d_trimmed,
                num_segments, vsl_path):
    """
    Run baseline + VSL simulations for one calibration variant.
    Returns a dict of metrics, or None if the VSL file is missing.
    """
    p_sim, v_sim, _, tts_sim = run_metanet_sim(
        time_step, L, init_state, data_inflow, ds_density_norm,
        model_params, lanes=lane_dict, vsl_speeds=None,
        plotting=True, real_data=True
    )
    p_sim = p_sim[:-1, :]
    v_sim = v_sim[:-1, :]

    try:
        vsl = np.load(vsl_path)
    except FileNotFoundError:
        print(f"  VSL file not found: {vsl_path}")
        return None

    p_opt, v_opt, _, tts_opt = run_metanet_sim(
        time_step, L, init_state, data_inflow, ds_density_norm,
        model_params, lanes=lane_dict, vsl_speeds=vsl,
        plotting=True, real_data=False
    )
    v_opt = v_opt[:-1, :]
    p_opt = p_opt[:-1, :]

    v_free = model_params['v_free']
    ff_ttt    = get_ff_tts(data_inflow, time_step, L, 
                           np.max(v_free, axis=0) if len(v_free.shape)==2 else v_free)
    delay     = tts_sim - ff_ttt
    opt_delay = tts_opt - ff_ttt
    cc        = (delay - opt_delay) / delay * 100
    sim_mape_ = mape(v_trimmed, v_sim)

    gt_tt      = time_step * sum(np.sum(d_trimmed[:, i]) * lane_counts[i] * L
                                 for i in range(num_segments))
    metanet_tt = time_step * sum(np.sum(p_sim[:, i])     * lane_counts[i] * L
                                 for i in range(num_segments))
    tts_mape_  = np.abs(gt_tt - metanet_tt) / gt_tt * 100

    return {'sim_mape': sim_mape_, 'tts_mape': tts_mape_, 'cc': cc}


# ── Main loop ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = []

    for date in DATES:
        print(f"\n── {date} ──────────────────────────────────")
        root_path = f"/Users/shreyaar/Desktop/PhD/research/MPC/data/i24/i24_{date}"

        flow_data    = np.load(root_path + "/q_hat.npy")
        density_data = np.load(root_path + "/rho_hat.npy")
        density_data = np.where(density_data == 0.0, 1e-3, density_data)
        flow_data    = np.where(flow_data    == 0.0, 1e-3, flow_data)
        velocity_data = flow_data / density_data

        # ── Shared boundary conditions ────────────────────────────────────
        true_density_initial  = density_data[0, 1:-1].reshape(-1)
        true_velocity_initial = velocity_data[0, 1:-1].reshape(-1)
        downstream_density    = smooth_inflow(density_data[0:, -1].reshape(-1), window_size=2)
        data_inflow           = smooth_inflow(
            (velocity_data[0:, 0] * density_data[0:, 0]).reshape(-1), window_size=2
        )
        num_segments = density_data.shape[1] - 2

        # ── Static calibration ────────────────────────────────────────────
        static_cal_path = (
            f"/Users/shreyaar/Desktop/PhD/research/MPC/data/i24/i24_{date}"
            "/calibration_static/fixed_ramping"
        )
        static_params = METANET_Params(
            path=static_cal_path,
            num_timesteps=flow_data.shape[0],
            num_segments=flow_data.shape[1]
        ).get_params()

        lane_counts = np.load(f'{static_cal_path}/num_lanes.npy').reshape(-1)
        lane_dict   = {i: lane_counts[i] for i in range(num_segments)}

        v_trimmed       = velocity_data[:, 1:-1]
        d_trimmed       = density_data[:, 1:-1] / lane_counts
        ds_density_norm = downstream_density / lane_dict[num_segments - 1]
        d0_norm         = true_density_initial / lane_counts
        init_state      = (d0_norm, true_velocity_initial, data_inflow[0], 0)

        static_vsl_path = (
            f"/Users/shreyaar/Desktop/PhD/research/MPC/results/i24/i24_{date}/calibration_static/fixed_ramping/optimal_vsl.npy"
        )

        print("  Running static calibration simulations...")
        static_metrics = run_one_day(
            init_state, data_inflow, ds_density_norm,
            static_params, lane_dict, lane_counts,
            v_trimmed, d_trimmed, num_segments, static_vsl_path
        )
        if static_metrics is None:
            print(f"  Skipping {date} (missing static VSL).")
            continue

        # ── Dynamic calibration ───────────────────────────────────────────
        dyn_cal_path = (
            f"/Users/shreyaar/Desktop/PhD/research/MPC/data/i24/i24_{date}/calibration_dynamic"
        )
        dyn_params = METANET_Params(
            path=dyn_cal_path,
            num_timesteps=flow_data.shape[0],
            num_segments=flow_data.shape[1],
            control_h=90
        ).get_params()

        dyn_vsl_path = (
            f"/Users/shreyaar/Desktop/PhD/research/MPC/results/i24/i24_{date}"
            "/calibration_dynamic/control_h_90/optimal_vsl.npy"
        )

        print("  Running dynamic calibration simulations...")
        dyn_metrics = run_one_day(
            init_state, data_inflow, ds_density_norm,
            dyn_params, lane_dict, lane_counts,
            v_trimmed, d_trimmed, num_segments, dyn_vsl_path
        )
        if dyn_metrics is None:
            print(f"  Skipping {date} (missing dynamic VSL).")
            continue

        print(
            f"  Static  — Sim MAPE: {static_metrics['sim_mape']:.2f}%  "
            f"TTS MAPE: {static_metrics['tts_mape']:.2f}%  "
            f"CC: {static_metrics['cc']:.2f}%"
        )
        print(
            f"  Dynamic — Sim MAPE: {dyn_metrics['sim_mape']:.2f}%  "
            f"TTS MAPE: {dyn_metrics['tts_mape']:.2f}%  "
            f"CC: {dyn_metrics['cc']:.2f}%"
        )

        results.append({
            'date':            date,
            'static_sim_mape': static_metrics['sim_mape'],
            'static_tts_mape': static_metrics['tts_mape'],
            'static_cc':       static_metrics['cc'],
            'dyn_sim_mape':    dyn_metrics['sim_mape'],
            'dyn_tts_mape':    dyn_metrics['tts_mape'],
            'dyn_cc':          dyn_metrics['cc'],
        })

    # ── Plot summary table ────────────────────────────────────────────────────
    if results:
        plot_results_table(results, save_path="figs/i24_cc_table_withdyn.png")
    else:
        print("No results to display.")