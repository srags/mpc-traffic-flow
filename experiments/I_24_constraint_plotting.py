"""
Plot controllable congestion (%) vs. minimum VSL speed limit, for the real
I-24 corridor -- styled to match plot_constraints.ipynb's speedlimit plot.

Data loading follows metanet_params.ipynb (cells 3-8, 24, 41).
Plot styling follows plot_constraints.ipynb (cell 7).
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import AutoMinorLocator

sys.path.append('../src')  # ADAPT: match your notebook's sys.path setup
from traffic_sim import run_metanet_sim
from param_loader import METANET_Params
from generate_demand_synthetic import get_ff_tts

def smooth_inflow(inflow, window_size=2):
    # Create averaging kernel
    kernel = np.ones(window_size) / window_size
    
    # Compute asymmetric padding for even window sizes
    pad_left = window_size // 2
    pad_right = window_size - pad_left - 1

    # Pad using boundary values (edge padding)
    if inflow.ndim == 1:
        padded = np.pad(inflow, (pad_left, pad_right), mode='edge')
    else:
        padded = np.pad(inflow, ((pad_left, pad_right), (0, 0)), mode='edge')

    # Convolve along time dimension
    smoothed = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode="valid"), axis=0, arr=padded
    )
    return smoothed

# ---------------------------------------------------------------------------
# Config -- mirrors metanet_params.ipynb cells 3-4 and cell 24's save location
# ---------------------------------------------------------------------------
L = 0.4
time_step = 10 / 3600
start_time = 0  # hours
start_time_step = int(start_time / time_step)

date = "i24_11_30"
data_path = f"/Users/shreyaar/Desktop/PhD/research/MPC/data/i24/{date}"
calibration_id = "calibration_static/fixed_ramping"
calibration_interval = None  # None if static
cal_path = (f'{data_path}/{calibration_id}' if calibration_interval is None
            else f'{data_path}/{calibration_id}/control_h_{calibration_interval}')

# Where mpc_find_vsl runs were saved -- ADAPT to match how you're saving per-speedlb
# runs now (cell 41 used f"src/sim_params/{exp_path}/optimal_vsl_speedlb{speed}.npy";
# your current mpc_opt.py setup saves to `results_path`, see metanet_params.ipynb cell 24)
constraint = "hold_length"
vsl_dir = f"/Users/shreyaar/Desktop/PhD/research/MPC/results/i24/{date}/{calibration_id}/{constraint}"  # ADAPT

# Sweep values -- ADAPT to whatever speed_lb values you've actually run
speeds = [1, 2, 3, 4]

save_name = "/Users/shreyaar/Desktop/PhD/research/MPC/figs/i24_hold_len.png"

# ---------------------------------------------------------------------------
# Load I-24 data + calibrated params (metanet_params.ipynb cells 5-8)
# ---------------------------------------------------------------------------
assert os.path.exists(cal_path), f"Calibration parameters not found at {cal_path}"
assert os.path.exists(f'{data_path}/v_hat.npy') and os.path.exists(f'{data_path}/rho_hat.npy'), \
    f"Data not found at {data_path}"

density_data = np.load(f'{data_path}/rho_hat.npy')
flow_data = np.load(f'{data_path}/q_hat.npy')
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

lane_counts = (np.load(f'{cal_path}/num_lanes.npy').reshape(-1) if calibration_interval is None
               else np.load(f'{cal_path}/params_1/num_lanes.npy').reshape(-1))
lane_dict = {i: lane_counts[i] for i in range(num_segments)}

downstream_density = downstream_density / lane_dict[num_segments - 1]
true_density_initial = true_density_initial / lane_counts

subfolders = [f for f in os.listdir(cal_path)
              if os.path.isdir(os.path.join(cal_path, f)) and f.startswith("params_")]
if subfolders:
    control_h = time_steps // len(subfolders)
    model_params = METANET_Params(path=f"{data_path}/{calibration_id}",
                                   control_h=control_h, num_timesteps=time_steps,
                                   num_segments=num_segments).get_params()
else:
    model_params = METANET_Params(path=cal_path, num_timesteps=time_steps,
                                   num_segments=num_segments).get_params()

init_state = (true_density_initial, true_velocity_initial, data_inflow[start_time_step], 0)

# ---------------------------------------------------------------------------
# Baseline (uncontrolled) run + free-flow travel time (metanet_params.ipynb cell 39)
# ---------------------------------------------------------------------------
_, _, _, tts_baseline = run_metanet_sim(
    time_step, L, init_state,
    data_inflow[start_time:], downstream_density[start_time:],
    model_params, lanes=lane_dict, vsl_speeds=None,
    plotting=True, real_data=True,
)
ff_ttt = get_ff_tts(data_inflow, time_step, L, model_params['v_free'])
delay_baseline = tts_baseline - ff_ttt

# ---------------------------------------------------------------------------
# Sweep over min speed limit values (metanet_params.ipynb cell 41)
# ---------------------------------------------------------------------------
opt_time = time_steps  # ADAPT if your saved optimal_vsl arrays cover a different horizon
ccs, valid_speeds = [], []

for speed in speeds:
    vsl_path = f"{vsl_dir}/optimal_vsl_{speed}.npy"
    if not os.path.exists(vsl_path):
        print(f"[skip] no saved run for speed_lb={speed} at {vsl_path}")
        continue

    opt_vsl = np.load(vsl_path)
    print(opt_vsl.shape)
    _, _, _, tts_opt = run_metanet_sim(
        time_step, L, init_state,
        data_inflow[start_time:start_time + opt_time],
        downstream_density[start_time:start_time + opt_time],
        model_params, vsl_speeds=opt_vsl, lanes=lane_dict,
        plotting=True, real_data=False,
    )

    opt_delay = tts_opt - ff_ttt
    cc = (delay_baseline - opt_delay) / delay_baseline * 100
    ccs.append(cc)
    valid_speeds.append(speed)

# ---------------------------------------------------------------------------
# Plot -- style matches plot_constraints.ipynb cell 7 (speedlimit branch)
# ---------------------------------------------------------------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16
text_fontsize = 20

fig, ax1 = plt.subplots(figsize=(10, 5))

colors = ["darkred", "red", "orange", "green", "darkgreen"]
vals = [0, 0.15, 0.5, 0.9, 1.0]
cmap = LinearSegmentedColormap.from_list("rg", list(zip(vals, colors)), N=256).reversed()

x = np.array(valid_speeds)
y = np.array(ccs)
ax1.plot(x, y, color=cmap(0.7), linewidth=3, marker="o")

ax1.set_xlabel("Min Speed Limit (kmph)", fontsize=text_fontsize)
ax1.set_ylabel("Controllable Congestion (%)", fontsize=text_fontsize)
ax1.set_ylim(-1, 60)

ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())

for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
    label.set_fontname("Times New Roman")
    label.set_fontsize(text_fontsize - 2)

ax1.grid(which="both", linestyle="-", linewidth=0.5)

plt.tight_layout()
plt.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()

print("Speeds:", valid_speeds)
print("Controllable congestion (%):", ccs)