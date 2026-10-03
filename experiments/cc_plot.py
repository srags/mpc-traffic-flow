import sys, os, pathlib

REPO_DIR = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_DIR / "src"))

from traffic_sim import METANET_Simulator
from viz import Plotter, colored
from tabulate import tabulate
from sim_types import *

# Setup freeway info
from data_loader import Freeway

freeway = Freeway(freeway="i24", date="11_30", L=0.4, num_segments=14, time_step=10/3600, time_steps=360, start_time=0.0)
start_time_step = freeway.start_time_step

# Read in the data
CONSTRAINT = None # or "speed_lb", "hold_length", "safety_sweep"
speed_lb = 0
assert(CONSTRAINT is not None or speed_lb == 0)
calibration_id = "calibration_static/fixed_ramping" # default: "calibration_static/fixed_ramping"
calibration_interval = None # None if static, else interval in time steps

p_data, q_data, v_data, lane_cts, data_inflow, downstream_density, p_init, v_init \
             = freeway.load_real_data()
results_path = freeway.results_path(calibration_id, calibration_interval, CONSTRAINT)
model_params = freeway.get_params(calibration_id, calibration_interval)

lane_dict: lane_map = {i: lane_count for i, lane_count in enumerate(lane_cts)}

print(f"{colored('No Control', 'bold', 'yellow')}")

simulator = METANET_Simulator(T=freeway.time_step, l=freeway.L, params=model_params, lanes=lane_dict, real_data=True)
p_sim, v_sim, _, _ = simulator.run_with_history(
  demand = data_inflow[start_time_step:],
  downstream_density = downstream_density[start_time_step:],
  init_traffic_state = MetanetState(p_init, v_init, data_inflow[start_time_step], 0)
)
p_sim: time_space = p_sim[:-1, :]
v_sim: time_space = v_sim[:-1, :]

q_sim = v_sim * p_sim * lane_cts

def _mape(y_true, y_pred): return np.mean(np.abs((y_true - y_pred) / y_true) * 100)
def _rmse(y_true, y_pred): return np.sqrt(np.mean((y_true - y_pred) ** 2))
gt_tt, metanet_tt = (freeway.time_step * freeway.L * (np.sum(rhos * lane_cts[np.newaxis,:])) for rhos in [p_data, p_sim])

print(tabulate(({
   qt: f"MAPE {_mape(y, y_pred):.2f}%, RMSE {_rmse(y, y_pred):.2f}" 
   for qt, y, y_pred in [("Velocity", v_data, v_sim), ("Density", p_data, p_sim), ("Flow", q_data, q_sim)]
   } | { "Travel Time": f"{gt_tt:.2f} veh-hr vs {metanet_tt:.2f} veh-hr (MAPE {abs(gt_tt - metanet_tt) / gt_tt * 100:.2f}%)"
    }).items(), headers=("Quantity", "Result"), tablefmt="outline"))

# plot optimized VSL

print(f"{colored('Optimized VSLs', 'bold', 'yellow')}")

path_id = f"optimal_vsl_{speed_lb}" if CONSTRAINT == "speed_lb" else f"optimal_vsl"
optimal_vsl: np.ndarray = np.load(results_path / f'{path_id}.npy')

v_free_extended = np.tile(model_params['v_free'], (optimal_vsl.shape[0], 1))
optimal_vsl = np.where(optimal_vsl > v_free_extended, 150, optimal_vsl)
p = Plotter(1, 1)
p.fig.colorbar(p[0].imshow(optimal_vsl.T, cmap='RdYlGn', aspect='auto', interpolation='none', vmin=0, vmax=model_params['v_free'].max()),
  label='VSL speed (km/hr)')
p[0] = {'title': 'Optimal VSL Speeds', 'xlabel': 'Time (min)', 'ylabel': 'Distance (km)',
        'xticks': np.arange(0, freeway.time_steps+1, 60), 'xticklabels': (np.arange(0, (freeway.time_steps+1) * freeway.time_step * 60, 60 * freeway.time_step * 60)).astype(int),
        'yticks': np.arange(0, freeway.num_segments, 2), 'yticklabels': np.round(np.arange(0, freeway.num_segments*freeway.L, 2*freeway.L), 2)}
p[0].invert_yaxis()
p[0].grid()
p.savefig(REPO_DIR / "fig_new" / "optimal_vsl.png")
print(f"Saved to {colored('fig_new/optimal_vsl.png', 'green')}")

(p_baseline, v_baseline, _, tts_baseline), (p_opt, v_opt, queue, tts_opt) = \
  (simulator.run_with_history(
    demand = data_inflow[start_time_step:],
    downstream_density = downstream_density[start_time_step:],
    init_traffic_state = MetanetState(p_init, v_init, data_inflow[start_time_step], 0),
    vsl_speeds = vsl
  ) for vsl in [np.full((freeway.time_steps, freeway.num_segments), 150), optimal_vsl])

from generate_demand_synthetic import get_ff_tts

ff_ttt = get_ff_tts(data_inflow, freeway.time_step, freeway.L, model_params['v_free'])
delay  = tts_baseline - ff_ttt
opt_delay = tts_opt - ff_ttt

start_wallclock_time = 7.5 # 7

n_time = v_data.shape[0] +1
minutes = np.arange(n_time) * freeway.time_step * 60
tick_pos = np.linspace(0, n_time - 1, 5, dtype=int)

def format_time(start_hour, minutes):
    hour, minute = divmod(int(start_hour * 60 + minutes), 60)
    return f"{hour % 24:02d}:{minute:02d}"

tick_labels = [format_time(start_wallclock_time, m) for m in minutes[tick_pos]]

# Compute y ticks in km — add this before the imshow calls
n_segments = v_data.shape[1]
ytick_pos = np.linspace(0, n_segments - 1, 5, dtype=int)
ytick_labels = [f"{i * 0.4:.1f}" for i in ytick_pos]  # 400m per segment → km

# pointwise delay
# --- Compute Delay ---
# free_flow_speed shape: (n_segments,) → reshape to (n_segments, 1) to broadcast over time
free_flow_speed_2d = np.array(model_params['v_free'])[:, np.newaxis]  # or .reshape(-1, 1)

delay_gt        = (0.4 / v_data.T - 0.4 / free_flow_speed_2d) * 60
delay_noctrl    = (0.4 / v_baseline[0:-1, :].T - 0.4 / free_flow_speed_2d) * 60
delay_ctrl      = (0.4 / v_opt[0:-1, :].T - 0.4 / free_flow_speed_2d) * 60
pct_decrease = np.where(delay_gt > 0.01, (delay_gt - delay_ctrl) / delay_gt * 100, 0)

print(tabulate({
    "Total free flow travel time": f"{ff_ttt:.2f} veh-hrs",
    "Controllable congestion": f"{(delay - opt_delay) / delay * 100:.2f}%",
    "Delay (ground truth)": f"{delay_gt.min():.2f} min - {delay_gt.max():.2f} min",
    "Pct decrease": f"{pct_decrease.min():.2f}% - {pct_decrease.max():.2f}%",
}.items(), headers=("Metric", "Value"), tablefmt="outline"))

p = Plotter(1, 4)
for i, mat in enumerate([v_data, v_baseline, v_opt]):
  p.fig.colorbar(
    mappable=p[i].imshow(mat.T, cmap='RdYlGn', aspect='auto', interpolation='none', vmin=0, vmax=max(v_data.max(), v_sim.max())),
    ax=p[i], label='Velocity (km/hr)', orientation='horizontal'
  )
p.fig.colorbar(
    mappable=p[3].imshow(pct_decrease, cmap='inferno', aspect='auto', vmin=-100, vmax=100),
    ax=p[3], label='% Decrease in Delay', orientation='horizontal', location='bottom', pad=0.15, shrink=0.8, aspect=20
)
for i, ax in enumerate(p):
  p[i] = {'xlabel': 'Time', 'ylabel': 'Distance (km)', 'xticks': tick_pos, 'yticks': ytick_pos, 
          'xticklabels': tick_labels, 'yticklabels': ytick_labels}
  p[i].invert_yaxis()

p[0] = {'title': f'I-24 Ground Truth'}
p[1] = {'title': f'METANET Simulation\n(TT: {np.round(tts_baseline, 2)} veh-hr, Delay: {np.round(delay, 2)} veh-hr)'}
p[2] = {'title': f'VSL Control\n(TT: {np.round(tts_opt, 2)} veh-hr, Delay: {np.round(opt_delay, 2)} veh-hr)'}
p[3] = {'title': f'% Decrease in Delay'}

p.savefig(REPO_DIR / 'fig_new' / 'controlled.png')
print(f"Saved to {colored('fig_new/controlled.png', 'green')}")
