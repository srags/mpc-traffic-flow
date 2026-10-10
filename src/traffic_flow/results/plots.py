import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize

from ..model.simulation import *
from ..types import time_space

preset_figsizes = {
  (1, 1): (10, 5),
  (1, 2): (15, 5),
  (1, 3): (17, 5),
  (1, 4): (20, 5),
}

plt.rcParams.update({'font.family': 'Times New Roman'})

class Plotter:
  def __init__(self, rows: int, cols: int, figsize = None):
    if figsize is None and (rows, cols) in preset_figsizes: 
      figsize = preset_figsizes[(rows, cols)]
    self.rows = rows
    self.cols = cols
    self.fig, self.axs = plt.subplots(rows, cols, figsize = figsize, squeeze=False)

  def __getitem__(self, id: int | tuple) -> Axes:
    if isinstance(id, tuple):
        return self.axs[id] # type: ignore
    return self.axs.flat[id]

  def __setitem__(self, id: int | tuple, kwargs: dict):
    for key, value in kwargs.items(): getattr(self[id], "set_" + key)(value)

  def __iter__(self):
    for i in range(self.rows * self.cols): yield self[i]

  def colorbar(self, image: ScalarMappable, *, axes: list[int], label: str = "",) -> Colorbar:
    return self.fig.colorbar(
      image, ax=[self[i] for i in axes],
      label=label, pad=0.02,
    )

  def show(self):
    for ax in self: 
      if ax.get_legend_handles_labels()[1]: ax.legend()
    self.fig.tight_layout()
    plt.show()

  def savefig(self, *args, **kwargs):
    for ax in self: 
      if ax.get_legend_handles_labels()[1]: ax.legend()
    # plt.tight_layout()
    self.fig.savefig(*args, **kwargs)



def plot_velocity_comparison(
    observed: time_space,
    simulated: time_space,
) -> Plotter:
    """Compare aligned (time, segment) velocity arrays in km/hr."""
    if observed.ndim != 2 or simulated.ndim != 2:
        raise ValueError("Both arrays must have shape (time, segment)")
    if observed.shape != simulated.shape or observed.size == 0:
        raise ValueError("Arrays must have matching, nonempty shapes")

    norm = Normalize(vmin=float(min(observed.min(), simulated.min())), vmax=float(max(observed.max(), simulated.max())))

    p = Plotter(1, 2)
    for i, (values, title) in enumerate([(observed, "Observed velocity"), (simulated, "Simulated velocity")]):
        image = p[i].imshow(values.T, aspect="auto", origin="lower", cmap="RdYlGn", interpolation="none", norm=norm)
        p[i] = {"title": title, "xlabel": "Time step", "ylabel": "Segment index"}

    p.colorbar(image, axes=[0, 1], label="Velocity (km/hr)")
    return p

def plot_vsls(vsl_matrix: np.ndarray, time_steps: int, num_segm: int, v_free: float, T=10/3600, l=0.5):
    '''
    Given a matrix of VSL speeds, plot the VSL speeds as a heatmap. 
    (0,0) is the top left corner of the matrix. Rows represent time steps and columns represent segments.
    '''
    p = Plotter(1, 1)
    p.fig.colorbar(p[0].imshow(vsl_matrix.T, cmap='RdYlGn', aspect='auto', interpolation='none', vmin=0, vmax=v_free),
      label='VSL speed (km/hr)')
    p[0] = {'title': 'Optimal VSL Speeds', 'xlabel': 'Time (min)', 'ylabel': 'Distance (km)',
            'xticks': np.arange(0, time_steps, 60), 'xticklabels': (np.arange(0, time_steps * T * 60, 60 * T * 60)).astype(int),
            'yticks': np.arange(0, num_segm, 2), 'yticklabels': np.round(np.arange(0, num_segm*l, 2*l), 2)}
    p[0].invert_yaxis()
    p[0].grid()
    p.show()

def plot_outflow(traffic_demand, downstream_density, vsl_control, lane_map, v_free, T=10/3600, l=500/1000, seg=0):
    time_steps, num_segm = vsl_control.shape

    start_state = MetanetState(np.full(num_segm, 0), np.full(num_segm, 0), traffic_demand[0], 0)
    
    # p_nocontrol, v_nocontrol, queue_nocontrol, tts_nocontrol = run_metanet_sim_plottable(T, l, start_state, np.full((time_steps, num_segm), v_free), traffic_demand, downstream_density, lanes=lane_map)
    # p_optimal, v_optimal, queue_optimal, tts_optimal = run_metanet_sim_plottable(T, l, start_state, vsl_control, traffic_demand, downstream_density, lanes=lane_map)
    p_nocontrol, v_nocontrol, queue_nocontrol, tts_nocontrol = np.zeros((4, time_steps, num_segm))
    p_optimal, v_optimal, queue_optimal, tts_optimal = np.zeros((4, time_steps, num_segm))

    improvement_tts = round((tts_nocontrol - tts_optimal)/tts_nocontrol * 100, 1)

    lane_list = np.array(list(lane_map.values()))
    lane_array = np.tile(lane_list, (time_steps+1, 1))
    outflow_opt = v_optimal * p_optimal * lane_array
    outflow_nocontrol = v_nocontrol * p_nocontrol * lane_array

    p = Plotter(1, 1, figsize=(20, 10))
    p[0] = {'title': f'Flow out of segment {seg}',
            'xlabel': 'Time (min)', 'ylabel': 'Flow (veh/hr)', 
            'xticks': np.arange(0, time_steps, 120), 'xticklabels': (np.arange(0, time_steps * T * 60, 120 * T * 60)).astype(int),}
    p[0].plot(outflow_opt[:, seg], label='Optimal')
    p[0].plot(outflow_nocontrol[:, seg], label='No Control')
    p[0].axhline(y=2200 * lane_map[seg+1], color='r', linestyle='--', label='Capacity of next segment')
    p[0].legend()
    p.show()

def plot_nocontrol_control(traffic_demand, downstream_density, vsl_control, lane_map, v_free, T=10/3600, l=500/1000, plot_v=True, plot_p=False, plot_q=False, params=None):
    plot_var = np.array([plot_v, plot_p, plot_q])
    h = plot_var.sum()
    w = 3 if plot_v else 2
    widths = [2.4, 3, 3] if plot_v else [2.4, 3]
    time_steps, num_segm = vsl_control.shape
    print(time_steps, num_segm)

    start_state = MetanetState(np.full(num_segm, traffic_demand[0]/(lane_map[0] * 90)), np.full(num_segm, 90), traffic_demand[0], 0)

    
    if params is not None:
        sim = METANET_Simulator(T=T, l=l, params=params, lanes=lane_map, real_data=False)
        p_nocontrol, v_nocontrol, queue_nocontrol, tts_nocontrol = sim.run_with_history(
            traffic_demand, downstream_density, start_state
        )
        p_optimal, v_optimal, queue_optimal, tts_optimal = sim.run_with_history(
            traffic_demand, downstream_density, start_state, vsl_control
        )
    else:
        p_nocontrol, v_nocontrol, queue_nocontrol, tts_nocontrol= np.array([0, 0, 0, 0])#metanet_sim(T, l, start_state, np.full((time_steps, num_segm), v_free), traffic_demand, downstream_density, plotting=True, lanes=lane_map)
        p_optimal, v_optimal, queue_optimal, tts_optimal, = np.array([0, 0, 0, 0])#metanet_sim(T, l, start_state, vsl_control, traffic_demand, downstream_density, plotting=True, lanes=lane_map)

    total_inflow = np.sum(np.array(traffic_demand[0:time_steps]) * T)
    freeflow_tts = total_inflow * (num_segm * l) / v_free

    improvement_tts = np.round(((tts_nocontrol - freeflow_tts) - (tts_optimal - freeflow_tts))/(tts_nocontrol - freeflow_tts) * 100, 1)
    delay_nocontrol = tts_nocontrol - freeflow_tts
    delay_optimal = tts_optimal - freeflow_tts
    lane_list = np.array(list(lane_map.values()))
    lane_array = np.tile(lane_list, (time_steps+1, 1))
    
    #plot velocity, density, amd flow in order of plot_var
    p = Plotter(h, w, figsize=(30, 3.5*h))
    i = 0
    #set title for first and second column

    if plot_v:
        p[i].imshow(v_nocontrol.T, cmap='RdYlGn', aspect='auto', extent=(0, time_steps+1, num_segm, 0), vmin=0, vmax=v_free, interpolation='None')
        p[i] = {
            'title': 'Delay: {:.1f} veh-hr'.format(delay_nocontrol),
            'xticks': []
        }
        im_v = p[i+1].imshow(v_optimal.T, cmap='RdYlGn', aspect='auto', extent=(0, time_steps+1, num_segm, 0), vmin=0, vmax=v_free, interpolation='None')
        p.fig.colorbar(im_v, label='Velocity (km/hr)', ax=p[i+1])
        p[i+1] = {
            'title': f'Delay: {delay_optimal:.1f} veh-hr',
            'xticks': np.arange(0, time_steps, 120), 'xticklabels': (np.arange(0, time_steps * T * 60, 120 * T * 60)).astype(int),
            'xlabel': 'Time (min)'
        }
        im_cc = p[i+2].imshow(v_optimal.T - v_nocontrol.T, cmap='RdYlBu', extent=(0, time_steps+1, num_segm, 0), aspect='auto', vmin=-100, vmax=100, interpolation='None')
        p[i+2] = {'title': f'{improvement_tts}% delay reduction\nVelocity Improvement'}
        p.fig.colorbar(im_cc, label='Velocity Difference (km/hr)', ax=p[i+2], extend='both')
        i += 3
    if plot_p:
        p[i].imshow(p_nocontrol.T, cmap='RdYlGn_r', aspect='auto', extent=(0, time_steps+1, num_segm, 0), vmin=0, vmax=180, interpolation='None')
        p[i] = {'title': 'Density without control'}
        im_p = p[i+1].imshow(p_optimal.T, cmap='RdYlGn_r', aspect='auto', extent=(0, time_steps+1, num_segm, 0), vmin=0, vmax=180)
        p[i+1] = {'title': 'Density with control'}
        p.fig.colorbar(im_p, label='Density (veh/km)', ax=p[i+1])

        if plot_v:
            p.fig.delaxes(p[1, 2])
        i += 3 if plot_v else 2
    if plot_q:
        max_val = max((v_nocontrol*p_nocontrol* lane_array).max(), (v_optimal*p_optimal*lane_array).max())
        p[i].imshow(v_nocontrol.T * p_nocontrol.T * lane_array.T, cmap='viridis', aspect='auto', extent=(0, time_steps+1, num_segm, 0), vmin=0, vmax=max_val, interpolation='None')
        p[i] = {'title': 'Flow without control'}
        im_q = p[i+1].imshow(v_optimal.T * p_optimal.T * lane_array.T, cmap='viridis', aspect='auto', extent=(0, time_steps+1, num_segm, 0), vmin=0, vmax=max_val)
        p[i+1] = {'title': 'Flow with control'}
        p.fig.colorbar(im_q, label='Flow (veh/hr)', ax=p[i+1])

        if plot_v: p.fig.delaxes(p[i//3,2])
    
    # reverse y axis for all subplots
    tick_freq = int(num_segm / 5)
    for ax in p:
        ax.invert_yaxis()
        ax.set_yticks(np.arange(0, num_segm+1, tick_freq), np.round(np.arange(0, num_segm*l + 1, tick_freq*l), 2))
        ax.set_ylabel('Distance (km)', fontsize=22, fontname='Times New Roman')
        ax.tick_params(labelsize=20)
    
    # Plot a horizontal line at 2.5 km (segment 5)
    # for ax in axs:
    #     ax.axhline(10, color='black', linestyle='--')
    p.savefig('vsl_control.png', dpi=300, bbox_inches='tight')
    p.fig.subplots_adjust(hspace=0.4)
    p.show()

    

from pathlib import Path
from .analysis import MPCStats
from .console import colored
from ..paths import REPO_DIR


def save_cc_plots(pct_decrease: time_space, run_result: RunResult, stats: MPCStats, output_dir: Path, *, start_hour: float = 7.5) -> tuple[Path, Path]:
    """Save optimal_vsl.png and controlled.png, preserving the cc_plot layout.

    No solver/simulator runs or policy writes. Matplotlib is imported only here.
    """
    spec, traffic = run_result.scenario.spec, run_result.scenario.traffic
    result = run_result.optimization
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def save(plot: Plotter, name: str) -> Path:
        path = output_dir / name
        try: plot.savefig(path)
        finally: plt.close(plot.fig)
        display = path.relative_to(REPO_DIR) if path.is_relative_to(REPO_DIR) else path
        print(f"Saved to {colored(display, 'green')}")
        return path

    p = Plotter(1, 1)
    display_vsl = np.where(result.vsl > run_result.params["v_free"][None, :], 150, result.vsl)
    p.fig.colorbar(p[0].imshow(display_vsl.T, cmap='RdYlGn', aspect='auto', interpolation='none',
                             vmin=0, vmax=run_result.params['v_free'].max()), label='VSL speed (km/hr)')
    p[0] = {
        'title': 'Optimal VSL Speeds', 'xlabel': 'Time (min)', 'ylabel': 'Distance (km)',
        'xticks': np.arange(0, spec.time_steps + 1, 60),
        'xticklabels': np.arange(0, (spec.time_steps + 1) * spec.time_step * 60, 60 * spec.time_step * 60).astype(int),
        'yticks': np.arange(0, spec.num_segments, 2),
        'yticklabels': np.round(np.arange(0, spec.num_segments * spec.L, 2 * spec.L), 2),
    }
    p[0].invert_yaxis()
    p[0].grid()
    policy_path = save(p, "optimal_vsl.png")

    n_time = traffic.velocity.shape[0] + 1
    minutes = np.arange(n_time) * spec.time_step * 60
    tick_pos = np.linspace(0, n_time - 1, 5, dtype=int)

    def format_time(minutes: float) -> str:
        hour, minute = divmod(int(start_hour * 60 + minutes), 60)
        return f"{hour % 24:02d}:{minute:02d}"

    tick_labels = [format_time(m) for m in minutes[tick_pos]]
    ytick_pos = np.linspace(0, traffic.velocity.shape[1] - 1, 5, dtype=int)
    ytick_labels = [f"{i * spec.L:.1f}" for i in ytick_pos]
    p = Plotter(1, 4)
    for i, mat in enumerate((traffic.velocity, result.baseline.velocity, result.controlled.velocity)):
        p.fig.colorbar(
            mappable=p[i].imshow(mat.T, cmap='RdYlGn', aspect='auto', interpolation='none',
                                vmin=0, vmax=max(traffic.velocity.max(), run_result.optimization.baseline.velocity[:-1].max())),
            ax=p[i], label='Velocity (km/hr)', orientation='horizontal',
        )
    p.fig.colorbar(
        mappable=p[3].imshow(pct_decrease, cmap='inferno', aspect='auto', vmin=-100, vmax=100),
        ax=p[3], label='% Decrease in Delay', orientation='horizontal', location='bottom',
        pad=0.15, shrink=0.8, aspect=20,
    )
    for i, _ in enumerate(p):
        p[i] = {'xlabel': 'Time', 'ylabel': 'Distance (km)', 'xticks': tick_pos, 'yticks': ytick_pos,
                'xticklabels': tick_labels, 'yticklabels': ytick_labels}
        p[i].invert_yaxis()
    p[0] = {'title': f'{spec.freeway.upper().replace("I24", "I-24")} Ground Truth'}
    p[1] = {'title': f'METANET Simulation\n(TT: {np.round(result.baseline.total_travel_time, 2)} veh-hr, Delay: {np.round(stats.sim_tt - stats.ff_tt, 2)} veh-hr)'}
    p[2] = {'title': f'VSL Control\n(TT: {np.round(result.controlled.total_travel_time, 2)} veh-hr, Delay: {np.round(stats.opt_tt - stats.ff_tt, 2)} veh-hr)'}
    p[3] = {'title': '% Decrease in Delay'}
    controlled_path = save(p, "controlled.png")


    return policy_path, controlled_path



def plot_virtual_trajectories(macro_velocity_field, virtual_trajectories):
    m, time_steps = macro_velocity_field.shape
    p = Plotter(1, 1)
    p.fig.colorbar(
        p[0].imshow(
            macro_velocity_field, aspect="auto", interpolation="None",
            cmap="RdYlGn", extent=(0, time_steps, 0, m),
        ),
        ax=p[0], label="Velocity (km/hr)",
    )
    p[0] = {
        "xlabel": "Time (min)",
        "ylabel": "Space (segments)",
        "title": "Virtual Trajectory",
        "xlim": (0, time_steps),
        "xticks": np.arange(0, time_steps + 1, 10),
        "yticks": np.arange(0, m + 1, 1),
        "xticklabels": tuple(
            str(int(i * 10 / 60)) if i % 120 == 0 else None
            for i in range(0, time_steps + 1, 10)
        ),
    }
    for time_points, space_points in virtual_trajectories:
        p[0].plot(
            time_points, 15 - np.array(space_points),
            color="blue", linewidth=1,
        )
    p[0].grid()
    p.show()



from traffic_flow import evaluate

def save_reveal_plots(result: RunResult, output_dir: Path, checkpoints: int = 16, batches: int = 4) -> Path:
  """Save the evaluate-based reveal using modeled inflow/queue and a 150 km/hr baseline."""
  import numpy as np
  from traffic_flow.results.plots import Plotter

  if checkpoints < 2 or batches < 1 or checkpoints % batches:
    raise ValueError("CHECKPOINTS must be at least 2 and divisible by BATCHES")
  scenario, params = result.scenario, result.params
  spec = scenario.spec
  baseline = result.optimization.baseline.velocity[:-1]
  # Keep the notebook's speed substitution to isolate the origin-mode change.
  optimal_vsl = np.where(result.optimization.vsl > params["v_free"], 150, result.optimization.vsl)
  free_vsl = np.full_like(optimal_vsl, 150)

  times = np.rint(np.linspace(0, len(optimal_vsl), checkpoints)).astype(int)
  columns = checkpoints // batches
  p = Plotter(2 * batches, columns, figsize=(5 * columns, 5 * batches))
  vmax = max(scenario.traffic.velocity.max(), baseline.max())
  extent = (0, spec.time_steps * spec.time_step * 60, 0, spec.num_segments * spec.L)

  for k, t in enumerate(times):
    vsl = free_vsl.copy()
    vsl[:t] = optimal_vsl[:t]
    velocity = baseline.copy()
    if t:
      replay = evaluate(scenario, params, vsl)
      velocity[:t] = replay.controlled.velocity[:t]
    batch, column = divmod(k, columns)
    for row, data, label, first, last in (
      (2 * batch, vsl, "VSL Speeds", "Free VSL Speeds", "Optimal VSL Speeds"),
      (2 * batch + 1, velocity, "Time Space", "Baseline Time Space", "Optimized Time Space"),
    ):
      ax = p[row, column]
      ax.imshow(data.T, cmap="RdYlGn", aspect="auto", interpolation="none",
                origin="lower", extent=extent, vmin=0, vmax=vmax)
      if 0 < t < len(optimal_vsl): ax.axvline(t * spec.time_step * 60, color="black", linewidth=3)
      ax.set(title=first if k == 0 else last if k == checkpoints - 1 else f"{label} {k}",
              xlabel="Time (min)", ylabel="Distance (km)")

  p.fig.tight_layout()
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  path = output_dir / "cc_reveal_evaluate.png"
  p.savefig(path, dpi=150)
  display = path.relative_to(REPO_DIR) if path.is_relative_to(REPO_DIR) else path
  print(f"Saved to {colored(display, 'green')}")
  return path
