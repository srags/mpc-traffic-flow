import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize
from matplotlib.projections import register_projection

from traffic_flow.config import ScenarioConfig
from traffic_flow.console import print_save

from ..model.simulation import *
from ..types import time_space
from typing import Literal

preset_figsizes = {
  (1, 1): (10, 5),
  (1, 2): (15, 5),
  (1, 3): (17, 5),
  (1, 4): (20, 5),
}

plt.rcParams.update({'font.family': 'Times New Roman'})

def format_time(start_hour: hr, minutes: float) -> str:
  hour, minute = divmod(int(start_hour * 60 + minutes), 60)
  return f"{hour % 24:02d}:{minute:02d}"

class PlotterAxes(Axes):
  def parent(self, p: "Plotter"):
    self._parent = p
    return self
  def plot_time_space(self, data: time_space, spec: ScenarioConfig, start_time: hr, vmax: float, title: str = "",
                    colorbar: None | tuple[Literal["h", "v"], str] = None, grid: bool = False,
                    cmap: str = 'RdYlGn', vmin = 0., smooth: bool = False, **kwargs) -> ScalarMappable:
    """Plot a time-space diagram of the given data."""
    image = self.imshow(data.T, cmap=cmap, aspect='auto', extent=(0, data.shape[0], data.shape[1], 0),
                        interpolation='antialiased' if smooth else 'none', vmin=vmin, vmax=vmax)
    time_steps, num_segments = data.shape
    minutes = np.arange(0, time_steps + 1, 60)
    self.set(title=title, xlabel='Time', ylabel='Distance (km)',
      xticks=np.arange(0, time_steps + 1, 60), xticklabels=[format_time(start_time, m * spec.time_step * 60) for m in minutes],
      yticks=np.arange(0, num_segments, 1/spec.L), yticklabels=np.round(np.arange(0, num_segments * spec.L, 1), 2))
    if colorbar is not None:
      self._parent.fig.colorbar(image, ax=self, label=colorbar[1],
        orientation="horizontal" if colorbar[0] == "h" else None, **kwargs)
    self.invert_yaxis()
    if grid: self.grid()
    return image

register_projection(PlotterAxes)

class Plotter:
  def __init__(self, rows: int, cols: int, figsize = None):
    if figsize is None and (rows, cols) in preset_figsizes: 
      figsize = preset_figsizes[(rows, cols)]
    self.rows = rows
    self.cols = cols
    self.fig, self.axs = plt.subplots(rows, cols, figsize = figsize, squeeze=False)

  def __getitem__(self, id: int | tuple) -> PlotterAxes:
    axes = cast(PlotterAxes, self.axs[id] if isinstance(id, tuple) else self.axs.flat[id])
    axes.parent(self)
    return axes

  def __iter__(self):
    for i in range(self.rows * self.cols): yield self[i]

  def colorbar(self, image: ScalarMappable, axes: list[int], label: str = "",) -> Colorbar:
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
    print_save(args[0])

def plot_velocity_comparison(observed: time_space, simulated: time_space) -> Plotter:
    """Compare aligned (time, segment) velocity arrays in km/hr."""
    if observed.ndim != 2 or simulated.ndim != 2:
        raise ValueError("Both arrays must have shape (time, segment)")
    if observed.shape != simulated.shape or observed.size == 0:
        raise ValueError("Arrays must have matching, nonempty shapes")

    norm = Normalize(vmin=float(min(observed.min(), simulated.min())), vmax=float(max(observed.max(), simulated.max())))

    p = Plotter(1, 2)
    for i, (values, title) in enumerate([(observed, "Observed velocity"), (simulated, "Simulated velocity")]):
        image = p[i].imshow(values.T, aspect="auto", origin="lower", cmap="RdYlGn", interpolation="none", norm=norm)
        p[i].set(title=title, xlabel="Time step", ylabel="Segment index")

    p.colorbar(image, axes=[0, 1], label="Velocity (km/hr)")
    return p

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
    p = Plotter(h, w, figsize=(30, 7*h))
    i = 0
    #set title for first and second column

    if plot_v:
        p[i].imshow(v_nocontrol.T, cmap='RdYlGn', aspect='auto', extent=(0, time_steps+1, num_segm, 0), vmin=0, vmax=v_free, interpolation='None')
        p[i].set(title='Delay: {:.1f} veh-hr'.format(delay_nocontrol), xticks=[])
        im_v = p[i+1].imshow(v_optimal.T, cmap='RdYlGn', aspect='auto', extent=(0, time_steps+1, num_segm, 0), vmin=0, vmax=v_free, interpolation='None')
        p.fig.colorbar(im_v, label='Velocity (km/hr)', ax=p[i+1])
        p[i+1].set(
            title=f'Delay: {delay_optimal:.1f} veh-hr',
            xticks=np.arange(0, time_steps, 120), xticklabels=(np.arange(0, time_steps * T * 60, 120 * T * 60)).astype(int),
            xlabel='Time (min)',
        )
        im_cc = p[i+2].imshow(v_optimal.T - v_nocontrol.T, cmap='RdYlBu', extent=(0, time_steps+1, num_segm, 0), aspect='auto', vmin=-100, vmax=100, interpolation='None')
        p[i+2].set(title=f'{improvement_tts}% delay reduction\nVelocity Improvement')
        p.fig.colorbar(im_cc, label='Velocity Difference (km/hr)', ax=p[i+2], extend='both')
        i += 3
    if plot_p:
        p[i].imshow(p_nocontrol.T, cmap='RdYlGn_r', aspect='auto', extent=(0, time_steps+1, num_segm, 0), vmin=0, vmax=180, interpolation='None')
        p[i].set(title='Density without control')
        im_p = p[i+1].imshow(p_optimal.T, cmap='RdYlGn_r', aspect='auto', extent=(0, time_steps+1, num_segm, 0), vmin=0, vmax=180)
        p[i+1].set(title='Density with control')
        p.fig.colorbar(im_p, label='Density (veh/km)', ax=p[i+1])

        if plot_v:
            p.fig.delaxes(p[1, 2])
        i += 3 if plot_v else 2
    if plot_q:
        max_val = max((v_nocontrol*p_nocontrol* lane_array).max(), (v_optimal*p_optimal*lane_array).max())
        p[i].imshow(v_nocontrol.T * p_nocontrol.T * lane_array.T, cmap='viridis', aspect='auto', extent=(0, time_steps+1, num_segm, 0), vmin=0, vmax=max_val, interpolation='None')
        p[i].set(title='Flow without control')
        im_q = p[i+1].imshow(v_optimal.T * p_optimal.T * lane_array.T, cmap='viridis', aspect='auto', extent=(0, time_steps+1, num_segm, 0), vmin=0, vmax=max_val)
        p[i+1].set(title='Flow with control')
        p.fig.colorbar(im_q, label='Flow (veh/hr)', ax=p[i+1])

        if plot_v: p.fig.delaxes(p[i//3,2])
    
    # reverse y axis for all subplots
    tick_freq = int(num_segm / 5)
    for ax in p:
        ax.invert_yaxis()
        ax.set_yticks(np.arange(0, num_segm+1, tick_freq), np.round(np.arange(0, num_segm*l + 1, tick_freq*l), 2))
        ax.set_ylabel('Distance (km)', fontsize=22, fontname='Times New Roman')
        ax.tick_params(labelsize=20)
    
    # p.savefig('vsl_control.png', dpi=300, bbox_inches='tight')
    p.fig.subplots_adjust(hspace=0.4)
    p.show()

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
    p[0].set(
        xlabel="Time (min)",
        ylabel="Space (segments)",
        title="Virtual Trajectory",
        xlim=(0, time_steps),
        xticks=np.arange(0, time_steps + 1, 10),
        yticks=np.arange(0, m + 1, 1),
        xticklabels=tuple(
            str(int(i * 10 / 60)) if i % 120 == 0 else None
            for i in range(0, time_steps + 1, 10)
        ),
    )
    for time_points, space_points in virtual_trajectories:
        p[0].plot(
            time_points, 15 - np.array(space_points),
            color="blue", linewidth=1,
        )
    p[0].grid()
    p.show()
