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
