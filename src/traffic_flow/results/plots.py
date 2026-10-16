import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.colorbar import Colorbar
from matplotlib.projections import register_projection

from ..console import print_save
from ..model.simulation import *
from ..types import time_space, ScenarioSpec, DepartureCurves
from .sensitivity import SensitivityResult
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

  def plot_time_space(self, data: time_space, spec: ScenarioSpec, vmax: float, title: str = "",
                    colorbar: None | tuple[Literal["h", "v"], str] = None, grid: bool = False,
                    cmap: str = 'RdYlGn', vmin = 0., smooth: bool = False, **kwargs) -> AxesImage:
    """Plot a time-space diagram of the given data."""
    image = self.imshow(data.T, cmap=cmap, aspect='auto', extent=(0, data.shape[0], data.shape[1], 0),
                        interpolation='antialiased' if smooth else 'none', vmin=vmin, vmax=vmax)
    time_steps, num_segments = data.shape
    minutes = np.arange(0, time_steps + 1, 60)
    self.set(title=title, xlabel='Time', ylabel='Distance (km)',
      xticks=np.arange(0, time_steps + 1, 60), xticklabels=[format_time(spec.start_hour, m * spec.time_step * 60) for m in minutes],
      yticks=np.arange(0, num_segments, 1/spec.L), yticklabels=np.round(np.arange(0, num_segments * spec.L, 1), 2))
    if colorbar is not None:
      self._parent.fig.colorbar(image, ax=self, label=colorbar[1],
        orientation="horizontal" if colorbar[0] == "h" else None, **kwargs)
    self.invert_yaxis()
    if grid: self.grid()
    return image

  def plot_departures(
    self,
    curves: DepartureCurves,
    duration_min: float,
    oblique: bool = True,
  ):
      x = curves.time_min
      offset = curves.arrivals[-1] * x / duration_min if oblique else 0.0

      ff = curves.free_flow - offset
      baseline = curves.baseline - offset
      controlled = curves.controlled - offset

      self.fill_between(x, ff, baseline, color="#d33b19", alpha=0.15)
      self.fill_between(x, ff, controlled, color="#4e9858", alpha=0.30)
      self.plot(x, ff, "k--", label="Free flow")
      self.plot(x, baseline, color="#d33b19", label="No control")
      self.plot(x, controlled, color="#4e9858", label="Optimal control")
      self.axvline(duration_min, color="0.7", linewidth=1)
      self.set(
          xlabel="Time (min)",
          ylabel="Departures − background" if oblique else "Departures",
          xlim=(0, x[-1]),
      )
      self.grid(alpha=0.3)

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

  def plot_param_sensitivity(self, result: SensitivityResult, show_mape: bool = True):
    if (self.rows, self.cols) != (2, 3):
      raise ValueError("Parameter sensitivity needs Plotter(2, 3).")

    titles = {
      "eta_high": r"$\eta$", "K": r"$\kappa$", "tau": r"$\tau$",
      "p_crit": r"$\rho_{\mathrm{cr}}$",
      "v_free": r"$v_{\mathrm{free}}$", "a": r"$a$",
    }

    def limits(series, nominal):
      values = np.concatenate([*series.values(), [nominal]])
      lo, hi = np.nanmin(values), np.nanmax(values)
      pad = 0.05 * (hi - lo) if hi > lo else 1.0
      return lo - pad, hi + pad

    x = result.perturb_pct * 100
    nominal_cc = result.nominal.cc_unclipped
    nominal_mape = result.nominal.sim_error
    cc_limits = limits(result.cc, nominal_cc)
    cc_color = "blue" if show_mape else "orange"
    if show_mape:
      mape_limits = limits(result.mape, nominal_mape)

    for i, key in enumerate(result.cc):
      ax = self[i]
      ax.plot(x, result.cc[key], color=cc_color, linewidth=2)
      ax.plot(0, nominal_cc, "ko", markersize=5)
      ax.axvline(0, color="black", linestyle=":", alpha=0.7)
      ax.axhline(nominal_cc, color="black", linestyle=":", alpha=0.7)
      ax.grid(alpha=0.3)
      ax.set(
        title=titles[key], xlim=(x[0], x[-1]), ylim=cc_limits,
        xlabel="Perturbation (%)" if i >= 3 else "",
        ylabel="CC (%)" if i % 3 == 0 else "",
      )
      ax.tick_params(axis="y", labelcolor=cc_color, labelleft=i % 3 == 0)

      if show_mape:
        right = ax.twinx()
        right.plot(x, result.mape[key], color="orange", linewidth=2)
        right.plot(0, nominal_mape, "o", color="orange", markeredgecolor="black")
        right.set(
          ylim=mape_limits,
          ylabel="Sim MAPE (%)" if i % 3 == 2 else "",
        )
        right.tick_params(axis="y", labelcolor="orange", labelright=i % 3 == 2)

    self.fig.tight_layout()
