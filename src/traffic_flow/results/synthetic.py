"""Replay legacy synthetic CSV policies and report their bottleneck study.

No MPC, calibration, policy conversion or writes to the input directory.
Matplotlib is imported only by the figure writers.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from ..config import SyntheticConfig
from ..inputs.generation import generate_demand
from ..model.parameters import default_metanet_params
from ..model.simulation import METANET_Simulator
from ..types import MetanetParams, MetanetState, OptimizationResult, SimulationResult
from .analysis import get_ff_tts, get_num_veh
from .console import colored


@dataclass(frozen=True)
class SyntheticResult:
    config: SyntheticConfig
    peak: float
    optimization: OptimizationResult
    num_veh: float
    free_flow_ttt: float

    @property
    def delay(self) -> float:
        return self.optimization.baseline.total_travel_time - self.free_flow_ttt

    @property
    def controlled_delay(self) -> float:
        return self.optimization.controlled.total_travel_time - self.free_flow_ttt

    @property
    def cc(self) -> float:
        return (self.delay - self.controlled_delay) / self.delay * 100 if self.delay else float("nan")


def load_synthetic_results(
    peaks: Iterable[float],
    policy_dir: Path,
    *,
    config: SyntheticConfig = SyntheticConfig(),
    params: MetanetParams | None = None,
    skip_missing: bool = False,
) -> list[SyntheticResult]:
    """Read demand<peak>_duration<duration>.csv and replay both trajectories.

    Preserve the study's conventions: modeled origin/queue, unrestricted
    baseline (VSL=None), uniform initial density based on the upstream lanes,
    and T+1 unpadded demand samples in vehicle/free-flow totals. The free-flow
    reference uses the first segment's speed, historically uniform at 120.
    Both simulations retain T+1 history rows. No fake measured data is created
    to fit the observed-data Scenario interface.
    """
    policy_dir = Path(policy_dir)
    params = default_metanet_params(config.num_segments) if params is None else params
    v_free = params["v_free"]
    if v_free.shape != (config.num_segments,) or not np.isfinite(v_free[0]) or v_free[0] <= 0:
        raise ValueError("Synthetic reports require static free-flow speeds with a positive first value")

    results = []
    for peak in peaks:
        peak = float(peak)
        if not np.isfinite(peak) or peak < 0:
            raise ValueError("Peak demand must be finite and nonnegative")
        policy = policy_dir / f"demand{peak}_duration{float(config.peak_duration)}.csv"
        try:
            vsl = np.loadtxt(policy, delimiter=",", ndmin=2)
        except FileNotFoundError:
            if not skip_missing:
                raise FileNotFoundError(f"No saved VSL policy for peak demand {peak}: {policy}") from None
            print(f"  [skip] no saved policy for peak demand {peak:g}: {policy}")
            continue
        if vsl.shape != (config.time_steps, config.num_segments):
            raise ValueError(f"{policy}: expected VSL shape {(config.time_steps, config.num_segments)}, got {vsl.shape}")
        if not np.isfinite(vsl).all() or np.any(vsl < 0):
            raise ValueError(f"{policy}: VSL values must be finite and nonnegative")

        demand = generate_demand(
            config.duration, config.time_step,
            peak_start=config.peak_start, peak_end=config.peak_start + config.peak_duration,
            flow_standard=float(config.flow_standard), flow_peak=peak, # type: ignore
        )
        state = MetanetState(
            np.full(config.num_segments, demand[0] / (config.lanes[0] * config.initial_speed)),
            np.full(config.num_segments, config.initial_speed), float(demand[0]), 0.0,
        )
        simulator = METANET_Simulator(
            T=config.time_step, l=config.L, params=params,
            lanes=dict(enumerate(config.lanes)), real_data=False,
        )
        # Historical uncontrolled replay receives the terminal demand sample;
        # controlled replay receives T samples. Neither needs forecast padding.
        baseline = SimulationResult(*simulator.run_with_history(
            demand, np.zeros(config.time_steps), state,
        ))
        controlled = SimulationResult(*simulator.run_with_history(
            demand[:config.time_steps], np.zeros(config.time_steps), state, vsl,
        ))
        results.append(SyntheticResult(
            config, peak, OptimizationResult(vsl, baseline, controlled),
            float(get_num_veh(demand, config.time_step)),
            float(get_ff_tts(demand, config.time_step, config.total_distance, [float(v_free[0])])),
        ))

    if not results:
        raise FileNotFoundError(f"No requested synthetic policies found in {policy_dir}")
    return results


def print_synthetic_report(results: Sequence[SyntheticResult]) -> None:
    from tabulate import tabulate

    print(tabulate(
        [(r.peak, r.delay, r.controlled_delay, r.cc) for r in results],
        headers=("Peak (veh/hr)", "Delay (veh-hr)", "Controlled (veh-hr)", "CC (%)"),
        floatfmt=(".0f", ".2f", ".2f", ".2f"), tablefmt="outline",
    ))
    if results:
        best = max(results, key=lambda r: r.cc)
        print(f"CC peaks at {best.peak:.0f} veh/hr with CC = {best.cc:.2f}%")
    if len(results) > 1:
        before, after = min(zip(results, results[1:]), key=lambda pair: pair[1].cc - pair[0].cc)
        print(f"Largest single-step drop: {before.peak:.0f} veh/hr ({before.cc:.2f}%)"
              f" -> {after.peak:.0f} veh/hr ({after.cc:.2f}%)")


def save_synthetic_tsd(
    results: Sequence[SyntheticResult], output_dir: Path, *, control_start_km: float | None = 4.0,
) -> Path:
    """Save the existing baseline/controlled velocity panels with one row per peak.

    Lane-change markers come from config.lanes. The blue control-zone marker
    is explicit because the legacy CSV does not record the allowed zone.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("At least one synthetic result is required")
    config = results[0].config
    if any(r.config != config for r in results):
        raise ValueError("Time-space panels must share geometry and timing settings")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "synthetic_demand_tsd.png"
    duration_min = config.duration * 60
    lane_changes = (np.flatnonzero(np.diff(config.lanes)) + 1) * config.L

    with mpl.rc_context({"font.family": "serif", "font.serif": ["Times New Roman"]}):
        fig, axes = plt.subplots(len(results), 2, figsize=(13, 4.6 * len(results)),
                                 sharex=True, sharey=True, squeeze=False)
        try:
            for row, result in enumerate(results):
                for col, (sim, delay, title) in enumerate((
                    (result.optimization.baseline, result.delay, "No control"),
                    (result.optimization.controlled, result.controlled_delay, "Optimal speed limit control"),
                )):
                    ax = axes[row, col]
                    im = ax.imshow(sim.velocity.T, aspect="auto", origin="lower",
                                   cmap="RdYlGn", interpolation="none", vmin=0, vmax=120,
                                   extent=(0, duration_min, 0, config.total_distance))
                    for position in lane_changes:
                        ax.axhline(position, color="black", linestyle="--", linewidth=2.5)
                    if control_start_km is not None:
                        ax.axhline(control_start_km, color="blue", linestyle="--", linewidth=2.5)
                    label = chr(ord("a") + row * 2 + col)
                    ax.set_title(f"({label}) {result.peak:.0f} veh/hr, {title}\nDelay = {delay:.0f} veh-hrs",
                                 fontsize=14, fontname="Times New Roman")
                    if col == 0:
                        ax.set_ylabel("Distance (km)", fontsize=16, fontname="Times New Roman")
                    if row == len(results) - 1:
                        ax.set_xlabel("Time (min)", fontsize=16, fontname="Times New Roman")
                    ax.set_xticks(np.arange(0, duration_min + 1, 20))
                    ax.set_yticks(np.arange(0, config.total_distance + 1, 2))
                    ax.tick_params(labelsize=12)
                    for tick in ax.get_xticklabels() + ax.get_yticklabels():
                        tick.set_fontname("Times New Roman")

            fig.subplots_adjust(right=0.88, hspace=0.28, wspace=0.08)
            cbar = fig.colorbar(im, cax=fig.add_axes((0.90, 0.12, 0.02, 0.76)))
            cbar.set_label("Velocity (km/hr)", fontsize=16, fontname="Times New Roman")
            cbar.ax.tick_params(labelsize=12)
            for tick in cbar.ax.get_yticklabels():
                tick.set_fontname("Times New Roman")
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        finally:
            plt.close(fig)
    print(f"Saved to {colored(path, 'green')}")
    return path


def save_synthetic_cc(
    results: Sequence[SyntheticResult], output_dir: Path, *, units: Literal["min", "hr"] = "min",
) -> Path:
    """Save the existing CC-vs-demand bars, without running any simulations."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("At least one synthetic result is required")
    if units not in ("min", "hr"):
        raise ValueError("units must be 'min' or 'hr'")
    if any(r.num_veh <= 0 for r in results):
        raise ValueError("Average delay requires a positive vehicle count")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "synthetic_cc_demand.png"
    scale = 60.0 if units == "min" else 1.0
    peaks = np.array([r.peak for r in results])
    uncontrolled = np.array([r.delay / r.num_veh * scale for r in results])
    controlled = np.array([r.controlled_delay / r.num_veh * scale for r in results])
    scenarios = np.arange(1, len(results) + 1)

    with mpl.rc_context({"font.family": "serif", "font.serif": ["Times New Roman"]}):
        fig, ax = plt.subplots(figsize=(15, 6))
        try:
            ax.grid()
            ax.set_axisbelow(True)
            ax.bar(scenarios, uncontrolled, label="Delay without control", color="#d33b19", width=0.85)
            ax.bar(scenarios, uncontrolled - controlled, label="Delay reduced by control", color="#4e9858", width=0.85)
            for i, result in enumerate(results):
                ax.text(scenarios[i], uncontrolled[i] + 0.01 * np.max(uncontrolled), f"{result.cc:.0f}",
                        ha="center", va="bottom", fontsize=12, fontname="Times New Roman", fontweight="bold")
            unit_label = "min" if units == "min" else "hrs"
            ax.set_xlabel("Peak demand of scenario (veh/hr)", fontsize=18, fontname="Times New Roman")
            ax.set_ylabel(f"Average delay per vehicle ({unit_label})", fontsize=18, fontname="Times New Roman")
            ax.set_xticks(scenarios)
            ax.set_xticklabels(peaks.astype(int), rotation=45, ha="center")
            ax.legend(prop={"family": "Times New Roman", "size": 18})
            ax.tick_params(labelsize=14)
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontname("Times New Roman")
            ax.set_ylim(0, np.max(uncontrolled) * 1.12)
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        finally:
            plt.close(fig)
    print(f"Saved to {colored(path, 'green')}")
    return path
