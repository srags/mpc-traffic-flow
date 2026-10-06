"""The existing CC tables and two figures, consuming prepared report data."""

from pathlib import Path

import numpy as np

from .analysis import CCReport
from .console import colored
from ..paths import REPO_DIR


def print_cc_report(report: CCReport) -> None:
    from tabulate import tabulate

    print(colored("No Control", "bold", "yellow"))
    print(tabulate(report.fit_rows.items(), headers=("Quantity", "Result"), tablefmt="outline"))
    print(colored("Optimized VSLs", "bold", "yellow"))
    print(tabulate(report.cc_rows.items(), headers=("Metric", "Value"), tablefmt="outline"))


def save_cc_plots(report: CCReport, output_dir: Path, *, start_hour: float = 7.5) -> tuple[Path, Path]:
    """Save optimal_vsl.png and controlled.png, preserving the cc_plot layout.

    No solver/simulator runs or policy writes. Matplotlib is imported only here.
    """
    from .plots import Plotter
    import matplotlib.pyplot as plt

    spec, traffic = report.scenario.spec, report.scenario.traffic
    result = report.result
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
    p.fig.colorbar(p[0].imshow(report.display_vsl.T, cmap='RdYlGn', aspect='auto', interpolation='none',
                             vmin=0, vmax=report.params['v_free'].max()), label='VSL speed (km/hr)')
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
                                vmin=0, vmax=max(traffic.velocity.max(), report.diagnostic.velocity[:-1].max())),
            ax=p[i], label='Velocity (km/hr)', orientation='horizontal',
        )
    p.fig.colorbar(
        mappable=p[3].imshow(report.pct_decrease, cmap='inferno', aspect='auto', vmin=-100, vmax=100),
        ax=p[3], label='% Decrease in Delay', orientation='horizontal', location='bottom',
        pad=0.15, shrink=0.8, aspect=20,
    )
    for i, _ in enumerate(p):
        p[i] = {'xlabel': 'Time', 'ylabel': 'Distance (km)', 'xticks': tick_pos, 'yticks': ytick_pos,
                'xticklabels': tick_labels, 'yticklabels': ytick_labels}
        p[i].invert_yaxis()
    p[0] = {'title': f'{spec.freeway.upper().replace("I24", "I-24")} Ground Truth'}
    p[1] = {'title': f'METANET Simulation\n(TT: {np.round(result.baseline.total_travel_time, 2)} veh-hr, Delay: {np.round(report.delay, 2)} veh-hr)'}
    p[2] = {'title': f'VSL Control\n(TT: {np.round(result.controlled.total_travel_time, 2)} veh-hr, Delay: {np.round(report.controlled_delay, 2)} veh-hr)'}
    p[3] = {'title': '% Decrease in Delay'}
    controlled_path = save(p, "controlled.png")



    return policy_path, controlled_path


