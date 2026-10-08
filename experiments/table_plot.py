import os
import sys
from tabulate import tabulate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from traffic_flow.paths import REPO_DIR, cut_repo
from traffic_flow.results.console import colored
from traffic_flow.results.i24 import run_static_dynamic_analysis, run_one_day
from traffic_flow.results.plots import Plotter
from traffic_flow.config import CalRef, CalSource
from traffic_flow.pipeline import init_state
from traffic_flow.results.io import load_some_run

# ── Dates to sweep ───────────────────────────────────────────────────────────
STATIC_DATES = ["11_30"]
# STATIC_DATES = [
#     "11_21", "11_22", "11_23", "11_24", "11_25",
#     "11_28", "11_29", "11_30", "12_01", "12_02"
# ]

DYNAMIC_DATES = ["11_30"]

def main():
    print(colored("Static-only CC sweep", "bold", "yellow"))
    run_static_analysis(STATIC_DATES, save_path=REPO_DIR / "figs" / "i24_cc_table.png",
                        bar_chart_save_path=REPO_DIR / "figs" / "i24_cc_bar_chart.png")
    print(colored("Static + dynamic CC sweep", "bold", "yellow"))
    run_static_dynamic_analysis(DYNAMIC_DATES, save_path=REPO_DIR / "figs" / "i24_cc_table_withdyn.png")

def run_static_analysis(dates, save_path=REPO_DIR / "figs" / "i24_cc_table.png",
                        bar_chart_save_path=REPO_DIR / "figs" / "i24_avg_delay_bar.png"):
    results = []
    for date in dates:
        print(f"Date: {colored(date, 'yellow')}")
        run = load_some_run("i24", date, CalRef(CalSource.FIXED_RAMPS, interval=None), study=None)
                        #  where=lambda config: all([config.init_fixed == 150.0]))
        spec, traffic = run.scenario.spec, run.scenario.traffic
        vsl = run.optimization.vsl
        metrics = run_one_day(
            init_state(traffic), traffic.inflow, traffic.downstream_density,
            run.params, dict(enumerate(traffic.lanes)), traffic.lanes,
            traffic.velocity, traffic.density, spec.num_segments, vsl
        )
        sim_mape = (abs(traffic.velocity - run.optimization.baseline.velocity[:-1]) / traffic.velocity).mean() * 100
        uncontrolled_tts, controlled_tts = run.optimization.baseline.total_travel_time, run.optimization.controlled.total_travel_time
        if metrics is None:
            print(f"  Skipping {date} (missing static VSL).")
            continue
        print(tabulate(tablefmt='grid', headers=['', 'Sim MAPE %', 'TTS MAPE %', 'CC %'], 
                       tabular_data =[["Static", metrics['sim_mape'], metrics['tts_mape'], metrics['cc']]]))
        results.append({'date': date, **metrics})

    KEYS = ["date", "tts_mape", "uncontrolled_tts", "controlled_tts", "cc", "avg_tt_reduced"]
    HEADERS = ["Date", "TTS Error\n(MAPE)", "Uncontrolled TTT\n(veh-hrs)", "Controlled TTT\n(veh-hrs)", "Controllable Congestion\n(%)", "Avg. TT Reduced / Veh\n(min)"]
    
    if results:
        p = Plotter(1, 1)
        p[0].axis('off')
        p[0].table(cellText=[[result[c] for c in KEYS] for result in results], 
                   colLabels=[h.replace("\n", " ") for h in HEADERS], 
                   loc='center', cellLoc='center')
        p.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Table saved to {colored(cut_repo(save_path), 'green', 'bold')}")

        print(tabulate(headers=[h.replace("\n", " ") for h in HEADERS], tablefmt="grid",
                    tabular_data=[[result[c] for c in KEYS] for result in results]))
        tex_path = REPO_DIR / "figs" / "i24_cc_table.tex"
        with open(tex_path, "w") as f: 
            f.write(latex_table(
                headers=HEADERS,
                rows=[[static_result[c] for c in KEYS] for static_result in results],
                caption="Simulation error and controllable congestion by dat",
                label="tab:cc_static"
            ))
        print(f"LaTeX table saved to {colored(cut_repo(tex_path), 'green', 'bold')}\n")

        plot_avg_delay_bar_chart(results, save_path=bar_chart_save_path)
    else:
        print("No results to display.")

    return results

def latex_table(headers, rows, caption="", label=""):
    fmt = lambda x: str(round(x, 2) if isinstance(x, (int, float)) else x).replace("%", r"\%").replace("_", r"\_")
    a = " & ".join(r"\textbf{" + (r"\shortstack{" + fmt(h).replace("\n", r" \\ ") + "}" if "\n" in str(h) else fmt(h)) + "}" for h in headers)
    b = chr(10).join(" & ".join(map(fmt, row)) + r" \\" for row in rows)
    return rf"""\begin{{table}}[ht]
    \centering
    \caption{{{caption}}}
    \label{{{label}}}
    \begin{{tabular}}{{l{"c" * (len(headers)-1)}}}
    \hline
    {a} \\
    \hline
    {b}
    \hline
    \end{{tabular}}
    \end{{table}}
"""

def plot_avg_delay_bar_chart(results, save_path=None):
    """
    Bar chart of average delay per vehicle for each day. The red bar is the
    total uncontrolled delay; the green overlay is the portion of that delay
    which was removed by control (i.e. uncontrolled * CC%), so the green
    fraction of each red bar visually equals that day's CC%. The exposed red
    remainder on top is the delay left over even with control. Styling
    follows src/generate_demand_synthetic.py's delay_reduction plot.
    """
    import numpy as np
    dates        = [format_date_label(r['date']) for r in results]
    uncontrolled = np.array([r['avg_delay_uncontrolled'] for r in results])
    controlled   = np.array([r['avg_delay_controlled'] for r in results])
    cc           = np.array([r['cc'] for r in results])
    reduced      = uncontrolled - controlled
    scenarios = np.arange(1, len(dates) + 1)

    # _, ax = plt.subplots(figsize=(max(8, 1.2 * len(dates)), 5))
    p = Plotter(1, 1, figsize=(max(8, 1.2 * len(dates)), 5))
    p[0].grid()
    p[0].set_axisbelow(True)

    p[0].bar(scenarios, uncontrolled, label="Delay without control", color="#d33b19")
    p[0].bar(scenarios, reduced, label="Delay reduced by control", color="#4e9858")

    for i in range(len(scenarios)):
        p[0].text(scenarios[i], uncontrolled[i] + 0.02 * np.max(uncontrolled),
                f"{cc[i]:.0f}", ha='center', va='bottom', fontweight='bold')

    p[0] = {'xlabel': 'Date', 'ylabel': 'Average delay per vehicle (min)', 
            'xticks': scenarios, 'xticklabels': dates,
            'ylim': (0, np.max(uncontrolled) * 1.15)}
    # p[0].set_xticklabels(dates, rotation=45, ha='center')
    p[0].legend()

    if save_path:
        p.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        print(f"Bar chart saved to {colored(cut_repo(save_path), 'green', 'bold')}")

HOLIDAY_DATES = {(11, 24), (11, 25)}
def format_date_label(date_str, year=2022):
    """'11_30' -> '11/30 (Wed)'; holiday dates -> '11/24 (Holiday)'."""
    import datetime
    month, day = (int(p) for p in date_str.split('_'))
    if (month, day) in HOLIDAY_DATES: tag = 'Holiday'
    else: tag = datetime.date(year, month, day).strftime('%a')
    return f"{month:02d}/{day:02d} ({tag})"

if __name__ == "__main__":
    main()