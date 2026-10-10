import os
import sys, numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from traffic_flow.paths import REPO_DIR, cut_repo
from traffic_flow.results.console import colored
from traffic_flow.results.plots import Plotter
from traffic_flow.results.analysis import get_ff_tts
from traffic_flow.config import CalRef, CalSource, StudyChoice
from traffic_flow.results.io import load_some_run

# ── Dates to sweep ───────────────────────────────────────────────────────────
STATIC_DATES = ["11_28", "11_29", "11_30"]
# STATIC_DATES = [
#     "11_21", "11_22", "11_23", "11_24", "11_25",
#     "11_28", "11_29", "11_30", "12_01", "12_02"
# ]

DYNAMIC_DATES = ["11_30"]

def main():
    run_static_analysis(STATIC_DATES, 
                        table_save=REPO_DIR / "figs" / "i24_cc_table.png",
                        tex_save=REPO_DIR / "figs" / "i24_cc_table.tex",
                        bar_chart_save=REPO_DIR / "figs" / "i24_cc_bar_chart.png")
    run_static_dynamic_analysis(DYNAMIC_DATES, 
                                table_save=REPO_DIR / "figs" / "i24_cc_table_withdyn.png")

from typing import NamedTuple

class MPCStats(NamedTuple):
    sim_error: float
    tts_error: float
    uncontrolled_ttt: float
    controlled_ttt: float
    cc: float
    avg_tt_reduced_per_veh: float
    uncontrolled_avg_tts: float
    controlled_avg_tts: float

def run_analysis(date, calibration: CalRef, study: StudyChoice):
    run = load_some_run("i24", date, calibration, study)
    spec, traffic = run.scenario.spec, run.scenario.traffic
    gt_tt = spec.time_step * spec.L * (traffic.density.sum(axis=0) @ traffic.lanes) 
    sim_tt, opt_tt = run.optimization.baseline.total_travel_time, run.optimization.controlled.total_travel_time
    ffv = run.params['v_free']
    ff_tts = get_ff_tts(traffic.inflow, spec.time_step, spec.L, ffv.max(axis=0) if calibration.interval else ffv) # static
    num_veh = traffic.inflow.sum() * spec.time_step
    return MPCStats(
        (abs(traffic.velocity - run.optimization.baseline.velocity[:-1]) / traffic.velocity).mean() * 100, 
        (abs(gt_tt - sim_tt) / gt_tt) * 100, sim_tt, opt_tt, 
        np.clip((sim_tt - opt_tt) / (sim_tt - ff_tts) * 100, 0, 100), 
        (sim_tt - opt_tt) / num_veh * 60,
        (sim_tt - ff_tts) / num_veh * 60, 
        (opt_tt - ff_tts) / num_veh * 60
    )

def run_static_analysis(dates, table_save, tex_save, bar_chart_save):
    print(colored("Static-only CC sweep", "bold", "yellow"))
    HEADERS = ["Date", "Sim Error", "TTS Error", "Uncontrolled TTT", "Controlled TTT", "Controllable Congestion", "Avg. TT Reduced / Veh"]
    UNITS = ["% MAPE", "% MAPE", "veh-hrs", "veh-hrs", "%", "min"]
    rows = np.empty((len(dates), len(HEADERS)-1))
    avg_tts = np.empty((len(dates), 2))
    print(f"Dates: {colored(', '.join(dates), 'yellow')}")
    for i, date in enumerate(dates):
        stats = run_analysis(date, CalRef(CalSource.FIXED_RAMPS, interval=None), study=None)
        rows[i] = stats[:6]
        avg_tts[i] = stats[6:]

    # TABLE PNG AND TEX
    table = [[date] + [f"{x:.2f} {u}" for x, u in zip(row, UNITS)] 
             for date, row in zip(map(format_date_label, dates), rows)]
    save_table(table_save, HEADERS, table)
    with open(tex_save, "w") as f: 
        f.write(latex_table(headers=HEADERS, rows=table, 
                            label="tab:cc_static", caption="Simulation error and controllable congestion by date"))
    print(f"LaTeX table saved to {colored(cut_repo(tex_save), 'green', 'bold')}")

    # BAR CHART
    uncontrolled, controlled = avg_tts.T
    scenarios = range(1, len(dates) + 1)
    p = Plotter(1, 1, figsize=(max(8, 1.2 * len(dates)), 5))
    p[0].grid()
    p[0].bar(scenarios, uncontrolled, label="Delay without control", color="#d33b19")
    p[0].bar(scenarios, uncontrolled - controlled, label="Delay reduced by control", color="#4e9858")
    for i, (height, c) in enumerate(zip(uncontrolled, rows[:, 4])):
        p[0].text(i+1, height + 0.02 * uncontrolled.max(), f"{c:.0f}", ha='center', va='bottom', fontweight='bold')
    p[0] = {'xlabel': 'Date', 'ylabel': 'Average delay per vehicle (min)', 
            'xticks': scenarios, 'xticklabels': map(format_date_label, dates),
            'ylim': (0, uncontrolled.max() * 1.15), 'axisbelow': True}
    p[0].set_xticklabels(dates, rotation=20, ha='center')
    p.savefig(bar_chart_save, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Bar chart saved to {colored(cut_repo(bar_chart_save), 'green', 'bold')}")

def run_static_dynamic_analysis(dates, table_save):
    print(colored("Static + dynamic CC sweep", "bold", "yellow"))
    HEADERS = ['Date', 'Static Sim', 'Static TTS', 'Static CC', 'Dynamic Sim', 'Dynamic TTS', 'Dynamic CC']
    UNITS = ["% MAPE", "% MAPE", "%", "% MAPE", "% MAPE", "%"]

    rows = np.empty((len(dates), len(HEADERS)-1))
    print(f"Dates: {colored(', '.join(dates), 'yellow')}")

    for i, date in enumerate(dates):
        stats = run_analysis(date, CalRef(CalSource.FIXED_RAMPS, interval=None), study=None)
        rows[i][:3] = stats.sim_error, stats.tts_error, stats.cc
        stats = run_analysis(date, CalRef(CalSource.DYNAMIC, interval=90), study=None)
        rows[i][3:] = stats.sim_error, stats.tts_error, stats.cc

    save_table(table_save, HEADERS, 
               [[date]+[f"{x:.2f} {unit}" for x, unit in zip(row, UNITS)] 
                for date, row in zip(map(format_date_label, dates), rows)])

def save_table(save_path, headers, rows):
    row_height    = 0.5
    header_height = 0.6
    fig_height    = header_height + len(rows) * row_height + 0.8
    p = Plotter(1, 1, figsize=(13, fig_height))
    p[0].axis('off')
    p[0].table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    p.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Table saved to {colored(cut_repo(save_path), 'green', 'bold')}")


def latex_table(headers, rows, caption="", label=""):
    fmt = lambda x: str(x).replace("%", r"\%").replace("_", r"\_")
    a = " & ".join(r"\textbf{" + (r"\shortstack{" + fmt(h).replace("\n", r" \\ ") + "}" if "\n" in str(h) else fmt(h)) + "}" for h in headers)
    b = chr(10).join(" & ".join(map(fmt, row)) + r" \\" for row in rows)
    return rf"""\begin{{table}}[ht]
    \centering
    \caption{{{caption}}} \label{{{label}}}
    \begin{{tabular}}{{l{"c" * (len(headers)-1)}}}
    \hline
    {a} \\
    \hline
    {b}
    \hline
    \end{{tabular}} \end{{table}}
"""

HOLIDAY_DATES = {(11, 24), (11, 25)}
def format_date_label(date_str: str, year=2022):
    """'11_30' -> '11/30 (Wed)'; holiday dates -> '11/24 (Holiday)'."""
    import datetime
    month, day = (int(p) for p in date_str.split('_'))
    tag = 'Holiday' if (month, day) in HOLIDAY_DATES else datetime.date(year, month, day).strftime('%a')
    return f"{month:02d}/{day:02d} ({tag})"

if __name__ == "__main__":
    main()