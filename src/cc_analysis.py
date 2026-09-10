import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from param_loader import METANET_Params
from traffic_sim import run_metanet_sim

# ── Simulation parameters ────────────────────────────────────────────────────
L          = 0.4
time_step  = 10 / 3600
total_time = 1   # hours

# Re-exported under their historical names: experiments/i24_tsd.py and
# experiments/i24_constraints.py import RESULTS_ROOT from this module.
from paths import I24_DATA as DATA_ROOT, I24_RESULTS as RESULTS_ROOT, fig


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def get_num_veh(demand_profile, time_step):
    return sum(demand_profile) * time_step

def get_ff_tts(demand_profile, time_step, length, v_free):
    num_vehicles   = sum(demand_profile) * time_step
    ff_travel_time = sum(length / v for v in v_free)
    return num_vehicles * ff_travel_time

def smooth_inflow(inflow, window_size=2):
    kernel    = np.ones(window_size) / window_size
    pad_left  = window_size // 2
    pad_right = window_size - pad_left - 1
    if inflow.ndim == 1:
        padded = np.pad(inflow, (pad_left, pad_right), mode='edge')
    else:
        padded = np.pad(inflow, ((pad_left, pad_right), (0, 0)), mode='edge')
    return np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode="valid"), axis=0, arr=padded
    )


# Dates (month, day) that are holidays rather than a regular weekday, tagged
# "Holiday" instead of a day name (Nov 24/25, 2022 = Thanksgiving + the day after).
HOLIDAY_DATES = {(11, 24), (11, 25)}

def format_date_label(date_str, year=2022):
    """'11_30' -> '11/30 (Wed)'; holiday dates -> '11/24 (Holiday)'."""
    month, day = (int(p) for p in date_str.split('_'))
    if (month, day) in HOLIDAY_DATES:
        tag = 'Holiday'
    else:
        tag = datetime.date(year, month, day).strftime('%a')
    return f"{month:02d}/{day:02d} ({tag})"


def load_day_data(date):
    """
    Load MOTION data for a single date and build the shared boundary
    conditions / static-calibration parameters needed by any VSL run.
    """
    root_path = f"{DATA_ROOT}/i24_{date}"

    flow_data    = np.load(root_path + "/q_hat.npy")
    density_data = np.load(root_path + "/rho_hat.npy")
    density_data = np.where(density_data == 0.0, 1e-3, density_data)
    flow_data    = np.where(flow_data    == 0.0, 1e-3, flow_data)
    velocity_data = flow_data / density_data

    true_density_initial  = density_data[0, 1:-1].reshape(-1)
    true_velocity_initial = velocity_data[0, 1:-1].reshape(-1)
    downstream_density    = smooth_inflow(density_data[0:, -1].reshape(-1), window_size=2)
    data_inflow           = smooth_inflow(
        (velocity_data[0:, 0] * density_data[0:, 0]).reshape(-1), window_size=2
    )
    num_segments = density_data.shape[1] - 2

    static_cal_path = f"{root_path}/calibration_static/fixed_ramping"
    static_params = METANET_Params(
        path=static_cal_path,
        num_timesteps=flow_data.shape[0],
        num_segments=flow_data.shape[1]
    ).get_params()

    lane_counts = np.load(f'{static_cal_path}/num_lanes.npy').reshape(-1)
    lane_dict   = {i: lane_counts[i] for i in range(num_segments)}

    v_trimmed       = velocity_data[:, 1:-1]
    d_trimmed       = density_data[:, 1:-1] / lane_counts
    ds_density_norm = downstream_density / lane_dict[num_segments - 1]
    d0_norm         = true_density_initial / lane_counts
    init_state      = (d0_norm, true_velocity_initial, data_inflow[0], 0)

    return {
        'flow_data':       flow_data,
        'data_inflow':     data_inflow,
        'ds_density_norm': ds_density_norm,
        'init_state':      init_state,
        'static_params':   static_params,
        'lane_dict':       lane_dict,
        'lane_counts':     lane_counts,
        'v_trimmed':       v_trimmed,
        'd_trimmed':       d_trimmed,
        'num_segments':    num_segments,
    }


def run_one_day(init_state, data_inflow, ds_density_norm, model_params,
                lane_dict, lane_counts, v_trimmed, d_trimmed,
                num_segments, vsl_path):
    """
    Run baseline + VSL simulations for one calibration variant.
    Returns a dict of metrics, or None if the VSL file is missing.
    """
    p_sim, v_sim, _, tts_sim = run_metanet_sim(
        time_step, L, init_state, data_inflow, ds_density_norm,
        model_params, lanes=lane_dict, vsl_speeds=None,
        plotting=True, real_data=True
    )
    p_sim = p_sim[:-1, :]
    v_sim = v_sim[:-1, :]

    try:
        vsl = np.load(vsl_path)
    except FileNotFoundError:
        print(f"  VSL file not found: {vsl_path}")
        return None

    p_opt, v_opt, _, tts_opt = run_metanet_sim(
        time_step, L, init_state, data_inflow, ds_density_norm,
        model_params, lanes=lane_dict, vsl_speeds=vsl,
        plotting=True, real_data=False
    )
    v_opt = v_opt[:-1, :]
    p_opt = p_opt[:-1, :]

    v_free = model_params['v_free']
    ff_ttt    = get_ff_tts(data_inflow, time_step, L,
                           np.max(v_free, axis=0) if len(v_free.shape)==2 else v_free)
    delay     = tts_sim - ff_ttt
    opt_delay = tts_opt - ff_ttt
    cc        = (delay - opt_delay) / delay * 100
    cc = np.clip(cc, 0, 100)  # avoid negative CC due to numerical issues
    sim_mape_ = mape(v_trimmed, v_sim)

    gt_tt      = time_step * sum(np.sum(d_trimmed[:, i]) * lane_counts[i] * L
                                 for i in range(num_segments))
    metanet_tt = time_step * sum(np.sum(p_sim[:, i])     * lane_counts[i] * L
                                 for i in range(num_segments))
    tts_mape_  = np.abs(gt_tt - metanet_tt) / gt_tt * 100

    num_vehicles    = get_num_veh(data_inflow, time_step)
    avg_tt_reduced_ = (tts_sim - tts_opt) / num_vehicles * 60  # minutes/vehicle

    return {
        'sim_mape':               sim_mape_,
        'tts_mape':                tts_mape_,
        'cc':                      cc,
        'uncontrolled_tts':        tts_sim,
        'controlled_tts':          tts_opt,
        'avg_tt_reduced':          avg_tt_reduced_,
        'avg_delay_uncontrolled':  delay     / num_vehicles * 60,  # minutes/vehicle
        'avg_delay_controlled':    opt_delay / num_vehicles * 60,  # minutes/vehicle
    }


def run_static_for_date(date):
    """Run the static-calibration VSL simulation for one date."""
    day = load_day_data(date)
    static_vsl_path = f"{RESULTS_ROOT}/i24_{date}/calibration_static/fixed_ramping/optimal_vsl.npy"

    print("  Running static calibration simulations...")
    return run_one_day(
        day['init_state'], day['data_inflow'], day['ds_density_norm'],
        day['static_params'], day['lane_dict'], day['lane_counts'],
        day['v_trimmed'], day['d_trimmed'], day['num_segments'], static_vsl_path
    )


def run_static_and_dynamic_for_date(date):
    """Run both the static- and dynamic-calibration VSL simulations for one date."""
    day = load_day_data(date)
    static_vsl_path = f"{RESULTS_ROOT}/i24_{date}/calibration_static/fixed_ramping/optimal_vsl.npy"

    print("  Running static calibration simulations...")
    static_metrics = run_one_day(
        day['init_state'], day['data_inflow'], day['ds_density_norm'],
        day['static_params'], day['lane_dict'], day['lane_counts'],
        day['v_trimmed'], day['d_trimmed'], day['num_segments'], static_vsl_path
    )
    if static_metrics is None:
        return None, None

    dyn_cal_path = f"{DATA_ROOT}/i24_{date}/calibration_dynamic"
    dyn_params = METANET_Params(
        path=dyn_cal_path,
        num_timesteps=day['flow_data'].shape[0],
        num_segments=day['flow_data'].shape[1],
        control_h=90
    ).get_params()

    dyn_vsl_path = f"{RESULTS_ROOT}/i24_{date}/calibration_dynamic/control_h_90/optimal_vsl.npy"

    print("  Running dynamic calibration simulations...")
    dyn_metrics = run_one_day(
        day['init_state'], day['data_inflow'], day['ds_density_norm'],
        dyn_params, day['lane_dict'], day['lane_counts'],
        day['v_trimmed'], day['d_trimmed'], day['num_segments'], dyn_vsl_path
    )

    return static_metrics, dyn_metrics


def _style_table(fig, ax, table, columns, n_rows):
    HEADER_BG = '#1a2744'
    HEADER_FG = 'white'
    for col_idx in range(len(columns)):
        cell = table[0, col_idx]
        cell.set_facecolor(HEADER_BG)
        cell.set_text_props(color=HEADER_FG, fontweight='bold',
                            fontfamily='serif', fontname='Times New Roman')
        cell.set_edgecolor(HEADER_BG)

    ROW_ODD  = '#f5f6fa'
    ROW_EVEN = 'white'
    EDGE     = '#d0d4e0'
    for row_idx in range(1, n_rows + 1):
        bg = ROW_ODD if row_idx % 2 != 0 else ROW_EVEN
        for col_idx in range(len(columns)):
            cell = table[row_idx, col_idx]
            cell.set_facecolor(bg)
            cell.set_edgecolor(EDGE)
            cell.set_text_props(color='#1a1a2e',
                                fontfamily='serif', fontname='Times New Roman')


# Available columns for plot_static_table: key -> (header label, formatter).
STATIC_COLUMNS = {
    'date':                    ('Date',                       lambda r: format_date_label(r['date'])),
    'sim_mape':                ('Sim MAPE',                   lambda r: f"{r['sim_mape']:.2f}%"),
    'tts_mape':                ('TTS MAPE',                   lambda r: f"{r['tts_mape']:.1f}%"),
    'cc':                      ('CC (%)',                     lambda r: f"{r['cc']:.0f}%"),
    'uncontrolled_tts':        ('Uncontrolled TTT (veh-hrs)',  lambda r: f"{r['uncontrolled_tts']:.0f}"),
    'controlled_tts':          ('Controlled TTT (veh-hrs)',    lambda r: f"{r['controlled_tts']:.0f}"),
    'avg_tt_reduced':          ('Avg TT Reduced/Veh (min)',   lambda r: f"{r['avg_tt_reduced']:.1f}"),
    'avg_delay_uncontrolled':  ('Avg Uncontrolled Delay/Veh (min)', lambda r: f"{r['avg_delay_uncontrolled']:.2f}"),
    'avg_delay_controlled':    ('Avg Controlled Delay/Veh (min)',   lambda r: f"{r['avg_delay_controlled']:.2f}"),
}

DEFAULT_STATIC_COLUMNS = [
    'date', 'sim_mape', 'tts_mape', 'cc',
    'uncontrolled_tts', 'controlled_tts', 'avg_tt_reduced',
]


def plot_static_table(results, columns=None, save_path=None):
    """
    Table for the static calibration only.

    `columns` selects and orders which fields to show, using keys from
    STATIC_COLUMNS (e.g. ['date', 'cc', 'avg_tt_reduced']). Defaults to
    DEFAULT_STATIC_COLUMNS.
    """
    import matplotlib as mpl

    if columns is None:
        columns = DEFAULT_STATIC_COLUMNS
    unknown = [c for c in columns if c not in STATIC_COLUMNS]
    if unknown:
        raise ValueError(f"Unknown column(s) {unknown}. Available: {list(STATIC_COLUMNS)}")

    original_font = mpl.rcParams['font.family']
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']

    headers = [STATIC_COLUMNS[c][0] for c in columns]
    rows    = [tuple(STATIC_COLUMNS[c][1](r) for c in columns) for r in results]
    n_rows  = len(rows)
    n_cols  = len(columns)

    # Size each column (and the figure) to the longest string it holds, so
    # long headers like "Avg Uncontrolled Delay/Veh (min)" aren't cut off.
    col_chars = [
        max(len(headers[col_idx]), *(len(row[col_idx]) for row in rows)) if rows else len(headers[col_idx])
        for col_idx in range(n_cols)
    ]
    total_chars = sum(col_chars)
    col_widths  = [c / total_chars for c in col_chars]

    row_height    = 0.5
    header_height = 0.6
    fig_height    = header_height + n_rows * row_height + 0.8
    fig_width     = max(8, 0.22 * total_chars)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')

    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, row_height / 0.22)

    _style_table(fig, ax, table, headers, n_rows)

    for col_idx, w in enumerate(col_widths):
        for row_idx in range(n_rows + 1):
            table[row_idx, col_idx].set_width(w)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Table saved to: {save_path}")

    plt.show()
    mpl.rcParams['font.family'] = original_font


def plot_avg_delay_bar_chart(results, save_path=None, text_fontsize=16):
    """
    Bar chart of average delay per vehicle for each day. The red bar is the
    total uncontrolled delay; the green overlay is the portion of that delay
    which was removed by control (i.e. uncontrolled * CC%), so the green
    fraction of each red bar visually equals that day's CC%. The exposed red
    remainder on top is the delay left over even with control. Styling
    follows src/generate_demand_synthetic.py's delay_reduction plot.
    """
    import matplotlib as mpl

    original_font = mpl.rcParams['font.family']
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']

    dates        = [format_date_label(r['date']) for r in results]
    uncontrolled = np.array([r['avg_delay_uncontrolled'] for r in results])
    controlled   = np.array([r['avg_delay_controlled'] for r in results])
    cc           = np.array([r['cc'] for r in results])
    reduced      = uncontrolled - controlled

    scenarios = np.arange(1, len(dates) + 1)
    width = 0.85

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(dates)), 5))
    plt.grid()
    ax.set_axisbelow(True)

    ax.bar(scenarios, uncontrolled, label="Delay without control",       color="#d33b19", width=width)
    ax.bar(scenarios, reduced,      label="Delay reduced by control", color="#4e9858", width=width)

    for i in range(len(scenarios)):
        ax.text(scenarios[i], uncontrolled[i] + 0.02 * np.max(uncontrolled),
                f"{cc[i]:.0f}", ha='center', va='bottom',
                fontsize=text_fontsize - 4, fontname="Times New Roman", fontweight='bold')

    ax.set_xlabel("Date", fontsize=text_fontsize, fontname="Times New Roman")
    ax.set_ylabel("Average delay per vehicle (min)", fontsize=text_fontsize, fontname="Times New Roman")
    ax.set_xticks(scenarios)
    ax.set_xticklabels(dates, rotation=45, ha='center')
    ax.legend(prop={'family': 'Times New Roman', 'size': text_fontsize})

    ax.tick_params(labelsize=text_fontsize - 4)
    for label in ax.get_xticklabels():
        label.set_fontname('Times New Roman')
    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')

    ax.set_ylim(0, np.max(uncontrolled) * 1.15)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        print(f"Bar chart saved to: {save_path}")

    plt.show()
    mpl.rcParams['font.family'] = original_font


def plot_static_dynamic_table(results, save_path=None):
    """Table with static and dynamic columns side by side."""
    import matplotlib as mpl

    original_font = mpl.rcParams['font.family']
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']

    dates            = [format_date_label(r['date']) for r in results]
    static_sim_mapes = [f"{r['static_sim_mape']:.2f}%" for r in results]
    static_tts_mapes = [f"{r['static_tts_mape']:.2f}%" for r in results]
    static_ccs       = [f"{r['static_cc']:.2f}%" for r in results]
    dyn_sim_mapes    = [f"{r['dyn_sim_mape']:.2f}%" for r in results]
    dyn_tts_mapes    = [f"{r['dyn_tts_mape']:.2f}%" for r in results]
    dyn_ccs          = [f"{r['dyn_cc']:.2f}%" for r in results]

    columns = [
        'Date',
        'Static Sim MAPE', 'Static TTS MAPE', 'Static CC (%)',
        'Dynamic Sim MAPE', 'Dynamic TTS MAPE', 'Dynamic CC (%)',
    ]
    rows = list(zip(dates, static_sim_mapes, static_tts_mapes, static_ccs,
                    dyn_sim_mapes, dyn_tts_mapes, dyn_ccs))
    n_rows = len(rows)

    row_height    = 0.5
    header_height = 0.6
    fig_height    = header_height + n_rows * row_height + 0.8
    fig, ax = plt.subplots(figsize=(13, fig_height))
    ax.axis('off')

    table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, row_height / 0.22)

    _style_table(fig, ax, table, columns, n_rows)

    col_widths = [0.10, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
    for col_idx, w in enumerate(col_widths):
        for row_idx in range(n_rows + 1):
            table[row_idx, col_idx].set_width(w)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Table saved to: {save_path}")

    plt.show()
    mpl.rcParams['font.family'] = original_font


# Publication-style headers for static_results_to_latex: (label, unit).
# unit=None renders as a single line; otherwise the unit sits on its own
# line below the label via \shortstack. Falls back to the STATIC_COLUMNS
# header (no unit) if a column has no override here.
LATEX_HEADERS = {
    'date':                    ('Date',                      None),
    'sim_mape':                ('Simulation Error',          '(MAPE)'),
    'tts_mape':                ('TTS Error',                  '(MAPE)'),
    'cc':                      ('Controllable Congestion',    '(\\%)'),
    'uncontrolled_tts':        ('Uncontrolled TTT',           '(veh-hrs)'),
    'controlled_tts':          ('Controlled TTT',             '(veh-hrs)'),
    'avg_tt_reduced':          ('Avg.\\ TT Reduced / Veh',    '(min)'),
    'avg_delay_uncontrolled':  ('Avg.\\ Uncontrolled Delay / Veh', '(min)'),
    'avg_delay_controlled':    ('Avg.\\ Controlled Delay / Veh',   '(min)'),
}


def _latex_header_cell(col_key):
    label, unit = LATEX_HEADERS.get(col_key, (STATIC_COLUMNS[col_key][0], None))
    if unit:
        return rf"\textbf{{\shortstack{{{label} \\ {unit}}}}}"
    return rf"\textbf{{{label}}}"


# Value-formatter overrides for static_results_to_latex only (the PNG table
# via plot_static_table keeps STATIC_COLUMNS' 2-decimal formatting).
LATEX_VALUE_FORMATTERS = {
    'uncontrolled_tts': lambda r: f"{round(r['uncontrolled_tts'])}",
    'controlled_tts':   lambda r: f"{round(r['controlled_tts'])}",
}


def static_results_to_latex(results, columns=None,
                            caption="Simulation error and controllable congestion by date",
                            label="tab:cc_static"):
    """
    Render `results` (from run_static_analysis) as a booktabs LaTeX table
    string. `columns` picks which STATIC_COLUMNS fields appear (defaults to
    ['date', 'sim_mape', 'cc']); the Date column is left-aligned, all others
    centered. Column headers with a unit (e.g. "(min)") render it on its own
    line below the label.
    """
    if columns is None:
        columns = ['date', 'sim_mape', 'cc']
    unknown = [c for c in columns if c not in STATIC_COLUMNS]
    if unknown:
        raise ValueError(f"Unknown column(s) {unknown}. Available: {list(STATIC_COLUMNS)}")

    header_cells = [_latex_header_cell(c) for c in columns]
    align        = 'l' + 'c' * (len(columns) - 1)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(header_cells) + r" \\",
        r"\midrule",
    ]
    for r in results:
        row_vals = [
            LATEX_VALUE_FORMATTERS.get(c, STATIC_COLUMNS[c][1])(r).replace('%', r'\%')
            for c in columns
        ]
        lines.append(" & ".join(row_vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    return "\n".join(lines)


def run_static_analysis(dates, columns=None, save_path=None,
                        bar_chart_save_path=None):
    """
    Run the static-only sweep over `dates` and save/plot the results table
    and the average-delay-per-vehicle bar chart.

    `columns` picks which fields (and in what order) appear in the table —
    see STATIC_COLUMNS for available keys. Defaults to DEFAULT_STATIC_COLUMNS.
    """
    # Resolved here rather than in the signature so the defaults follow
    # FIGS_ROOT instead of the caller's working directory.
    save_path = fig("i24_cc_table.png") if save_path is None else save_path
    bar_chart_save_path = (fig("i24_avg_delay_bar.png")
                           if bar_chart_save_path is None else bar_chart_save_path)

    results = []
    for date in dates:
        print(f"\n── {date} ──────────────────────────────────")
        metrics = run_static_for_date(date)
        if metrics is None:
            print(f"  Skipping {date} (missing static VSL).")
            continue

        print(
            f"  Static — Sim MAPE: {metrics['sim_mape']:.2f}%  "
            f"TTS MAPE: {metrics['tts_mape']:.2f}%  "
            f"CC: {metrics['cc']:.2f}%"
        )
        results.append({'date': date, **metrics})

    if results:
        plot_static_table(results, columns=columns, save_path=save_path)
        plot_avg_delay_bar_chart(results, save_path=bar_chart_save_path)
    else:
        print("No results to display.")
    return results


def run_static_dynamic_analysis(dates, save_path=None):
    """Run the static+dynamic sweep over `dates` and save/plot the results table."""
    save_path = fig("i24_cc_table_withdyn.png") if save_path is None else save_path
    results = []
    for date in dates:
        print(f"\n── {date} ──────────────────────────────────")
        static_metrics, dyn_metrics = run_static_and_dynamic_for_date(date)
        if static_metrics is None:
            print(f"  Skipping {date} (missing static VSL).")
            continue
        if dyn_metrics is None:
            print(f"  Skipping {date} (missing dynamic VSL).")
            continue

        print(
            f"  Static  — Sim MAPE: {static_metrics['sim_mape']:.2f}%  "
            f"TTS MAPE: {static_metrics['tts_mape']:.2f}%  "
            f"CC: {static_metrics['cc']:.2f}%"
        )
        print(
            f"  Dynamic — Sim MAPE: {dyn_metrics['sim_mape']:.2f}%  "
            f"TTS MAPE: {dyn_metrics['tts_mape']:.2f}%  "
            f"CC: {dyn_metrics['cc']:.2f}%"
        )

        results.append({
            'date':            date,
            'static_sim_mape': static_metrics['sim_mape'],
            'static_tts_mape': static_metrics['tts_mape'],
            'static_cc':       static_metrics['cc'],
            'dyn_sim_mape':    dyn_metrics['sim_mape'],
            'dyn_tts_mape':    dyn_metrics['tts_mape'],
            'dyn_cc':          dyn_metrics['cc'],
        })

    if results:
        plot_static_dynamic_table(results, save_path=save_path)
    else:
        print("No results to display.")
    return results
