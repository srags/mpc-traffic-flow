import datetime
import numpy as np

from traffic_flow.config import CalRef, CalSource
from traffic_flow.inputs.scenario import load_params, load_scenario
from traffic_flow.pipeline import init_state
from traffic_flow.results.console import colored
from traffic_flow.results.io import load_runs
from traffic_flow.results.plots import Plotter
from traffic_flow.model.parameters import load_metanet_params
from ..model.simulation import METANET_Simulator
from tabulate import tabulate

# ── Simulation parameters ────────────────────────────────────────────────────
L          = 0.4
time_step  = 10 / 3600
total_time = 1   # hours

# Re-exported under their historical names: experiments/i24_tsd.py
from ..paths import REPO_DIR, cut_repo
from .analysis import get_num_veh, get_ff_tts

# Dates (month, day) that are holidays rather than a regular weekday, tagged
# "Holiday" instead of a day name (Nov 24/25, 2022 = Thanksgiving + the day after).
HOLIDAY_DATES = {(11, 24), (11, 25)}

def format_date_label(date_str, year=2022):
    """'11_30' -> '11/30 (Wed)'; holiday dates -> '11/24 (Holiday)'."""
    month, day = (int(p) for p in date_str.split('_'))
    if (month, day) in HOLIDAY_DATES: tag = 'Holiday'
    else: tag = datetime.date(year, month, day).strftime('%a')
    return f"{month:02d}/{day:02d} ({tag})"

def run_one_day(init_state, data_inflow, ds_density_norm, model_params,
                lane_dict, lane_counts, v_trimmed, d_trimmed,
                num_segments, vsl):
    """
    Run baseline + VSL simulations for one calibration variant.
    Returns a dict of metrics, or None if the VSL file is missing.
    """
    sim = METANET_Simulator(T=time_step, l=L, params=model_params, lanes=lane_dict, real_data=True)
    p_sim, v_sim, _, tts_sim = sim.run_with_history(data_inflow, ds_density_norm, init_state)
    p_sim = p_sim[:-1, :]
    v_sim = v_sim[:-1, :]
    print(f"{tts_sim=}")

    sim.real_data = False
    p_opt, v_opt, _, tts_opt = sim.run_with_history(data_inflow, ds_density_norm, init_state, vsl)

    v_opt = v_opt[:-1, :]
    p_opt = p_opt[:-1, :]
    print(f"{tts_opt=}")

    v_free = model_params['v_free']
    ff_ttt    = get_ff_tts(data_inflow, time_step, L,
                           np.max(v_free, axis=0) if len(v_free.shape)==2 else v_free)
    delay     = tts_sim - ff_ttt
    opt_delay = tts_opt - ff_ttt
    cc        = (delay - opt_delay) / delay * 100
    cc = np.clip(cc, 0, 100)  # avoid negative CC due to numerical issues
    sim_mape_ = np.mean(np.abs((v_trimmed - v_sim) / v_trimmed)) * 100

    gt_tt      = time_step * sum(np.sum(d_trimmed[:, i]) * lane_counts[i] * L for i in range(num_segments))
    print(f"{gt_tt=}")
    metanet_tt = time_step * sum(np.sum(p_sim[:, i])     * lane_counts[i] * L for i in range(num_segments))
    print(f"{metanet_tt=}")
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


def run_static_and_dynamic_for_date(date):
    """Run both the static- and dynamic-calibration VSL simulations for one date."""
    scenario = load_scenario("i24", date)
    spec, traffic = scenario.spec, scenario.traffic
    params = load_params(scenario, CalRef(CalSource.FIXED_RAMPS, interval=None))
    runs = load_runs("i24", date, CalRef(CalSource.FIXED_RAMPS, interval=None), study=None)
                        #  where=lambda config: all([config.init_fixed == 150.0]))
    print(f"{date}: {len(runs)} runs found, picking one")
    vsl = runs[0].optimization.vsl
    
    print("  Running static calibration simulations...")
    static_metrics = run_one_day(
        init_state(traffic), traffic.inflow, traffic.downstream_density,
        params, dict(enumerate(traffic.lanes)), traffic.lanes,
        traffic.velocity, traffic.density, spec.num_segments, vsl
    )
    if static_metrics is None:
        return None, None

    dyn_cal_path = REPO_DIR / "data" / "i24" / f"i24_{date}" / "calibration_dynamic"
    dyn_params = load_metanet_params(
        path=dyn_cal_path,
        num_timesteps=spec.time_steps,
        num_segments=spec.num_segments,
        control_h=90
    )

    dyn_vsl_path = REPO_DIR / "results_bu" / "i24" / f"i24_{date}" / "calibration_dynamic" / "control_h_90" / "optimal_vsl.npy"
    try: vsl = np.load(dyn_vsl_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"  VSL file not found: {dyn_vsl_path}")

    print("  Running dynamic calibration simulations...")
    dyn_metrics = run_one_day(
        init_state(traffic), traffic.inflow, traffic.downstream_density,
        dyn_params, dict(enumerate(traffic.lanes)), traffic.lanes,
        traffic.velocity, traffic.density, spec.num_segments, vsl
    )

    return static_metrics, dyn_metrics


def plot_static_dynamic_table(results, save_path=None):
    """Table with static and dynamic columns side by side."""
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

    row_height    = 0.5
    header_height = 0.6
    fig_height    = header_height + len(rows) * row_height + 0.8
    p = Plotter(1, 1, figsize=(13, fig_height))
    p[0].axis('off')

    p[0].table(cellText=rows, colLabels=columns, loc='center', cellLoc='center')

    if save_path:
        p.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Table saved to {colored(cut_repo(save_path), 'green', 'bold')}")


def run_static_dynamic_analysis(dates, save_path=None):
    """Run the static+dynamic sweep over `dates` and save/plot the results table."""
    save_path = REPO_DIR / "figs" / "i24_cc_table_withdyn.png" if save_path is None else save_path
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

        print(tabulate(tablefmt='grid', headers=['', 'Sim MAPE %', 'TTS MAPE %', 'CC %'], 
                        tabular_data=[
                            ["Static", static_metrics['sim_mape'], static_metrics['tts_mape'], static_metrics['cc']],
                            ["Dynamic", dyn_metrics['sim_mape'], dyn_metrics['tts_mape'], dyn_metrics['cc']]
                        ]))

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
