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
    """Free-flow TTS charging every mainline arrival one full corridor traversal.

    Superseded by `ff_tts_vkm`, which also counts vehicles entering at on-ramps and
    charges each vehicle only the distance it travels. Kept so the numbers in results
    produced before that change can still be reproduced.
    """
    num_vehicles   = sum(demand_profile) * time_step
    ff_travel_time = sum(length / v for v in v_free)
    return num_vehicles * ff_travel_time

def ff_tts_vkm(density, velocity, lane_counts, time_step, length, v_free):
    """Free-flow TTS for the travel a run actually served, in vehicle-hours.

    Vehicle-km in each segment, charged at that segment's free-flow speed:

        veh_km_i = sum_t rho_i(t) v_i(t) lanes_i * time_step * length
        TTS_ff   = sum_i veh_km_i / v_free_i

    Measuring travel rather than counting trips is what makes this correct when
    ramps are present: a vehicle entering at an on-ramp contributes vehicle-km only
    to the segments it uses, and one leaving at an off-ramp stops contributing there.
    `get_ff_tts` instead counts mainline arrivals only and charges each the whole
    corridor, which on I-24 ignores the on-ramp inflows (roughly twice the mainline
    demand) and overcharges everything exiting at an off-ramp.

    Args:
        density, velocity: (time_steps, num_segments) per-lane fields from one run.
        lane_counts: (num_segments,) lanes per segment.
        time_step, length: hours per step, km per segment.
        v_free: (num_segments,) free-flow speeds, or (time_steps, num_segments) for a
            time-varying calibration, in which case the max over time is used.

    Returns:
        free flow TTS (vehicle-hours)
    """
    v_free = np.max(v_free, axis=0) if np.ndim(v_free) == 2 else v_free
    veh_km = (density * velocity * lane_counts).sum(axis=0) * time_step * length
    return float((veh_km / v_free).sum())

def vehicles_served(density, velocity, lane_counts, beta, time_step):
    """Vehicles that left the corridor over a run.

    A vehicle departs either out of the last segment or at an off-ramp, which takes
    `q_i * beta_i / (1 - beta_i)` because `density_dynamics` removes `outflow/(1-beta)`.
    `get_num_veh` counts mainline arrivals only, so on I-24 it misses the on-ramp
    inflows -- roughly twice the mainline demand -- and per-vehicle metrics built on it
    are too large by that factor. Over a run that drains (`until_ff=True`) this equals
    the vehicles that entered.
    """
    q = density * velocity * lane_counts
    departures = q[:, -1] + (q * beta / (1 - beta)).sum(axis=1)
    return float(departures.sum() * time_step)

def free_flow_departures(day, params, n_steps):
    """Cumulative arrivals and departures if every vehicle ran at v_free.

    Same arrivals and splits as the simulated runs, on a grid of `n_steps` steps of
    `time_step`. Each arrival stream (mainline demand, each on-ramp, and the vehicles
    already in the corridor at t=0) is spread over its free-flow exit times. METANET
    removes ramp traffic as a sink on a segment's own outflow, so those vehicles cover
    no distance in the segment they leave at: split first, then charge the traversal
    time. Returns (arrivals, departures), both cumulative vehicle counts.

    Ported from experiments/capacity_drop_diagnostics.ipynb.
    """
    n_seg = day["num_segments"]
    lanes = np.asarray(day["lane_counts"], dtype=float)
    t_grid = np.arange(n_steps) * time_step                # hours
    tau = L / np.asarray(params["v_free"], dtype=float)    # free-flow traversal, hours
    beta, r = np.asarray(params["beta"], float), np.asarray(params["r"], float)
    n_dem = day["data_inflow"].shape[0]
    D = np.zeros(n_steps)
    A = np.zeros(n_steps)

    def on_grid(per_step_veh):
        c = np.cumsum(per_step_veh)
        out = np.full(n_steps, c[-1])
        out[: len(c)] = c
        return out

    def add_stream(cum_arrivals, enter_seg, head_start=0.0):
        nonlocal D, A
        A = A + cum_arrivals
        surv, elapsed = 1.0, head_start
        for j in range(enter_seg, n_seg):
            if beta[j] > 0:
                D = D + surv * beta[j] * np.interp(t_grid - elapsed, t_grid, cum_arrivals, left=0.0)
            surv *= 1 - beta[j]
            elapsed += tau[j]
        D = D + surv * np.interp(t_grid - elapsed, t_grid, cum_arrivals, left=0.0)

    add_stream(on_grid(day["data_inflow"] * time_step), 0)
    n0 = day["init_state"][0] * lanes * L                  # vehicles already in each segment
    for i in range(n_seg):
        if r[i] > 1e-6:
            add_stream(on_grid(np.full(n_dem, r[i] * time_step)), i)
        if n0[i] > 1e-9:                                   # released from mid-segment
            add_stream(on_grid(np.r_[n0[i], np.zeros(n_dem - 1)]), i, head_start=-0.5 * tau[i])
    return A, D


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

def is_holiday(date_str):
    """'11_24' -> True."""
    month, day = (int(p) for p in date_str.split('_'))
    return (month, day) in HOLIDAY_DATES

def format_date_label(date_str, year=2022):
    """'11_30' -> '11/30 (Wed)'; holiday dates -> '11/24 (Holiday)'."""
    month, day = (int(p) for p in date_str.split('_'))
    if is_holiday(date_str):
        tag = 'Holiday'
    else:
        tag = datetime.date(year, month, day).strftime('%a')
    return f"{month:02d}/{day:02d} ({tag})"


def load_day_data(date, cal_path = "calibration_static/fixed_ramping"):
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

    static_cal_path = f"{root_path}/{cal_path}"
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
    num_timesteps = v_trimmed.shape[0]
    p_sim, v_sim, _, tts_sim = run_metanet_sim(
        time_step, L, init_state, data_inflow, ds_density_norm,
        model_params, lanes=lane_dict, vsl_speeds=None,
        plotting=True, real_data=True, until_ff=True
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
        plotting=True, real_data=False, until_ff=True
    )
    v_opt = v_opt[:-1, :]
    p_opt = p_opt[:-1, :]

    # One free-flow baseline for both runs, taken from the uncontrolled one.
    ff_ttt    = ff_tts_vkm(p_sim, v_sim, lane_counts, time_step, L, model_params['v_free'])
    delay     = tts_sim - ff_ttt
    opt_delay = tts_opt - ff_ttt
    cc_unclipped = (delay - opt_delay) / delay * 100
    cc = np.clip(cc_unclipped, 0, 100)  # avoid negative CC due to numerical issues
    sim_mape_ = mape(v_trimmed, v_sim[0:num_timesteps, :])

    gt_tt      = time_step * sum(np.sum(d_trimmed[:, i]) * lane_counts[i] * L
                                 for i in range(num_segments))
    metanet_tt = time_step * sum(np.sum(p_sim[0:num_timesteps, i])     * lane_counts[i] * L
                                 for i in range(num_segments))
    tts_mape_  = np.abs(gt_tt - metanet_tt) / gt_tt * 100

    # Everyone the corridor served, ramp entries included, to match what TTS counts.
    num_vehicles    = vehicles_served(p_sim, v_sim, lane_counts,
                                      model_params['beta'], time_step)
    avg_tt_reduced_ = (tts_sim - tts_opt) / num_vehicles * 60  # minutes/vehicle

    return {
        'sim_mape':               sim_mape_,
        'tts_mape':                tts_mape_,
        'cc':                      cc,
        'cc_unclipped':            cc_unclipped,  # negative = policy increased delay
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


# Calibration bounds from `metanet_param_fit` in src/optimization_utils.py (cold
# start). Keys follow METANET_Params, so rho_crit is "p_crit". Perturbed
# parameters are clipped to these so they stay where the model was fit.
CALIBRATION_BOUNDS = {
    'tau':      (10 / 3600, 60 / 3600),
    'K':        (5.0, 60.0),
    'eta_high': (10.0, 90.0),
    'p_crit':   (15.0, 95.0),
    'v_free':   (70.0, 150.0),
    'a':        (0.5, 5.0),
}


def perturb_params(params, rng, delta=0.10, segment_delta_frac=0.5, clip_params=True):
    """
    Randomly perturb the calibrated METANET parameters (those in
    CALIBRATION_BOUNDS), leaving q_capacity, r, beta and gamma untouched.

    Each parameter is multiplied by (1 + shared) * (1 + local):
      shared ~ U(-delta, delta), one draw per parameter for the whole corridor
               (systematic error);
      local  ~ U(-delta * segment_delta_frac, +delta * segment_delta_frac), one
               draw per segment (local error).
    The result is clipped to CALIBRATION_BOUNDS, except that a calibrated value
    already outside its bound (e.g. a `tau` stored 1e-12 below the lower bound)
    is allowed to stay where it is, so delta=0 returns the nominal parameters
    exactly. Pass `clip_params=False` to skip the clipping entirely. Returns a
    new dict.
    """
    segment_delta = delta * segment_delta_frac
    perturbed = dict(params)
    for key, (lo, hi) in CALIBRATION_BOUNDS.items():
        values = np.asarray(params[key], dtype=float)
        shared = 1 + rng.uniform(-delta, delta)
        local  = 1 + rng.uniform(-segment_delta, segment_delta, size=values.shape[-1])
        new_values = values * shared * local
        if clip_params:
            new_values = np.clip(new_values, np.minimum(lo, values), np.maximum(hi, values))
        perturbed[key] = new_values
    return perturbed


def monte_carlo_cc(date, n_samples=50, delta=0.10, segment_delta_frac=0.5,
                   max_mape=100.0, mape_tolerance=None, clip_cc=False,
                   clip_params=True, seed=None):
    """
    Monte Carlo sensitivity of controllable congestion to parameter error, for
    one date, holding the VSL policy fixed (no re-optimization).

    The nominal-optimal policy is optimized once on the nominal calibration. For
    each of `n_samples` perturbed parameter sets (see `perturb_params`), the
    uncontrolled run, the controlled run and the free-flow baseline are all
    re-simulated, so CC is recomputed from that model's own TTS_nocontrol,
    TTS_control and ff_tts. A sample whose sim MAPE is not below `max_mape` (or
    that produces a non-finite CC) is printed and thrown out.

    Fit screening: if `mape_tolerance` is given, the limit becomes the nominal
    sim MAPE plus `mape_tolerance` (percentage points) instead of `max_mape`, so
    only models that fit the data about as well as the nominal one are kept.

    CC is reported unclipped by default, so a negative value means the nominal
    policy made delay worse under that perturbed model. Pass `clip_cc=True` to
    clip to [0, 100] as `run_one_day`'s own `cc` does. Pass `clip_params=False`
    to let perturbed parameters leave the calibration bounds.

    Returns:
        nominal_cc: CC (%) under the nominal calibration.
        cc_samples: list of CC (%) for the kept samples (at most `n_samples`).
    """
    cc_key = 'cc' if clip_cc else 'cc_unclipped'
    day = load_day_data(date)
    vsl_path = f"{RESULTS_ROOT}/i24_{date}/calibration_static/fixed_ramping/optimal_vsl.npy"

    def simulate(params):
        return run_one_day(
            day['init_state'], day['data_inflow'], day['ds_density_norm'],
            params, day['lane_dict'], day['lane_counts'],
            day['v_trimmed'], day['d_trimmed'], day['num_segments'], vsl_path
        )

    nominal = simulate(day['static_params'])
    if nominal is None:
        raise FileNotFoundError(f"No optimal VSL for {date}: {vsl_path}")
    nominal_cc = float(nominal[cc_key])
    mape_limit = max_mape if mape_tolerance is None else nominal['sim_mape'] + mape_tolerance

    rng = np.random.default_rng(seed)
    cc_samples = []
    for i in range(n_samples):
        params  = perturb_params(day['static_params'], rng, delta, segment_delta_frac, clip_params)
        metrics = simulate(params)

        if not metrics['sim_mape'] < mape_limit:  # also catches NaN
            print(f"  Sample {i + 1}/{n_samples} thrown out: "
                  f"sim MAPE {metrics['sim_mape']:.1f}% is not below {mape_limit:.1f}%")
            continue
        if not np.isfinite(metrics[cc_key]):
            print(f"  Sample {i + 1}/{n_samples} thrown out: CC is not finite")
            continue
        cc_samples.append(float(metrics[cc_key]))

    print(f"  {date}: kept {len(cc_samples)}/{n_samples} samples (delta={delta:.0%}); "
          f"nominal CC {nominal_cc:.1f}%")
    return nominal_cc, cc_samples


# Order and Greek-letter titles for the parameter-sensitivity grid.
SENSITIVITY_PARAMS = ['eta_high', 'K', 'tau', 'p_crit', 'v_free', 'a']
SENSITIVITY_TITLES = {
    'eta_high': r'$\eta$',
    'K':        r'$\kappa$',
    'tau':      r'$\tau$',
    'p_crit':   r'$\rho_{\mathrm{cr}}$',
    'v_free':   r'$v_{\mathrm{free}}$',
    'a':        r'$a$',
}


def one_at_a_time_cc(date, perturb_pct=None, return_mape=False):
    """
    CC as each calibrated parameter is scaled, on its own, by (1 + pct) in every
    segment while all others stay nominal; the VSL policy is held fixed.

    `perturb_pct` is in fractions (default: -1% to +1% in 0.05% steps, which
    includes exactly 0). Returns (perturb_pct, results, nominal_cc) where
    `results[param]` is the array of unclipped CC (%) over `perturb_pct`.

    With `return_mape=True` it returns (perturb_pct, results, nominal_cc,
    mape_results, nominal_mape), where `mape_results[param]` is the sim MAPE (%)
    over `perturb_pct` and `nominal_mape` is the nominal run's sim MAPE.
    """
    if perturb_pct is None:
        perturb_pct = np.arange(-20, 21) * 0.0005
    day = load_day_data(date)
    vsl_path = f"{RESULTS_ROOT}/i24_{date}/calibration_static/fixed_ramping/optimal_vsl.npy"
    params = day['static_params']

    def simulate(model_params):
        return run_one_day(
            day['init_state'], day['data_inflow'], day['ds_density_norm'],
            model_params, day['lane_dict'], day['lane_counts'],
            day['v_trimmed'], day['d_trimmed'], day['num_segments'], vsl_path
        )

    nominal = simulate(params)
    if nominal is None:
        raise FileNotFoundError(f"No optimal VSL for {date}: {vsl_path}")

    results, mape_results = {}, {}
    for key in SENSITIVITY_PARAMS:
        base = np.asarray(params[key], dtype=float)
        ccs, mapes = [], []
        for pct in perturb_pct:
            scaled = dict(params)
            scaled[key] = base * (1 + pct)
            metrics = simulate(scaled)
            ccs.append(float(metrics['cc_unclipped']))
            mapes.append(float(metrics['sim_mape']))
        results[key] = np.array(ccs)
        mape_results[key] = np.array(mapes)

    out = (np.asarray(perturb_pct), results, float(nominal['cc_unclipped']))
    if return_mape:
        out += (mape_results, float(nominal['sim_mape']))
    return out


def plot_param_sensitivity(perturb_pct, results, nominal_cc, save_path=None,
                           g_fontsize=20, mape_results=None, nominal_mape=None):
    """
    2x3 grid, one panel per parameter: CC (%) against the parameter's perturbation
    (%), with the nominal CC marked at x=0. Styling follows plotting.py in
    dyn-traffic-macrosim (serif/Times, Greek titles, dotted reference lines, light
    grid, x labels on the bottom row and y labels on the outer columns only).

    Pass `mape_results` (param -> sim MAPE (%) array over `perturb_pct`, from
    `one_at_a_time_cc(..., return_mape=True)`) to also draw the sim MAPE in each
    panel on a right-hand axis; `nominal_mape` marks its value at x=0.
    """
    import matplotlib as mpl
    from matplotlib.lines import Line2D

    show_mape = mape_results is not None
    # CC-only plots keep the original orange; with MAPE shown, CC is blue and MAPE orange.
    CC_COLOR, MAPE_COLOR = ("blue", "orange") if show_mape else ("orange", None)
    tick_fs, label_fs, title_fs = g_fontsize, g_fontsize + 1, g_fontsize + 8
    perturb_x = np.asarray(perturb_pct) * 100
    n_keys = len(SENSITIVITY_PARAMS)
    nrows, ncols = 2, 3

    def shared_limits(arrays, nominal):
        lo = min(np.nanmin(a) for a in arrays)
        hi = max(np.nanmax(a) for a in arrays)
        if nominal is not None:
            lo, hi = min(lo, nominal), max(hi, nominal)
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        return lo - pad, hi + pad

    # Shared y-limits so panels are directly comparable.
    cc_lim = shared_limits(results.values(), nominal_cc)
    if show_mape:
        mape_lim = shared_limits(mape_results.values(), nominal_mape)

    with mpl.rc_context({
        "font.family": "serif",
        "font.serif": "Times New Roman",
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "text.usetex": False,
    }):
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        axes = np.atleast_1d(axes).ravel()

        for plot_idx, (ax, key) in enumerate(zip(axes[:n_keys], SENSITIVITY_PARAMS)):
            is_bottom_row = plot_idx // ncols == nrows - 1
            is_left_col   = plot_idx % ncols == 0
            is_right_col  = plot_idx % ncols == ncols - 1

            ax.set_facecolor('white')
            ax.plot(perturb_x, results[key], color=CC_COLOR, linewidth=2)
            ax.plot(0, nominal_cc, marker='o', color='black', markersize=6, zorder=3)

            ax.axvline(0, color='black', linewidth=0.8, linestyle=':', alpha=0.7)
            ax.axhline(nominal_cc, color='black', linewidth=0.8, linestyle=':', alpha=0.7)
            ax.grid(True, color='lightgrey', linewidth=0.5, linestyle='-', alpha=0.8)
            ax.set_axisbelow(True)

            ax.set_title(SENSITIVITY_TITLES[key], fontsize=title_fs, fontweight='normal', pad=6)
            ax.set_xlim(perturb_x[0], perturb_x[-1])
            ax.set_ylim(*cc_lim)
            ax.tick_params(labelsize=tick_fs)

            if is_bottom_row:
                ax.set_xlabel('Perturbation (%)', fontsize=label_fs)
            else:
                ax.tick_params(axis='x', which='both', labelbottom=False, bottom=False)

            if is_left_col:
                ax.set_ylabel('CC (%)', fontsize=label_fs,
                              color=CC_COLOR if show_mape else 'black')
            else:
                ax.tick_params(axis='y', which='both', labelleft=False)
            if show_mape:
                ax.tick_params(axis='y', labelcolor=CC_COLOR)

            for spine in ax.spines.values():
                spine.set_edgecolor('#cccccc')
                spine.set_linewidth(0.8)

            if show_mape:
                ax2 = ax.twinx()
                # Draw the CC axes above the MAPE axes so CC is plotted on top.
                ax.set_zorder(ax2.get_zorder() + 1)
                ax.patch.set_visible(False)
                ax2.plot(perturb_x, mape_results[key], color=MAPE_COLOR, linewidth=2)
                if nominal_mape is not None:
                    ax2.plot(0, nominal_mape, marker='o', color=MAPE_COLOR,
                             markeredgecolor='black', markersize=6, zorder=3)
                ax2.set_ylim(*mape_lim)
                ax2.tick_params(axis='y', labelsize=tick_fs, labelcolor=MAPE_COLOR)
                for spine in ax2.spines.values():
                    spine.set_edgecolor('#cccccc')
                    spine.set_linewidth(0.8)
                if is_right_col:
                    ax2.set_ylabel('Sim MAPE (%)', fontsize=label_fs, color=MAPE_COLOR)
                else:
                    ax2.tick_params(axis='y', which='both', labelright=False, right=False)

        for ax in axes[n_keys:]:
            ax.set_visible(False)

        if show_mape:
            fig.legend(
                handles=[Line2D([0], [0], color=CC_COLOR, linewidth=2, label='CC'),
                         Line2D([0], [0], color=MAPE_COLOR, linewidth=2, label='Sim MAPE')],
                loc='lower center', ncol=2, bbox_to_anchor=(0.5, 1.0),
                frameon=True, fontsize=label_fs,
            )

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Saved plot to {save_path}")
        plt.show()
    return fig


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


# Publication-style headers for static_results_to_latex: (line 1, line 2).
# line 2 = None renders a single line; otherwise both lines are stacked via
# \shortstack. Falls back to the STATIC_COLUMNS header if a column has no
# override here.
LATEX_HEADERS = {
    'date':                    ('Date',               None),
    'sim_mape':                ('Simulation Error',   '(\\%)'),
    'tts_mape':                ('TTS Error',          '(\\%)'),
    'cc':                      ('Controllable',       'Congestion (\\%)'),
    'uncontrolled_tts':        ('Uncontrolled',       'TTS (veh-hrs)'),
    'controlled_tts':          ('Controlled',         'TTS (veh-hrs)'),
    'avg_tt_reduced':          ('TT Reduced',         'per Veh (min)'),
    'avg_delay_uncontrolled':  ('Uncontrolled Delay', 'per Veh (min)'),
    'avg_delay_controlled':    ('Controlled Delay',   'per Veh (min)'),
}


def _latex_header_cell(col_key):
    line1, line2 = LATEX_HEADERS.get(col_key, (STATIC_COLUMNS[col_key][0], None))
    if line2:
        return rf"\textbf{{\shortstack{{{line1} \\ {line2}}}}}"
    return rf"\textbf{{{line1}}}"


# Value-formatter overrides for static_results_to_latex only; every other
# column uses STATIC_COLUMNS' formatter (same rounding as the PNG table).
LATEX_VALUE_FORMATTERS = {
    'cc': lambda r: rf"\textbf{{{r['cc']:.0f}%}}",
}


def static_results_to_latex(results, columns=None,
                            caption="Simulation error and controllable congestion by date",
                            label="tab:cc_static"):
    """
    Render `results` (from run_static_analysis) as a booktabs LaTeX table
    string. `columns` picks which STATIC_COLUMNS fields appear (defaults to
    ['date', 'sim_mape', 'cc']); the Date column is left-aligned, all others
    centered. Headers are two-line stacks (see LATEX_HEADERS), CC values are
    bold, and a \\midrule separates runs of holiday and non-holiday dates.
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
    prev_holiday = None
    for r in results:
        holiday = is_holiday(r['date'])
        if prev_holiday is not None and holiday != prev_holiday:
            lines.append(r"\midrule")
        prev_holiday = holiday

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
    bar_chart_save_path = (fig("i24_avg_delay_bar.pdf")
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
