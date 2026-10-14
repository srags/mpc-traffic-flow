"""Numerical CC report inputs; no plotting or file writes."""
import numpy as np
from ..types import *
from ..inputs.scenario import Scenario

def mape(observed: np.ndarray | float, predicted: np.ndarray | float):
    return (np.abs((observed - predicted) / observed) * 100).mean()

def rmse(observed: np.ndarray | float, predicted: np.ndarray | float):
    return np.sqrt(np.mean((observed - predicted) ** 2))

def get_ff_tts(demand_profile, time_step, length, v_free):
    """Free-flow TTS charging every mainline arrival one full corridor traversal.

    Superseded by `ff_tts_vkm`, which also counts vehicles entering at on-ramps and
    charges each vehicle only the distance it travels. Kept so the numbers in results
    produced before that change can still be reproduced.
    """
    num_vehicles  = sum(demand_profile) * time_step
    ff_travel_time = sum(length / v for v in v_free)  # hours
    return num_vehicles * ff_travel_time

def ff_tts_vkm(density: time_space, velocity: time_space, lane_counts: space_vec, 
               time_step: hr, length, v_free: time_space | space_vec):
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

def vehicles_served(density: time_space, velocity: time_space, lane_counts: space_vec, 
                    beta: space_vec | time_space, time_step: hr):
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

HOLIDAY_DATES = {(11, 24), (11, 25)}
def format_date_label(date_str: str, year=2022):
    """'11_30' -> '11/30 (Wed)'; holiday dates -> '11/24 (Holiday)'."""
    import datetime
    month, day = (int(p) for p in date_str.split('_'))
    tag = 'Holiday' if (month, day) in HOLIDAY_DATES else datetime.date(year, month, day).strftime('%a')
    return f"{month:02d}/{day:02d} ({tag})"

def run_analysis(run: RunResult) -> MPCStats:
    spec, traffic = run.scenario.spec, run.scenario.traffic
    gt_tt = (spec.time_step * spec.L * (traffic.density.sum(axis=0) @ spec.lanes)
             if len(traffic.density) else float('nan'))
    sim_tt, opt_tt = run.optimization.baseline.total_travel_time, run.optimization.controlled.total_travel_time
    ffv = run.params['v_free']
    rho, vel = run.optimization.baseline.density[:-1], run.optimization.baseline.velocity[:-1]
    beta = run.params["beta"]
    if beta.ndim == 2: beta = beta[np.minimum(np.arange(len(rho)), len(beta)-1)]
    return MPCStats(
        gt_tt = gt_tt, sim_tt = sim_tt, opt_tt = opt_tt,
        sim_window_tt = float(spec.time_step * spec.L * (rho[:spec.time_steps].sum(axis=0) @ spec.lanes)),
        sim_error = float(mape(traffic.velocity, vel[:spec.time_steps])) if len(traffic.velocity) else float('nan'),
        ff_tt = ff_tts_vkm(rho, vel, spec.lanes, spec.time_step, spec.L, run.params["v_free"])
                if LATEST_UPDATE else 
                get_ff_tts(traffic.inflow, spec.time_step, spec.L, ffv.max(axis=0) if ffv.ndim==2 else ffv),        
        num_veh=vehicles_served(rho, vel, spec.lanes, beta, spec.time_step)
                if LATEST_UPDATE else 
                traffic.inflow.sum() * spec.time_step
    )

def run_analysis_direct(scenario: Scenario, params: MetanetParams, result: OptimizationResult) -> MPCStats:
    return run_analysis(RunResult(scenario, params, scenario.spec.to_mpc_config(), result))
