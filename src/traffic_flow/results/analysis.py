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

def departure_rates(
    density: time_space,
    velocity: time_space,
    lane_counts: space_vec,
    beta: space_vec | time_space,
) -> time_vec:
    if beta.ndim == 2: beta = beta[np.minimum(np.arange(len(density)), len(beta) - 1)]
    q = density * velocity * lane_counts
    return q[:, -1] + (q * beta / (1 - beta)).sum(axis=1)

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
    return float(departure_rates(density, velocity, lane_counts, beta).sum() * time_step)

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






def free_flow_departures(
    scenario: Scenario,
    params: MetanetParams,
    n_steps: int,
) -> tuple[time_vec, time_vec]:
    """Cumulative arrivals and departures if every vehicle ran at v_free.

    Same arrivals and splits as the simulated runs, on a grid of `n_steps` steps of
    `time_step`. Each arrival stream (mainline demand, each on-ramp, and the vehicles
    already in the corridor at t=0) is spread over its free-flow exit times. METANET
    removes ramp traffic as a sink on a segment's own outflow, so those vehicles cover
    no distance in the segment they leave at: split first, then charge the traversal
    time. Returns (arrivals, departures), both cumulative vehicle counts.

    Ported from experiments/capacity_drop_diagnostics.ipynb.
    """
    spec, traffic = scenario.spec, scenario.traffic
    time_step, L = spec.time_step, spec.L

    if any(x.ndim != 1 for x in (params["v_free"], params["r"], params["beta"])):
        raise ValueError("Free-flow departure curves currently require static parameters.")

    n_seg = spec.num_segments
    lanes = np.asarray(spec.lanes, dtype=float)
    t_grid = np.arange(n_steps) * time_step                # hours
    tau = L / np.asarray(params["v_free"], dtype=float)    # free-flow traversal, hours
    beta, r = np.asarray(params["beta"], float), np.asarray(params["r"], float)
    n_dem = traffic.inflow.shape[0]
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

    add_stream(on_grid(traffic.inflow * time_step), 0)
    n0 = traffic.initial_density * lanes * L                  # vehicles already in each segment
    for i in range(n_seg):
        if r[i] > 1e-6:
            add_stream(on_grid(np.full(n_dem, r[i] * time_step)), i)
        if n0[i] > 1e-9:                                   # released from mid-segment
            add_stream(on_grid(np.r_[n0[i], np.zeros(n_dem - 1)]), i, head_start=-0.5 * tau[i])
    return A, D


def departure_curves(
    run: RunResult,
    drain_tol: float = 3.0,
) -> DepartureCurves:
    from ..pipeline import simulate

    scenario, params = run.scenario, run.params
    spec = scenario.spec

    baseline, controlled = (
        simulate(
            scenario, params, vsl=vsl,
            real_data=False, until_ff=True, drain_tol=drain_tol,
        )
        for vsl in (None, run.optimization.vsl)
    )

    n_steps = max(len(baseline.density), len(controlled.density)) - 1

    def cumulative(sim: SimulationResult) -> time_vec:
        rates = departure_rates(
            sim.density[:-1], sim.velocity[:-1],
            spec.lanes, params["beta"],
        )
        counts = np.cumsum(rates) * spec.time_step
        return np.pad(counts, (0, n_steps - len(counts)), mode="edge")

    arrivals, free_flow = free_flow_departures(scenario, params, n_steps)

    return DepartureCurves(
        np.arange(n_steps) * spec.time_step * 60,
        arrivals,
        free_flow,
        cumulative(baseline),
        cumulative(controlled),
    )