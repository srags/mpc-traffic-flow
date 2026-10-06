"""Numerical CC report inputs; no plotting or file writes."""

from dataclasses import dataclass

import numpy as np

from ..inputs.scenario import Scenario
from ..types import MetanetParams, OptimizationResult, SimulationResult, time_space

def get_num_veh(demand_profile, time_step):
    """Integrate demand (veh/hr) over time steps (hr), returning vehicles."""
    return sum(demand_profile) * time_step


def get_ff_tts(demand_profile, time_step, length, v_free):
    """
    Calculate total time spent (TTS) under free flow conditions.

    Parameters:
        demand_profile : list of demands at each time step (veh/h)
        time_step      : simulation time step (hours)
        length         : segment length (km)
        v_free         : list of free flow speeds, one per segment (km/h)

    Returns:
        free flow TTS (vehicle-hours)
    """
    num_vehicles  = sum(demand_profile) * time_step
    ff_travel_time = sum(length / v for v in v_free)  # hours
    return num_vehicles * ff_travel_time


@dataclass(frozen=True)
class CCReport:
    scenario: Scenario
    params: MetanetParams
    result: OptimizationResult
    diagnostic: SimulationResult
    display_vsl: time_space
    free_flow_ttt: float
    delay: float
    controlled_delay: float
    observed_delay: time_space
    pct_decrease: time_space

    @property
    def fit_rows(self) -> dict[str, str]:
        """Original measured-inflow diagnostic, excluding the terminal row."""
        spec, traffic = self.scenario.spec, self.scenario.traffic
        p_sim, v_sim = self.diagnostic.density[:-1], self.diagnostic.velocity[:-1]
        q_sim = v_sim * p_sim * traffic.lanes
        rows = {
            name: f"MAPE {np.mean(np.abs((observed - predicted) / observed) * 100):.2f}%, "
                  f"RMSE {np.sqrt(np.mean((observed - predicted) ** 2)):.2f}"
            for name, observed, predicted in (
                ("Velocity", traffic.velocity, v_sim),
                ("Density", traffic.density, p_sim),
                ("Flow", traffic.flow, q_sim),
            )
        }
        gt_tt, sim_tt = (spec.time_step * spec.L * np.sum(rho * traffic.lanes[np.newaxis, :])
                         for rho in (traffic.density, p_sim))
        rows["Travel Time"] = f"{gt_tt:.2f} veh-hr vs {sim_tt:.2f} veh-hr (MAPE {abs(gt_tt - sim_tt) / gt_tt * 100:.2f}%)"
        return rows

    @property
    def cc_rows(self) -> dict[str, str]:
        return {
            "Total free flow travel time": f"{self.free_flow_ttt:.2f} veh-hrs",
            "Controllable congestion": f"{np.divide(self.delay - self.controlled_delay, self.delay) * 100:.2f}%",
            "Delay (ground truth)": f"{self.observed_delay.min():.2f} min - {self.observed_delay.max():.2f} min",
            "Pct decrease": f"{self.pct_decrease.min():.2f}% - {self.pct_decrease.max():.2f}%",
        }


def cc_report(scenario: Scenario, params: MetanetParams, result: OptimizationResult) -> CCReport:
    """Prepare the existing cc_plot report, keeping its two baseline conventions.

    Calibration diagnostics use measured inflow; the supplied policy result uses
    the modeled origin/queue. Only the diagnostic is recomputed here, never MPC.
    This report expects static free-flow speeds, as the original cc_plot did.
    """
    from ..pipeline import simulate_scenario

    spec, traffic = scenario.spec, scenario.traffic
    v_free = params["v_free"]
    if v_free.shape != (spec.num_segments,):
        raise ValueError("The CC report requires one static free-flow speed per segment")
    if result.vsl.shape != (spec.time_steps, spec.num_segments):
        raise ValueError("Policy dimensions do not match the report scenario")

    diagnostic = simulate_scenario(
        traffic, params, T=spec.time_step, l=spec.L, steps=spec.time_steps, real_data=True,
    )
    # Display-only transformation: simulation always uses result.vsl unchanged.
    display_vsl = np.where(result.vsl > np.tile(v_free, (len(result.vsl), 1)), 150, result.vsl)
    ff_ttt = get_ff_tts(traffic.inflow, spec.time_step, spec.L, v_free)
    delay = result.baseline.total_travel_time - ff_ttt
    controlled_delay = result.controlled.total_travel_time - ff_ttt
    free_flow_speed = np.array(v_free)[:, np.newaxis]
    observed_delay = (spec.L / traffic.velocity.T - spec.L / free_flow_speed) * 60
    controlled_point_delay = (spec.L / result.controlled.velocity[:-1].T - spec.L / free_flow_speed) * 60
    pct_decrease = np.where(observed_delay > 0.01,
                            (observed_delay - controlled_point_delay) / observed_delay * 100, 0)
    return CCReport(scenario, params, result, diagnostic, display_vsl,
                    float(ff_ttt), float(delay), float(controlled_delay), observed_delay, pct_decrease)

