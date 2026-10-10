"""Numerical CC report inputs; no plotting or file writes."""
from dataclasses import dataclass
import numpy as np
from .console import colored
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
        gt_tt, sim_tt = (spec.time_step * spec.L * (rho * traffic.lanes[np.newaxis, :]).sum()
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


def print_cc_report(report: CCReport) -> None:
    from tabulate import tabulate

    print(colored("No Control", "bold", "yellow"))
    print(tabulate(report.fit_rows.items(), headers=("Quantity", "Result"), tablefmt="outline"))
    print(colored("Optimized VSLs", "bold", "yellow"))
    print(tabulate(report.cc_rows.items(), headers=("Metric", "Value"), tablefmt="outline"))

from math import ceil, floor

def get_virtual_trajectory(speed_field, time_start, space_start, d_time, d_space, direction=-1):
    # Initialize the trajectory mask with zeros
    trajectory_mask = np.zeros_like(speed_field)

    # Let's change the units of the speed field so what we can use it as the slope of the line in the time-space plot
    # d_space is in miles per y_bin, and d_time is in seconds per x_bin 
    # The speed field is in mile per hour
    # Let's convert the speed field to miles per second
    speed_field = speed_field / 3600
    # Now let's account for the direction of the vechicles on the road
    speed_field *= direction
    # and let's convert the speed field to y_bin per x_bin  
    speed_field = speed_field * d_time / d_space

    # Forward propagation
    forward_time_points = []
    forward_space_points = []

    time_current, space_current = time_start, space_start
    while 0 <= time_current < speed_field.shape[1] and 0 <= space_current < speed_field.shape[0]:
        # Store the current points in the trajectory points list
        forward_time_points.append(time_current)
        forward_space_points.append(space_current)
        # Get the speed at the current point
        speed = speed_field[int(space_current), int(time_current)]

        # The speed is the slope of the line in the time-space plot
        # For each time step, let's draw a line from the current point and follow the slope until we reach another bin
        # Let's get the time step to reach the next bin, which is the minimum between the time to reach the next time bin and the time to reach the next space bin
        time_to_next_time_bin = floor(time_current + 1) - time_current
        # and now the time to reach the next space bin
        if direction == 1:
            time_to_next_space_bin = (floor(space_current + 1) - space_current) / speed
        elif direction == -1:
            time_to_next_space_bin = (space_current - ceil(space_current - 1)) / speed
        # Let's get the minimum of the two
        time_step = min(abs(time_to_next_time_bin), abs(time_to_next_space_bin))
        # Let's move to the next point in time and space
        time_current += time_step
        space_current += time_step * speed
    
    
    # Backward propagation
    backward_time_points = []
    backward_space_points = []
    time_current, space_current = time_start, space_start
    while 0 <= time_current < speed_field.shape[1] and 0 <= space_current < speed_field.shape[0]:
        # Store the current points in the trajectory points list
        backward_time_points.append(time_current)
        backward_space_points.append(space_current)
        # Get the speed at the current point
        speed = speed_field[int(space_current), int(time_current)]

        # The speed is the slope of the line in the time-space plot
        # For each time step, let's draw a line from the current point and follow the slope until we reach another bin
        # Let's get the time step to reach the previous bin, which is the minimum between the time to reach the next time bin and the time to reach the next space bin
        time_to_previous_time_bin = time_current - ceil(time_current - 1) 
        # and now the time to reach the previous space bin
        if direction == 1: 
            time_to_previous_space_bin = (space_current - ceil(space_current - 1)) / speed
        elif direction == -1: 
            time_to_previous_space_bin = (floor(space_current + 1) - space_current) / speed
        # Let's get the minimum of the two
        time_step = min(abs(time_to_previous_time_bin), abs(time_to_previous_space_bin))
        # Let's move to the next point in time and space
        time_current -= time_step
        space_current -= time_step * speed
    
    # Let's reverse the backward points and add them to the trajectory points 
    backward_time_points.reverse()
    backward_space_points.reverse()
    
    # and now we add them to the beginning of the time_points and space_points
    time_points = backward_time_points + forward_time_points
    space_points = backward_space_points + forward_space_points

    return time_points, space_points       

def vt_travel_time_stats(macro_velocity_field, time_step=10/3600, num_samples=50, plotting=False):
    m, time_steps = macro_velocity_field.shape
    # Let's use the get_virtual_trajectory and visualize the results
    x_starts = np.linspace(0, time_steps-1, num_samples) #np.random.uniform(0, time_steps, num_samples)
    y_starts = [m - 0.0001 for i in range(num_samples)]
    d_time = 4
    d_space =  0.5
    virtual_trajectories = [get_virtual_trajectory(macro_velocity_field, x_start, y_start, d_time, d_space) for x_start, y_start in zip(x_starts, y_starts)]

    vt_times = [(time_points[-1] - time_points[0]) * time_step * 60 for time_points, space_points in virtual_trajectories]

    if plotting:
        from .plots import plot_virtual_trajectories

        print(f"Mean vehicle travel time: {np.mean(vt_times)} min")
        print(f"Standard deviation of vehicle travel time: {np.std(vt_times, ddof=1)} min")
        plot_virtual_trajectories(macro_velocity_field, virtual_trajectories)

    return vt_times
