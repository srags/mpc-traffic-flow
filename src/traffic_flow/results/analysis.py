"""Numerical CC report inputs; no plotting or file writes."""
from dataclasses import dataclass
import numpy as np

from ..types import *

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


HOLIDAY_DATES = {(11, 24), (11, 25)}
def format_date_label(date_str: str, year=2022):
    """'11_30' -> '11/30 (Wed)'; holiday dates -> '11/24 (Holiday)'."""
    import datetime
    month, day = (int(p) for p in date_str.split('_'))
    tag = 'Holiday' if (month, day) in HOLIDAY_DATES else datetime.date(year, month, day).strftime('%a')
    return f"{month:02d}/{day:02d} ({tag})"


@dataclass(frozen=True)
class MPCStats:
    gt_tt: veh_hr; sim_tt: veh_hr; opt_tt: veh_hr; ff_tt: veh_hr
    sim_error: float
    num_veh: float
    @property
    def tts_error(self): return (abs(self.gt_tt - self.sim_tt) / self.gt_tt) * 100
    @property
    def cc(self): return np.clip((self.sim_tt - self.opt_tt) / (self.sim_tt - self.ff_tt) * 100, 0, 100)
    @property
    def avg_tt_reduced_per_veh(self): return (self.sim_tt - self.opt_tt) / self.num_veh * 60
    @property
    def uncontrolled_avg_tts(self): return (self.sim_tt - self.ff_tt) / self.num_veh * 60
    @property
    def controlled_avg_tts(self): return (self.opt_tt - self.ff_tt) / self.num_veh * 60

def run_analysis(run: RunResult) -> MPCStats:
    spec, traffic = run.scenario.spec, run.scenario.traffic
    gt_tt = spec.time_step * spec.L * (traffic.density.sum(axis=0) @ traffic.lanes) 
    sim_tt, opt_tt = run.optimization.baseline.total_travel_time, run.optimization.controlled.total_travel_time
    ffv = run.params['v_free']
    return MPCStats(
        gt_tt     = gt_tt,
        sim_tt    = sim_tt,
        opt_tt    = opt_tt,
        ff_tt     = get_ff_tts(traffic.inflow, spec.time_step, spec.L, ffv.max(axis=0) if ffv.ndim==2 else ffv),
        sim_error = (abs(traffic.velocity - run.optimization.baseline.velocity[:-1]) / traffic.velocity).mean() * 100, 
        num_veh   = traffic.inflow.sum() * spec.time_step
    )