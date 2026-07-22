# v2 get virtual trajectory

from math import floor
from math import ceil
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

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
        print(f"Mean vehicle travel time: {np.mean(vt_times)} min")
        print(f"Standard deviation of vehicle travel time: {np.std(vt_times, ddof=1)} min")

        # Let's visualize the virtual trajectory
        plt.figure(figsize=(15, 5))
        plt.imshow(macro_velocity_field, aspect='auto', interpolation='None', cmap='RdYlGn', extent=[0, time_steps, 0, m])
        plt.xlim(0, time_steps)
        plt.yticks(np.arange(0, m + 1, 1))
        plt.xticks(np.arange(0, time_steps + 1, 10), labels=[int(i * 10 / 60) if i % 120 == 0 else None for i in range(0, time_steps+1, 10)])
        plt.colorbar(label='Velocity (km/hr)')
        plt.xlabel('Time (min)')
        plt.ylabel('Space (segments)')
        for time_points, space_points in virtual_trajectories:
            plt.plot(time_points, 15 - np.array(space_points), color='blue', linewidth=1)
        plt.title('Virtual Trajectory')
        plt.grid()
        plt.show()
    return vt_times