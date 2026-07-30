import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from itertools import product
from traffic_sim import *
from mpc_metanet import *
from param_loader import METANET_Params
import matplotlib.colors as mcolors


def generate_demand(sim_time, time_step, peak_start=0.25, peak_end=0.75, flow_standard=2400, flow_peak=3000):
    """
    Generate a traffic demand profile for a given simulation time.

    Parameters:
    - sim_time: Total simulation time in hours.
    - time_step: Time step for the simulation in seconds.
    - demand_start: Start time of the demand increase in seconds.
    - demand_end: End time of the demand decrease in seconds.
    - demand_peak: Peak traffic demand during the peak period.
    - demand_duration: Duration of the peak period in seconds.

    Returns:
    - A list representing the traffic demand at each second of the simulation.
    """
    total_time_steps = int(sim_time / time_step)
    traffic_demand = [0] * (total_time_steps + 1)
    
    for i in range(total_time_steps+1):
        if i < int(peak_start / time_step):
            traffic_demand[i] = flow_standard
        elif i < int(peak_end / time_step):
            traffic_demand[i] = flow_peak
        else:
            traffic_demand[i] = flow_standard

    return np.array(traffic_demand)

def generate_demand_options(num_options, sim_time, time_step, num_segments, seg_length, sim_params, lanes, peak_min=2400, peak_max=3400, flow_standard=2400, duration_range=[0.1, 0.2, 0.3, 0.4, 0.5]):
    """
    Generate a list of traffic demand options for different scenarios.

    Parameters:
    - num_options: Number of demand options to generate.
    - peak_min: Minimum peak traffic demand.
    - peak_max: Maximum peak traffic demand.
    - flow_standard: Standard traffic demand.

    Returns:
    - A list of tuples representing the traffic demand options.
    """
    demand_options = dict()
    travel_times = dict()

    peak_options = np.linspace(peak_min, peak_max, num_options, endpoint=True)
    demand_scenarios = list(product(peak_options, duration_range))
    for peak_demand, peak_duration in demand_scenarios:
        demand_profile = generate_demand(sim_time, time_step, flow_standard=flow_standard, flow_peak=peak_demand, peak_start=0.055, peak_end=0.055 + peak_duration)
        demand_options[(peak_demand, peak_duration)] = demand_profile
        start_state = (np.full(num_segments, demand_profile[0]/(sim_lanes[0] * 90)), np.full(num_segments, 90), demand_profile[0], 0)

        num_time_steps = int(sim_time / time_step)
        travel_time = run_metanet_sim(time_step, seg_length, start_state, demand_profile, np.zeros(num_time_steps), 
                                      sim_params, lanes=lanes, real_data=False, vsl_speeds=None)[1]
 

        travel_times[(peak_demand, peak_duration)] = travel_time

    return demand_options, travel_times

def get_num_veh(demand_profile, time_step):
    """
    Calculate the average travel time for a given demand profile.

    Parameters:
    - demand_profile: The traffic demand profile.
    - time_step: The time step for the simulation in seconds.
    - total_tt: Total travel time from the simulation.

    Returns:
    - Average travel time in seconds.
    """
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

def optimize_scenarios(num_options, filepath, sim_time, time_step, num_segments, seg_length, sim_params, lanes, peak_min=2400, peak_max=3400, flow_standard=2400, duration_range=[0.1, 0.2, 0.3, 0.4, 0.5]):
    """
    Optimize traffic demand scenarios by reading from a CSV file and generating demand options.

    Parameters:
    - filepath: Path to the CSV file containing traffic demand scenarios.
    - sim_time: Total simulation time in hours.
    - time_step: Time step for the simulation in seconds.
    - num_segments: Number of segments in the simulation.
    - seg_length: Length of each segment in kilometers.
    - sim_params: Simulation parameters.
    - lanes: Dictionary mapping segment indices to the number of lanes.

    Returns:
    - A list of optimized traffic demand options.
    """

    peak_options = np.linspace(peak_min, peak_max, num_options, endpoint=True)
    demand_scenarios = list(product(peak_options, duration_range))
    print(f"Number of options: {len(demand_scenarios)}, Peak options: {demand_scenarios}")

    demands, tts = generate_demand_options(num_options, sim_time, time_step, num_segments, seg_length, sim_params, lanes, peak_min=peak_min, peak_max=peak_max, flow_standard=flow_standard, duration_range=duration_range)

    # Prepare 2D arrays: rows = durations, cols = peak_demands
    num_peaks = len(np.linspace(peak_min, peak_max, num_options, endpoint=True))
    num_durations = len(duration_range)
    opt_tts_arr = np.zeros((num_durations, num_peaks))
    tts_arr = np.zeros((num_durations, num_peaks))
    ff_tt_arr = np.zeros((num_durations, num_peaks))
    avg_tt_arr = np.zeros((num_durations, num_peaks))
    avg_speed_arr = np.zeros((num_durations, num_peaks))

    peak_options = np.linspace(peak_min, peak_max, num_options, endpoint=True)
    for i_dur, peak_duration in enumerate(duration_range):
        for i_peak, peak_demand in enumerate(peak_options):
            tts_arr[i_dur, i_peak] = tts[(peak_demand, peak_duration)]

            print(f"optimizing for peak demand and duration: {peak_demand}, {peak_duration}")
            demand_profile = demands[(peak_demand, peak_duration)]
            time_steps = int(sim_time / time_step)
            mpc_time_steps = time_steps + 40 - 5
            downstream_density = np.zeros(mpc_time_steps + 1)

            while len(demand_profile) < mpc_time_steps + 1:
                demand_profile = np.append(demand_profile, demand_profile[-1])

            policy_dir = filepath + f"/demand{peak_demand}_duration{peak_duration}.csv" #+ f"/policy_demand{peak_demand}_duration{peak_duration}"
            start_state = (np.full(num_segments, demand_profile[0]/(sim_lanes[0] * 90)), np.full(num_segments, 90), demand_profile[0], 0)
            init_vsl = np.full((time_steps, num_segments), 40)
            #print(init_vsl.shape, demand_profile.shape, downstream_density.shape, time_steps)
            if not os.path.exists(policy_dir):
                optimal_vsl = mpc_find_vsl(
                        mpc_time_steps,
                        demand_profile,
                        downstream_density,
                        sim_lanes,
                        T=time_step,
                        l=seg_length,
                        num_segments=num_segments,
                        pred_horizon=40,
                        control_horizon=5,
                        hold_length=1,
                        initialize_vsl=init_vsl,
                        control_one_segment=None,
                        control_changepoints=None,
                        safety_spatial=None,
                        safety_temporal=None,
                        params=sim_params,
                        verbose=False,
                        speed_lb=0,
                        v_fd_penalty=10000,
                        control_zone=np.arange(10, num_segments) 
                )   
                _, velocities, _, vsl_travel_time = run_metanet_sim(time_step, seg_length, start_state, demand_profile[0:time_steps], downstream_density[0:time_steps], 
                                                                    sim_params, lanes=lanes, vsl_speeds=optimal_vsl, real_data=False, plotting=True)
                opt_tts_arr[i_dur, i_peak] = vsl_travel_time
                np.savetxt(policy_dir, optimal_vsl, delimiter=',')
            else:
                optimal_vsl = np.loadtxt(policy_dir, delimiter=',')
                print(sim_time)
                _, velocities, _, vsl_travel_time = run_metanet_sim(time_step, seg_length, start_state, demand_profile[0:time_steps], downstream_density[0:time_steps], 
                                                                    sim_params, lanes=lanes, vsl_speeds=optimal_vsl, real_data=False, plotting=True)
                opt_tts_arr[i_dur, i_peak] = vsl_travel_time

            print(optimal_vsl.shape)
            print(f"Peak demand / duration: {peak_demand} / {peak_duration}, Travel time: {tts[(peak_demand, peak_duration)]}, Optimal travel time: {vsl_travel_time}")

            ff_tt_arr[i_dur, i_peak] = get_ff_tt(demand_profile, time_step, seg_length * num_segments, sim_params['v_free'][0])
            avg_tt_arr[i_dur, i_peak] = tts[(peak_demand, peak_duration)] / get_num_veh(demand_profile, time_step) * 60
            # print(np.shape(velocities[int(35/(60 * time_step)), 10:]))
            _, nc_velocities, _, _ = run_metanet_sim(time_step, seg_length, start_state, demand_profile[0:time_steps], 
                              np.zeros(time_steps), sim_params, lanes=lanes, real_data=False, vsl_speeds=None, plotting=True)
            avg_speed_arr[i_dur, i_peak] = np.mean(nc_velocities[100:, 10:]) * 0.62

    return tts_arr, opt_tts_arr, ff_tt_arr, avg_tt_arr, avg_speed_arr

def set_saturation(color, sat_scale=0.8):
    # Convert color to RGB
    rgb = mcolors.to_rgb(color)
    
    # Convert RGB → HSV
    h, s, v = mcolors.rgb_to_hsv(rgb)
    
    # Scale saturation
    s = s * sat_scale
    if s > 1:
        s = 1
        
    # Convert back HSV → RGB
    return mcolors.hsv_to_rgb([h, s, v])

if __name__ == "__main__":
    total_distance = 10
    num_segments = int(total_distance/0.4)

    sim_lanes = {i: 4 if i < num_segments-5 else 2 for i in range(num_segments)}
    params = METANET_Params(path=None, num_segments=num_segments).get_params()
    num_scenarios = 14 * 2 + 1
    p_min = 5100
    p_max = 6500
    durations = [0.5]
    text_fontsize = 20


    tts, opt_tts, ff_tt, avg_tt, avg_speed = optimize_scenarios(num_scenarios,"/Users/shreyaar/Desktop/PhD/research/MPC/results/final_extended/demand/", 2, 10/3600, num_segments, 0.4, params, sim_lanes, peak_min=p_min, peak_max=p_max, flow_standard=4000, duration_range=durations)
    peak_demand = np.linspace(p_min, p_max, num_scenarios, endpoint=True)
    demand_options, _ = generate_demand_options(num_scenarios, 2, 10/3600, num_segments, 0.4, params, sim_lanes, peak_min=p_min, peak_max=p_max, flow_standard=4000, duration_range= durations)

    # ff_tt = np.apply_along_axis(get_ff_tt, 1, demand_options, 10/3600, 0.4 * 15, params['v_free'][0])
    # avg_tt = tts / np.apply_along_axis(get_num_veh, 1, demand_options, 10/3600) * 60
    delay = tts - ff_tt
    opt_delay = opt_tts - ff_tt
    percent_improvement = np.round((delay - opt_delay) / delay * 100, 1).reshape(-1)
    print(percent_improvement)
    print(avg_speed)
    num_veh = []
    for demand in demand_options:
        num_veh.append(get_num_veh(demand_options[demand], 10/3600))
    num_veh = np.array(num_veh)
    print(num_veh)
    avg_delay = delay.reshape(-1)/num_veh

    scenarios = np.arange(1, percent_improvement.shape[0]+1) # labels 1–20
    width = 0.85  # bar width

    fig, ax = plt.subplots(figsize=(15, 6))
    plt.grid()
    ax.set_axisbelow(True)

    # Stacked bars
    p1 = ax.bar(scenarios, avg_delay, label="Delay without control", color="#d33b19", width=width)
    p2 = ax.bar(scenarios, percent_improvement * avg_delay/100,
                label="Delay reduced from control", color="#4e9858", width=width)

    # # Add percentage annotations
    # total = delay + percent_improvement
    # percentages = 100 * percent_improvement / total

    for i in range(len(scenarios)):
        ax.text(scenarios[i], avg_delay[i] +0.001,   # slightly above bar
                f"{percent_improvement[i]:.0f}",
                ha='center', va='bottom', fontsize=text_fontsize-4, fontname="Times New Roman", fontweight='bold')

    # Labels & layout
    ax.set_xlabel("Peak demand of scenario (veh/hr)", fontsize=text_fontsize, fontname="Times New Roman")
    ax.set_ylabel("Average delay per vehicle (hrs)", fontsize=text_fontsize, fontname="Times New Roman")
    # ax.set_title("Stacked Delay and Controllable Congestion Across 20 Scenarios", fontsize=text_fontsize, fontname="Times New Roman")
    ax.set_xticks(scenarios)
    ax.set_xticklabels(peak_demand.astype(int), rotation=45, ha='center')
    ax.legend(prop={'family': 'Times New Roman', 'size': text_fontsize})


    ax.tick_params(labelsize=text_fontsize - 4)
    # set tick label font
    for label in ax.get_xticklabels():
        label.set_fontname('Times New Roman')
    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')
    
    ax.set_ylim(0, np.max(avg_delay)*1.1)
    # move legend outside the plot
    # plt.legend( loc='upper left')
    # plt.legend(loc='upper left', fontsize=14)
    plt.savefig(f"figs/delay_reduction.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    

    # plt.show()

    # ax2.cla()
    # plt.xlabel('Average Travel Time without Control (min)', fontname='Times New Roman', fontsize=18)
    # ax.set_xlabel('Demand During 30 min Peak Period (veh/hr)', fontname='Times New Roman', fontsize=18)
    # ax.set_xlim(p_min-10, p_max+10)

    ## Controllable congestion

    # ax.plot(peak_demand, percent_improvement.reshape(-1), color='blue', marker='o')
    # ax.set_ylabel('Controllable Congestion (%)', fontname='Times New Roman', fontsize=18)
    # ax.set_ylim(0, np.max(percent_improvement)*1.1)

    ### Opportunity gap plot

    # los_thresholds = {'B':59, 'C':54, 'D':46, 'E':30, 'F':0}
    # los_tracker = 'B'
    # demand_threshold = []
    # for i in range(len(peak_demand)):
    #     if avg_speed.reshape(-1)[i] < los_thresholds[los_tracker]:
    #         los_tracker = chr(ord(los_tracker) + 1)
    #         demand_threshold.append((peak_demand[i-1] + peak_demand[i]) / 2 if i > 0 else peak_demand[i])

    # for i in range(len(demand_threshold)):
    #     if i == len(demand_threshold) - 1:
    #         ax.axvline(demand_threshold[i], color='C'+str(i+1), linestyle='--', alpha=0.5)
    #         ax.axvline(p_max+10, color='C'+str(i+1), linestyle='--', alpha=0.5)
    #         span = ax.axvspan(demand_threshold[i], p_max+10, alpha=0.1, color='C'+str(i+1))
    #         # Annotate the shaded region
    #         ax.annotate(f'LOS {chr(ord("A")+i + 1)}', 
    #                     xy=((demand_threshold[i]+p_max)/2, ax.get_ylim()[1]-2), 
    #                     xycoords='data', ha='center', va='bottom', fontsize=18, color='C'+str(i+1), fontweight='bold', fontname='Times New Roman')
    #     else:
    #         if i == 0:
    #             ax.axvspan(p_min-10, demand_threshold[i+1], alpha=0.1, color='C'+str(i+1))
    #         else:
    #             ax.axvspan(demand_threshold[i], demand_threshold[i+1], alpha=0.1, color='C'+str(i+1))
    #             ax.axvline(demand_threshold[i], color='C'+str(i+1), linestyle='--', alpha=0.5)

    #         # Annotate the shaded region
    #         ax.annotate(f'LOS {chr(ord("A")+i + 1)}', 
    #                     xy=((demand_threshold[i]+demand_threshold[i+1])/2, ax.get_ylim()[1]-2), 
    #                     xycoords='data', ha='center', va='bottom', fontsize=18, color='C'+str(i+1), fontweight='bold', fontname='Times New Roman')
    


    # ax.plot(peak_demand, delay.reshape(-1), color='red', marker='o', label='No Control')
    # ax.plot(peak_demand, opt_delay.reshape(-1), color='green', marker='o', label='With VSL Control')
    # #Shade area between two curves
    # ax.fill_between(peak_demand, delay.reshape(-1), opt_delay.reshape(-1), color='gray', alpha=0.3)
    # ax.set_ylabel('Total delay (veh-hr)', fontname='Times New Roman', fontsize=18)
    # ax.set_ylim(0, np.max(delay)*1.1)

    ## Average speed

    # ax2 = ax.twinx()
    # ax2.set_ylabel('Average Speed (mph)', fontname='Times New Roman', fontsize=18, fontweight='bold')
    # ax2.tick_params(labelsize=14)
    # ax2.set_ylim(30, 60)
    # ax2.plot(peak_demand, avg_speed.reshape(-1), color='purple', marker='o')
    # ax2.yaxis.label.set_color('purple')


    # assert demand.shape == (10, 1800)
    # time_ = np.arange(0, demand_options.shape[1], 1) * 10/3600  # Convert to hours
    # for i in range(demand_options.shape[0]):
    #     plt.plot(time_, demand_options[i], label=f'Avg TT: {avg_tt[i]} min')
    # plt.xlabel('Time (hours)')
    # plt.ylabel('Traffic Demand (veh/hr)')
    # # plt.legend()
    # plt.show()