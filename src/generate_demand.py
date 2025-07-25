import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import product
from traffic_sim import metanet_sim_params
from mpc_metanet import *
from sim_params.I24_400mParams import I24_400mParams
from sim_params.default import DefaultParams

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
        demand_profile = generate_demand(sim_time, time_step, flow_standard=flow_standard, flow_peak=peak_demand, peak_start=0.05, peak_end=0.05 + peak_duration)
        demand_options[(peak_demand, peak_duration)] = demand_profile

        num_time_steps = int(sim_time / time_step)
        travel_time = metanet_sim_params(time_step, seg_length, (np.full(num_segments, 0), np.full(num_segments, 0), demand_profile[0], 0), 
                              np.full((num_time_steps, num_segments), sim_params['v_free']), demand_profile, 
                              np.zeros(num_time_steps), sim_params, lanes=lanes, real_data=False, plotting=False)[1]

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

def get_ff_tt(demand_profile, time_step, length, v_free):
    num_veh = sum(demand_profile) * time_step
    return num_veh * length / v_free

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

            policy_dir = filepath + f"/policy_demand{peak_demand}_duration{peak_duration}"
            if not os.path.exists(policy_dir):
                optimal_vsl = mpc_find_vsl(mpc_time_steps, demand_profile, downstream_density, sim_lanes, T=time_step,
                                        l=seg_length, num_segments=num_segments, pred_horizon=40, control_horizon=5, params=sim_params, speed_lb=0)

                vsl_travel_time = metanet_sim_params(time_step, seg_length, (np.full(num_segments, 0), np.full(num_segments, 0), demand_profile[0], 0), 
                                    optimal_vsl, demand_profile, downstream_density, sim_params, lanes=lanes, real_data=False, plotting=False)[1]
                opt_tts_arr[i_dur, i_peak] = vsl_travel_time
                os.makedirs(policy_dir)
                np.savetxt(policy_dir + "/optimal_vsl.csv", optimal_vsl, delimiter=',')
            else:
                optimal_vsl = np.loadtxt(policy_dir + "/optimal_vsl.csv", delimiter=',')
                vsl_travel_time = metanet_sim_params(time_step, seg_length, (np.full(num_segments, 0), np.full(num_segments, 0), demand_profile[0], 0), 
                                    optimal_vsl, demand_profile, downstream_density, sim_params, lanes=lanes, real_data=False, plotting=False)[1]
                opt_tts_arr[i_dur, i_peak] = vsl_travel_time

            print(f"Peak demand / duration: {peak_demand} / {peak_duration}, Travel time: {tts[(peak_demand, peak_duration)]}, Optimal travel time: {vsl_travel_time}")

            ff_tt_arr[i_dur, i_peak] = get_ff_tt(demand_profile, time_step, seg_length * num_segments, sim_params['v_free'][0])
            avg_tt_arr[i_dur, i_peak] = tts[(peak_demand, peak_duration)] / get_num_veh(demand_profile, time_step) * 60

    return tts_arr, opt_tts_arr, ff_tt_arr, avg_tt_arr

if __name__ == "__main__":
    sim_lanes = {i: 4 if i < 15-5 else 2 for i in range(15)}
    params = DefaultParams(15).get_params()
    num_scenarios = 10 * 5 + 1
    p_min = 5000
    p_max = 6000
    durations = [0.1, 0.2, 0.3, 0.4, 0.5]


    tts, opt_tts, ff_tt, avg_tt = optimize_scenarios(num_scenarios,"/Users/shreyaar/Desktop/PhD/research/MPC/results/default_0speed_heatmap", 2, 10/3600, 15, 0.4, params, sim_lanes, peak_min=p_min, peak_max=p_max, flow_standard=4000)
    peak_demand = np.linspace(p_min, p_max, num_scenarios, endpoint=True)
    demand_options, _ = generate_demand_options(num_scenarios, 2, 10/3600, 15, 0.4, params, sim_lanes, peak_min=p_min, peak_max=p_max, flow_standard=4000, duration_range= durations)

    # ff_tt = np.apply_along_axis(get_ff_tt, 1, demand_options, 10/3600, 0.4 * 15, params['v_free'][0])
    # avg_tt = tts / np.apply_along_axis(get_num_veh, 1, demand_options, 10/3600) * 60
    delay = tts - ff_tt
    opt_delay = opt_tts - ff_tt
    percent_improvement = np.round((delay - opt_delay) / delay * 100, 1)

    # for k in demand_options.keys():
    #     print(f"Peak demand: {k[0]}, Duration: {k[1]}, Controllable Congestion: {percent_improvement[k]}%")

    # test_date = pd.read_csv('/Users/shreyaar/Desktop/PhD/research/MPC/results/default_0speed/tt_0.25/policy_4900.0/optimal_vsl.csv').to_numpy()
    # print(test_date.shape)

    fig, ax = plt.subplots(figsize=(10, 6))

    duration_gap = (durations[1] - durations[0])/2
    demand_gap = (peak_demand[1] - peak_demand[0])/2
    # Heatmap of percent improvement, peak demand on x-axis, peak duration on y-axis
    print(percent_improvement.shape)
    heatmap_data = np.array([percent_improvement[i][j] for i in range(len(durations)) for j in range(len(peak_demand))]).reshape(len(durations), len(peak_demand))
    print(heatmap_data.shape)
    cax = ax.imshow(heatmap_data.T, aspect='auto', cmap='viridis', origin='lower',
                   extent=[durations[0] - duration_gap, durations[-1] + duration_gap, p_min - demand_gap, p_max + demand_gap],
                   vmin=0, vmax=max(percent_improvement.flatten()))

    # Center x and y axis tick labels
    # ax.set_xticks(durations)
    # ax.set_xticklabels([f"{d:.2f}" for d in durations], ha='center')
    # yticks = np.linspace(p_min, p_max, len(peak_demand))
    # ax.set_yticks(yticks)
    # ax.set_yticklabels([f"{int(y)}" for y in yticks], va='center')
    ax.set_xlabel('Peak Duration (hours)', fontname='Times New Roman', fontsize=18)
    ax.set_ylabel('Peak Demand (veh/hr)', fontname='Times New Roman', fontsize=18)
    # ax.set_title('Controllable Congestion (%)', fontname='Times New Roman', fontsize=20)
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label('Controllable Congestion (%)', fontname='Times New Roman', fontsize=18)
    cbar.ax.tick_params(labelsize=14)   

    plt.show()

    # ax.scatter(peak_demand, percent_improvement)

    # for i, txt in enumerate(peak_demand):
    #     if i% 5 == 0:
    #         ax.annotate(int(txt), (avg_tt[i], percent_improvement[i]), xytext=(avg_tt[i], percent_improvement[i]+1), fontname='Times New Roman')

    # Add another axis on top that is peak_demand instead of percent improvement
    # ax2 = ax.twiny()
    # print(peak_demand)
    # ax2.plot(peak_demand, percent_improvement)
    # ax2.cla()
    # plt.xlabel('Average Travel Time without Control (min)', fontname='Times New Roman', fontsize=18)
    # plt.xlabel('Demand During 30 min Peak Period (veh/hr)', fontname='Times New Roman', fontsize=18)

    # plt.ylabel('Controllable Congestion (%)', fontname='Times New Roman', fontsize=18)
    # #Increase size of ticks
    # ax.tick_params(labelsize=14)
    # # move legend outside the plot
    # # plt.legend( loc='upper left')
    # plt.grid()
    # plt.show()

    # assert demand.shape == (10, 1800)
    # time_ = np.arange(0, demand_options.shape[1], 1) * 10/3600  # Convert to hours
    # for i in range(demand_options.shape[0]):
    #     plt.plot(time_, demand_options[i], label=f'Avg TT: {avg_tt[i]} min')
    # plt.xlabel('Time (hours)')
    # plt.ylabel('Traffic Demand (veh/hr)')
    # # plt.legend()
    # plt.show()