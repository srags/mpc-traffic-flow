import pandas as pd
from old_code.traffic_sim import *
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    hold_length_dict = {
        1: (6, 40),
        2: (6, 40),
        3: (6, 40),
        4: (8, 40),
        5: (10, 40),
        6: (12, 40),
        9: (18, 60),
        12: (12, 60),
        15: (15, 60),
        18: (18, 60)
    }

    hold_lens = np.array(list(hold_length_dict.keys()))

    sim_time = 1200
    traffic_demand = [5000 if i in range(150, 150 + 360) else 4000 for i in range(sim_time + 1)]
    downstream_density = np.full(sim_time + 1, 0)
    m = 15
    sim_lanes = {i: 4 if i < 10 else 2 for i in range(m)}
    time_step = 10/3600
    segment_len = 0.5

    no_control = metanet_sim(time_step, segment_len, (np.full(m, 0), np.full(m, 0), traffic_demand[0], 0), np.full((sim_time, m), v_free), traffic_demand, downstream_density, lanes=sim_lanes)[1]
    delay = 1002
    ttt_dict = dict()
    for hold_length in hold_lens:
        vsl = pd.read_csv(f"optimal_vsl_{sim_time}_holdlen{hold_length}_c{hold_length_dict[hold_length][0]}_p{hold_length_dict[hold_length][1]}.csv").to_numpy()

        # Simulate
        vsl_ttt = metanet_sim(time_step, segment_len, (np.full(m, 0), np.full(m, 0), traffic_demand[0], 0), vsl, traffic_demand, downstream_density, lanes=sim_lanes)[1]
        ttt_dict[hold_length] = vsl_ttt

    plt.plot(hold_lens*10, [(no_control - ttt_dict[i])/delay * 100 for i in hold_lens],marker='o', label='Total travel time')
    # X axis labels every 10
    plt.xticks(np.arange(0, 200, 20))
    # Overlay another bar behind to represent delay
    # plt.bar(hold_lens*10,delay, width=8, color='red', alpha=0.5, label='Delay')
    # plt.ylim(bottom=400)
    # Gridlines on plot
    plt.grid(axis='y')
    plt.grid(axis='x')
    
    plt.xlabel('Hold length (s)')
    plt.ylabel('Percentage of Delay reduced (%)')
    plt.savefig('hold_length.png')