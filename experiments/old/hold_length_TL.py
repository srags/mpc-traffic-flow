import pandas as pd
from old_code.traffic_sim import *
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    hold_length_dict = {
        1: (5, 20),
        2: (6, 20),
        3: (6, 30),
        4: (8, 40),
        5: (10, 30),
        6: (12, 40),
        # 9: (18, 40),
        # 12: (12, 60),
        # 15: (15, 60),
        # 18: (18, 60)
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
    
    same_init = dict()
    best_init_ttt = dict()
    best_init_length = dict()

    for hold_length in hold_lens:
        best_ttt = np.inf
        best_source = None

        for source in hold_lens:
            vsl = pd.read_csv(f"holdlen_TL/policy_holdlen_{hold_length}/init_holdlen{source}.csv").to_numpy()

            # Simulate
            vsl_ttt = metanet_sim(time_step, segment_len, (np.full(m, 0), np.full(m, 0), traffic_demand[0], 0), vsl, traffic_demand, downstream_density, lanes=sim_lanes)[1]
            print(hold_length, source, vsl_ttt)
            if vsl_ttt < best_ttt:
                best_ttt = vsl_ttt
                best_source = source
            
            if source == hold_length:
                same_init[hold_length] = vsl_ttt
            
        best_init_ttt[hold_length] = best_ttt
        best_init_length[hold_length] = best_source
    
    #plot same init
    plt.plot(hold_lens, [same_init[i] for i in hold_lens], label='Same Initialization',  marker='o')
    plt.plot(hold_lens, [best_init_ttt[i] for i in hold_lens], label='Best Initialization',  marker='o')
    
    print(best_init_length)
    plt.savefig('hold_length_TL.png')

