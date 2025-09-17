import numpy as np
import pandas as pd
from sympy import divisors

def warm_starts_min_speedlimit(i, sim_time, num_segments, params, control_zone, path="/Users/shreyaar/Desktop/PhD/research/MPC/results/final_extended/min_speedlimit"):
    warm_starts = dict()
    # warm_starts['free_flow'] = np.full((sim_time, num_segments), params['v_free'])
    warm_starts['min_speed'] = np.full((sim_time, num_segments), params['v_free'])
    warm_starts['min_speed'][:, control_zone] = i

    try:
        prev_vsl = np.array(pd.read_csv(f"{path}/speedlimit{i-5}.csv"))
        warm_starts['prev_vsl'] = prev_vsl

    except FileNotFoundError:
        pass
    
    try:
        opt_vsl = np.array(pd.read_csv(f"{path}/speedlimit{0}.csv"))
        warm_starts['opt_vsl'] = opt_vsl
        warm_starts['opt_vsl_clipped'] = np.clip(opt_vsl, i, None)
    except FileNotFoundError:
        pass

    return warm_starts

def warm_starts_hold_len(i, sim_time, num_segments, params, control_zone, path="/Users/shreyaar/Desktop/PhD/research/MPC/results/final_extended/holdlength"):
    warm_starts = dict()
    warm_starts['free_flow'] = np.full((sim_time, num_segments), params['v_free'])
    warm_starts['min_speed'] = np.full((sim_time, num_segments), params['v_free'])
    warm_starts['min_speed'][:, control_zone] = 40

    #warm_starts['current'] = np.array(pd.read_csv(f"{path}/holdlen{i}.csv"))

    if i > 1:
        try:
            opt_vsl = np.array(pd.read_csv(f"{path}/holdlen{1}.csv"))
            warm_starts['opt_vsl'] = opt_vsl

            opt_reshape = opt_vsl.reshape(sim_time//30, 30, -1)
            opt_means = np.mean(opt_reshape, axis=1, keepdims=True)
            warm_starts['opt_vsl_averaged'] = np.repeat(opt_means, 30, axis=1).reshape(sim_time, num_segments)
            assert warm_starts['opt_vsl_averaged'].shape == (sim_time, num_segments)
        except FileNotFoundError:
            pass

    try:
        divisors_list = divisors(sim_time)
        ind_i = divisors_list.index(i)
        if ind_i > 0:
            hold_len_prev = divisors_list[ind_i - 1]
            prev_vsl = np.array(pd.read_csv(f"{path}/holdlen{hold_len_prev}.csv"))
            warm_starts['prev_vsl'] = prev_vsl
    except FileNotFoundError:
        pass

    # divisors_list = divisors(sim_time)
    # for div in divisors_list:
    #     if div % i ==0 and div !=i:
    #         try:
    #             warm_starts[f'holdlen_div{div}'] = np.array(pd.read_csv(f"{path}/holdlen{div}.csv"))
    #         except FileNotFoundError:
    #             pass

    #         break

    try:
        divisors_list = divisors(sim_time)
        ind_i = divisors_list.index(i)
        hold_len_after = divisors_list[ind_i + 1]
        print(hold_len_after)
        after_vsl = np.array(pd.read_csv(f"{path}/holdlen{hold_len_after}.csv"))
        warm_starts['after_vsl'] = after_vsl
    except FileNotFoundError:
        pass
    
    # try:
    #     vsl_5 = np.array(pd.read_csv(f"{path}/holdlen{5}.csv"))
    #     warm_starts['vsl_5'] = vsl_5
    # except FileNotFoundError:
    #     pass

    # try:
    #     vsl_40 = np.array(pd.read_csv(f"{path}/holdlen{40}.csv"))
    #     warm_starts['vsl_40'] = vsl_40
    # except FileNotFoundError:
    #     pass

    return warm_starts

def warm_starts_hold_len_shifted(i, sim_time, num_segments, params, control_zone, path="/Users/shreyaar/Desktop/PhD/research/MPC/results/final_extended/holdlength_shifted"):
    warm_starts = dict()
    warm_starts['free_flow'] = np.full((sim_time, num_segments), params['v_free'])
    warm_starts['min_speed'] = np.full((sim_time, num_segments), params['v_free'])
    warm_starts['min_speed'][:, control_zone] = 40

    if i > 1:
        try:
            opt_vsl = np.array(pd.read_csv(f"{path}/holdlen{1}.csv"))
            warm_starts['opt_vsl'] = opt_vsl

            opt_reshape = opt_vsl.reshape(sim_time//30, 30, -1)
            opt_means = np.mean(opt_reshape, axis=1, keepdims=True)
            warm_starts['opt_vsl_averaged'] = np.repeat(opt_means, 30, axis=1).reshape(sim_time, num_segments)
            assert warm_starts['opt_vsl_averaged'].shape == (sim_time, num_segments)
        except FileNotFoundError:
            pass

    try:
        divisors_list = divisors(sim_time)
        ind_i = divisors_list.index(i)
        if ind_i > 0:
            hold_len_prev = divisors_list[ind_i - 1]
            prev_vsl = np.array(pd.read_csv(f"{path}/holdlen{hold_len_prev}.csv"))
            warm_starts['prev_vsl'] = prev_vsl
    except FileNotFoundError:
        pass
    
    try:
        unshifted = np.array(pd.read_csv(f"{path}/holdlen{i}.csv"))
        warm_starts['unshifted'] = unshifted
    except FileNotFoundError:
        pass

    return warm_starts

def warm_starts_spatial_safety(i, sim_time, num_segments, params, control_zone, path="/Users/shreyaar/Desktop/PhD/research/MPC/results/final_extended/spatial_safety"):
    warm_starts = dict()
    warm_starts['min_speed'] = np.full((sim_time, num_segments), params['v_free'])
    warm_starts['min_speed'][:, control_zone] = 40

    try:
        opt_vsl = np.array(pd.read_csv(f"{path}/safety{60}.csv"))
        warm_starts['opt_vsl'] = opt_vsl

    except FileNotFoundError:
        pass

    try:
        prev_vsl = np.array(pd.read_csv(f"{path}/safety{i+2}.csv"))
        warm_starts['prev_vsl'] = prev_vsl

    except FileNotFoundError:
        pass

    try:
        prev_vsl2 = np.array(pd.read_csv(f"{path}/safety{i-2}.csv"))
        warm_starts['prev_vsl2'] = prev_vsl2

    except FileNotFoundError:
        pass

    # try:
    #     prev_vsl = np.array(pd.read_csv(f"/Users/shreyaar/Desktop/PhD/research/MPC/results/final_extended/spatial_safety/safety{i-5}.csv"))
    #     warm_starts['prev_vsl'] = prev_vsl

    # except FileNotFoundError:
    #     pass

    # try:
    #     orig_vsl = np.array(pd.read_csv(f"/Users/shreyaar/Desktop/PhD/research/MPC/results/final_extended/spatial_safety/safety{i}.csv"))
    #     warm_starts['orig_vsl'] = orig_vsl

    # except FileNotFoundError:
    #     pass

    return warm_starts

def warm_starts_temporal_safety(i, sim_time, num_segments, params, control_zone, path="/Users/shreyaar/Desktop/PhD/research/MPC/results/final_extended/temporal_safety"):
    warm_starts = dict()
    warm_starts['free_flow'] = np.full((sim_time, num_segments), params['v_free'])
    warm_starts['min_speed'] = np.full((sim_time, num_segments), params['v_free'])
    warm_starts['min_speed'][:, control_zone] = 40

    # try:
    #     opt_vsl = np.array(pd.read_csv(f"/Users/shreyaar/Desktop/PhD/research/MPC/results/final_extended/temporal_safety/safety{40}.csv"))
    #     warm_starts['opt_vsl'] = opt_vsl

    # except FileNotFoundError:
    #     pass

    # try:
    #     prev_vsl = np.array(pd.read_csv(f"/Users/shreyaar/Desktop/PhD/research/MPC/results/final_extended/temporal_safety/safety{i+5}.csv"))
    #     warm_starts['prev_vsl'] = prev_vsl

    # except FileNotFoundError:
    #     pass

    # try:
    #     prev_vsl = np.array(pd.read_csv(f"/Users/shreyaar/Desktop/PhD/research/MPC/results/final_extended/temporal_safety/safety{i+1}.csv"))
    #     warm_starts['prev_vsl'] = prev_vsl

    # except FileNotFoundError:
    #     pass

    try:
        prev_vsl = np.array(pd.read_csv(f"{path}/safety{1}.csv"))
        warm_starts['prev_vsl'] = prev_vsl

    except FileNotFoundError:
        pass

    return warm_starts

def warm_starts_one_gantry(i, sim_time, num_segments, params, control_zone, path="/Users/shreyaar/Desktop/PhD/research/MPC/results/final_extended/gantry"):
    warm_starts = dict()
    warm_starts['free_flow'] = np.full((sim_time, num_segments), params['v_free'])
    warm_starts['min_speed'] = np.full((sim_time, num_segments), params['v_free'])
    warm_starts['min_speed'][:, control_zone] = 40

    try:
        opt_vsl = np.array(pd.read_csv(f"{path}/speedlimit{0}.csv"))
        warm_starts['opt_vsl'] = opt_vsl

    except FileNotFoundError:
        pass
    
    try:
        prev_vsl = np.array(pd.read_csv(f"{path}/location{i-1}.csv"))
        warm_starts['prev_vsl'] = prev_vsl
    except FileNotFoundError:
        pass

    return warm_starts
