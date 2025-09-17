import time
import json

import pyomo.environ as pyo
import numpy as np
import scipy.sparse as sp
from src.traffic_sim import *
import pandas as pd
import pyomo.contrib.parmest.utils.ipopt_solver_wrapper as ipopt_solver_wrapper
from pyomo.core import Constraint, value
from matplotlib import pyplot as plt

def append_number_csv(path, number):
    # 'a' mode: create file if it doesn't exist, then append
    with open(path, 'a', newline='') as f:
        f.write(f"{number}\n")
        f.flush()

def mpc_opt(T, l, num_segments, traffic_demand, downstream_density, horizon_p, horizon_c, starting_traffic_vars, lanes, hold_len, 
            initialize=None, init_fixed=None, control_one_segment=None, control_changepoints=None, safety_temporal=None, safety_spatial=None, prior_vsl=None, params=None, speed_lb=40, v_fd_penalty=0.1, control_zone=None):
    """
    Optimizes over horizon_p time steps given starting_traffic_vars as the boundary condition
    Parameters:
        T (float): METANET simulation time step
        l (float): METANET link length
        traffic_demand (list of size 'p_horizon'): traffic demand during prediction horizon
        downstream_density (list of size 'p_horizon'): downstream density during prediction horizon
        horizon_p (int): number of time steps in prediction horizon
        horizon_c (int): number of time steps in control horizon
        starting_traffic_vars: tuple of initial density, velocity, origin flow, and queue
        lanes (dict): dictionary of number of lanes for each segment
        hold_len (int): number of time steps to hold the control

    Returns:
        control (list of size horizon_c): list of vsl speeds for each time step in control horizon
    """

    # Set default parameters
    if params is None:
        v_free = [120 for i in range(num_segments)]
        a = [1.4 for i in range(num_segments)]
        p_crit = [37.45 for i in range(num_segments)]
        q_capacity = [2200 for i in range(num_segments)]
        K = [40 for i in range(num_segments)]
        tau = [18/3600 for i in range(num_segments)]
        eta_high = [30 for i in range(num_segments)]
        p_max = 180
    else:
        v_free = params['v_free']
        a = params['a']
        p_crit = params['p_crit']
        q_capacity = params['q_capacity']
        K = params['K']
        tau = params['tau']
        eta_high = params['eta_high']
        p_max = 180

    initial_density, initial_velocity, initial_flow_or, initial_queue = starting_traffic_vars
    time_steps = horizon_p + 1

    time_horizon = range(time_steps)
    control_horizon = range(horizon_c)
    segment_range = range(num_segments)

    if initialize is not None:
        if initialize.shape == (horizon_p, num_segments):
            init_provided_vsl = initialize
        else:
            add_vsl = horizon_p - initialize.shape[0]
            init_provided_vsl = np.vstack((initialize, np.ones((add_vsl, num_segments)) * min(v_free)))
    elif init_fixed is not None:
        init_provided_vsl = np.ones((horizon_p, num_segments)) * init_fixed
    else:
        init_provided_vsl = None

    def find_zone_vsl(i):
        if i < 5 and i >= 3:
            return 0.5 * v_free
        elif i < 8 and i >= 5:
            return 0.6 * v_free
        elif i >= 8:
            return 0.8 * v_free
        else:
            return v_free 
    
    def vsl_init(model, i, j):
        return init_provided_vsl[i, j] if (init_provided_vsl is not None and i < init_provided_vsl.shape[0]) else max(v_free)

    model = pyo.ConcreteModel()
    #suppress output

    model.vsl = pyo.Var(range(horizon_p), segment_range, bounds=(speed_lb,max(v_free)), within=pyo.NonNegativeReals)
    model.density = pyo.Var(time_horizon, segment_range, bounds=(1e-20,p_max), within=pyo.NonNegativeReals)
    model.velocity = pyo.Var(time_horizon, segment_range, bounds=(0, max(v_free)+30), within=pyo.NonNegativeReals)

    model.queue = pyo.Var(time_horizon, bounds=(-1e-6, 10000))
    model.v_fd = pyo.Var(range(horizon_p), segment_range, bounds=(0, max(v_free)), within=pyo.NonNegativeReals)
    model.queue_out = pyo.Var(time_horizon, bounds=(0, q_capacity[0] * lanes[0] + 50), within=pyo.NonNegativeReals) #q_capacity[0] * lanes[0]
    model.c = pyo.Var(time_horizon, bounds=(0,1), within=pyo.NonNegativeReals) # Aux variable for inflow min expression
    model.u_bin = pyo.Var(range(horizon_p), segment_range, bounds=(0,1))

    # if init_provided_vsl is not None:
    #     for r in range(init_provided_vsl.shape[0]):
    #         for c in segment_range:
    #             model.vsl[r, c] = init_provided_vsl[r, c]
    #             model.density[r, c] = density[r, c]
    #             model.velocity[r, c] = velocity[r, c]

    if init_provided_vsl is not None:
        # print("Warm starting solver with provided VSL speeds")
    # print(init_provided_vsl.shape)
        density, velocity, queue, flow_or, v_fd, tts = metanet_sim_params(T, l, starting_traffic_vars, init_provided_vsl, traffic_demand, downstream_density, params, real_data=False, lanes=lanes, opt=True)
        v_fd_min = np.minimum(v_fd, init_provided_vsl)
        v_fd_bin = np.where(v_fd < init_provided_vsl, 1, 0)
        # print(time_steps, num_segments)
        for r in range(0, time_steps):
            model.queue[r].value = queue[r, 0]
            model.queue_out[r].value = flow_or[r, 0]
            model.c[r].value = flow_or[r, 0] / (q_capacity[0] * lanes[0])

            for c in segment_range:
                model.density[r, c].value = density[r, c]
                model.velocity[r, c].value = velocity[r, c]
                if r < horizon_p:
                    model.vsl[r, c].value = init_provided_vsl[r, c]
                    model.v_fd[r, c].value = v_fd_min[r, c]
                    model.u_bin[r, c].value = v_fd_bin[r, c]


    # print(f"Initialized value of VSL: {pyo.value(model.vsl[0, 0])}")
    # print(f"Provided value of VSL: {initialize[0, 0]}")

    # print(f"Initialized value of density: {pyo.value(model.density[0, 0])}")
    # print(f"Provided value of density: {density[0, 0]}")

    # Slack variables for safety constraints
    # if safety_temporal is not None or safety_spatial is not None:
    #     model.slack_spatial_1 = pyo.Var(time_horizon, segment_range, within=pyo.NonNegativeReals)
    #     model.slack_spatial_2 = pyo.Var(time_horizon, segment_range, within=pyo.NonNegativeReals)

    #     model.slack_temporal_1 = pyo.Var(time_horizon, segment_range, within=pyo.NonNegativeReals)
    #     model.slack_temporal_2 = pyo.Var(time_horizon, segment_range, within=pyo.NonNegativeReals)

    # Constraints initialization
    model.constraints = pyo.ConstraintList()

    # Add hold length constraint
    for m in segment_range:
        for i in range(0, horizon_c, hold_len):
            for j in range(0, hold_len-1):
                model.constraints.add(model.vsl[i+j, m] == model.vsl[i+j+1, m])
    
    # Add changepoints constraint
    # For now, assume the list always starts with 0
    segment_lookup = dict()
    if control_changepoints is not None:
        # model.u_fixed = pyo.Var(time_horizon, range(len(control_changepoints)), within=pyo.NonNegativeReals)
        for i in range(len(control_changepoints)):
            start = control_changepoints[i]
            end = control_changepoints[i+1] if i+1 < len(control_changepoints) else num_segments
            for seg in range(start, end - 1):
                # segment_lookup[seg] = i
                for h in range(horizon_p):
                    model.constraints.add(model.vsl[h, seg] == model.vsl[h, seg+1])
            # for seg in range(start, end-1):
            #     for h in time_horizon:
            #         model.constraints.add(model.vsl[h, seg] == model.vsl[h, seg+1])
        
        # if control_changepoints[0] != 0:
        #     for m in range(control_changepoints[0]-1):
        #         for h in time_horizon:
        #             model.constraints.add(model.vsl[h, m] == model.vsl[h, m+1])
        
    #queueing and origin inflow constraints
    model.constraints.add(model.queue[0] == initial_queue)
    model.constraints.add(model.queue_out[0] == initial_flow_or)

    # for i in range(init_provided_vsl.shape[0] if init_provided_vsl is not None else 0):
    #     for m in segment_range:
    #         model.constraints.add(model.vsl[i, m] == init_provided_vsl[i, m])

    # for m in segment_range:
    #     if control_zone is not None and m not in control_zone:
    #         for h in range(horizon_p):
    #             model.constraints.add(model.vsl[h,m] == params['v_free'][m])
    
    for m in segment_range:
        # Initial state constraints
            model.constraints.add(model.density[0, m] == initial_density[m])
            model.constraints.add(model.velocity[0, m] == initial_velocity[m])



    for h in range(1, time_steps):
        # Queue constraints
        model.constraints.add(model.queue[h] == (model.queue[h-1] + T * (traffic_demand[h-1] - model.queue_out[h-1])))
        
        for m in segment_range:
            if (control_zone is not None and m not in control_zone) or (control_one_segment is not None and m != control_one_segment):
                model.constraints.add(model.v_fd[h-1, m] == v_free[m] * pyo.exp(-(1/a[m]) * (model.density[h-1, m] / p_crit[m])**a[m]))
                model.constraints.add(model.vsl[h-1, m] == v_free[m])
            # All segments are controlled independently
            else: #or (control_one_segment is None) or (control_one_segment is not None and m == control_one_segment):
                ## ORIGINAL FROM KIMIA PAPER
                # model.constraints.add(model.vsl[h-1, m] <= v_free[m] * pyo.exp(-(1/a[m]) * (model.density[h-1, m] / p_crit[m])**a[m]))
                # model.constraints.add(model.v_fd[h-1, m] == model.vsl[h-1, m])

                #Big M constraint
                M = max(v_free)
                model.constraints.add(model.v_fd[h-1, m] <= v_free[m] * pyo.exp(-(1/a[m]) * (model.density[h-1, m] / p_crit[m])**a[m]))
                model.constraints.add(model.v_fd[h-1, m] <= model.vsl[h-1, m])
                model.constraints.add(model.v_fd[h-1, m] >= v_free[m] * pyo.exp(-(1/a[m]) * (model.density[h-1, m] / p_crit[m])**a[m]) - M * (1 - model.u_bin[h-1, m]))
                model.constraints.add(model.v_fd[h-1, m] >= model.vsl[h-1, m] - M * model.u_bin[h-1, m])
                model.constraints.add(v_free[m] * pyo.exp(-(1/a[m]) * (model.density[h-1, m] / p_crit[m])**a[m]) - model.vsl[h-1, m] <= M * (1 - model.u_bin[h-1, m]))
                model.constraints.add(model.vsl[h-1, m] - v_free[m] * pyo.exp(-(1/a[m]) * (model.density[h-1, m] / p_crit[m])**a[m]) <= M * model.u_bin[h-1, m])

  
            if m == 0:
                # Density / Velocity constraints
                if num_segments == 1:
                    model.constraints.add(model.density[h, m] == model.density[h-1, m] + T/(l * lanes[m]) * (model.queue_out[h-1]  - model.density[h-1, m] * model.velocity[h-1, m] * lanes[m]))
                    model.constraints.add(model.velocity[h, m] == model.velocity[h-1, m] + (T / tau[m]) * (model.v_fd[h-1, m] - model.velocity[h-1, m]) 
                                            - (eta_high[m] * T / (tau[m] * l)) * ((downstream_density[h-1] - model.density[h-1, m]) / (model.density[h-1, m] + K[m])))
                else:
                    model.constraints.add(model.density[h, m] == model.density[h-1, m] + T/(l * lanes[m]) * (model.queue_out[h-1]  - model.density[h-1, m] * model.velocity[h-1, m] * lanes[m]))
                    model.constraints.add(model.velocity[h, m] == model.velocity[h-1, m] + (T / tau[m]) * (model.v_fd[h-1, m] - model.velocity[h-1, m]) 
                                            - (eta_high[m] * T / (tau[m] * l)) * ((model.density[h-1, m+1] - model.density[h-1, m]) / (model.density[h-1, m] + K[m])))
                
            elif m == num_segments - 1:
                # Density / Velocity constraints
                model.constraints.add(model.density[h, m] == model.density[h-1, m] 
                                        + T/(l * lanes[m]) * (model.density[h-1, m-1] * model.velocity[h-1, m-1] * lanes[m-1] - model.density[h-1, m] * model.velocity[h-1, m] * lanes[m]))
                model.constraints.add(model.velocity[h, m] == model.velocity[h-1, m] + (T / tau[m]) * (model.v_fd[h-1, m] - model.velocity[h-1, m]) 
                                        + (T/l) * model.velocity[h-1, m]* (model.velocity[h-1, m-1] - model.velocity[h-1, m])
                                        - (eta_high[m] * T / (tau[m] * l)) * ((downstream_density[h-1] - model.density[h-1, m]) / (model.density[h-1, m] + K[m])))
            else:
                # Density / Velocity constraints
                model.constraints.add(model.density[h, m] == model.density[h-1, m] 
                                        + T/(l * lanes[m]) * (model.density[h-1, m-1] * model.velocity[h-1, m-1] * lanes[m-1]  - model.density[h-1, m] * model.velocity[h-1, m] * lanes[m]))
                model.constraints.add(model.velocity[h, m] == model.velocity[h-1, m] + (T / tau[m]) * (model.v_fd[h-1, m] - model.velocity[h-1, m]) 
                                        + (T/l) * model.velocity[h-1, m] * (model.velocity[h-1, m-1] - model.velocity[h-1, m])
                                        - (eta_high[m] * T / (tau[m] * l)) * ((model.density[h-1, m+1] - model.density[h-1, m]) / (model.density[h-1, m] + K[m])))
        
        # Inflow constraint 
        model.constraints.add(model.c[h] <= (p_max - model.density[h, 0]) / (p_max - p_crit[0]))
        model.constraints.add(model.c[h] * q_capacity[0] * lanes[0] <= traffic_demand[h] + model.queue[h]/T)
        model.constraints.add(model.queue_out[h] == model.c[h] * q_capacity[0] * lanes[0])
        
    # Safety constraint such that vsl for a segment cannot change more than 10 km/hr in 10 seconds
    if safety_temporal is not None:
        for m in control_zone:
            if prior_vsl is not None:
                model.constraints.add(model.vsl[0, m] - prior_vsl[m] <= safety_temporal)
                model.constraints.add(prior_vsl[m] - model.vsl[0, m] <= safety_temporal)
            for h in range(1, horizon_c):
                model.constraints.add(model.vsl[h, m] - model.vsl[h-1, m] <= safety_temporal)
                model.constraints.add(model.vsl[h-1, m] - model.vsl[h, m] <= safety_temporal)

    if safety_spatial is not None:
        for m in control_zone[1:]:
            for h in range(0, horizon_c):
                model.constraints.add(model.vsl[h, m] - model.vsl[h, m-1] <= safety_spatial)
                model.constraints.add(model.vsl[h, m-1] - model.vsl[h, m] <= safety_spatial)

    # Define the objective function
    def objective_rule(model):
        return T * sum(model.density[h, m] * lanes[m] * l for h in range(1, time_steps) for m in segment_range) + T * sum(model.queue[h] for h in range(1, time_steps)) #+ sum(model.v_fd[h,m] for h in range(1, time_steps) for m in segment_range)  #- T * sum(model.velocity[h, m] for h in time_horizon for m in segment_range)

    def objective_rule_safety(model):
        return T * sum(model.density[h, m] * lanes[m] * l for h in range(1, time_steps) for m in segment_range) + T * sum(model.queue[h] for h in range(1, time_steps)) + T * sum(model.slack_temporal_1[h,m] + model.slack_temporal_2[h,m] + model.slack_spatial_1[h,m] + model.slack_spatial_2[h,m] for h in range(0, time_steps) for m in segment_range)

    def objective_rule_fd(model):
        return T * sum(model.density[h, m] * lanes[m] * l for h in range(1, time_steps) for m in segment_range) + T * sum(model.queue[h] for h in range(1, time_steps)) + v_fd_penalty * sum(model.u_bin[h, m] * (1 - model.u_bin[h,m]) for h in range(0, time_steps-1) for m in segment_range) # - v_fd_penalty * sum(model.v_fd[h,m] for h in range(0, time_steps-1) for m in segment_range)
    #model.objective = pyo.Objective(rule=objective_rule if (safety_temporal is not None or safety_spatial is not None) else objective_rule_safety, sense=pyo.minimize)
    model.objective = pyo.Objective(rule=objective_rule_fd, sense=pyo.minimize)
    # plt.imshow(density.T, aspect='auto', interpolation='nearest')
    # print(time_steps)

    # CHECK IF INIT IS FEASIBLE
    # if init_provided_vsl is not None:
    #     c_violated = 0
    #     for c in model.component_objects(Constraint, active=True):
    #         for idx in c:
    #             con = c[idx]
    #             body_val = value(con.body)
    #             if con.lower is not None and body_val < value(con.lower) - 1e-6:
    #                 # print(f"Violation (too low): {c.name}[{idx}] = {body_val} < {value(con.lower)}")
    #                 c_violated += 1
    #                 # con.pprint()
    #                 # print("Not using initialization")
    #             if con.upper is not None and body_val > value(con.upper) + 1e-6:
    #                 # print(f"Violation (too high): {c.name}[{idx}] = {body_val} > {value(con.upper)}")
    #                 c_violated += 1
    #                 # con.pprint()
        
    #     if c_violated > 0:
    #         print(f"Warm start initialization is infeasible, {c_violated} constraints violated. Not using initialization.")
    #     else:
    #         print(f"Warm start initialization is feasible, using initialization.")


    # Solve the problem with IPOPT
    # solver = pyo.SolverFactory('multistart')
    # results = solver.solve(model, solver='ipopt', strategy='midpoint_guess_and_bound', suppress_unbounded_warning=True)
    # print("Before solving obj:", pyo.value(model.objective))
    # print("Initialization TTS:", tts)
    # append_number_csv("/Users/shreyaar/Desktop/PhD/research/MPC/warmstart_obj_trends/extended_init.csv", pyo.value(model.objective))

    solver = pyo.SolverFactory('ipopt', executable='/usr/local/bin/ipopt')

    # solver.options['tol'] = 1e-4
    solver.options['acceptable_tol'] = 1e-3
    solver.options['warm_start_init_point'] = 'yes'
    solver.options['mu_init'] = 1e-6
    solver.options['warm_start_bound_push'] = 1e-6
    solver.options['warm_start_mult_bound_push'] = 1e-6
    # solver.options['max_iter'] = 5
    # solver.options['warm_start_init_point'] = 'yes'

    start_time = time.process_time()
    status, _, iters, _, _ = ipopt_solver_wrapper.ipopt_solve_with_stats(model, solver, max_iter=10000, warmstart=(init_provided_vsl is not None), tee=False)
    # result = solver.solve(model, tee=True)
    end_time = time.process_time()

    # print("After solving obj:", pyo.value(model.objective))
    # append_number_csv("/Users/shreyaar/Desktop/PhD/research/MPC/warmstart_obj_trends/bad_form_initialized_with_extended.csv", pyo.value(model.objective))

    # Extract results
    density = np.array([[pyo.value(model.density[i, j]) for j in range(num_segments)] for i in range(0, time_steps)])
    queue = np.array([pyo.value(model.queue[i]) for i in range(0, time_steps)])
    vsl_speeds = np.array([[pyo.value(model.vsl[i, j]) for j in range(num_segments)] for i in range(0, horizon_c)])

    vsl_speeds_p = np.array([[pyo.value(model.vsl[i, j]) for j in range(num_segments)] for i in range(0, horizon_p)])
    u_bin = np.array([[pyo.value(model.u_bin[i, j]) for j in range(num_segments)] for i in range(0, horizon_p)])

    # Check if u_bin is binary
    # if np.any(np.where((u_bin > 1e-5) | (u_bin < 1 - 1e-5), False, True), axis=None, out=None, keepdims=False):
    #         print("u_bin is not all binary, some values are:", u_bin)
    # else:
    #     print("u_bin is all binary")
    
    # print("TTS after solving:", T * sum(density[i, j] * lanes[j] * l for i in range(0, horizon_p + 1) for j in segment_range) + T * sum(queue[i] for i in range(0, horizon_p + 1)))
    # print("TTS after simulation:", metanet_sim_params(T, l, starting_traffic_vars, vsl_speeds_p, traffic_demand, downstream_density, params, real_data=False, lanes=lanes, opt=True)[-1])

    # plt.imshow(density.T, aspect='auto', interpolation='nearest')
    # flip y axis
    # plt.gca().invert_yaxis()
    # print("VSL speeds found:", vsl_speeds)
    # print("VSL speed initialized:", init_provided_vsl[0:horizon_c, :] if init_provided_vsl is not None else "No initialization provided")

    # v_fd_speeds = np.array([[pyo.value(model.v_fd[i, j]) for j in range(num_segments)] for i in range(0, horizon_c)])
    # print("VFD speeds:", v_fd_speeds)
    # print("VSL speeds:", vsl_speeds)

    # Print parts of the objective
    # if params is not None:
    #     print("Objective function value:", pyo.value(model.objective))
    #     print("Total density + queue:", T * sum(pyo.value(model.density[i, j]) * lanes[j] * l for i in range(1, time_steps) for j in segment_range) + T * sum(pyo.value(model.queue[i]) for i in range(1, time_steps)))
    #     print("Total slack variable:", T * sum(pyo.value(model.slack_spatial_1[h,m]) + pyo.value(model.slack_spatial_2[h,m]) for h in range(0, time_steps) for m in segment_range))
    #     print("--------")

    if status.solver.termination_condition == pyo.TerminationCondition.infeasible:
        print("Infeasible")
        
    ytime = end_time - start_time #result.solver.wallclock_time
    # iters = 0
    return iters, ytime, vsl_speeds

def mpc_find_vsl(total_time_steps, traffic_demand, downstream_density, lanes, T=5/3600, l=300/1000, num_segments=10, pred_horizon=20, control_horizon=20, hold_length=1, init_state=None,
                 control_one_segment = None, initialize_vsl=None, init_fixed=None, control_changepoints=None, safety_temporal=None, safety_spatial=None, params=None, verbose=False, speed_lb=40, v_fd_penalty=0.1, control_zone=None):
    
    t = 0
    if init_state is not None:
        init_traffic_state = init_state
    else:
        init_density = np.array([traffic_demand[0]/(lanes[0] * 90) for i in range(num_segments)]) # traffic_demand[0]/90)
        init_velocity = np.full(num_segments, 90) #90)
        init_flow_or = traffic_demand[0]
        init_queue = 0
        init_traffic_state = (init_density, init_velocity, init_flow_or, init_queue)
    # init_vsl = np.full(num_segments, 100)
    sim_time = total_time_steps - pred_horizon + control_horizon
    full_control = None
    
    solve_time = 0
    iterations = 0

    while t + pred_horizon <= total_time_steps:
        if initialize_vsl is not None:
            init_vsl_step = initialize_vsl[t:t+pred_horizon]
        else:
            init_vsl_step = None

        if verbose and t%pred_horizon == 0:
            print(f"MPC at time step {t}")

        prior_vsl = None if t == 0 else full_control[-1:, :][0]

        stime, ytime, vsl_control = mpc_opt(T, l, num_segments, traffic_demand[t:t+pred_horizon+1], downstream_density[t:t+pred_horizon+1], 
                                pred_horizon, control_horizon, init_traffic_state, lanes, 
                                hold_len=hold_length, initialize=init_vsl_step, init_fixed=init_fixed, 
                                control_one_segment=control_one_segment, control_changepoints=control_changepoints, 
                                safety_temporal=safety_temporal, safety_spatial=safety_spatial, prior_vsl=prior_vsl, 
                                params=params, speed_lb=speed_lb, v_fd_penalty=v_fd_penalty, control_zone=control_zone)
        # print(vsl_control)
        full_control = np.vstack((full_control, vsl_control)) if t > 0 else vsl_control
        if params is not None:
            new_traffic_state = metanet_sim_params(T, l, init_traffic_state, vsl_control, traffic_demand[t:t+control_horizon+1], downstream_density[t:t+control_horizon], params, real_data=False, lanes=lanes)[0] # traffic state at (t + C)
        else:
            new_traffic_state = metanet_sim(T, l, init_traffic_state, vsl_control, traffic_demand[t:t+control_horizon+1], downstream_density[t:t+control_horizon], lanes=lanes)[0]
        init_traffic_state = new_traffic_state
        t += control_horizon
        solve_time += ytime
        iterations += stime
    
    if t < sim_time:
        # print(t)
        init_vsl_step = initialize_vsl[t:] if initialize_vsl is not None else None
        # print(init_vsl_step.shape)

        # print(f"MPC at time step {t}")
        stime, ytime, vsl_control = mpc_opt(T, l, num_segments, traffic_demand[t:], downstream_density[t:], 
                                total_time_steps-t, sim_time-t, init_traffic_state, lanes, 
                                hold_len=hold_length, initialize=init_vsl_step, init_fixed=init_fixed, 
                                control_one_segment=control_one_segment, control_changepoints=control_changepoints,
                                safety_temporal=safety_temporal, safety_spatial=safety_spatial, prior_vsl=full_control[-1:, :][0], 
                                params=params, speed_lb=speed_lb, v_fd_penalty=v_fd_penalty, control_zone=control_zone)
        full_control = np.vstack((full_control, vsl_control))
        solve_time += ytime
        iterations += stime

    # print(f"{solve_time} seconds")
    # print(f"{iterations} iterations")
    # num_optimizations = total_time_steps // control_horizon if total_time_steps % control_horizon == 0 else total_time_steps // control_horizon + 1
    # print(f"{solve_time/num_optimizations} seconds per optimization problem")
    # print(f"{iterations/num_optimizations} iterations per optimization problem")
    print("MPC Done")
    
    return full_control
