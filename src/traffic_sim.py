import numpy as np

# # Network parameters
# tau = 18/3600 #33
# p_crit = 37.45 #27.6 #31.1
# a = 1.4 #2.5 #2.9
# v_free = 120
# p_max = 180 #84.5
# K = 40 #10
# eta_high = 30 #21.3
# eta_low = 60
# q_capacity = 2200
# v_start = 100 # starting speed when vehicles enter the freeway

## Dynamics for METANET
def density_dynamics(current, inflow, outflow, lanes, T, l):
    return current + T/(l * lanes) * (inflow - outflow)

def flow_dynamics(density, velocity, lanes):
    # if density * velocity > q_capacity:
    #     print(density, velocity, lanes)
    return density * velocity * lanes

def queue_dynamics(current, demand, flow_origin, T):
    return current + T * (demand - flow_origin)

def calculate_V(rho, v_ctrl, a, p_crit, v_free):
    return min(v_free * np.exp(-1/a * (rho / p_crit)**a), v_ctrl)

def calculate_V_arr(rho_arr, v_ctrl_arr, a, p_crit, v_free):
    return np.minimum(v_free * np.exp(-1/a * (rho_arr / p_crit)**a), v_ctrl_arr)

def velocity_dynamics_MN(current, prev_state, density, next_density, v_ctrl, T, l, eta_high=30, K=40, tau=18/3600, a=1.4, p_crit=37.45, v_free=120, perturbation=False):
    scaling = 0.2 if perturbation else 1
    next =  current + T/tau * (calculate_V(density, v_ctrl, a, p_crit, v_free) * scaling - current) + T/l * current * (prev_state - current) - (eta_high * T) / (tau * l) * (next_density - density) / (density + K)
    return max(0, next) #if current_density == 0 else min(max(0, next), q_capacity/current_density)

def origin_flow_dynamics_MN(demand, density_first, queue, lanes, T, p_max=180, p_crit=37.45, q_capacity=2200):
    return min(demand + queue / T, lanes * q_capacity * (p_max - density_first)/(p_max - p_crit), lanes * q_capacity)

# From linearized METANET
# def velocity_dynamics(current, current_estimate, prev_state, density, next_density, V_term, measured_density, T, l):
#     next =  current + T/tau * (V_term - current) + T/l * current_estimate * (prev_state - current) - (eta_high * T) / (tau * l) * (next_density - density) / (measured_density + K)
#     return next #max(0, next)

def metanet_sim(T, l, init_traffic_state, vsl_speeds, demand, downstream_density, lanes=None, plotting=False, perturbation=None):
    if not lanes:
        lanes = {i: 1 for i in range(vsl_speeds.shape[1])}

    initial_density, initial_velocity, initial_flow_or, initial_queue = init_traffic_state

    time_steps, num_segments = vsl_speeds.shape
    # Initialize state variables (velocity, density, flow, demand, queue, origin flow) for simulation
    velocity = np.zeros((time_steps + 1, num_segments))
    density = np.zeros((time_steps + 1, num_segments))
    flow = np.zeros((time_steps + 1, num_segments)) 
    queue = np.zeros((time_steps + 1, 1))
    flow_origin = np.zeros((time_steps + 1, 1))

    # initial conditions
    density[0] = initial_density
    velocity[0] = initial_velocity
    flow[0] = np.array([initial_density[i] * initial_velocity[i] * lanes[i] for i in range(num_segments)])
    flow_origin[0, 0] = initial_flow_or
    queue[0, 0] = initial_queue

    # Step forward simulation of velocity, density, and flow
    for t in range(0, time_steps):

        # Update density
        for i in range(num_segments):
            if i == 0:
                density[t + 1, i] = density_dynamics(density[t, i], flow_origin[t, 0], flow[t, i], lanes[i], T, l)
            else:
                density[t + 1, i] = density_dynamics(density[t, i], flow[t, i-1], flow[t, i], lanes[i], T, l)

        # Update velocity
        for i in range(num_segments):
            p = (perturbation is not None) and (i == perturbation[0]) and (t >= perturbation[1]) and (t < perturbation[1] + int(300/(T *  3600)))
            if i == 0:
                velocity[t + 1, i] = velocity_dynamics_MN(velocity[t, i], velocity[t, i], density[t, i], density[t, i + 1], vsl_speeds[t, i], T, l, perturbation=p)
            elif i == num_segments - 1:
                velocity[t + 1, i] = velocity_dynamics_MN(velocity[t, i], velocity[t, i - 1], density[t, i], downstream_density[t], vsl_speeds[t, i], T, l, perturbation=p)
            else:
                velocity[t + 1, i] = velocity_dynamics_MN(velocity[t, i], velocity[t, i - 1], density[t, i], density[t, i + 1], vsl_speeds[t, i], T, l, perturbation=p)

        # Update queue
        queue[t + 1, 0] = queue_dynamics(queue[t, 0], demand[t], flow_origin[t, 0], T)

        # Update flow
        for i in range(num_segments):
            flow[t+1, i] = flow_dynamics(density[t+1, i], velocity[t+1, i], lanes[i])
        
        # Update origin flow
        flow_origin[t+1, 0] = origin_flow_dynamics_MN(demand[t+1], density[t+1, 0], queue[t+1, 0], lanes[0], T)
    
    total_travel_time = T * (sum([np.sum(density[:, i]) * lanes[i] * l for i in range(num_segments)]) + np.sum(queue))
    # print(density)
    if plotting:
        return density, velocity, flow_origin, total_travel_time
    else:
        return (density[-1], velocity[-1], flow_origin[-1, 0], queue[-1, 0]), total_travel_time

def metanet_sim_params(T, l, init_traffic_state, vsl_speeds, demand, downstream_density, params, lanes=None, plotting=False, perturbation=None, real_data=False, opt=False):
    if not lanes:
        lanes = {i: 1 for i in range(vsl_speeds.shape[1])}

    initial_density, initial_velocity, initial_flow_or, initial_queue = init_traffic_state

    time_steps, num_segments = vsl_speeds.shape
    # Initialize state variables (velocity, density, flow, demand, queue, origin flow) for simulation
    velocity = np.zeros((time_steps + 1, num_segments))
    density = np.zeros((time_steps + 1, num_segments))
    flow = np.zeros((time_steps + 1, num_segments)) 
    queue = np.zeros((time_steps + 1, 1))
    flow_origin = np.zeros((time_steps + 1, 1))

    # initial conditions
    density[0] = initial_density
    velocity[0] = initial_velocity
    flow[0] = np.array([initial_density[i] * initial_velocity[i] * lanes[i] for i in range(num_segments)])
    flow_origin[0, 0] = initial_flow_or
    queue[0, 0] = initial_queue

    # Step forward simulation of velocity, density, and flow
    for t in range(0, time_steps):
        # Update density
        for i in range(num_segments):
            if i == 0:
                if real_data:
                    density[t + 1, i] = density_dynamics(density[t, i], demand[t], density[t, i] * velocity[t, i] * lanes[i], lanes[i], T, l)
                else:
                    density[t + 1, i] = density_dynamics(density[t, i], flow_origin[t, 0], density[t, i] * velocity[t, i] * lanes[i], lanes[i], T, l)
            else:
                density[t + 1, i] = density_dynamics(density[t, i], density[t, i-1] * velocity[t, i-1] * lanes[i-1], density[t, i] * velocity[t, i] * lanes[i], lanes[i], T, l)

        # Update velocity
        for i in range(num_segments):
            p = (perturbation is not None) and (i == perturbation[0]) and (t >= perturbation[1]) and (t < perturbation[1] + int(450/(T *  3600)))
            if num_segments == 1:
                velocity[t + 1, i] = velocity_dynamics_MN(velocity[t, i], velocity[t, i], density[t, i], downstream_density[t], vsl_speeds[t, i], T, l, 
                                                          eta_high=params['eta_high'][i], K=params['K'][i], tau=params['tau'][i], a=params['a'][i], p_crit=params['p_crit'][i], v_free=params['v_free'][i], perturbation=p)
            elif i == 0:
                velocity[t + 1, i] = velocity_dynamics_MN(velocity[t, i], velocity[t, i], density[t, i], density[t, i + 1], vsl_speeds[t, i], T, l, 
                                                          eta_high=params['eta_high'][i], K=params['K'][i], tau=params['tau'][i], a=params['a'][i], p_crit=params['p_crit'][i], v_free=params['v_free'][i], perturbation=p)
            elif i == num_segments - 1:
                velocity[t + 1, i] = velocity_dynamics_MN(velocity[t, i], velocity[t, i - 1], density[t, i], downstream_density[t], vsl_speeds[t, i], T, l,
                                                          eta_high=params['eta_high'][i], K=params['K'][i], tau=params['tau'][i], a=params['a'][i], p_crit=params['p_crit'][i], v_free=params['v_free'][i], perturbation=p)
            else:
                velocity[t + 1, i] = velocity_dynamics_MN(velocity[t, i], velocity[t, i - 1], density[t, i], density[t, i + 1], vsl_speeds[t, i], T, l,
                                                          eta_high=params['eta_high'][i], K=params['K'][i], tau=params['tau'][i], a=params['a'][i], p_crit=params['p_crit'][i], v_free=params['v_free'][i], perturbation=p)

        # Update queue
        if not real_data:
            queue[t + 1, 0] = queue_dynamics(queue[t, 0], demand[t], flow_origin[t, 0], T)

        # Update flow
        for i in range(num_segments):
            flow[t+1, i] = flow_dynamics(density[t+1, i], velocity[t+1, i], lanes[i])
        
        # Update origin flow
        if not real_data:
            flow_origin[t+1, 0] = origin_flow_dynamics_MN(demand[t+1], density[t+1, 0], queue[t+1, 0], lanes[0], T, p_max=180, p_crit=params['p_crit'][i], q_capacity=params['q_capacity'][i])
    
    total_travel_time = T * (sum([np.sum(density[:, i]) * lanes[i] * l for i in range(num_segments)]) + np.sum(queue))
    # print(density)
    if plotting:
        return density, velocity, queue, total_travel_time
    elif opt:
        V_fd = calculate_V_arr(density[0:-1], vsl_speeds, params['a'][0], params['p_crit'][0], params['v_free'][0])
        return density, velocity, queue, flow_origin, V_fd, total_travel_time
    else:
        return (density[-1], velocity[-1], flow_origin[-1, 0], queue[-1, 0]), total_travel_time
    
