import time
import json

import pyomo.environ as pyo
from amplpy import modules

import numpy as np
import scipy.sparse as sp
from traffic_sim import run_metanet_sim, _get_time_space_param
import pandas as pd
import pyomo.contrib.parmest.utils.ipopt_solver_wrapper as ipopt_solver_wrapper
from pyomo.util.infeasible import log_infeasible_constraints
import logging


def append_number_csv(path, number):
    with open(path, 'a', newline='') as f:
        f.write(f"{number}\n")
        f.flush()

from scipy.optimize import minimize

def mpc_opt_shooting(T, l, num_segments, traffic_demand, downstream_density,
                     horizon_p, horizon_c, starting_traffic_vars, lanes,
                     params, speed_lb=40):

    def objective(vsl_flat):
        vsl = vsl_flat.reshape(horizon_p, num_segments)
        # clip to bounds before simulating
        vsl = np.clip(vsl, speed_lb, 150)
        density, velocity, flow_or, queue = run_metanet_sim(
            T, l, starting_traffic_vars,
            traffic_demand, downstream_density,
            params, vsl_speeds=vsl, lanes=lanes, real_data=False
        )
        tts = T * np.sum(density * lanes * l) + T * np.sum(queue)
        return tts

    # warm start at free flow
    vsl0 = np.full((horizon_p, num_segments), 120.0).flatten()
    bounds = [(speed_lb, 150)] * len(vsl0)

    result = minimize(
        objective, vsl0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-6}
    )

    vsl_opt = result.x.reshape(horizon_p, num_segments)
    return vsl_opt[:horizon_c]


def mpc_opt(
    T, l, num_segments,
    traffic_demand, downstream_density,
    horizon_p, horizon_c,
    starting_traffic_vars, lanes, hold_len,
    initialize=None, init_fixed=None,
    control_one_segment=None, control_changepoints=None,
    safety_temporal=None, safety_spatial=None,
    prior_vsl=None, params=None,
    speed_lb=40, v_fd_penalty=0.1, control_zone=None, tee=False
):
    # initial_density, initial_velocity, initial_flow_or, initial_queue = starting_traffic_vars
    # print("density range:", initial_density.min(), initial_density.max())
    # print("velocity range:", initial_velocity.min(), initial_velocity.max())
    # print("queue:", initial_queue, "  queue_out/flow_or:", initial_flow_or)
    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    if params is None:
        v_free   = [120.0] * num_segments
        a        = [1.4]   * num_segments
        p_crit   = [37.45] * num_segments
        q_cap    = [2200.0]* num_segments
        K        = [40.0]  * num_segments
        tau      = [18/3600] * num_segments
        eta_high = [30.0]  * num_segments
        r        = [0.0]   * num_segments
        beta     = [0.0]   * num_segments
    else:
        v_free   = params['v_free']
        a        = params['a']
        p_crit   = params['p_crit']
        q_cap    = params['q_capacity']
        K        = params['K']
        tau      = params['tau']
        eta_high = params['eta_high']
        r        = params['r']
        beta     = params['beta']

    p_max = 180.0
    initial_density, initial_velocity, initial_flow_or, initial_queue = starting_traffic_vars

    time_steps   = horizon_p + 1
    time_horizon = range(time_steps)
    seg_range    = range(num_segments)

    # ------------------------------------------------------------------
    # Warm-start VSL array
    # ------------------------------------------------------------------
    if initialize is not None:
        if initialize.shape[0] >= horizon_p + 1:
            init_vsl = initialize[:horizon_p + 1]
        else:
            pad = np.ones((horizon_p + 1 - initialize.shape[0], num_segments)) * min(v_free)
            init_vsl = np.vstack((initialize, pad))
    elif init_fixed is not None:
        if init_fixed == "adaptive":
            init_fixed_value = float(np.mean(initial_velocity))
        else:
            init_fixed_value = init_fixed
        init_vsl = np.full((horizon_p + 1, num_segments), init_fixed_value)
    else:
        init_vsl = None

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = pyo.ConcreteModel()

    def vsl_bounds(model, h, m):
        if control_zone is not None and m not in control_zone:
            return (1e-4, 150)   # let it track v_free freely
        return (speed_lb, 150)

    model.vsl = pyo.Var(range(horizon_p), seg_range,
                        bounds=vsl_bounds, within=pyo.NonNegativeReals)
    model.density  = pyo.Var(time_horizon, seg_range,
                             bounds=(1e-4, 400), within=pyo.NonNegativeReals)
    model.velocity = pyo.Var(time_horizon, seg_range,
                             bounds=(1e-4, float(np.max(v_free)) + 30), within=pyo.NonNegativeReals)
    model.queue     = pyo.Var(time_horizon, bounds=(0, 10000))
    if len(q_cap.shape) == 2:
        model.queue_out = pyo.Var(time_horizon, bounds=(0, q_cap[0][0] * lanes[0]),
                                within=pyo.NonNegativeReals)
    else:
        model.queue_out = pyo.Var(time_horizon, bounds=(0, q_cap[0] * lanes[0]),
                        within=pyo.NonNegativeReals)
    model.v_fd  = pyo.Var(range(horizon_p), seg_range,
                          bounds=(0, float(np.max(v_free)) + 30), within=pyo.NonNegativeReals)
    # model.u_bin = pyo.Var(range(horizon_p), seg_range,
    #                       bounds=(0, 1), within=pyo.NonNegativeReals)
    # c[t] ∈ [0,1]: fraction of capacity entering the motorway (METANET merge model)
    model.c = pyo.Var(time_horizon, bounds=(0, 1), within=pyo.NonNegativeReals)

    # model.slack_v = pyo.Var(time_horizon, seg_range, bounds=(-2, 0), initialize=0)
    # model.slack_d = pyo.Var(time_horizon, seg_range, bounds=(-2, 0), initialize=0)

    # ------------------------------------------------------------------
    # Warm-start initialisation
    # ------------------------------------------------------------------
    if init_vsl is not None:
        density_ws, velocity_ws, queue_ws, flow_or_ws, v_fd_ws, _ = run_metanet_sim(
            T, l, starting_traffic_vars,
            traffic_demand[0:horizon_p+1], downstream_density[0:horizon_p],
            params, lanes=lanes,
            vsl_speeds=init_vsl[0:horizon_p, :], opt=True, real_data=False,
        )
        v_fd_ws = np.minimum(v_fd_ws[0:horizon_p, :], init_vsl[0:horizon_p, :])

        # eps = 0.5
        # vff_ws = v_fd_ws[0:horizon_p, :]
        # u_ws = init_vsl[0:horizon_p, :]
        # v_fd_ws = 0.5 * (vff_ws + u_ws - np.sqrt((vff_ws - u_ws)**2 + eps))

        c_ws = flow_or_ws[:, 0] / (np.maximum(_get_time_space_param(q_cap, 0, 0), 1e-6) * lanes[0])
        c_ws = np.clip(c_ws, 0, 1)
        for t in range(horizon_p):
            model.c[t].value = c_ws[t]
        model.c[horizon_p].value = c_ws[-1] if len(c_ws) > 0 else 0.5

        for t in range(horizon_p):
            model.queue[t].value     = queue_ws[t, 0]
            model.queue_out[t].value = flow_or_ws[t, 0]
            for m in seg_range:
                if t > 0:
                    model.density[t, m].value  = density_ws[t, m]
                    model.velocity[t, m].value = velocity_ws[t, m]
                if t < horizon_p:
                    model.vsl[t, m].value  = init_vsl[t, m]
                    model.v_fd[t, m].value = v_fd_ws[t, m]

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------
    con = model.constraints = pyo.ConstraintList()

    # Initial conditions
    con.add(model.queue[0]     == initial_queue)
    con.add(model.queue_out[0] == initial_flow_or)
    for m in seg_range:
        con.add(model.density[0, m]  == initial_density[m])
        con.add(model.velocity[0, m] == initial_velocity[m])

    # VSL hold-length
    for m in seg_range:
        uncontrolled = (
            (control_zone        is not None and m not in control_zone) or
            (control_one_segment is not None and m != control_one_segment)
        )
        if not uncontrolled:
            for i in range(0, horizon_p, hold_len):
                # con.add(model.vsl[i, m] == 150)
                for j in range(hold_len - 1):
                    if i + j + 1 < horizon_p:
                        con.add(model.vsl[i + j, m] == model.vsl[i + j + 1, m])

    # VSL zone grouping
    if control_changepoints is not None:
        for i, start in enumerate(control_changepoints):
            end = control_changepoints[i + 1] if i + 1 < len(control_changepoints) else num_segments
            for seg in range(start, end - 1):
                for h in range(horizon_p):
                    con.add(model.vsl[h, seg] == model.vsl[h, seg + 1])

    # Segments actually under VSL control (mirrors the `uncontrolled` check
    # used for the dynamics below) — needed so safety constraints aren't
    # applied to segments forced to v_free elsewhere.
    controlled_segs = [
        m for m in seg_range
        if not (
            (control_zone        is not None and m not in control_zone) or
            (control_one_segment is not None and m != control_one_segment)
        )
    ]

    # Temporal safety
    if safety_temporal is not None:
        for m in controlled_segs:
            if prior_vsl is not None:
                con.add(model.vsl[0, m] - prior_vsl[m] <=  safety_temporal)
                con.add(prior_vsl[m] - model.vsl[0, m] <=  safety_temporal)
            for h in range(1, horizon_p):
                con.add(model.vsl[h, m] - model.vsl[h - 1, m] <=  safety_temporal)
                con.add(model.vsl[h - 1, m] - model.vsl[h, m] <=  safety_temporal)

    # Spatial safety — only between pairs of segments that are both controlled
    if safety_spatial is not None:
        for m in controlled_segs:
            if (m - 1) not in controlled_segs:
                continue
            for h in range(horizon_p):
                con.add(model.vsl[h, m] - model.vsl[h, m - 1] <=  safety_spatial)
                con.add(model.vsl[h, m - 1] - model.vsl[h, m] <=  safety_spatial)

    # Dynamics

    for h in range(1, time_steps):
        # Queue dynamics
        con.add(
            model.queue[h] ==
            model.queue[h - 1] + T * (traffic_demand[h - 1] - model.queue_out[h - 1])
        )

        # Origin inflow — smooth METANET merge model via auxiliary c[h]
        #   queue_out[h] = c[h] * Q_cap * λ
        #   c[h] ≤ (p_max − ρ[h,0]) / (p_max − ρ_crit)   [space on motorway]
        #   c[h] * Q_cap * λ  ≤  d[h] + w[h]/T            [demand + queued vehicles]
        # con.add(model.c[h] <= (p_max - model.density[h, 0]) / (p_max - p_crit[0]))
        # con.add(model.c[h] * q_cap[0] * lanes[0] <= traffic_demand[h] + model.queue[h] / T)
        # con.add(model.queue_out[h] == model.c[h] * q_cap[0] * lanes[0])

        # con.add(model.c[h] <= (p_max - model.density[h, 0]) / (p_max - _get_time_space_param(p_crit, h, 0)))
        # con.add(model.c[h] * _get_time_space_param(q_cap, h, 0) * lanes[0] <= traffic_demand[h] + model.queue[h] / T)
        # con.add(model.queue_out[h] == model.c[h] * _get_time_space_param(q_cap, h, 0) * lanes[0])

        #######

        term1 = _get_time_space_param(q_cap, h, 0) * lanes[0] * (p_max - model.density[h, 0]) / (p_max - _get_time_space_param(p_crit, h, 0))
        term2 = traffic_demand[h] + model.queue[h] / T
        term3 = _get_time_space_param(q_cap, h, 0) * lanes[0]

        eps_merge = 1e-4
        min12 = 0.5 * (term1 + term2 - pyo.sqrt((term1 - term2)**2 + eps_merge))
        smooth_qout = 0.5 * (min12 + term3 - pyo.sqrt((min12 - term3)**2 + eps_merge))

        con.add(model.queue_out[h] == smooth_qout)

        # con.add(model.queue_out[h] ==traffic_demand[h])

        for m in seg_range:
            uncontrolled = (
                (control_zone        is not None and m not in control_zone) or
                (control_one_segment is not None and m != control_one_segment)
            )

            v_ff = _get_time_space_param(v_free, h-1, m) * pyo.exp(-(1/_get_time_space_param(a, h-1, m)) 
                                                                   * (model.density[h-1, m] / _get_time_space_param(p_crit, h-1, m))**_get_time_space_param(a, h-1, m))

            if uncontrolled:
                con.add(model.v_fd[h - 1, m] == v_ff)
                con.add(model.vsl[h - 1, m]  == _get_time_space_param(v_free, h-1, m))
            else:
                # v_fd = min(v_ff(ρ), vsl) via Big-M relaxation
                # u_bin → 1 when v_ff is binding, → 0 when VSL is binding
                # con.add(model.v_fd[h-1, m] <= v_ff)
                # con.add(model.v_fd[h-1, m] <= model.vsl[h-1, m])
                # con.add(model.v_fd[h-1, m] >= v_ff              - M_big * (1 - model.u_bin[h-1, m]))
                # con.add(model.v_fd[h-1, m] >= model.vsl[h-1, m] - M_big *      model.u_bin[h-1, m])
                # con.add(v_ff - model.vsl[h-1, m]  <= M_big * (1 - model.u_bin[h-1, m]))
                # con.add(model.vsl[h-1, m] - v_ff  <= M_big *      model.u_bin[h-1, m])


                # con.add(model.v_fd[h-1, m] == v_ff)
                # con.add(model.v_fd[h-1, m] <= model.vsl[h-1, m])

                #con.add(model.vsl[h-1, m] == 150)

                # With a smooth min equality:

                # con.add(model.vsl[h-1, m] <= v_free[m])
                
                eps = 1e-4  # smoothing parameter, tune between 1e-3 and 1.0
                diff = v_ff - model.vsl[h - 1, m]
                smooth_min = 0.5 * (v_ff + model.vsl[h - 1, m] - pyo.sqrt(diff**2 + eps))
                con.add(model.v_fd[h - 1, m] == smooth_min)
                # con.add(model.vsl[h-1, m] == 150)

            # METANET density/velocity update (unified across all segment positions)
            inflow  = (model.density[h-1, m-1] * model.velocity[h-1, m-1] * lanes[m-1]
                    if m > 0 else model.queue_out[h-1])
            inflow += _get_time_space_param(r, h-1, m)
            outflow = model.density[h-1, m] * model.velocity[h-1, m] * lanes[m] / (1 - _get_time_space_param(beta, h-1, m))

            con.add(
                model.density[h, m] ==
                model.density[h-1, m] + T / (l * lanes[m]) * (inflow - outflow)
            )

            rho_next = (model.density[h-1, m+1] if m < num_segments - 1
                        else downstream_density[h-1])
            convection = ((T / l) * model.velocity[h-1, m] * (model.velocity[h-1, m-1] - model.velocity[h-1, m])
                        if m > 0 else 0)

            # Raw (unclamped) METANET velocity update — matches velocity_dynamics_MN
            # in traffic_sim.py before its hard max(1e-4, ...) floor.
            velocity_next_raw = (
                model.velocity[h-1, m]
                + (T / _get_time_space_param(tau, h-1, m)) * (model.v_fd[h-1, m] - model.velocity[h-1, m])
                + convection
                - (_get_time_space_param(eta_high, h-1, m) * T / (_get_time_space_param(tau, h-1, m) * l)) * (
                    (rho_next - model.density[h-1, m]) / (model.density[h-1, m] + _get_time_space_param(K, h-1, m))
                )
            )

            # Smoothed floor: same role as traffic_sim.py's max(1e-4, nxt) clamp,
            # but differentiable so near-standstill states stay feasible instead
            # of colliding with the hard velocity lower bound.
            v_floor = 1e-4
            eps_floor = 1e-4
            diff_floor = velocity_next_raw - v_floor
            velocity_next = 0.5 * (velocity_next_raw + v_floor + pyo.sqrt(diff_floor**2 + eps_floor))

            con.add(model.velocity[h, m] == velocity_next)

    # ------------------------------------------------------------------
    # Objective: TTS + binary-relaxation penalty
    # ------------------------------------------------------------------
    #model.objective = pyo.Objective(expr=0.0, sense=pyo.minimize)

    model.objective = pyo.Objective(
        expr=(
            T * sum(
                model.density[h, m] * lanes[m] * l
                for h in range(1, time_steps) for m in seg_range
            )
            + T * sum(model.queue[h] for h in range(1, time_steps))
            # + 1e-3 * sum(
            #     (model.vsl[h, m] - model.v_fd[h, m])**2
            #     for h in range(horizon_p) for m in seg_range
            # )
            # + v_fd_penalty * sum(
            #     model.u_bin[h, m] * (1 - model.u_bin[h, m])
            #     for h in range(horizon_p) for m in seg_range
            # )
        ),
        sense=pyo.minimize,
    )

    # # 1. VSL array matching exactly what the model permits
    # vsl_chk = np.full((horizon_p, num_segments), 150.0)
    # for m in seg_range:
    #     unc = ((control_zone is not None and m not in control_zone) or
    #         (control_one_segment is not None and m != control_one_segment))
    #     if unc:
    #         for h in range(horizon_p):
    #             vsl_chk[h, m] = _get_time_space_param(v_free, h, m)

    # # 2. Simulate
    # d_s, v_s, q_s, fo_s, vfd_s, _ = run_metanet_sim(
    #     T, l, starting_traffic_vars,
    #     traffic_demand[0:horizon_p+1], downstream_density[0:horizon_p],
    #     params, lanes=lanes, vsl_speeds=vsl_chk, opt=True, real_data=False)

    # # 3. Load into the model
    # for h in range(horizon_p + 1):
    #     model.queue[h].value     = q_s[h, 0]
    #     model.queue_out[h].value = fo_s[h, 0]
    #     for m in seg_range:
    #         model.density[h, m].value  = d_s[h, m]
    #         model.velocity[h, m].value = v_s[h, m]
    # for h in range(horizon_p):
    #     for m in seg_range:
    #         model.vsl[h, m].value  = vsl_chk[h, m]
    #         model.v_fd[h, m].value = min(vfd_s[h, m], vsl_chk[h, m])

    # # 4. Residuals, no solve
    # log_infeasible_constraints(model, log_expression=True, log_variables=True, tol=1e-2)
    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------
    solver = pyo.SolverFactory('ipopt', executable='/usr/local/bin/ipopt')

    # solver.options['bound_relax_factor'] = 1e-8
    # solver.options['honor_original_bounds'] = 'yes'
    # solver.options['mu_strategy'] = 'adaptive'

    # solver.options['constr_viol_tol']            = 1e-3
    # solver.options['acceptable_constr_viol_tol'] = 1e-2
    # # solver.options['nlp_scaling_method']         = 'gradient-based'
    # solver.options['tol'] = 1e-4
    # solver.options['acceptable_tol'] = 1e-2
    # solver.options['acceptable_constr_viol_tol'] = 1e-2
    # solver.options['acceptable_dual_inf_tol'] = 1e-2
    # solver.options['acceptable_iter'] = 5

    if init_vsl is not None:
        solver.options['warm_start_init_point'] = 'yes'
        solver.options['mu_init'] = 0.1   # IPOPT default, instead of 1e-2
        solver.options['warm_start_bound_push'] = 1e-2
        solver.options['warm_start_mult_bound_push'] = 1e-2

        # solver.options['warm_start_init_point']      = 'yes'
        # solver.options['mu_init']                    = 1e-6
        # solver.options['warm_start_bound_push']      = 1e-3
        # solver.options['warm_start_mult_bound_push'] = 1e-3
        # solver.options['constr_viol_tol']            = 1e-6
        # solver.options['acceptable_constr_viol_tol'] = 1e-6

    t0 = time.process_time()
    status, _, iters, _, _ = ipopt_solver_wrapper.ipopt_solve_with_stats(
        model, solver,
        max_iter=10000, max_cpu_time=90,
        warmstart=(init_vsl is not None), tee=tee,
    )
    solve_time = time.process_time() - t0

    # logging.basicConfig(level=logging.INFO)
    # log_infeasible_constraints(model, log_expression=True, log_variables=True, tol=1e-6)

    if status.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise ValueError(
            f"Solver did not converge (termination condition: "
            f"{status.solver.termination_condition}).\n"
            # f"  downstream_density={downstream_density}\n"
            # f"  traffic_demand={traffic_demand}\n"
            # f"  starting_vars={starting_traffic_vars}"
        )

    vsl_speeds_c = np.array([[pyo.value(model.vsl[h, m]) for m in seg_range]
                            for h in range(horizon_c)])
    vsl_speeds_p = np.array([[pyo.value(model.vsl[h, m]) for m in seg_range]
                            for h in range(horizon_p)])

    # Print mismatch
    # After computing vsl_ctrl and new_state, run the optimizer's
    # own prediction forward and compare
    predicted_density = np.array([[pyo.value(model.density[h, m]) 
                                    for m in seg_range] 
                                for h in range(1, horizon_c)])
    predicted_velocity = np.array([[pyo.value(model.velocity[h, m]) 
                                    for m in seg_range] 
                                    for h in range(1, horizon_c)])

    actual_density, actual_velocity, _, _ = run_metanet_sim(
        T, l, starting_traffic_vars,
        traffic_demand[0: horizon_c+1],
        downstream_density[0: horizon_c],
        params, vsl_speeds=vsl_speeds_c, lanes=lanes, real_data=False, plotting=True
    )

    density_error = np.abs(predicted_density - actual_density[1:-1]).mean()
    velocity_error = np.abs(predicted_velocity - actual_velocity[1:-1]).mean()
    print(f"Density mismatch: {density_error:.4f}, Velocity mismatch: {velocity_error:.4f}")

    return iters, solve_time, vsl_speeds_c, vsl_speeds_p

def param_slice(params, start_time_step, end_time_step, total_time_steps, desired_length=None):
    sliced_params = {}
    for key, value in params.items():
        if isinstance(value, np.ndarray) and value.shape[0] == total_time_steps:
            sliced_params[key] = value[start_time_step:end_time_step, :].copy()

            if desired_length is not None and sliced_params[key].shape[0] < desired_length:
                sliced_params[key] = np.append(sliced_params[key], np.tile(value[-1], (desired_length - sliced_params[key].shape[0], 1)), axis=0)

                assert sliced_params[key].shape[0] == desired_length, f"Parameter {key} has length {sliced_params[key].shape[0]}, expected {desired_length}"
        else:
            sliced_params[key] = value
    return sliced_params

def mpc_find_vsl(
    total_time_steps, traffic_demand, downstream_density, lanes,
    T=5/3600, l=300/1000, num_segments=10,
    pred_horizon=20, control_horizon=20, hold_length=1,
    init_state=None,
    control_one_segment=None, initialize_vsl=None, init_fixed=None,
    control_changepoints=None, safety_temporal=None, safety_spatial=None,
    params=None, verbose=False,
    speed_lb=40, v_fd_penalty=0.1, control_zone=None, warmup_time=0, tee=False
):
    t     = 0
    state = init_state if init_state is not None else (
        np.array([traffic_demand[0] / (lanes[i] * 90) for i in range(num_segments)]),
        np.full(num_segments, 90.0),
        traffic_demand[0],
        0.0,
    )

    sim_time     = total_time_steps - pred_horizon + control_horizon
    full_control = None
    solve_time   = 0.0
    iterations   = 0

    def _solve(t_start, p_h, c_h, cur_state, init_slice, sliced_params):
        prior = full_control[-1] if (full_control is not None and full_control.ndim == 2) else None
        kwargs = dict(
            hold_len=hold_length, initialize=init_slice, init_fixed=init_fixed,
            control_one_segment=control_one_segment, control_changepoints=control_changepoints,
            safety_temporal=safety_temporal, safety_spatial=safety_spatial,
            prior_vsl=prior, params=sliced_params,
            speed_lb=speed_lb, v_fd_penalty=v_fd_penalty, control_zone=control_zone, tee=tee
        )
        try:
            return mpc_opt(T, l, num_segments,
                            traffic_demand[t_start: t_start + p_h + 1],
                            downstream_density[t_start: t_start + p_h + 1],
                            p_h, c_h, cur_state, lanes, **kwargs)

        except ValueError as exc:
            print(f"[MPC t={t_start}] Solver failed: {exc}\n  Retrying with init_fixed warm-start.")
            kwargs.update(initialize=None)
            try:
                return mpc_opt(T, l, num_segments,
                               traffic_demand[t_start: t_start + p_h + 1],
                               downstream_density[t_start: t_start + p_h + 1],
                               p_h, c_h, cur_state, lanes, **kwargs)

            except ValueError as exc2:
                print(f"[MPC t={t_start}] Solver failed again: {exc2}\n  Retrying fully cold (no warm-start).")
                kwargs.update(init_fixed=None)
                return mpc_opt(T, l, num_segments,
                               traffic_demand[t_start: t_start + p_h + 1],
                               downstream_density[t_start: t_start + p_h + 1],
                               p_h, c_h, cur_state, lanes, **kwargs)

    prev_full_solution = None

    while t + pred_horizon <= sim_time:
        if verbose and t % control_horizon == 0:
            print(f"[MPC] t = {t}")

        params_mpc = param_slice(params, t, t+pred_horizon, sim_time, desired_length=pred_horizon)

        # During warm-up, apply free-flow VSL and advance state without solving
        if t < warmup_time:
            vsl_ctrl = np.full((control_horizon, num_segments), 150.0)
            full_control = np.vstack((full_control, vsl_ctrl)) if full_control is not None else vsl_ctrl
            state = run_metanet_sim(
                T, l, state,
                traffic_demand[t: t + control_horizon + 1],
                downstream_density[t: t + control_horizon],
                params_mpc, vsl_speeds=vsl_ctrl, lanes=lanes, real_data=False,
            )[0]
            t += control_horizon
            continue

        # ------------------------------------------------------------------
        # Warm-start priority:
        #   1. user-provided initialize_vsl (explicit initialization wins)
        #   2. previous MPC step's solution, shifted by control_horizon
        #   3. None (falls through to init_fixed / cold start inside mpc_opt)
        # ------------------------------------------------------------------
        if initialize_vsl is not None:
            init_slice = initialize_vsl[t: t + pred_horizon + 1]
        elif prev_full_solution is not None:
            # Drop the first control_horizon rows (already executed),
            # pad the end with the last row repeated
            shifted = prev_full_solution[control_horizon:]
            pad = np.tile(shifted[-1], (pred_horizon + 1 - shifted.shape[0], 1))
            init_slice = np.vstack((shifted, pad))
        else:
            init_slice = None

        n_iters, ytime, vsl_ctrl, vsl_full = _solve(t, pred_horizon, control_horizon, state, init_slice, params_mpc)
        prev_full_solution = vsl_full.copy()   # save for next iteration
        full_control = np.vstack((full_control, vsl_ctrl)) if full_control is not None else vsl_ctrl

        state = run_metanet_sim(
            T, l, state,
            traffic_demand[t: t + control_horizon + 1],
            downstream_density[t: t + control_horizon],
            params_mpc, vsl_speeds=vsl_ctrl, lanes=lanes, real_data=False,
        )[0]

        t          += control_horizon
        solve_time += ytime
        iterations += n_iters

    # Tail step
    if t < sim_time:
        print(t)
        print(sim_time-t)
        params_mpc = param_slice(params, t, sim_time, sim_time, desired_length=pred_horizon+1)
        init_slice = initialize_vsl[t:] if initialize_vsl is not None else None

        n_iters, ytime, vsl_ctrl, vsl_full = _solve(t, sim_time - t, sim_time - t, state, init_slice, params_mpc)
        full_control = np.vstack((full_control, vsl_ctrl))
        solve_time  += ytime
        iterations  += n_iters

    if verbose:
        n_solves = (total_time_steps // control_horizon +
                    (1 if total_time_steps % control_horizon else 0))
        print(f"[MPC] Done. Total CPU: {solve_time:.1f}s  Avg/solve: {solve_time/n_solves:.2f}s")

    return full_control