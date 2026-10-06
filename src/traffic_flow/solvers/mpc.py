import sys, time

from ..types import *
from ..config import InitValue, InitChoices, InitMode

from ..model.simulation import param_at, METANET_Simulator

from tqdm import tqdm

class MPCModel(PyoModel):
  vsl: IndexedVar
  density: IndexedVar
  velocity: IndexedVar
  queue: IndexedVar
  queue_out: IndexedVar
  v_fd: IndexedVar
  objective: ScalarObjective
  constraints: ConstraintList

class MPCSolveError(RuntimeError):
  """Exception raised when the MPC solver fails to find a solution."""

from ..model.parameters import param_slice

def mpc_opt(
  T: hr, l: km, num_segments: int,
  traffic_demand: time_vec, downstream_density: time_vec,
  horizon_p, horizon_c,
  starting_traffic_vars, lanes: lane_map, hold_len,
  *, params: MetanetParams,
  initialize: time_space|None =None, init_fixed: InitValue | None = None,
  control_one_segment=None, control_changepoints=None,
  safety_temporal=None, safety_spatial=None,
  prior_vsl=None,
  speed_lb=40, v_fd_penalty=0.1, control_zone=None, tee=False
) -> MPCSolveResult:
    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
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

    time_steps = horizon_p + 1
    time_range = range(time_steps)
    segm_range = range(num_segments)

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
        if init_fixed == InitMode.ADAPTIVE:
            init_fixed_value = float(np.mean(initial_velocity))
        else:
            init_fixed_value = init_fixed
        init_vsl = np.full((horizon_p + 1, num_segments), init_fixed_value)
    else:
        init_vsl = None

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = MPCModel()

    def vsl_bounds(model, t, s):
        return (1e-4 if control_zone is not None and s not in control_zone else speed_lb, 150) 

    model.vsl = IndexedVar(range(horizon_p), segm_range, bounds=vsl_bounds, within=pyo.NonNegativeReals)
    model.density  = IndexedVar(time_range, segm_range, bounds=(1e-4, 400), within=pyo.NonNegativeReals)
    # Lower bound is deliberately below traffic_sim.py's velocity_dynamics_MN
    # clamp (max(1e-4, ...)). A gridlocked segment arrives here with velocity
    # exactly 1e-4, and if that equalled the bound the initial-condition
    # equality velocity[0, m] == 1e-4 would pin the variable onto its own
    # bound — a degenerate point that sends IPOPT into its restoration phase
    # and makes it report "locally infeasible" for every warm start.
    model.velocity = IndexedVar(time_range, segm_range, bounds=(1e-6, float(np.max(v_free)) + 30), within=pyo.NonNegativeReals)
    model.queue = IndexedVar(time_range, bounds=(0, 10000))
    model.queue_out = IndexedVar(time_range, bounds=(0, param_at(q_cap, 0, 0) * lanes[0]), within=pyo.NonNegativeReals)
    model.v_fd = IndexedVar(range(horizon_p), segm_range, bounds=(0, float(np.max(v_free)) + 30), within=pyo.NonNegativeReals)

    # ------------------------------------------------------------------
    # Warm-start initialisation
    # ------------------------------------------------------------------
    if init_vsl is not None:
        density_ws, velocity_ws, queue_ws, flow_or_ws, \
        v_fd_ws, _ = METANET_Simulator(T=T, l=l, params=params, lanes=lanes, real_data=False).run_with_opt(
            traffic_demand[0:horizon_p+1], downstream_density[0:horizon_p], starting_traffic_vars, init_vsl[0:horizon_p, :]
        )
        v_fd_ws = np.minimum(v_fd_ws[0:horizon_p, :], init_vsl[0:horizon_p, :])

        # eps = 0.5
        # vff_ws = v_fd_ws[0:horizon_p, :]
        # u_ws = init_vsl[0:horizon_p, :]
        # v_fd_ws = 0.5 * (vff_ws + u_ws - np.sqrt((vff_ws - u_ws)**2 + eps))

        for t in range(horizon_p):
            model.queue[t]     = queue_ws[t, 0]
            model.queue_out[t] = flow_or_ws[t, 0]
            for m in segm_range:
                if t > 0:
                    model.density[t, m]  = density_ws[t, m]
                    model.velocity[t, m] = velocity_ws[t, m]
                if t < horizon_p:
                    model.vsl[t, m]  = init_vsl[t, m]
                    model.v_fd[t, m] = v_fd_ws[t, m]

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------
    con = model.constraints = ConstraintList()

    # Initial conditions
    con.add(model.queue[0]     == initial_queue)
    con.add(model.queue_out[0] == initial_flow_or)
    for m in segm_range:
        con.add(model.density[0, m]  == initial_density[m])
        con.add(model.velocity[0, m] == initial_velocity[m])

    # VSL hold-length
    for m in segm_range:
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
        m for m in segm_range
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
            if (m - 1) not in controlled_segs: continue
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

        term1 = param_at(q_cap, h, 0) * lanes[0] * (p_max - model.density[h, 0]) / (p_max - param_at(p_crit, h, 0))
        term2 = traffic_demand[h] + model.queue[h] / T
        term3 = param_at(q_cap, h, 0) * lanes[0]

        eps_merge = 1e-4
        min12 = 0.5 * (term1 + term2 - pyo.sqrt((term1 - term2)**2 + eps_merge))
        smooth_qout = 0.5 * (min12 + term3 - pyo.sqrt((min12 - term3)**2 + eps_merge))

        con.add(model.queue_out[h] == smooth_qout)

        for m in segm_range:
            uncontrolled = (
                (control_zone        is not None and m not in control_zone) or
                (control_one_segment is not None and m != control_one_segment)
            )

            v_ff = param_at(v_free, h-1, m) * pyo.exp(-(1/param_at(a, h-1, m)) * (model.density[h-1, m] / param_at(p_crit, h-1, m))**param_at(a, h-1, m))

            if uncontrolled:
                con.add(model.v_fd[h - 1, m] == v_ff)
                con.add(model.vsl[h - 1, m]  == param_at(v_free, h-1, m))
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
            inflow = (model.density[h-1, m-1] * model.velocity[h-1, m-1] * lanes[m-1]
                    if m > 0 else model.queue_out[h-1])
            inflow += param_at(r, h-1, m)
            outflow = model.density[h-1, m] * model.velocity[h-1, m] * lanes[m] / (1 - param_at(beta, h-1, m))

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
                + (T / param_at(tau, h-1, m)) * (model.v_fd[h-1, m] - model.velocity[h-1, m])
                + convection
                - (param_at(eta_high, h-1, m) * T / (param_at(tau, h-1, m) * l)) * (
                    (rho_next - model.density[h-1, m]) / (model.density[h-1, m] + param_at(K, h-1, m))
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

    model.objective = ScalarObjective(sense=pyo.minimize,
        expr = (T * sum(model.density[h, m] * lanes[m] * l
                for h in range(1, time_steps) 
                for m in segm_range
                ) + T * sum(model.queue[h] for h in range(1, time_steps)))
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
    # sim = METANET_Simulator(T=T, l=l, params=params, lanes=lanes, real_data=False)
    # d_s, v_s, q_s, fo_s, vfd_s, _ = sim.run_with_opt(
    #     traffic_demand[0:horizon_p+1], downstream_density[0:horizon_p],
    #     starting_traffic_vars, vsl_speeds=vsl_chk)

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
    solver = pyo.SolverFactory('ipopt')

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
    # status, _, iters, _, _ = ipopt_solver_wrapper.ipopt_solve_with_stats(
    #     model, solver,
    #     max_iter=40000, max_cpu_time=180,
    #     warmstart=(init_vsl is not None), tee=tee,
    # )
    status = solver.solve(model, options={'max_iter': 40000, 'max_cpu_time': 180}, tee=tee, load_solutions=False)
    iters = 0
    solve_time = time.process_time() - t0

    solver_status = status.solver.status
    termination = status.solver.termination_condition

    if (
        solver_status not in (pyo.SolverStatus.ok, pyo.SolverStatus.warning)
        or termination != pyo.TerminationCondition.optimal
        or len(status.solution) == 0
    ):
        raise MPCSolveError(
            f"Solver failed: status={solver_status}, "
            f"termination={termination}, "
            f"message={status.solver.message}"
        )

    model.solutions.load_from(status)

    vsl_speeds_c = np.array([[pyo.value(model.vsl[h, m]) for m in segm_range] for h in range(horizon_c)])
    vsl_speeds_p = np.array([[pyo.value(model.vsl[h, m]) for m in segm_range] for h in range(horizon_p)])

    # Print mismatch
    # After computing vsl_ctrl and new_state, run the optimizer's
    # own prediction forward and compare
    predicted_density = np.array([[pyo.value(model.density[h, m]) for m in segm_range] for h in range(1, horizon_c)])
    predicted_velocity = np.array([[pyo.value(model.velocity[h, m]) for m in segm_range] for h in range(1, horizon_c)])

    sim = METANET_Simulator(T=T, l=l, params=params, lanes=lanes, real_data=False)
    
    actual_density, actual_velocity, _, _ = sim.run_with_history(
        traffic_demand[0: horizon_c+1],
        downstream_density[0: horizon_c],
        starting_traffic_vars, vsl_speeds=vsl_speeds_c
    )

    density_error = np.abs(predicted_density - actual_density[1:-1]).mean()
    velocity_error = np.abs(predicted_velocity - actual_velocity[1:-1]).mean()
    # print(f"Density mismatch: {density_error:.4f}, Velocity mismatch: {velocity_error:.4f}")

    return MPCSolveResult(iters, solve_time, vsl_speeds_c, vsl_speeds_p, density_error, velocity_error)

def _shift_vsl_warmstart(previous: time_space, steps: int, length: int) -> time_space:
    """Discard applied controls and extend using the last predicted row."""
    assert previous.ndim == 2 and previous.shape[0] > 0, ValueError("previous must be a nonempty time-space array")
    assert steps >= 0 and length > 0, ValueError("steps must be nonnegative and length positive")
    return previous[np.minimum(np.arange(length) + steps, previous.shape[0] - 1)]

def mpc_find_vsl(
  total_time_steps: int,
  traffic_demand: time_vec,
  downstream_density: time_vec,
  lanes: lane_map,
  *,
  params: MetanetParams,
  T: hr = 5/3600, l: km = 300/1000, num_segments: int = 10,
  pred_horizon = 20, control_horizon = 20, hold_length = 1,
  init_state: MetanetState | None = None,
  control_one_segment=None, initialize_vsl=None, init_fixed: InitChoices = None,
  control_changepoints=None, safety_temporal=None, safety_spatial=None,
  verbose=False,
  speed_lb=40, v_fd_penalty=0.1, control_zone=None, warmup_time=0, tee=False
) -> time_space:
    
  t = 0
  state: MetanetState = init_state if init_state is not None else MetanetState(
      np.array([traffic_demand[0] / (lanes[i] * 90) for i in range(num_segments)]),
      np.full(num_segments, 90.0),
      traffic_demand[0],
      0.0,
  )

  sim_time     = total_time_steps - pred_horizon + control_horizon
  full_control: time_space | None = None
  solve_time   = 0.0
  iterations   = 0

  progress = tqdm(
      total=sim_time, desc="MPC", unit="step", dynamic_ncols=True, 
      disable=not verbose, file=sys.stdout, leave=True,
  )
  progress.refresh()

  # init_fixed may be a single value/"adaptive" (kept for backward compatibility)
  # or a list of fallback constants to cycle through in order.
  if init_fixed is None: init_fixed_list = []
  elif isinstance(init_fixed, (list, tuple)): init_fixed_list = list(init_fixed)
  else: init_fixed_list = [init_fixed]

  def _solve(t_start, p_h, c_h, cur_state, init_slice, sliced_params) -> MPCSolveResult:
    prior = full_control[-1] if (full_control is not None and full_control.ndim == 2) else None
    base_kwargs = dict(
      hold_len=hold_length, control_one_segment=control_one_segment,
      control_changepoints=control_changepoints,
      safety_temporal=safety_temporal, safety_spatial=safety_spatial,
      prior_vsl=prior, params=sliced_params,
      speed_lb=speed_lb, v_fd_penalty=v_fd_penalty, control_zone=control_zone, tee=tee
    )

    # Attempt order: provided/shifted solution (if any) -> each init_fixed
    # value in turn -> fully cold. Stops at the first attempt that succeeds.
    attempts = []
    if init_slice is not None:
      attempts.append(("provided/shifted solution", dict(initialize=init_slice, init_fixed=None)))
    for value in init_fixed_list:
      attempts.append((f"init_fixed={value}", dict(initialize=None, init_fixed=value)))
    attempts.append(("fully cold", dict(initialize=None, init_fixed=None)))

    last_exc: MPCSolveError | None = None
    for i, (label, overrides) in enumerate(attempts):
      progress.set_description_str(f"MPC t={t_start} | {label}", refresh=True)
      try:
        return mpc_opt(T, l, num_segments,
                        traffic_demand[t_start: t_start + p_h + 1],
                        downstream_density[t_start: t_start + p_h + 1],
                        p_h, c_h, cur_state, lanes, **base_kwargs, **overrides)
      except MPCSolveError as exc:
        last_exc = exc
        if i < len(attempts) - 1:
          next_label = attempts[i + 1][0]
          progress.set_description(f"MPC t={t_start} | trying {next_label}", refresh=True)
    assert last_exc is not None
    raise last_exc

  prev_full_solution = None

  try:
    while t + pred_horizon <= sim_time:
      params_mpc: MetanetParams = param_slice(params, t, t+pred_horizon, desired_length=pred_horizon)
      sim = METANET_Simulator(T=T, l=l, params=params_mpc, lanes=lanes, real_data=False)

      # During warm-up, apply free-flow VSL and advance state without solving
      if t < warmup_time:
        vsl_ctrl = np.full((control_horizon, num_segments), 150.0)
        full_control = np.vstack((full_control, vsl_ctrl)) if full_control is not None else vsl_ctrl
        state, _ = sim.run(traffic_demand[t: t + control_horizon + 1], 
                           downstream_density[t: t + control_horizon], 
                           state, vsl_speeds=vsl_ctrl)
        t += control_horizon
        progress.update(control_horizon)
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
          init_slice = _shift_vsl_warmstart(prev_full_solution, steps=control_horizon, length=pred_horizon + 1)
      else:
          init_slice = None

      result = _solve(t, pred_horizon, control_horizon, state, init_slice, params_mpc)
      prev_full_solution = result.prediction_vsl.copy()
      full_control = np.vstack((full_control, result.control_vsl)) if full_control is not None else result.control_vsl
      state, _ = sim.run(traffic_demand[t: t + control_horizon + 1], 
                         downstream_density[t: t + control_horizon], 
                         state, vsl_speeds=result.control_vsl)
      t += control_horizon
      progress.update(control_horizon)

      mismatch_tol, messages = 1e-3, []
      if result.density_error > mismatch_tol: messages.append(f"density mismatch={result.density_error:.4g}")
      if result.velocity_error > mismatch_tol: messages.append(f"velocity mismatch={result.velocity_error:.4g}")
      if messages: tqdm.write(f"[MPC t={t}] " + ", ".join(messages))
      solve_time += result.cpu_time
      iterations += result.iterations

    # Tail step
    if t < sim_time:
      params_mpc = param_slice(params, t, sim_time, desired_length=pred_horizon+1)
      init_slice = initialize_vsl[t:] if initialize_vsl is not None else None

      result = _solve(t, sim_time - t, sim_time - t, state, init_slice, params_mpc)
      progress.update(sim_time - t)
      assert full_control is not None
      full_control = np.vstack((full_control, result.control_vsl))
      solve_time += result.cpu_time
      iterations += result.iterations

  except MPCSolveError as exc:
    raise MPCSolveError(f"MPC failed at t={t} after exhausting all initialization attempts.") from exc

  finally: progress.close()

  if verbose:
    n_solves = (total_time_steps // control_horizon + (1 if total_time_steps % control_horizon else 0))
    print(f"[MPC] Done. Total CPU: {solve_time:.1f}s  Avg/solve: {solve_time/n_solves:.2f}s")

  assert full_control is not None
  return full_control
