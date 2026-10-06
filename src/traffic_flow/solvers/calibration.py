import numpy as np
from ..inputs.generation import generate_perturbations
from ..model.parameters import load_metanet_params, default_metanet_params
from ..types import *
from ..config import RobustOptConfig, CalibrationConfig, LossMode, LossChoice

CALIB_PARAM_NAMES = ["eta_high", "tau", "K", "rho_crit", "v_free", "a"]   # <<<

CALIB_PARAM_BOUNDS = {                                                       # <<<
    "eta_high": (10.0,        90.0),                                         # <<<
    "tau":      (10.0/3600,   60.0/3600),                                    # <<<
    "K":        (5.0,         60.0),                                         # <<<
    "rho_crit": (15.0,        75.0),                                         # <<<
    "v_free":   (70.0,        150.0),                                        # <<<
    "a":        (0.5,         5.0),                                          # <<<
}        
# CALIB_PARAM_BOUNDS = {                                                       # <<<
#     "eta_high": (30.0,        31.0),                                         # <<<
#     "tau":      (15.0/3600,   16.0/3600),                                    # <<<
#     "K":        (39.0,         40.0),                                         # <<<
#     "rho_crit": (34.0,        35.0),                                         # <<<
#     "v_free":   (90.0,        91.0),                                        # <<<
#     "a":        (1.1,         1.2),                                          # <<<
# }                                                                            # <<<
 

class FDModel(PyoModel):
    # Observation indices
    k: FiniteScalarRangeSet
    rho_hat: IndexedParam
    q_hat: IndexedParam
    C: ScalarParam
    rho_crit: ScalarVar
    V_free: ScalarVar
    a: ScalarVar
    q_pred: IndexedExpression
    obj: ScalarObjective

def fit_fd1(
  flattened_rho_hat: np.ndarray, flattened_q_hat: np.ndarray,
  C_i: float|None = None, V_free_init=60,
  a_init=1.0,
  # solver_name="ipopt",
  plot=True,
  top_k_for_C=5,
):
  """
  Fit a smooth FD1 fundamental diagram to (rho_hat, q_hat) data using Pyomo.

  Parameters
  ----------
  flattened_rho_hat : array-like
      Density measurements (veh/km).
  flattened_q_hat : array-like
      Flow measurements (veh/h).
  C_i : float or None
      Fixed capacity value. If None, capacity is estimated from the top K flows.
  V_free_init : float
      Initial guess for free-flow speed.
  a_init : float
      Initial guess for shape parameter a.
  solver_name : str
      Solver to use (default: 'ipopt').
  plot : bool
      Whether to plot the fitted curve and data.
  top_k_for_C : int
      Number of top flow values to average for capacity estimate if C_i is None.

  Returns
  -------
  dict
      Optimized parameters {'rho_crit', 'V_free', 'a', 'C'}.
  """

  # Ensure arrays
  flattened_rho_hat = np.array(flattened_rho_hat)
  flattened_q_hat = np.array(flattened_q_hat)
  K = len(flattened_rho_hat)

  # Estimate capacity if not provided
  if C_i is None: C_i = np.mean(sorted(flattened_q_hat)[-top_k_for_C:]).item()

  # Build Pyomo model
  model = FDModel()
  model.k = FiniteScalarRangeSet(0, K - 1)
  model.rho_hat = IndexedParam(model.k, initialize={k: flattened_rho_hat[k] for k in range(K)})
  model.q_hat = IndexedParam(model.k, initialize={k: flattened_q_hat[k] for k in range(K)})
  model.C = ScalarParam(initialize=C_i)
  model.rho_crit = ScalarVar(bounds=(1e-2, max(flattened_rho_hat)), initialize=np.median(flattened_rho_hat))
  model.V_free = ScalarVar(bounds=(10, 150), initialize=V_free_init)
  model.a = ScalarVar(bounds=(0.01, 10), initialize=a_init)
  model.q_pred = IndexedExpression(model.k, expr={
    k: model.rho_hat[k] * model.V_free * pyo.exp(-1 / model.a * (model.rho_hat[k] / model.rho_crit) ** model.a)
    for k in model.k
  })
  model.obj = ScalarObjective(sense=pyo.minimize, expr=sum((model.q_pred[k] - model.q_hat[k]) ** 2 for k in model.k))

  # Solve
  solver = pyo.SolverFactory("ipopt")
  solver.solve(model, tee=False)

  # Extract parameters
  rho_crit_opt = pyo.value(model.rho_crit); assert isinstance(rho_crit_opt, float)
  V_free_opt = pyo.value(model.V_free); assert isinstance(V_free_opt, float)
  a_opt = pyo.value(model.a); assert isinstance(a_opt, float)
  C_opt = pyo.value(model.C); assert isinstance(C_opt, float)

  # Define fitted FD1 function
  def Q_fd1(rho):
    rho = np.array(rho)
    return V_free_opt * rho * np.exp(-1 / a_opt * (rho / rho_crit_opt) ** a_opt)

  # Plot if requested
  if plot:
    from traffic_flow.results.plots import Plotter
    p = Plotter(1, 1)
    rho_range = np.linspace(0, max(flattened_rho_hat) * 1.1, 500)
    q_fit = Q_fd1(rho_range)

    p[0].scatter(flattened_rho_hat, flattened_q_hat, color="gray", alpha=0.7, label="Data", s=1)
    p[0].plot(rho_range, q_fit, linewidth=2.5, label="Fitted FD1", zorder=10)
    p[0].axvline(rho_crit_opt, color="red", linestyle="--", label=f"ρ_crit = {rho_crit_opt:.1f}")
    p[0].axhline(C_opt, color="blue", linestyle=":", label=f"C = {C_opt:.1f}")
    p[0] = {'title': 'Fundamental Diagram Fit (FD1)', 'xlabel': 'Density ρ (veh/km)', 'ylabel': 'Flow q (veh/h)'}
    p[0].grid()
    p.show()

  return {"rho_crit": rho_crit_opt, "V_free": V_free_opt, "a": a_opt, "C": C_opt, "Q_fd1": Q_fd1,}

class CalibrationModel(PyoModel):
  # Index sets
  t: FiniteScalarRangeSet
  i: FiniteScalarRangeSet

  # Ordinary attributes
  num_segments: int

  # Scalar parameters
  T: ScalarParam
  l: ScalarParam

  # Either a fixed parameter or fitted variable
  n_lanes: IndexedParam

  # Fitted variables
  eta_high: IndexedVar
  tau: IndexedVar
  K: IndexedVar
  rho_crit: IndexedVar
  v_free: IndexedVar
  a: IndexedVar
  beta: IndexedVar
  r_inflow: IndexedVar

  # State variables
  v_pred: IndexedVar
  rho_pred: IndexedVar

  # Data parameters
  v_hat: IndexedParam
  rho_hat: IndexedParam

  constraints: ConstraintList
  rho_dyn: IndexedConstraint
  v_dyn: IndexedConstraint

  loss: ScalarObjective

def extract_metanet_params(model: CalibrationModel) -> MetanetParams:
  """Extract fitted values; retain defaults for parameters not calibrated."""
  segments = list(model.i)

  def vector(component: IndexedVar) -> space_vec:
    return np.array([pyo.value(component[i]) for i in segments], dtype=np.float64)

  params = default_metanet_params(len(segments))
  params.update(
    tau=vector(model.tau),
    K=vector(model.K),
    eta_high=vector(model.eta_high),
    p_crit=vector(model.rho_crit),
    v_free=vector(model.v_free),
    a=vector(model.a),
    beta=vector(model.beta),
  )

  params["r"] = np.array(
    [[pyo.value(model.r_inflow[t, i]) for i in segments] for t in model.t], dtype=np.float64,
  ) if model.r_inflow.dim() == 2 else vector(model.r_inflow)

  return params

def metanet_param_fit(
    v_hat: time_space,
    rho_hat: time_space,
    T: hr, l: km,
    lane_mapping: dict[str, float],
    initial_traffic_state: space_vec,
    downstream_density: space_vec,
    include_ramping=True,
    constraint_tol=1e-12,
    warmstart=None,
    prev_param_path=None,
    ramp_mapping=None,
    time_varying_ramps=False,
    fixed_inflows=None,
    use_A_regularizer=False,
    lambda_reg=1.0,
    x0=None,
    tee=True
):
    initial_flow_or = initial_traffic_state
    
    if initial_flow_or.ndim == 1:
        initial_flow_or = initial_flow_or.reshape(-1, 1)
    
    if downstream_density.ndim == 1:
        downstream_density = downstream_density.reshape(-1, 1)

    num_timesteps, num_segments = v_hat.shape

    model = CalibrationModel()

    model.t = FiniteScalarRangeSet(0, num_timesteps - 1)
    model.i = FiniteScalarRangeSet(0, num_segments - 1)
    model.num_segments = num_segments
    model.constraints = ConstraintList()

    # Fixed params
    model.T = ScalarParam(initialize=T)
    model.l = ScalarParam(initialize=l)

    # Number of lanes (per calibrated segment)
    model.n_lanes = IndexedParam(model.i, initialize=lane_mapping)

    # Parameters to estimate
    if warmstart is not None:
      params = load_metanet_params(path=warmstart, num_segments=num_segments)

      model.eta_high = IndexedVar(model.i, bounds=(10.0, 90.0))
      model.tau = IndexedVar(model.i, bounds=(10.0 / 3600, 60.0 / 3600))
      model.K = IndexedVar(model.i, bounds=(5.0, 60.0))
      model.rho_crit = IndexedVar(model.i, bounds=(15, 100))
      model.v_free = IndexedVar(model.i, bounds=(70, 140))
      model.a = IndexedVar(model.i, bounds=(0.5, 5))

      for i in range(num_segments):
        model.eta_high[i] = params["eta_high"][i]
        model.tau[i] = params["tau"][i]
        model.K[i] = params["K"][i]
        model.rho_crit[i] = params["p_crit"][i]
        model.v_free[i] = params["v_free"][i]
        model.a[i] = params["a"][i]

    else:
      model.eta_high = IndexedVar(model.i, bounds=(10.0, 90.0), initialize=30.0)
      model.tau = IndexedVar(model.i, bounds=(10.0 / 3600, 60.0 / 3600), initialize=18 / 3600)
      model.K = IndexedVar(model.i, bounds=(5.0, 60.0), initialize=40.0)
      model.rho_crit = IndexedVar(model.i, bounds=(15, 95), initialize=37.45)
      model.v_free = IndexedVar(model.i, bounds=(70, 150), initialize=120.0)
      model.a = IndexedVar(model.i, bounds=(0.5, 5), initialize=1.4)

    if include_ramping:    
      if time_varying_ramps:
        model.beta = IndexedVar(model.i, bounds=(1e-3, 0.6), initialize=1e-3)
        model.r_inflow = IndexedVar(model.t, model.i, bounds=(1e-3, 2000), initialize=1e-3)
        
        if ramp_mapping is not None:
          assert("on_ramps" in ramp_mapping and "off_ramps" in ramp_mapping), \
          "If include_ramping is True, ramp_mapping must contain 'on_ramps' and 'off_ramps' keys."

          on_ramps = ramp_mapping["on_ramps"]
          off_ramps = ramp_mapping["off_ramps"]
          for i in range(num_segments):
            if not on_ramps[i]:
              for t in range(num_timesteps):
                model.r_inflow[t, i].set_value(1e-3)
                model.r_inflow[t, i].fix()

            if not off_ramps[i]:
              if tee: print(f"seg {i} is not an off-ramp, fixing beta to 1e-3")
              model.beta[i].set_value(1e-3)
              model.beta[i].fix()

            if fixed_inflows is not None and i in fixed_inflows:
              for t in range(num_timesteps):
                model.r_inflow[t, i].fix(fixed_inflows[i][t])

        if warmstart is not None:
          for i in range(num_segments):
            for t in range(num_timesteps):
              model.beta[i] = params["beta"][i]
              model.r_inflow[t, i] = params["r"][t, i]
      else:
        model.beta = IndexedVar(model.i, bounds=(1e-3, 0.9), initialize=1e-3)
        model.r_inflow = IndexedVar(model.i, bounds=(1e-3, 2000), initialize=1e-3)
        
        if ramp_mapping is not None:
          assert("on_ramps" in ramp_mapping and "off_ramps" in ramp_mapping), \
            "If include_ramping is True, ramp_mapping must contain 'on_ramps' and 'off_ramps' keys."

          on_ramps = ramp_mapping["on_ramps"]
          off_ramps = ramp_mapping["off_ramps"]
          for i in range(num_segments):
            if not on_ramps[i]:
              model.r_inflow[i].set_value(1e-3)
              model.r_inflow[i].fix()

            if not off_ramps[i]:
              if tee: print(f"seg {i} is not an off-ramp, fixing beta to 1e-3")
              model.beta[i].set_value(1e-3)
              model.beta[i].fix()

        if warmstart is not None:
          for i in range(num_segments):
            for t in range(num_timesteps):
              model.beta[i] = params["beta"][i]
              model.r_inflow[i] = params["r"][i]

    else:
      model.beta = IndexedVar(model.i, bounds=(0.0, 0.0), initialize=0.0)
      model.r_inflow = IndexedVar(model.i, bounds=(0.0, 0.0), initialize=0.0)
    # Variables to predict (per-lane values)

    model.v_pred = IndexedVar(
      model.t, model.i, bounds=(1e-3, 150),
      initialize={(t, i): float(v_hat[t, i]) for t in model.t for i in model.i},
    )
    model.rho_pred = IndexedVar(
        model.t, model.i, bounds=(1e-3, 400),
        initialize={(t, i): float(rho_hat[t, i]) for t in model.t for i in model.i},
    )

    # Initial conditions
    for i in range(num_segments):
      model.constraints.add(model.v_pred[0, i] == v_hat[0, i].item())
      model.constraints.add(model.rho_pred[0, i] == rho_hat[0, i].item())

    # Observed data
    model.v_hat = IndexedParam(
      model.t, model.i,
      initialize={(t, i): float(v_hat[t, i]) for t in model.t for i in model.i},
    )
    model.rho_hat = IndexedParam(
      model.t, model.i,
      initialize={(t, i): float(rho_hat[t, i]) for t in model.t for i in model.i},
    )

    # Dynamics functions
    def density_dynamics(current, inflow, outflow, T, l, lanes, beta, r_inflow):
      return current + T / l * (inflow - outflow / (1 - beta) + r_inflow) # add r(t)- s(t)

    def calculate_V(m: CalibrationModel, rho, VSL, seg):
      return m.v_free[seg] * pyo.exp(-1 / m.a[seg] * (rho / m.rho_crit[seg]) ** m.a[seg])

    def velocity_dynamics(
      m: CalibrationModel, current, prev_state, density, next_density, VSL, T, l, seg
    ):
      tau = m.tau[seg]
      eta = m.eta_high[seg]
      K = m.K[seg]
      v_eq = calculate_V(m, density, VSL, seg)
      term1 = T / tau * (v_eq - current)
      term2 = T / l * current * (prev_state - current)
      term3 = (eta * T) / (tau * l) * (next_density - density) / (density + K)
      return current + term1 + term2 - term3

    # Density dynamics
    def rho_update(m: CalibrationModel, t, i):
      if t == 0: return pyo.Constraint.Skip
      seg = i
      if i == 0:
        current = m.rho_pred[t - 1, 0]
        inflow = initial_flow_or[t - 1, 0]
        outflow = m.rho_pred[t - 1, i] * m.v_pred[t - 1, i]
      else:
        current = m.rho_pred[t - 1, i]
        inflow = m.rho_pred[t - 1, i - 1] * m.v_pred[t - 1, i - 1]
        outflow = m.rho_pred[t - 1, i] * m.v_pred[t - 1, i]
      if include_ramping:
        return m.rho_pred[t, i] == density_dynamics(
          current,
          inflow,
          outflow,
          model.T,
          model.l,
          model.n_lanes[i],
          model.beta[i],
          model.r_inflow[t, i] if time_varying_ramps else model.r_inflow[i],
        )
      else:
        return m.rho_pred[t, i] == density_dynamics(
          current, inflow, outflow, model.T, model.l, model.n_lanes[i], 0.0, 0.0
        )

    model.rho_dyn = IndexedConstraint(model.t, model.i, rule=rho_update)
    # Velocity dynamics
    VSL = 150

    def v_update(m: CalibrationModel, t, i):
      seg = i
      if t == 0: return pyo.Constraint.Skip

      current = m.v_pred[t - 1, i]
      prev_state = m.v_pred[t - 1, i]
      density = m.rho_pred[t - 1, i] / m.n_lanes[seg]

      if num_segments == 1:
        # single-segment case
        next_density = downstream_density[t - 1] / m.n_lanes[seg]
      elif i == 0:
        # first segment in a multi-segment block
        next_density = m.rho_pred[t - 1, i + 1] / m.n_lanes[seg + 1]
      elif i == num_segments - 1:
        # last segment in block
        prev_state = m.v_pred[t - 1, i - 1]
        next_density = downstream_density[t - 1] / m.n_lanes[seg]
      else:
        # interior segment
        prev_state = m.v_pred[t - 1, i - 1]
        next_density = m.rho_pred[t - 1, i + 1] / m.n_lanes[seg + 1]

      return m.v_pred[t, i] == velocity_dynamics(
          m, current, prev_state, density, next_density, VSL, m.T, m.l, seg
      )

    model.v_dyn = IndexedConstraint(model.t, model.i, rule=v_update)

    if x0 is not None:                                                # <<<
      _num_seg = num_segments                                        # <<<
      for i in range(_num_seg):                                      # <<<
        if "eta_high"  in x0: model.eta_high[i].value  = float(x0["eta_high"][i])   # <<<
        if "tau"       in x0: model.tau[i].value        = float(x0["tau"][i])        # <<<
        if "K"         in x0: model.K[i].value          = float(x0["K"][i])          # <<<
        if "rho_crit"  in x0: model.rho_crit[i].value   = float(x0["rho_crit"][i])  # <<<
        if "v_free"    in x0: model.v_free[i].value     = float(x0["v_free"][i])     # <<<
        if "a"         in x0: model.a[i].value          = float(x0["a"][i])          # <<<
        if include_ramping:                                         # <<<
          if "beta"     in x0: model.beta[i].value     = float(x0["beta"][i])      # <<<
          if "r_inflow" in x0 and not time_varying_ramps:        # <<<
            model.r_inflow[i].value = float(x0["r_inflow"][i]) # <<<
 

    def compute_A_regularizer_pyomo(m: CalibrationModel, t):
      """
      Compute the Frobenius norm squared of the parameter-dependent
      blocks (3 and 4) of A_t using Pyomo variables, summed over
      all segments at time step t.

      Block 3: d(v_{i+1})/d(rho)  -- depends on tau, eta_high, K, v_free, rho_crit, a
      Block 4: d(v_{i+1})/d(v)    -- depends on tau
      """
      reg = 0.0
      N = m.num_segments

      for i in m.i:
        # --- segment-specific Pyomo variables ---
        tau_i    = m.tau[i]
        nu_i     = m.eta_high[i]
        kap_i    = m.K[i]
        v_f_i    = m.v_free[i]
        rho_cr_i = m.rho_crit[i]
        alpha_i  = m.a[i]

        # density is per-lane, matching your v_update convention
        rho_i    = m.rho_pred[t, i] / m.n_lanes[i]
        rho_next = (m.rho_pred[t, i + 1] / m.n_lanes[i + 1]
                    if i < N - 1
                    else downstream_density[t] / m.n_lanes[i])

        # V'(rho_i) using Pyomo exp
        dV_i = -(v_f_i / rho_cr_i) \
                * (rho_i / rho_cr_i) ** (alpha_i - 1) \
                * pyo.exp(-(1.0 / alpha_i) * (rho_i / rho_cr_i) ** alpha_i)

        # --- Block 3 entries ---
        # self: d(v_{i,t+1})/d(rho_{i,t})
        b3_self = (m.T / tau_i) * dV_i \
                + (nu_i * m.T / (tau_i * m.l)) \
                * (rho_next + kap_i) / (rho_i + kap_i) ** 2

        # downstream neighbour: d(v_{i,t+1})/d(rho_{i+1,t})
        b3_down = (-(nu_i * m.T) / (tau_i * m.l * (rho_i + kap_i))
                    if i < N - 1 else 0.0)

        # --- Block 4 entries ---
        v_i    = m.v_pred[t, i]
        v_prev = m.v_pred[t, i - 1] if i > 0 else m.v_pred[t, i]

        # self: d(v_{i,t+1})/d(v_{i,t})
        b4_self = 1.0 - m.T / tau_i + (m.T / m.l) * (v_prev - 2.0 * v_i)

        # upstream neighbour: d(v_{i,t+1})/d(v_{i-1,t})
        b4_up = ((m.T / m.l) * v_i if i > 0 else 0.0)

        reg += b3_self ** 2 + b3_down ** 2 + b4_self ** 2 + b4_up ** 2

      return reg

    # Objective: per-lane error
    def loss_fn(m: CalibrationModel):
      v_max = max(m.v_hat[t, i] for t in m.t for i in m.i)
      rho_max = max(m.rho_hat[t, i] for t in m.t for i in m.i)
      #q_max = max(m.q_hat[t, i] for t in m.t for i in m.i)
      loss_fn = sum(
          (20 * ((m.v_pred[t, i] - m.v_hat[t, i]) / v_max) ** 2)
          + ((m.rho_pred[t, i] - m.rho_hat[t, i]) / rho_max) ** 2
          #+ ((m.q_pred[t, i] - m.q_hat[t, i]) / q_max) ** 2
          for t in m.t
          for i in m.i
      )
      # Add smoothness constraint to params from prev_param_path
      if prev_param_path is not None:
          prev_params = load_metanet_params(path=prev_param_path, num_segments=num_segments)
          for i in m.i:
              loss_fn += 10.0 * ((m.eta_high[i] - prev_params["eta_high"][i]) / 90.0) ** 2
              loss_fn += 10.0 * ((m.tau[i] - prev_params["tau"][i]) / (60.0 / 3600)) ** 2
              loss_fn += 10.0 * ((m.K[i] - prev_params["K"][i]) / 60.0) ** 2
              loss_fn += 10.0 * ((m.rho_crit[i] - prev_params["p_crit"][i]) / 100.0) ** 2
              loss_fn += 10.0 * ((m.v_free[i] - prev_params["v_free"][i]) / 150.0) ** 2
              loss_fn += 10.0 * ((m.a[i] - prev_params["a"][i]) / 5.0) ** 2
              loss_fn += 10.0 * ((m.beta[i] - prev_params["beta"][i]) / 0.9) ** 2
              loss_fn += 10.0 * ((m.r_inflow[i] - prev_params["r"][i]) / 2000.0) ** 2
      
        # Jacobian regularizer: penalize large ||A_t||_F over trajectory
      if use_A_regularizer:
          reg_sum = sum( compute_A_regularizer_pyomo(m, t) for t in m.t if t > 0)
          loss_fn += lambda_reg * reg_sum

      return loss_fn
    
    model.loss = ScalarObjective(rule=loss_fn, sense=pyo.minimize)

    # Solve
    solver = pyo.SolverFactory("ipopt")   
    solver.options["max_iter"] = 20000
    solver.options['acceptable_constr_viol_tol'] = constraint_tol
    solver.options['constr_viol_tol'] = constraint_tol
    if warmstart is not None:
        solver.options['warm_start_init_point'] = 'yes'
        solver.options['mu_init'] = 1e-6
        solver.options['warm_start_bound_push'] = 0.001
        solver.options['warm_start_mult_bound_push'] = 0.001
    solver_results = solver.solve(model, tee=tee)

    return model, solver_results

class RobustCalibrationModel(CalibrationModel):
  s: FiniteScalarRangeSet

  initial_flow: IndexedParam

  init_v: IndexedConstraint
  init_rho: IndexedConstraint
  rho_dyn: IndexedConstraint
  v_dyn: IndexedConstraint
  z_con: IndexedConstraint

  L: IndexedExpression
  z: ScalarVar
  obj: ScalarObjective

def metanet_param_fit_robust(
    v_hat,
    rho_hat,
    T, l,
    lane_mapping,
    initial_traffic_state,
    downstream_density,
    include_ramping=True,
    constraint_tol=1e-12,
    # robust-specific
    S=25,                     # number of scenarios
    bc_noise_percent=10.0,     # passed into generate_perturbations (see note above)
    seed=0,
    objective_mode: LossChoice = LossMode.MINMAX,
    lam_worst=0.2,              # only used for "mean_plus_worst"
    warmstart=None,
    ramp_mapping=None,
    prev_param_path=None
):
    """
    Scenario-based robust calibration:
      - shared parameters across scenarios
      - scenario-specific states (rho_pred, v_pred)
      - objective uses rho_pred*v_pred for flow fit (no q_pred anywhere)

    Uncertainty is applied to boundary conditions:
      - upstream inflow (initial_traffic_state)
      - downstream density (downstream_density)
    """
    print("Number of robust scenarios S =", S, "with bc noise percent =", bc_noise_percent)
    # --- reshape boundary conditions to (T, 1) like your current code ---
    initial_flow_or = np.asarray(initial_traffic_state)
    if initial_flow_or.ndim == 1:
        initial_flow_or = initial_flow_or.reshape(-1, 1)

    downstream_density = np.asarray(downstream_density)
    if downstream_density.ndim == 1:
        downstream_density = downstream_density.reshape(-1, 1)

    v_hat = np.asarray(v_hat, dtype=float)
    rho_hat = np.asarray(rho_hat, dtype=float)

    num_timesteps, num_segments = v_hat.shape

    # --- generate scenario perturbations for inflows ---
    inflow_s = generate_perturbations(
        initial_flow_or, percent_noise=bc_noise_percent, seed=seed, num=S
    )

    # down_s = generate_perturbations(
    #     downstream_density, percent_noise=bc_noise_percent, seed=seed + 1, num=S
    # )

    # Optional: clip to keep physical positivity
    # inflow_s = np.clip(inflow_s, 1e-3, None)

    m = RobustCalibrationModel()
    m.s = FiniteScalarRangeSet(0, S - 1)
    m.t = FiniteScalarRangeSet(0, num_timesteps - 1)
    m.i = FiniteScalarRangeSet(0, num_segments - 1)
    m.constraints = ConstraintList()

    # Fixed params
    m.T = ScalarParam(initialize=float(T))
    m.l = ScalarParam(initialize=float(l))

    # Number of lanes (per segment) — matches your pattern
    m.n_lanes = IndexedParam(m.i, initialize=lane_mapping)
    
    # Shared parameters to estimate (same as your metanet_param_fit)
    if warmstart is not None:
        params = load_metanet_params(path=warmstart, num_segments=num_segments)

        m.eta_high = IndexedVar(m.i, bounds=(10.0, 90.0))
        m.tau = IndexedVar(m.i, bounds=(10.0 / 3600, 60.0 / 3600))
        m.K = IndexedVar(m.i, bounds=(5.0, 60.0))
        m.rho_crit = IndexedVar(m.i, bounds=(15, 100))
        m.v_free = IndexedVar(m.i, bounds=(70, 150))
        m.a = IndexedVar(m.i, bounds=(0.5, 5))

        for i in range(num_segments):
            m.eta_high[i] = params["eta_high"][i]
            m.tau[i] = params["tau"][i]
            m.K[i] = params["K"][i]
            m.rho_crit[i] = params["p_crit"][i]
            m.v_free[i] = params["v_free"][i]
            m.a[i] = params["a"][i]
    
    else:
        m.eta_high = IndexedVar(m.i, bounds=(10.0, 90.0), initialize=30.0)
        m.tau = IndexedVar(m.i, bounds=(10.0 / 3600, 60.0 / 3600), initialize=18 / 3600)
        m.K = IndexedVar(m.i, bounds=(5.0, 60.0), initialize=40.0)
        m.rho_crit = IndexedVar(m.i, bounds=(15, 100), initialize=37.45)
        m.v_free = IndexedVar(m.i, bounds=(70, 150), initialize=120.0)
        m.a = IndexedVar(m.i, bounds=(0.5, 5), initialize=1.4)

    # m.eta_high = Var(m.i, bounds=(10.0, 90.0), initialize=30.0)
    # m.tau = Var(m.i, bounds=(1.0 / 3600, 60.0 / 3600), initialize=18 / 3600)
    # m.K = Var(m.i, bounds=(5.0, 60.0), initialize=40.0)
    # m.rho_crit = Var(m.i, bounds=(15, float(np.max(rho_hat))), initialize=37.45)
    # m.v_free = Var(m.i, bounds=(70, 150), initialize=120.0)
    # m.a = Var(m.i, bounds=(0.5, 5), initialize=1.4)

    if include_ramping:
      assert ramp_mapping is not None
      assert "on_ramps" in ramp_mapping and "off_ramps" in ramp_mapping, "If include_ramping is True, ramp_mapping must contain 'on_ramps' and 'off_ramps' keys."

      m.beta = IndexedVar(m.i, bounds=(1e-3, 0.9), initialize=1e-3)
      m.r_inflow = IndexedVar(m.i, bounds=(0, 2000), initialize=1e-3)

      if warmstart is not None:
        for i in m.i:
          m.beta[i] = params["beta"][i]
          m.r_inflow[i] = params["r"][i]

      for i in m.i:
        if not ramp_mapping["on_ramps"][i]:
          m.r_inflow[i] = 0.0
          m.r_inflow[i].fix()
        if not ramp_mapping["off_ramps"][i]:
          m.beta[i] = 1e-3
          m.beta[i].fix()
    else:
      m.beta = IndexedVar(m.i, bounds=(0.0, 0.0), initialize=0.0)
      m.r_inflow = IndexedVar(m.i, bounds=(0.0, 0.0), initialize=0.0)

    m.v_pred = IndexedVar(m.s, m.t, m.i, bounds=(1e-3, 150),
                   initialize={(s, t, i): float(v_hat[t, i]) for s in m.s for t in m.t for i in m.i})
    m.rho_pred = IndexedVar(m.s, m.t, m.i, bounds=(1e-3, 400),
                     initialize={(s, t, i): float(rho_hat[t, i]) for s in m.s for t in m.t for i in m.i})

    # Observed data params (shared across scenarios)
    m.v_hat = IndexedParam(m.t, m.i, initialize={(t, i): float(v_hat[t, i]) for t in range(num_timesteps) for i in range(num_segments)})
    m.rho_hat = IndexedParam(m.t, m.i, initialize={(t, i): float(rho_hat[t, i]) for t in range(num_timesteps) for i in range(num_segments)})
    #m.q_hat = Param(m.t, m.i, initialize={(t, i): float(q_hat[t, i]) for t in range(num_timesteps) for i in range(num_segments)})

    # Boundary conditions (scenario-specific)
    m.initial_flow = IndexedParam(m.s, m.t, initialize={(s, t): float(inflow_s[s, t, 0]) for s in range(S) for t in range(num_timesteps)})
    #m.downstream_density = Param(m.s, m.t, initialize={(s, t): float(down_s[s, t, 0]) for s in range(S) for t in range(num_timesteps)})

    # --- dynamics functions (same form as your code) ---
    def density_dynamics(current, inflow, outflow, T_, l_, lanes_, beta_, r_inflow_):
        # lanes_ is unused here, but kept to mirror your signature
        return current + T_ / l_ * (inflow - outflow / (1 - beta_) + r_inflow_)

    def calculate_V(mm: RobustCalibrationModel, rho_per_lane, VSL, seg):
        return mm.v_free[seg] * pyo.exp(-1 / mm.a[seg] * (rho_per_lane / mm.rho_crit[seg]) ** mm.a[seg])

    def velocity_dynamics(mm: RobustCalibrationModel, current, prev_state, density, next_density, VSL, T_, l_, seg):
        tau = mm.tau[seg]
        eta = mm.eta_high[seg]
        K = mm.K[seg]
        v_eq = calculate_V(mm, density, VSL, seg)
        term1 = T_ / tau * (v_eq - current)
        term2 = T_ / l_ * current * (prev_state - current)
        term3 = (eta * T_) / (tau * l_) * (next_density - density) / (density + K)
        return current + term1 + term2 - term3

    # --- initial conditions (for every scenario) ---
    def init_v_rule(mm: RobustCalibrationModel, s, i):
        return mm.v_pred[s, 0, i] == mm.v_hat[0, i]
    m.init_v = IndexedConstraint(m.s, m.i, rule=init_v_rule)

    def init_rho_rule(mm: RobustCalibrationModel, s, i):
        return mm.rho_pred[s, 0, i] == mm.rho_hat[0, i]
    m.init_rho = IndexedConstraint(m.s, m.i, rule=init_rho_rule)

    # --- density dynamics ---
    def rho_update(mm: RobustCalibrationModel, s, t, i):
        if t == 0:
            return pyo.Constraint.Skip

        if i == 0:
            current = mm.rho_pred[s, t - 1, 0]
            inflow = mm.initial_flow[s, t - 1]
            outflow = mm.rho_pred[s, t - 1, i] * mm.v_pred[s, t - 1, i]
        else:
            current = mm.rho_pred[s, t - 1, i]
            inflow = mm.rho_pred[s, t - 1, i - 1] * mm.v_pred[s, t - 1, i - 1]
            outflow = mm.rho_pred[s, t - 1, i] * mm.v_pred[s, t - 1, i]

        beta_ = mm.beta[i] if include_ramping else 0.0
        r_in_ = mm.r_inflow[i] if include_ramping else 0.0

        return mm.rho_pred[s, t, i] == density_dynamics(
            current, inflow, outflow, mm.T, mm.l, mm.n_lanes[i], beta_, r_in_
        )

    m.rho_dyn = IndexedConstraint(m.s, m.t, m.i, rule=rho_update)

    # --- velocity dynamics ---
    VSL = 150

    def v_update(mm: RobustCalibrationModel, s, t, i):
        if t == 0:
            return pyo.Constraint.Skip

        seg = i
        current = mm.v_pred[s, t - 1, i]
        prev_state = mm.v_pred[s, t - 1, i]
        density = mm.rho_pred[s, t - 1, i] / mm.n_lanes[seg]

        if num_segments == 1:
            next_density = downstream_density[t - 1] / mm.n_lanes[seg]
        elif i == 0:
            next_density = mm.rho_pred[s, t - 1, i + 1] / mm.n_lanes[seg + 1]
        elif i == num_segments - 1:
            prev_state = mm.v_pred[s, t - 1, i - 1]
            next_density = downstream_density[t - 1] / mm.n_lanes[seg]
        else:
            prev_state = mm.v_pred[s, t - 1, i - 1]
            next_density = mm.rho_pred[s, t - 1, i + 1] / mm.n_lanes[seg + 1]

        return mm.v_pred[s, t, i] == velocity_dynamics(
            mm, current, prev_state, density, next_density, VSL, mm.T, mm.l, seg
        )

    m.v_dyn = IndexedConstraint(m.s, m.t, m.i, rule=v_update)

    # --- robust objective ---
    v_max = max(m.v_hat[t, i] for t in m.t for i in m.i)
    rho_max = max(m.rho_hat[t, i] for t in m.t for i in m.i)
    # q_max = max(m.q_hat[t, i] for t in m.t for i in m.i)

    # scenario loss expression (uses rho*v directly; no q_pred)
    def scenario_loss(mm: RobustCalibrationModel, s):
        loss_fn = sum(
            (20 * ((mm.v_pred[s, t, i] - mm.v_hat[t, i]) / v_max) ** 2)
            + ((mm.rho_pred[s, t, i] - mm.rho_hat[t, i]) / rho_max) ** 2
            #+ ((mm.q_pred[s, t, i] - mm.q_hat[t, i]) / q_max) ** 2
            for t in mm.t
            for i in mm.i
        )
        # Add smoothness constraint to params from prev_param_path
        if prev_param_path is not None:
            prev_params = load_metanet_params(path=prev_param_path, num_segments=num_segments)
            for i in mm.i:
                loss_fn += 10.0 * ((mm.eta_high[i] - prev_params["eta_high"][i]) / 90.0) ** 2
                loss_fn += 10.0 * ((mm.tau[i] - prev_params["tau"][i]) / (60.0 / 3600)) ** 2
                loss_fn += 10.0 * ((mm.K[i] - prev_params["K"][i]) / 60.0) ** 2
                loss_fn += 10.0 * ((mm.rho_crit[i] - prev_params["p_crit"][i]) / 100.0) ** 2
                loss_fn += 10.0 * ((mm.v_free[i] - prev_params["v_free"][i]) / 150.0) ** 2
                loss_fn += 10.0 * ((mm.a[i] - prev_params["a"][i]) / 5.0) ** 2
                loss_fn += 10.0 * ((mm.beta[i] - prev_params["beta"][i]) / 0.9) ** 2
                loss_fn += 10.0 * ((mm.r_inflow[i] - prev_params["r"][i]) / 2000.0) ** 2
                
        return loss_fn

    m.L = IndexedExpression(m.s, rule=scenario_loss)

    # epigraph for worst-case loss
    if objective_mode == LossMode.MEAN:
        m.obj = ScalarObjective(expr=(pyo.quicksum(m.L[s] for s in m.s) / S), sense=pyo.minimize)
        
    elif objective_mode in {LossMode.MINMAX, LossMode.MEAN_PLUS_WORST}:
        m.z = ScalarVar(bounds=(0, None), initialize=0.0)

        def z_ge_loss(mm, s): return mm.z >= mm.L[s]
        m.z_con = IndexedConstraint(m.s, rule=z_ge_loss)

        if objective_mode == LossMode.MINMAX:
            m.obj = ScalarObjective(expr=m.z, sense=pyo.minimize)
        elif objective_mode == LossMode.MEAN_PLUS_WORST:
            m.obj = ScalarObjective(
                expr=(1.0 - lam_worst) * (pyo.quicksum(m.L[s] for s in m.s) / S) + lam_worst * m.z,
                sense=pyo.minimize
            )
    else:
        raise ValueError("objective_mode must be 'minmax' or 'mean_plus_worst' or 'mean'")

    # Solve
    solver = pyo.SolverFactory("ipopt")
    solver.options["max_iter"] = 20000
    solver.options["acceptable_constr_viol_tol"] = constraint_tol
    solver.options["constr_viol_tol"] = constraint_tol

    if warmstart is not None:
        solver.options['warm_start_init_point'] = 'yes'
        solver.options['mu_init'] = 1e-3
        # solver.options['warm_start_bound_push'] = 0.001
        # solver.options['warm_start_mult_bound_push'] = 0.001
    # solver.options['hessian_approximation'] = 'limited-memory'

    solver_results = solver.solve(m, tee=True)
    return m, solver_results


def run_calibration(traffic: TrafficData, T: hr, l: km, *, config: CalibrationConfig) -> MetanetParams:
    """Fit METANET parameters from prepared traffic observations.

    Observations use per-lane density and total flow across lanes.
    Returns simulation-ready parameters; raises RuntimeError if the
    solver does not report optimal termination.
    """

    # Existing fitters expect total density, not per-lane density.
    inputs = (
      traffic.velocity,
      traffic.density * traffic.lanes,
      T, l,
      {i: float(count) for i, count in enumerate(traffic.lanes)},
      traffic.inflow,
      traffic.downstream_density * traffic.lanes[-1],
    )

    common_options = dict(
      include_ramping=config.include_ramping,
      constraint_tol=config.constraint_tol,
      warmstart=config.warmstart,
      prev_param_path=config.prev_param_path,
      ramp_mapping=config.ramp_mapping,
    )

    if config.warmstart is not None:
      print("Using warmstart from:", config.warmstart)

    robust = config.robust_opt

    if robust is None:
      model, solver_result = metanet_param_fit(
        *inputs,
        **common_options, # type: ignore
        time_varying_ramps=config.time_varying_ramps,
        fixed_inflows=config.fixed_inflows,
        use_A_regularizer=config.use_A_regularizer,
        lambda_reg=config.lambda_reg,
        x0=config.x0,
        tee=config.tee,
      )
    else:
      assert isinstance(robust, RobustOptConfig)
      assert robust.objective_mode in tuple(LossMode)

      model, solver_result = metanet_param_fit_robust(
        *inputs,
        **common_options, # type: ignore
        S=robust.S,
        bc_noise_percent=robust.bc_noise_percent,
        seed=robust.seed,
        objective_mode=robust.objective_mode,
        lam_worst=robust.lam_worst,
      )

    termination = solver_result.solver.termination_condition
    if termination != pyo.TerminationCondition.optimal:
      raise RuntimeError(f"Calibration did not converge: {termination}")

    return extract_metanet_params(model)
