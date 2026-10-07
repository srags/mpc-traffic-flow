
import numpy as np
from typing import Dict, Tuple, Optional

# ----------------------------
# Core METANET helper functions
# ----------------------------

def density_dynamics(current: float, inflow: float, outflow: float, lanes: int, T: float, l: float,
                     gamma: float = 1.0, beta: float = 0.0, r: float = 0.0) -> float:
    """Update density with conservation equation (per segment).
    """
    return max(1e-4, current + T / (l * lanes) * (inflow - outflow / (1 - beta) + r))


def flow_dynamics(density: float, velocity: float, lanes: int) -> float:
    """Fundamental relation: q = rho * v * lanes."""
    return density * velocity * lanes


def queue_dynamics(current: float, demand: float, flow_origin: float, T: float) -> float:
    return current + T * (demand - flow_origin)


def calculate_V(rho: float, v_ctrl: float, a: float, p_crit: float, v_free: float = 150.0) -> float:
    """Desired speed function V(rho), capped by control speed v_ctrl."""
    # prevent overflow warnings
    # p_crit += 1e-4
    # a += 1e-4
    # assert - (rho / p_crit) ** a / a < 700, f"Overflow in desired speed calculation, pow too large: { - (rho / p_crit) ** a / a}, rho={rho}, p_crit={p_crit}, a={a}"
    return min(v_free * np.exp(- (rho / p_crit) ** a / a), v_ctrl)


def calculate_V_arr(rho_arr: np.ndarray, v_ctrl_arr: np.ndarray, a: float, p_crit: float, v_free: float) -> np.ndarray:
    """Vectorized desired speed function. Not used in sim, but useful for plotting."""
    return np.minimum(v_free * np.exp(- (rho_arr / p_crit) ** a / a), v_ctrl_arr)


def velocity_dynamics_MN(current: float,
                         prev_state: float,
                         density: float,
                         next_density: float,
                         v_ctrl: float,
                         T: float,
                         l: float,
                         eta_high: float = 30.0,
                         K: float = 40.0,
                         tau: float = 18 / 3600,
                         a: float = 1.4,
                         p_crit: float = 37.45,
                         v_free: float = 120.0) -> float:
    """One-step METANET velocity update with standard terms.
    Returns a small positive floor to avoid non-physical negative velocities.
    """
    nxt = (
        current
        + T / tau * (calculate_V(density, v_ctrl, a, p_crit, v_free) - current)
        + T / l * current * (prev_state - current)
        - (eta_high * T) / (tau * l) * (next_density - density) / (density + K)
    )
    return max(1e-4, nxt)


def origin_flow_dynamics_MN(demand: float,
                            density_first: float,
                            queue: float,
                            lanes: int,
                            T: float,
                            p_max: float = 180.0,
                            p_crit: float = 37.45,
                            q_capacity: float = 2200.0) -> float:
    """Origin (on-ramp) sending/merging flow constraint."""
    return min(
        demand + queue / T,
        lanes * q_capacity * (p_max - density_first) / (p_max - p_crit),
        lanes * q_capacity,
    )

def _get_time_space_param(param, t: int, i: int):
    """Helper to index params that may be scalar, 1D (over i), or 2D (over t,i)."""
    if np.ndim(param) == 0:
        return float(param)
    if np.ndim(param) == 1:
        return float(param[i])
    # assume 2D
    return float(param[t, i])


def metanet_step(t: int,
                 density_t: np.ndarray,
                 velocity_t: np.ndarray,
                 queue_t: float,
                 flow_origin_t: float,
                 *,
                 T: float,
                 l: float,
                 vsl_speeds: np.ndarray,
                 demand: np.ndarray,
                 downstream_density: np.ndarray,
                 params: Dict[str, np.ndarray],
                 lanes: Dict[int, int],
                 real_data: bool = False,
                 ) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Compute a single simulation step (t -> t+1) for METANET.

    Args:
        t: current time index.
        density_t, velocity_t: arrays of shape (num_segments,).
        queue_t, flow_origin_t: scalars at time t.
        T, l: discretization time and segment length.
        vsl_speeds: (time_steps, num_segments) control speeds.
        demand: (time_steps,) exogenous demand at origin.
        downstream_density: (time_steps,) boundary density at downstream end.
        params: dict with keys 'beta','r','gamma','eta_high','K','tau','a','p_crit','v_free','q_capacity'.
        lanes: dict mapping segment index -> number of lanes.
        real_data: if True, use demand directly for first cell inflow; otherwise use origin flow.
        upstream_velocity: optional (time_steps,) boundary velocity for the first cell.

    Returns:
        density_tp1, velocity_tp1, queue_tp1, flow_origin_tp1, flow_tp1

    Notes:
        - flow_tp1 is per-segment flow at t+1 (shape (num_segments,)).
    """
    num_segments = density_t.shape[0]
    density_tp1 = np.empty_like(density_t, dtype=float)

    # --- density update ---
    for i in range(num_segments):
        beta = _get_time_space_param(params["beta"], t if np.ndim(params["beta"]) == 2 else 0, i)
        r = _get_time_space_param(params["r"], t if np.ndim(params["r"]) == 2 else 0, i)
        gamma = _get_time_space_param(params["gamma"], 0, i)

        if i == 0:
            inflow = demand[t] if real_data else flow_origin_t
            outflow = density_t[i] * velocity_t[i] * lanes[i]
            density_tp1[i] = density_dynamics(density_t[i], inflow, outflow, lanes[i], T, l,
                                              gamma=gamma, beta=beta, r=r)
        else:
            inflow = density_t[i - 1] * velocity_t[i - 1] * lanes[i - 1]
            outflow = density_t[i] * velocity_t[i] * lanes[i]
            density_tp1[i] = density_dynamics(density_t[i], inflow, outflow, lanes[i], T, l,
                                              gamma=gamma, beta=beta, r=r)

    # --- velocity update ---
    velocity_tp1 = np.empty_like(velocity_t, dtype=float)
    for i in range(num_segments):
        kwargs = dict(
            eta_high=_get_time_space_param(params["eta_high"], t, i),
            K=_get_time_space_param(params["K"], t, i),
            tau=_get_time_space_param(params["tau"], t, i),
            a=_get_time_space_param(params["a"], t, i),
            p_crit=_get_time_space_param(params["p_crit"], t, i),
            v_free=_get_time_space_param(params["v_free"], t, i),
        )
        if i == 0:
            prev_vel = velocity_t[i]
            next_dens = density_t[i + 1] if num_segments > 1 else downstream_density[t]
            velocity_tp1[i] = velocity_dynamics_MN(
                velocity_t[i], prev_vel, density_t[i], next_dens, vsl_speeds[t, i], T, l, **kwargs
            )
        elif i == num_segments - 1:
            velocity_tp1[i] = velocity_dynamics_MN(
                velocity_t[i], velocity_t[i - 1], density_t[i], downstream_density[t], vsl_speeds[t, i], T, l, **kwargs
            )
        else:
            velocity_tp1[i] = velocity_dynamics_MN(
                velocity_t[i], velocity_t[i - 1], density_t[i], density_t[i + 1], vsl_speeds[t, i], T, l, **kwargs
            )

    # --- queue update (only when not using real data) ---
    if not real_data:
        queue_tp1 = queue_dynamics(queue_t, demand[t], flow_origin_t, T)
    else:
        queue_tp1 = queue_t

    # --- per-segment flow at t+1 ---
    flow_tp1 = np.empty_like(density_tp1, dtype=float)
    for i in range(num_segments):
        flow_tp1[i] = flow_dynamics(density_tp1[i], velocity_tp1[i], lanes[i])

    # --- origin flow update (only when not using real data) ---
    if t+1 >= demand.shape[0]:
        flow_origin_tp1 = 0
    elif real_data:
        flow_origin_tp1 = flow_origin_t
    else:
        pcrit0 = _get_time_space_param(params["p_crit"], t+1, 0)
        qcap0 = _get_time_space_param(params["q_capacity"], t+1, 0)
        flow_origin_tp1 = origin_flow_dynamics_MN(
            demand[t + 1], density_tp1[0], queue_tp1, lanes[0], T, p_max=180.0, p_crit=pcrit0, q_capacity=qcap0
        )

    return density_tp1, velocity_tp1, queue_tp1, flow_origin_tp1, flow_tp1


def _extend_params_for_drain(params: Dict[str, np.ndarray], time_steps: int,
                             total_steps: int, num_segments: int) -> Dict[str, np.ndarray]:
    """Copy of `params` covering the drain period, with on-ramp inflow switched off.

    Time-varying (2D) parameters hold their last row. The ramp term `r` is made 2D so
    it can be zeroed after `time_steps`: it is a constant inflow, so leaving it on would
    keep feeding the corridor and it would never empty.
    """
    extended = dict(params)
    pad = total_steps - time_steps

    for key, value in params.items():
        if np.ndim(value) == 2:
            extended[key] = np.vstack([value, np.tile(value[-1], (pad, 1))])

    r = np.asarray(params.get("r", 0.0), dtype=float)
    if r.ndim == 0:
        r = np.full(num_segments, float(r))
    if r.ndim == 1:
        r = np.tile(r, (time_steps, 1))
    extended["r"] = np.vstack([r[:time_steps], np.zeros((pad, num_segments))])

    return extended


def run_metanet_sim(T: float,
                    l: float,
                    init_traffic_state: Tuple[np.ndarray, np.ndarray, float, float],
                    demand: np.ndarray,
                    downstream_density: np.ndarray,
                    params: Dict[str, np.ndarray],
                    vsl_speeds: Optional[np.ndarray] = None,
                    lanes: Optional[Dict[int, int]] = None,
                    plotting: bool = False,
                    real_data: bool = False,
                    opt: bool = False,
                    until_ff: bool = False,
                    drain_tol: Optional[float] = None,
                    max_drain_steps: Optional[int] = None):
    """Run a METANET simulation for the provided horizon by repeatedly calling `metanet_step`.

    This preserves the original function's outputs for compatibility:
      - If plotting=True: returns (density, velocity, queue, total_travel_time)
      - If opt=True: returns (density, velocity, queue, flow_origin, V_fd, total_travel_time)
      - Else: returns ((density_T, velocity_T, flow_origin_T, queue_T), total_travel_time)

    With ``until_ff=True`` the simulation keeps running past the demand horizon until the
    corridor is empty, so every vehicle finishes its trip inside the returned arrays. Over
    a fixed horizon a congested run leaves more vehicles mid-trip than an uncontrolled one,
    so the two serve different vehicle-km and are not directly comparable; draining removes
    that difference. During the drain nothing enters: origin demand and on-ramp inflow are
    zero, speed limits are released, and the downstream boundary density holds its last
    value until the horizon ends, after which it follows the last segment's own density
    (zero-gradient free outflow). Holding the observed final density instead can leave a
    congested boundary pinning the exit, and on several I-24 days the corridor then never
    empties. The run ends when fewer than ``drain_tol`` vehicles remain, or after
    ``max_drain_steps`` extra steps (default 10x the horizon), which prints a warning.

    ``density_dynamics`` floors each segment at 1e-4 veh/km/lane, so the corridor can
    never hold fewer than that floor summed over segments. ``drain_tol`` defaults to ten
    times that floor; a smaller value can never be reached and is rejected. Where the run
    stops within this range barely matters: on I-24 11/30, stopping at 0.009 vehicles
    rather than 0.003 moves TTS by 0.03 veh-h out of 392.
    """
    time_steps = downstream_density.shape[0]
    num_segments = init_traffic_state[0].shape[0]
    # print(f"Running sim for time steps {time_steps}, segments {num_segments}")

    if vsl_speeds is None:
        vsl_speeds = np.full((time_steps, num_segments), 1000)  # effectively no speed limit
    if lanes is None or len(lanes) == 0:
        lanes = {i: 1 for i in range(num_segments)}

    total_steps = time_steps
    if until_ff:
        # Every segment is floored at 1e-4 veh/km/lane, so this is the emptiest the
        # corridor can ever be; a tolerance at or below it would never be reached.
        empty_floor = 1e-4 * sum(lanes[i] for i in range(num_segments)) * l
        if drain_tol is None:
            drain_tol = 10 * empty_floor
        elif drain_tol <= empty_floor:
            raise ValueError(
                f"drain_tol={drain_tol:g} is at or below the density floor "
                f"({empty_floor:.2e} vehicles), so the corridor can never get that empty.")
        pad = int(10 * time_steps if max_drain_steps is None else max_drain_steps)
        total_steps = time_steps + pad
        demand = np.concatenate([demand, np.zeros(pad)])
        downstream_density = np.concatenate(
            [downstream_density, np.full(pad, downstream_density[-1])])
        vsl_speeds = np.vstack([vsl_speeds, np.full((pad, num_segments), 1e4)])
        params = _extend_params_for_drain(params, time_steps, total_steps, num_segments)

    initial_density, initial_velocity, initial_flow_or, initial_queue = init_traffic_state
    init_flow_or_real = origin_flow_dynamics_MN(
        demand[0], initial_density[0], initial_queue, lanes[0], T,
        p_max=180.0, p_crit=_get_time_space_param(params["p_crit"], 0, 0), q_capacity=_get_time_space_param(params["q_capacity"], 0, 0)
        )

    # Allocate histories
    density = np.zeros((total_steps + 1, num_segments), dtype=float)
    velocity = np.zeros((total_steps + 1, num_segments), dtype=float)
    flow = np.zeros((total_steps + 1, num_segments), dtype=float)
    queue = np.zeros((total_steps + 1, 1), dtype=float)
    flow_origin = np.zeros((total_steps + 1, 1), dtype=float)

    # Initial conditions
    density[0] = initial_density
    velocity[0] = initial_velocity
    flow[0] = np.array([initial_density[i] * initial_velocity[i] * lanes[i] for i in range(num_segments)], dtype=float)
    flow_origin[0, 0] = initial_flow_or if real_data else init_flow_or_real
    queue[0, 0] = initial_queue

    # Main loop
    steps_run = total_steps
    for t in range(total_steps):
        if until_ff and t >= time_steps:
            # Free outflow while draining: the boundary follows the last segment, so the
            # anticipation term stops holding traffic back and the corridor can empty.
            downstream_density[t] = density[t, -1]
        d_tp1, v_tp1, q_tp1, fo_tp1, f_tp1 = metanet_step(
            t,
            density[t],
            velocity[t],
            queue[t, 0],
            flow_origin[t, 0],
            T=T,
            l=l,
            vsl_speeds=vsl_speeds,
            demand=demand,
            downstream_density=downstream_density,
            params=params,
            lanes=lanes,
            real_data=real_data
        )
        density[t + 1] = d_tp1
        velocity[t + 1] = v_tp1
        queue[t + 1, 0] = q_tp1
        flow[t + 1] = f_tp1
        flow_origin[t + 1, 0] = fo_tp1

        if until_ff and t + 1 >= time_steps:
            vehicles = sum(density[t + 1, i] * lanes[i] * l for i in range(num_segments))
            if vehicles <= drain_tol:
                steps_run = t + 1
                break

        # if t < time_steps - 1:
            # print(t, fo_tp1, demand[t+1])
    # print("Final flow origin:", flow_origin.T)
    # print("Density sim:", density[:,0].T)
    # print("Queue sim:", np.round(queue, 3).T)
    # print(demand)
    # print(params["p_crit"][0])

    if until_ff:
        if steps_run == total_steps:
            remaining = sum(density[-1, i] * lanes[i] * l for i in range(num_segments))
            print(f"Warning: corridor not empty after {total_steps - time_steps} drain steps "
                  f"({remaining:.2f} vehicles left); raise max_drain_steps.")
        density = density[:steps_run + 1]
        velocity = velocity[:steps_run + 1]
        flow = flow[:steps_run + 1]
        queue = queue[:steps_run + 1]
        flow_origin = flow_origin[:steps_run + 1]
        vsl_speeds = vsl_speeds[:steps_run]

    # Compute total travel time
    total_travel_time = T * (
        sum([np.sum(density[:, i]) * lanes[i] * l for i in range(num_segments)])
        + np.sum(queue)
    )

    if plotting:
        return density, velocity, queue, total_travel_time
    elif opt:
        V_fd = calculate_V_arr(
            density[0:-1],
            vsl_speeds,
            params["a"],
            params["p_crit"],
            params["v_free"],
        )
        return density, velocity, queue, flow_origin, V_fd, total_travel_time
    else:
        final_tuple = (density[-1], velocity[-1], float(flow_origin[-1, 0]), float(queue[-1, 0]))
        return final_tuple, total_travel_time
