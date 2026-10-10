from ..types import *

# ----------------------------
# Core METANET helper functions
# ----------------------------

def density_dynamics(current: float, inflow: float, outflow: float, lanes: float, T: hr, l: km,
                     gamma: float = 1.0, beta: float = 0.0, r: float = 0.0) -> float:
    """Update density with conservation equation (per segment). """
    return max(1e-4, current + T / (l * lanes) * (inflow - outflow / (1 - beta) + r))

def queue_dynamics(current: float, demand: float, flow_origin: float, T: hr) -> float:
    return current + T * (demand - flow_origin)

def calculate_V(rho: float, v_ctrl: float, a: float, p_crit: float, v_free: float = 150.0) -> float:
    """Desired speed function V(rho), capped by control speed v_ctrl."""
    return min(v_free * np.exp(- (rho / p_crit) ** a / a), v_ctrl)

def calculate_V_arr(rho_arr: np.ndarray, v_ctrl_arr: np.ndarray, a: np.ndarray, p_crit: np.ndarray, v_free: np.ndarray) -> np.ndarray:
    """Vectorized desired speed function. Not used in sim, but useful for plotting."""
    return np.minimum(v_free * np.exp(- (rho_arr / p_crit) ** a / a), v_ctrl_arr)

def velocity_dynamics_MN(current: float,
                         prev_state: float,
                         density: float,
                         next_density: float,
                         v_ctrl: float,
                         T: hr,
                         l: km,
                         eta_high: float = 30.0,
                         K: float = 40.0,
                         tau: hr = 18 / 3600,
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
                            lanes: float, # allow fractional lanes
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


def param_at(param, t: int, i: int):
    """Helper to index params that may be scalar, 1D (over i), or 2D (over t,i)."""
    if np.ndim(param) == 0: return float(param)
    if np.ndim(param) == 1: return float(param[i])
    if np.ndim(param) == 2: return float(param[t, i])
    raise ValueError("Unexpected parameter dimensions")


class METANET_Simulator:
    """Wrapper class for running METANET simulations with a given set of parameters and initial state."""
    def __init__(self, T: hr, l: km, params: MetanetParams, lanes: lane_map | None = None, real_data: bool = False):
        self.real_data: bool = real_data
        self.T: hr = T
        self.l: km = l
        self.params: MetanetParams = params
        self.num_segments = params['tau'].shape[-1]
        self.lanes_array = np.array([lanes[i] for i in range(len(lanes))]) if lanes is not None else np.array([1.0] * self.num_segments)

    def _initialize(self, demand: time_vec, downstream_density: time_vec, 
                    init_traffic_state: MetanetState, vsl_speeds: time_space | None = None):
        self.time_steps: int = len(downstream_density)
        assert self.num_segments == len(init_traffic_state.density)
        
        self.downstream_density: time_vec = downstream_density
        self.demand: time_vec = demand
        self.vsl_speeds: time_space = vsl_speeds if vsl_speeds is not None else np.full((self.time_steps, self.num_segments), 1000)

        initial_density, initial_velocity, initial_flow_or, initial_queue = init_traffic_state
        self.cur_state = MetanetState(
            initial_density,
            initial_velocity,
            initial_flow_or if self.real_data else origin_flow_dynamics_MN(
                self.demand[0], initial_density[0], initial_queue, self.lanes_array[0], self.T,
                p_max=180.0, p_crit=param_at(self.params["p_crit"], 0, 0), q_capacity=param_at(self.params["q_capacity"], 0, 0)
            ),
            initial_queue
        )

    def _step(self, t: int, cur_state: MetanetState) -> MetanetState:
        density_t, velocity_t, flow_origin_t, queue_t = cur_state
        density_tp1 = np.empty_like(density_t, dtype=float)
        # --- density update ---
        for i in range(self.num_segments):
            beta = param_at(self.params["beta"], t if np.ndim(self.params["beta"]) == 2 else 0, i)
            r = param_at(self.params["r"], t if np.ndim(self.params["r"]) == 2 else 0, i)
            gamma = param_at(self.params["gamma"], 0, i)
    
            if i == 0:
                inflow = self.demand[t] if self.real_data else flow_origin_t
                outflow = density_t[i] * velocity_t[i] * self.lanes_array[i]
                density_tp1[i] = density_dynamics(density_t[i], inflow, outflow, self.lanes_array[i], self.T, self.l,
                                                    gamma=gamma, beta=beta, r=r)
            else:
                inflow = density_t[i - 1] * velocity_t[i - 1] * self.lanes_array[i - 1]
                outflow = density_t[i] * velocity_t[i] * self.lanes_array[i]
                density_tp1[i] = density_dynamics(density_t[i], inflow, outflow, self.lanes_array[i], self.T, self.l,
                                                    gamma=gamma, beta=beta, r=r)
        # --- velocity update ---
        velocity_tp1 = np.empty_like(velocity_t, dtype=float)
        for i in range(self.num_segments):
            kwargs = dict(
                eta_high=param_at(self.params["eta_high"], t, i),
                K=param_at(self.params["K"], t, i),
                tau=param_at(self.params["tau"], t, i),
                a=param_at(self.params["a"], t, i),
                p_crit=param_at(self.params["p_crit"], t, i),
                v_free=param_at(self.params["v_free"], t, i),
            )
            if i == 0:
                prev_vel = velocity_t[i]
                next_dens = density_t[i + 1] if self.num_segments > 1 else self.downstream_density[t]
                velocity_tp1[i] = velocity_dynamics_MN(
                    velocity_t[i], prev_vel, density_t[i], next_dens, self.vsl_speeds[t, i], self.T, self.l, **kwargs
                )
            elif i == self.num_segments - 1:
                velocity_tp1[i] = velocity_dynamics_MN(
                    velocity_t[i], velocity_t[i - 1], density_t[i], self.downstream_density[t], self.vsl_speeds[t, i], self.T, self.l, **kwargs
                )
            else:
                velocity_tp1[i] = velocity_dynamics_MN(
                    velocity_t[i], velocity_t[i - 1], density_t[i], density_t[i + 1], self.vsl_speeds[t, i], self.T, self.l, **kwargs
                )
    
        # --- queue update (only when not using real data) ---
        if not self.real_data:
            queue_tp1 = queue_dynamics(queue_t, self.demand[t], flow_origin_t, self.T)
        else:
            queue_tp1 = queue_t
        # --- origin flow update (only when not using real data) ---
        if t+1 >= self.demand.shape[0]:
            flow_origin_tp1 = 0
        elif self.real_data:
            flow_origin_tp1 = flow_origin_t
        else:
            pcrit0 = param_at(self.params["p_crit"], t+1, 0)
            qcap0 = param_at(self.params["q_capacity"], t+1, 0)
            flow_origin_tp1 = origin_flow_dynamics_MN(
                self.demand[t + 1], density_tp1[0], queue_tp1, self.lanes_array[0], self.T, p_max=180.0, p_crit=pcrit0, q_capacity=qcap0
            )
    
        return MetanetState(density_tp1, velocity_tp1, flow_origin_tp1, queue_tp1)

    def run(self, demand: time_vec, downstream_density: time_vec, init_traffic_state: MetanetState, 
        vsl_speeds: time_space | None = None) -> tuple[MetanetState, veh_hr]:
        self._initialize(demand, downstream_density, init_traffic_state, vsl_speeds)
        ttt: veh_hr = self.T * float(self.l * (self.cur_state.density @ self.lanes_array) + self.cur_state.queue)
        for t in range(self.time_steps):
            self.cur_state = self._step(t, self.cur_state)
            ttt += self.T * float(self.l * (self.cur_state.density @ self.lanes_array) + self.cur_state.queue)
        return self.cur_state, ttt

    def run_with_history(self, demand: time_vec, downstream_density: time_vec, init_traffic_state: MetanetState,
                        vsl_speeds: time_space | None = None) -> tuple[time_space, time_space, time_space, veh_hr]:

        self._initialize(demand, downstream_density, init_traffic_state, vsl_speeds)
        # Allocate histories
        density: time_space = np.empty((self.time_steps + 1, self.num_segments), dtype=float)
        velocity: time_space = np.empty_like(density)
        queue: time_space = np.empty((self.time_steps + 1, 1), dtype=float)

        for t in range(self.time_steps + 1):
            state = self.cur_state
            density[t], velocity[t], _, queue[t, 0] = state
            if t < self.time_steps: self.cur_state = self._step(t, state)
        
        total_travel_time = self.T * (self.l * (density[:-1,:].sum(axis=0) @ self.lanes_array) + queue.sum())

        return density, velocity, queue, total_travel_time

    def run_with_opt(self, demand: time_vec, downstream_density: time_vec, 
                     init_traffic_state: MetanetState, vsl_speeds: time_space | None = None
                     ) -> tuple[time_space, time_space, time_space, time_space, time_space, veh_hr]:
        self._initialize(demand, downstream_density, init_traffic_state, vsl_speeds)
        # Allocate histories
        density: time_space = np.empty((self.time_steps + 1, self.num_segments), dtype=float)
        velocity: time_space = np.empty_like(density)
        queue: time_space = np.empty((self.time_steps + 1, 1), dtype=float)
        flow_origin: time_space = np.empty((self.time_steps + 1, 1), dtype=float)

        for t in range(self.time_steps + 1):
            state = self.cur_state
            density[t], velocity[t], flow_origin[t, 0], queue[t, 0] = state
            if t < self.time_steps: self.cur_state = self._step(t, state)
            
        total_travel_time: veh_hr = self.T * (self.l * (density.sum(axis=0) @ self.lanes_array) + queue.sum())
        V_fd = calculate_V_arr(density[0:-1], self.vsl_speeds, self.params["a"], self.params["p_crit"], self.params["v_free"])
        return density, velocity, queue, flow_origin, V_fd, total_travel_time
