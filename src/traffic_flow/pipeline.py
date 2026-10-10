from dataclasses import fields

import numpy as np

from .config import CalibrationConfig, MPCConfig

from .inputs.scenario import Scenario
from .model.simulation import METANET_Simulator
from .types import (
  TrafficData, MetanetParams, MetanetState,
  SimulationResult, OptimizationResult,
  time_space, hr, km, RunResult
)

from copy import deepcopy

def init_state(traffic: TrafficData) -> MetanetState:
  return MetanetState(
    density=traffic.initial_density,
    velocity=traffic.initial_velocity,
    demand=float(traffic.inflow[0]),
    queue=0.0,
  )

def simulate_scenario(
    traffic: TrafficData,
    params: MetanetParams,
    T: hr, l: km,
    vsl: time_space | None = None,
    steps: int | None = None,
    real_data: bool = False,
) -> SimulationResult:
  """Simulate aligned inputs, returning histories including the terminal state.

  With no VSL, use the simulator's unrestricted-speed default. `real_data=True`
  uses measured origin inflow for calibration diagnostics; policy comparisons
  use the modeled origin and queue (`False`). No files are read or written.
  """
  steps = len(traffic.inflow) if steps is None else steps
  if vsl is not None and vsl.shape != (steps, len(traffic.lanes)):
      raise ValueError(f"Unexpected VSL shape: {vsl.shape}")
  simulator = METANET_Simulator(
    T=T, l=l, params=params,
    lanes={i: float(count) for i, count in enumerate(traffic.lanes)},
    real_data=real_data,
  )
  return SimulationResult(*simulator.run_with_history(
    demand=traffic.inflow[:steps],
    downstream_density=traffic.downstream_density[:steps],
    init_traffic_state=init_state(traffic),
    vsl_speeds=vsl,
  ))


def calibrate(scenario: Scenario, config: CalibrationConfig) -> MetanetParams:
  """Fit this scenario's observations; no saved calibration is required."""
  from .solvers.calibration import run_calibration
  return run_calibration(scenario.traffic, scenario.spec.time_step, scenario.spec.L, config=config)


def optimize(scenario: Scenario, params: MetanetParams, config: MPCConfig) -> RunResult:
  """Optimize using independent snapshots of the supplied inputs."""
  steps = scenario.spec.time_steps
  num_segments = len(scenario.traffic.lanes)
  assert 1 <= config.control_horizon <= config.pred_horizon, "Require 1 <= control_horizon <= pred_horizon"
  assert config.pred_horizon <= steps, "Duration must cover at least one prediction horizon"
  lanes = {i: float(count) for i, count in enumerate(scenario.traffic.lanes)}

  opt_horizon = steps + config.pred_horizon - config.control_horizon

  from .solvers.mpc import mpc_find_vsl
  
  vsl = mpc_find_vsl(
    opt_horizon,
    np.pad(scenario.traffic.inflow, (0, max(0, opt_horizon + 1 - len(scenario.traffic.inflow))), mode="edge"),
    np.pad(scenario.traffic.downstream_density, (0, max(0, opt_horizon + 1 - len(scenario.traffic.downstream_density))), mode="edge"),
    lanes, params=params,
    T=scenario.spec.time_step, l=scenario.spec.L,
    num_segments=num_segments,
    init_state=init_state(scenario.traffic),
    **{field.name: getattr(config, field.name) for field in fields(config)}, # asdict but no deep copy
  )
  scenario, params, config = deepcopy((scenario, params, config))
  return RunResult(scenario, params, config, optimization = evaluate(scenario, params, vsl))

def evaluate(scenario: Scenario, params: MetanetParams, vsl: time_space) -> OptimizationResult:
  """Replay a supplied policy against the 150 km/hr modeled-origin baseline.

  No optimization or file writes; the raw policy is not display-clipped.
  """
  spec = scenario.spec
  steps = spec.time_steps
  assert vsl.shape == (steps, len(scenario.traffic.lanes)), f"Unexpected VSL shape: {vsl.shape}"
  return OptimizationResult(vsl=vsl, 
    baseline=simulate_scenario(scenario.traffic, params, T=spec.time_step, l=spec.L, steps=steps, vsl=np.full(vsl.shape, 150.0)), 
    controlled=simulate_scenario(scenario.traffic, params, T=spec.time_step, l=spec.L, steps=steps, vsl=vsl)
  )
