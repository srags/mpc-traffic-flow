from enum import Enum

import numpy as np
from dataclasses import dataclass, replace
from typing import NamedTuple, TypedDict
from .config import MPCConfig

LATEST_UPDATE = True # SWITCH TO MATCH NEWEST MAIN

hr = float
km = float
veh_hr = float

time_space = np.ndarray[tuple[int, int], np.dtype[np.float64]]
time_vec = np.ndarray[tuple[int], np.dtype[np.float64]]
space_vec = np.ndarray[tuple[int], np.dtype[np.float64]]
space_mask = np.ndarray[tuple[int], np.dtype[np.bool_]]

@dataclass(frozen=True)
class ScenarioSpec:
  """Resolved dataset identity, geometry and timing, without calibration settings."""
  freeway: str;         date: str
  L: km;                num_segments: int
  time_step: hr;        time_steps: int
  on_ramps: space_mask; off_ramps: space_mask
  lanes: space_vec;     start_hour: hr = 8

  def to_mpc_config(self, **kwargs) -> "MPCConfig":
    return replace(MPCConfig(
      pred_horizon=45, control_horizon=5, hold_length=1,
      speed_lb=0, v_fd_penalty=1e14,
      control_zone=tuple(range(2, self.num_segments)),
      init_fixed=150.0, verbose=True, tee=False,
    ), **kwargs)

class TrafficData(NamedTuple):
  """Prepared observations and boundary inputs; geometry belongs to the spec."""
  density: time_space          # veh/km/lane
  flow: time_space             # veh/hr, across all lanes
  velocity: time_space         # km/hr
  inflow: time_vec             # veh/hr
  downstream_density: time_vec # veh/km/lane
  initial_density: space_vec
  initial_velocity: space_vec

@dataclass(frozen=True)
class Scenario:
  spec: ScenarioSpec
  traffic: TrafficData

class _StringChoice(str, Enum):
  # Python 3.10 equivalent of StrEnum's string/path formatting behavior.
  def __str__(self) -> str: return self.value
    
class CalSource(_StringChoice):
  FIXED_RAMPS = "calibration_static/fixed_ramping"
  VARYING_RAMPS = "calibration_static/time_varying_ramping"
  DYNAMIC = "calibration_dynamic"

@dataclass(frozen=True)
class CalRef:
  """Which saved calibration to load—not settings for fitting a new one."""
  source: CalSource | str = CalSource.FIXED_RAMPS
  interval: int | None = None

class MetanetParams(TypedDict):
  tau: space_vec | time_space
  K: space_vec | time_space
  eta_high: space_vec | time_space
  p_crit: space_vec | time_space
  v_free: space_vec | time_space
  a: space_vec | time_space
  q_capacity: space_vec | time_space
  r: space_vec | time_space
  beta: space_vec | time_space
  gamma: space_vec | time_space

class SimulationResult(NamedTuple):
  density: time_space
  velocity: time_space
  queue: time_space
  total_travel_time: veh_hr
  @property
  def flow_per_lane(self) -> time_space:
    return self.density * self.velocity

class OptimizationResult(NamedTuple):
  vsl: time_space
  baseline: SimulationResult
  controlled: SimulationResult
  @property
  def travel_time_saved(self) -> veh_hr:
    return self.baseline.total_travel_time - self.controlled.total_travel_time

@dataclass(frozen=True)
class RunResult:
  """A complete optimization run: inputs, settings, and numerical outputs."""
  scenario: Scenario
  params: MetanetParams
  config: MPCConfig
  optimization: OptimizationResult

@dataclass(frozen=True)
class MPCStats:
  gt_tt: veh_hr; sim_tt: veh_hr; opt_tt: veh_hr; ff_tt: veh_hr; sim_window_tt: veh_hr
  sim_error: float
  num_veh: float
  @property
  def tts_error(self): return abs(self.gt_tt - self.sim_window_tt) / self.gt_tt * 100 
  @property
  def cc_unclipped(self) -> float: return (self.sim_tt - self.opt_tt) / (self.sim_tt - self.ff_tt) * 100
  @property
  def cc(self): return float(np.clip(self.cc_unclipped, 0, 100))
  @property
  def avg_tt_reduced_per_veh(self): return (self.sim_tt - self.opt_tt) / self.num_veh * 60
  @property
  def uncontrolled_avg_tts(self): return (self.sim_tt - self.ff_tt) / self.num_veh * 60
  @property
  def controlled_avg_tts(self): return (self.opt_tt - self.ff_tt) / self.num_veh * 60

class DepartureCurves(NamedTuple):
    time_min: time_vec
    arrivals: time_vec
    free_flow: time_vec
    baseline: time_vec
    controlled: time_vec

# pyo types
import pyomo.environ as pyo
from pyomo.core.base import (
  var as pyo_var, set as pyo_set, param as pyo_param,
  objective as pyo_obj, constraint as pyo_cstr, expression as pyo_expr
)
from typing_extensions import Self

class IndexedVar(pyo_var.IndexedVar): 
  def __new__(cls, *_, **__) -> Self: return object.__new__(cls)
class ScalarVar(pyo_var.ScalarVar):
  def __new__(cls, *_, **__) -> Self: return object.__new__(cls)

class FiniteScalarRangeSet(pyo_set.FiniteScalarRangeSet):
  def __new__(cls, *_, **__) -> Self: return object.__new__(cls)

class IndexedParam(pyo_param.IndexedParam):
  def __new__(cls, *_, **__) -> Self: return object.__new__(cls)
class ScalarParam(pyo_param.ScalarParam):
  def __new__(cls, *_, **__) -> Self: return object.__new__(cls)

class IndexedConstraint(pyo_cstr.IndexedConstraint):
  def __new__(cls, *_, **__) -> Self: return object.__new__(cls)
class ConstraintList(pyo_cstr.ConstraintList):
  def __new__(cls, *_, **__) -> Self: return object.__new__(cls)

class ScalarObjective(pyo_obj.ScalarObjective):
  def __new__(cls, *_, **__) -> Self: return object.__new__(cls)
class IndexedExpression(pyo_expr.IndexedExpression):
  def __new__(cls, *_, **__) -> Self: return object.__new__(cls)
class PyoModel(pyo.ConcreteModel):
  def __new__(cls, *_, **__) -> Self: return object.__new__(cls)
