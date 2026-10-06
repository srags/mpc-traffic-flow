import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, TypedDict

if TYPE_CHECKING:
  from .config import MPCConfig
  from .inputs.scenario import Scenario

hr = float
km = float
veh_hr = float
lane_map = dict[int, float]

time_space = np.ndarray[tuple[int, int], np.dtype[np.float64]]
time_vec = np.ndarray[tuple[int], np.dtype[np.float64]]
space_vec = np.ndarray[tuple[int], np.dtype[np.float64]]

class MetanetState(NamedTuple):
  density: space_vec
  velocity: space_vec
  demand: float
  queue: float

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

class TrafficData(NamedTuple):
  """Prepared observations; arrays retain the full input time range."""
  density: time_space          # veh/km/lane
  flow: time_space             # veh/hr, across all lanes
  velocity: time_space         # km/hr
  lanes: space_vec
  inflow: time_vec             # veh/hr
  downstream_density: time_vec # veh/km/lane
  initial_density: space_vec
  initial_velocity: space_vec

class MPCSolveResult(NamedTuple):
  iterations: int
  cpu_time: float              # seconds
  control_vsl: time_space      # control-horizon speed limits
  prediction_vsl: time_space   # full prediction-horizon speed limits
  density_error: float
  velocity_error: float

class SimulationResult(NamedTuple):
  density: time_space
  velocity: time_space
  queue: time_space
  total_travel_time: veh_hr

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
  scenario: "Scenario"
  params: MetanetParams
  config: "MPCConfig"
  optimization: OptimizationResult

# pyo types
import pyomo.environ as pyo
from pyomo.core.base import (
  var as pyo_var, set as pyo_set, param as pyo_param,
  objective as pyo_obj, constraint as pyo_cstr, expression as pyo_expr
)
from typing_extensions import Self

class IndexedVar(pyo_var.IndexedVar): 
  def __new__(cls, *args, **kwargs) -> Self: return object.__new__(cls)
class ScalarVar(pyo_var.ScalarVar):
  def __new__(cls, *args, **kwargs) -> Self: return object.__new__(cls)

class FiniteScalarRangeSet(pyo_set.FiniteScalarRangeSet):
  def __new__(cls, *args, **kwargs) -> Self: return object.__new__(cls)

class IndexedParam(pyo_param.IndexedParam):
  def __new__(cls, *args, **kwargs) -> Self: return object.__new__(cls)
class ScalarParam(pyo_param.ScalarParam):
  def __new__(cls, *args, **kwargs) -> Self: return object.__new__(cls)

class IndexedConstraint(pyo_cstr.IndexedConstraint):
  def __new__(cls, *args, **kwargs) -> Self: return object.__new__(cls)
class ConstraintList(pyo_cstr.ConstraintList):
  def __new__(cls, *args, **kwargs) -> Self: return object.__new__(cls)

class ScalarObjective(pyo_obj.ScalarObjective):
  def __new__(cls, *args, **kwargs) -> Self: return object.__new__(cls)
class IndexedExpression(pyo_expr.IndexedExpression):
  def __new__(cls, *args, **kwargs) -> Self: return object.__new__(cls)
class PyoModel(pyo.ConcreteModel):
  def __new__(cls, *args, **kwargs) -> Self: return object.__new__(cls)
