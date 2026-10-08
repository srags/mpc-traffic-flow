"""Dataset metadata, observation preparation, and scenario loading."""

from dataclasses import dataclass

import numpy as np

from ..config import CalRef, ScenarioConfig
from ..model.parameters import load_calibrated_params
from ..paths import REPO_DIR
from ..types import MetanetParams, TrafficData, space_vec, time_space, NamedTuple

@dataclass(frozen=True)
class DatasetMeta:
  segment_km: float
  step_seconds: float

_DATASETS: dict[str, DatasetMeta] = {
  "i24": DatasetMeta(segment_km=0.4, step_seconds=10),
}

@dataclass(frozen=True)
class Scenario:
  spec: ScenarioConfig
  traffic: TrafficData

def smooth_inflow(inflow: np.ndarray, window_size: int = 2) -> np.ndarray:
  from scipy.ndimage import uniform_filter1d
  return uniform_filter1d(inflow, window_size, axis=0, mode="nearest", output=np.float64)

def prepare_traffic_data(raw_density: time_space, raw_flow: time_space, lanes: space_vec,
    *, initial_step: int = 0, smoothing_window: int | None = 2) -> TrafficData:
  """Prepare measurements without reading files or running models.

  Input density is total across lanes.
  Input matrices include upstream/downstream boundary columns.
  `lanes` contains only interior-segment lane counts.
  All timesteps are retained; initial_step selects initial conditions.
  """
  if raw_density.ndim != 2 or raw_flow.ndim != 2: raise ValueError("Measurements must be two-dimensional")
  if raw_density.shape != raw_flow.shape: raise ValueError("Density and flow must have matching shapes")
  if lanes.ndim != 1 or lanes.size == 0: raise ValueError("lanes must be a nonempty one-dimensional array")
  if raw_density.shape[1] != lanes.size + 2: raise ValueError("Expected one column per segment plus two boundaries")
  if not np.all(np.isfinite(lanes)) or np.any(lanes <= 0): raise ValueError("Lane counts must be finite and positive")
  if not 0 <= initial_step < raw_density.shape[0]: raise ValueError("initial_step is outside the measurement time range")
  if smoothing_window is not None and smoothing_window < 1: raise ValueError("smoothing_window must be positive or None")

  density_total = np.clip(raw_density, 1e-3, None)
  flow_total = np.clip(raw_flow, 1e-3, None)

  density = density_total[:, 1:-1] / lanes
  flow = flow_total[:, 1:-1]
  velocity = (flow_total / density_total)[:, 1:-1]

  inflow = flow_total[:, 0]
  downstream = density_total[:, -1] / lanes[-1]

  if smoothing_window is not None:
    inflow = smooth_inflow(inflow, smoothing_window)
    downstream = smooth_inflow(downstream, smoothing_window)

  return TrafficData(density, flow, velocity, lanes, inflow, downstream, 
                     density[initial_step].copy(), velocity[initial_step].copy())

def load_scenario(dataset: str, date: str) -> Scenario:
    try: meta = _DATASETS[dataset]
    except KeyError: raise ValueError(f"Unknown dataset. Available: {', '.join(_DATASETS)}") from None
    directory = REPO_DIR / "data" / dataset / f"{dataset}_{date}"
    density: time_space = np.load(directory / "rho_hat.npy", allow_pickle=False)
    flow: time_space = np.load(directory / "q_hat.npy", allow_pickle=False)
    lanes: space_vec = np.load(directory / "lane_mapping.npy", allow_pickle=False)
    assert density.ndim == 2 and lanes.ndim == 1 and lanes.size == density.shape[1], "Incompatible shapes for density or lane mapping"
    traffic = prepare_traffic_data(density, flow, lanes[1:-1], smoothing_window=2)
    time_steps, num_segments = traffic.density.shape
    spec = ScenarioConfig(dataset, date, meta.segment_km, num_segments, meta.step_seconds / 3600, time_steps)
    return Scenario(spec, traffic)


def load_params(scenario: Scenario, calibration: CalRef) -> MetanetParams:
    spec = scenario.spec
    return load_calibrated_params(
        REPO_DIR / "data" / spec.freeway / f"{spec.freeway}_{spec.date}" / calibration.source, 
        interval=calibration.interval, num_timesteps=spec.time_steps, num_segments=spec.num_segments,
    )
