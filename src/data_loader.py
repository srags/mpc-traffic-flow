from paths import REPO_DIR
from sim_types import *
from param_loader import load_metanet_params
import os, pathlib
from paths import ensure_dir
from typing import Literal

def smooth_inflow(inflow: np.ndarray, window_size: int = 2) -> np.ndarray:
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(inflow, window_size, axis=0, mode="nearest", output=np.float64)

class RealData(NamedTuple):
    p: time_space
    q: time_space
    v: time_space
    lanes: space_vec
    data_inflow: time_vec
    downstream_density: time_vec
    p_init: space_vec
    v_init: space_vec

class Freeway:
  def __init__(self, freeway: str, date: str, 
               L: km, num_segments: int,
               time_step: hr, time_steps: int, start_time: hr):
    self.L = L
    self.time_step = time_step
    self.time_steps = time_steps
    self.start_time = start_time
    self.start_time_step: int = int(start_time / time_step)
    self.data_path = REPO_DIR / "data" / freeway / f"{freeway}_{date}"
    self.results_base_path = REPO_DIR / "results" / freeway / f"{freeway}_{date}"
    
    self.num_segments = num_segments

  def load_real_data(self) -> RealData:
    p: time_space = np.clip(np.load(self.data_path/'rho_hat.npy'), 1e-3, None)
    q: time_space = np.clip(np.load(self.data_path/'q_hat.npy'), 1e-3, None)
    v: time_space = q / p

    lane_cts = np.load(self.data_path/'lane_mapping.npy')[1:-1]
    
    p_init: space_vec = p[self.start_time_step, 1:-1].reshape(-1) / lane_cts
    v_init: space_vec = v[self.start_time_step, 1:-1].reshape(-1)

    assert p.shape[0] == self.time_steps
    assert len(lane_cts) == self.num_segments

    data_inflow: time_vec = smooth_inflow(q[:, 0].reshape(-1), window_size=2)
    downstream_density: time_vec = smooth_inflow(p[:, -1].reshape(-1), window_size=2) / lane_cts[self.num_segments-1]

    p: time_space = p[:, 1:-1] / lane_cts
    q: time_space = q[:, 1:-1]
    v: time_space = v[:, 1:-1]

    return RealData(p, q, v, lane_cts, data_inflow, downstream_density, p_init, v_init)

  def get_params(self, 
                 calibration_id: str = "calibration_static/fixed_ramping", 
                 calibration_interval = None
                 ) -> MetanetParams:
    calibration_dir = f"control_h_{calibration_interval}" if calibration_interval else ""
    cal_path = self.data_path / calibration_id / calibration_dir

    subfolders = [f for f in os.listdir(cal_path)
                  if os.path.isdir(cal_path / f) and f.startswith("params_")]

    if subfolders:
      print(f"Dynamic parameters detected: {len(subfolders)} subfolder(s) found")
      if calibration_interval is None or calibration_interval <= 0:
        raise ValueError("Dynamic parameters require a positive calibration_interval")
      return load_metanet_params(path=self.data_path/calibration_id, control_h=calibration_interval, num_timesteps=self.time_steps, num_segments=self.num_segments)
    else:
      print("No dynamic parameter subfolders found — using static calibration parameters")
      return load_metanet_params(path=cal_path, num_timesteps=self.time_steps, num_segments=self.num_segments)

  def results_path(self, 
                   calibration_id: str = "calibration_static/fixed_ramping", 
                   calibration_interval = None,
                   constraint: None | Literal["", "speed_lb", "hold_length", "safety_sweep"] = None
                   ) -> pathlib.Path:
    constraint = constraint or ""
    calibration_dir = f"control_h_{calibration_interval}" if calibration_interval else ""
    return ensure_dir(self.results_base_path / calibration_id / calibration_dir / constraint)

  