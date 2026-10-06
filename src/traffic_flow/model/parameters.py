from pathlib import Path
from typing import cast

import numpy as np

from ..types import MetanetParams

def param_slice(params: MetanetParams, start_time_step, end_time_step, desired_length=None) -> MetanetParams:
    sliced_params = {}
    for key, value in params.items():
        if isinstance(value, np.ndarray) and value.ndim == 2:
            sliced_params[key] = value[start_time_step:end_time_step, :].copy()
            if desired_length is not None and sliced_params[key].shape[0] < desired_length:
                sliced_params[key] = np.append(sliced_params[key], np.tile(value[-1], (desired_length - sliced_params[key].shape[0], 1)), axis=0)
                assert sliced_params[key].shape[0] == desired_length, f"Parameter {key} has length {sliced_params[key].shape[0]}, expected {desired_length}"
        else:
            sliced_params[key] = value
    return cast(MetanetParams, sliced_params)

def default_metanet_params(num_segments: int) -> MetanetParams:
    defaults = {
        "tau": 18 / 3600, "K": 40, "eta_high": 30, "p_crit": 37.45, "v_free": 120, 
        "a": 1.4, "q_capacity": 2400, "r": 0, "beta": 0, "gamma": 1,
    }
    return cast(MetanetParams, {key: np.full(num_segments, value) for key, value in defaults.items()})

def load_calibrated_params(directory: Path, interval: int | None, num_timesteps: int, num_segments: int) -> MetanetParams:
    """Read a saved calibration, detecting static files or interval blocks.

    directory is the calibration-source folder, before any control_h_* suffix.
    An interval folder can also contain a single static parameter block.
    """
    parameter_dir = directory / f"control_h_{interval}" if interval else directory
    blocks = [path for path in parameter_dir.iterdir() if path.is_dir() and path.name.startswith("params_")]
    if blocks:
        print(f"Dynamic parameters detected: {len(blocks)} subfolder(s) found")
        return load_metanet_params(
            path=directory, control_h=interval,
            num_timesteps=num_timesteps, num_segments=num_segments,
        )

    print("No dynamic parameter subfolders found — using static calibration parameters")
    return load_metanet_params(path=parameter_dir, num_timesteps=num_timesteps, num_segments=num_segments)

def _load_parameter_block(path: Path, num_segments: int) -> MetanetParams:
    defaults = default_metanet_params(num_segments)
    def optional(filename: str, default: np.ndarray) -> np.ndarray:
        try: return np.load(path / filename)
        except FileNotFoundError: return default
        
    return {
        "tau": np.load(path / "tau.npy"),
        "K": np.load(path / "K.npy"),
        "eta_high": np.load(path / "eta_high.npy"),
        "p_crit": np.load(path / "rho_crit.npy"),
        "v_free": np.load(path / "v_free.npy"),
        "a": np.load(path / "a.npy"),
        "q_capacity": defaults["q_capacity"],
        "r": optional("r_inflow_array.npy", defaults["r"]),
        "beta": optional("beta_array.npy", defaults["beta"]),
        "gamma": optional("gamma_array.npy", defaults["gamma"]),
    }

def load_metanet_params(path: Path | None = None, control_h: int | None = None, num_timesteps: int = 360, num_segments: int = 14) -> MetanetParams:
    if path is not None and control_h is not None:
        assert control_h > 0 and num_timesteps > 0, "control_h and num_timesteps must be positive"

        blocks: dict[str, list[np.ndarray]] = {}
        base_path = Path(path) / f"control_h_{control_h}"

        for i, start in enumerate(range(0, num_timesteps, control_h), start=1):
            duration = min(control_h, num_timesteps - start)
            for key, values in _load_parameter_block(base_path / f"params_{i}", num_segments).items():
                values = cast(np.ndarray, values).reshape(-1)
                assert values.size == num_segments, f"params_{i}/{key}: expected {num_segments} values, got {values.size}"
                block = np.tile(values, (duration, 1))
                blocks.setdefault(key, []).append(block)
        return cast(MetanetParams, {key: np.concatenate(parts, axis=0) for key, parts in blocks.items()})
    elif path is not None:
        return _load_parameter_block(path, num_segments)
    else:
        return default_metanet_params(num_segments)
