import numpy as np
from sim_types import MetanetParams
from pathlib import Path
from typing import cast

def _load_parameter_block(path: Path, num_segments: int) -> MetanetParams:
    def optional(filename: str, default: float) -> np.ndarray:
        try: return np.load(path / filename)
        except FileNotFoundError: return np.full(num_segments, default)

    return {
        "tau": np.load(path / "tau.npy"),
        "K": np.load(path / "K.npy"),
        "eta_high": np.load(path / "eta_high.npy"),
        "p_crit": np.load(path / "rho_crit.npy"),
        "v_free": np.load(path / "v_free.npy"),
        "a": np.load(path / "a.npy"),
        "q_capacity": np.full(num_segments, 2400),
        "r": optional("r_inflow_array.npy", default=0),
        "beta": optional("beta_array.npy", default=0),
        "gamma": optional("gamma_array.npy", default=1),
    }

def param_slice(params: MetanetParams, start_time_step, end_time_step, desired_length=None
                ) -> MetanetParams:
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

def load_metanet_params(path: Path | None = None, control_h: int | None = None, num_timesteps: int = 360, num_segments: int = 14) -> MetanetParams:
    if path is not None and control_h is not None:
        if control_h <= 0 or num_timesteps <= 0:
            raise ValueError("control_h and num_timesteps must be positive")

        blocks: dict[str, list[np.ndarray]] = {}
        base_path = Path(path) / f"control_h_{control_h}"

        for i, start in enumerate(range(0, num_timesteps, control_h), start=1):
            duration = min(control_h, num_timesteps - start)
            for key, values in _load_parameter_block(base_path / f"params_{i}", num_segments).items():
                values = cast(np.ndarray, values).reshape(-1)
                if values.size != num_segments:
                    raise ValueError(
                        f"params_{i}/{key}: expected {num_segments} "
                        f"values, got {values.size}"
                    )
                block = np.tile(values, (duration, 1))
                blocks.setdefault(key, []).append(block)
        return cast(MetanetParams, {key: np.concatenate(parts, axis=0) for key, parts in blocks.items()})
    elif path is not None:
        return _load_parameter_block(Path(path), num_segments)
    else:
        # Use default
        defaults = {
            "tau": 18 / 3600,
            "K": 40,
            "eta_high": 30,
            "p_crit": 37.45,
            "v_free": 120,
            "a": 1.4,
            "q_capacity": 2400,
            "r": 0,
            "beta": 0,
            "gamma": 1,
        }
        return cast(MetanetParams, {key: np.full(num_segments, value) for key, value in defaults.items()})
