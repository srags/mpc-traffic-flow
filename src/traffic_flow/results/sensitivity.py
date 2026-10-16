from dataclasses import replace
from typing import Literal, NamedTuple

import numpy as np
from tqdm.auto import tqdm

from ..pipeline import replay
from ..types import MPCStats, RunResult, time_vec
from .analysis import run_analysis


ParamKey = Literal["eta_high", "K", "tau", "p_crit", "v_free", "a"]
PARAM_KEYS: tuple[ParamKey, ...] = (
    "eta_high", "K", "tau", "p_crit", "v_free", "a",
)


CALIBRATION_BOUNDS = {
    'tau':      (10 / 3600, 60 / 3600),
    'K':        (5.0, 60.0),
    'eta_high': (10.0, 90.0),
    'p_crit':   (15.0, 95.0),
    'v_free':   (70.0, 150.0),
    'a':        (0.5, 5.0),
}


def perturb_params(params, rng, delta=0.10, segment_delta_frac=0.5, clip_params=True):
    """
    Randomly perturb the calibrated METANET parameters (those in
    CALIBRATION_BOUNDS), leaving q_capacity, r, beta and gamma untouched.

    Each parameter is multiplied by (1 + shared) * (1 + local):
      shared ~ U(-delta, delta), one draw per parameter for the whole corridor
               (systematic error);
      local  ~ U(-delta * segment_delta_frac, +delta * segment_delta_frac), one
               draw per segment (local error).
    The result is clipped to CALIBRATION_BOUNDS, except that a calibrated value
    already outside its bound (e.g. a `tau` stored 1e-12 below the lower bound)
    is allowed to stay where it is, so delta=0 returns the nominal parameters
    exactly. Pass `clip_params=False` to skip the clipping entirely. Returns a
    new dict.
    """
    segment_delta = delta * segment_delta_frac
    perturbed = params.copy()
    for key, (lo, hi) in CALIBRATION_BOUNDS.items():
        values = np.asarray(params[key], dtype=float)
        shared = 1 + rng.uniform(-delta, delta)
        local  = 1 + rng.uniform(-segment_delta, segment_delta, size=values.shape[-1])
        new_values = values * shared * local
        if clip_params:
            new_values = np.clip(new_values, np.minimum(lo, values), np.maximum(hi, values))
        perturbed[key] = new_values
    return perturbed


class SensitivityResult(NamedTuple):
    perturb_pct: time_vec
    cc: dict[ParamKey, time_vec]
    mape: dict[ParamKey, time_vec]
    nominal: MPCStats


def one_at_a_time_cc(
    run: RunResult,
    perturb_pct: time_vec | None = None,
) -> SensitivityResult:
    """Scale one parameter at a time, keeping the saved VSL policy fixed."""
    if perturb_pct is None:
        perturb_pct = np.arange(-20, 21) * 0.0005

    nominal = run_analysis(replay(run, until_ff=True))
    cc: dict[ParamKey, time_vec] = {}
    mape: dict[ParamKey, time_vec] = {}

    for key in tqdm(PARAM_KEYS, desc="Sensitivity", unit="parameter"):
        stats = []
        for pct in perturb_pct:
            params = run.params.copy()
            params[key] = run.params[key] * (1 + pct)
            varied = replay(replace(run, params=params), until_ff=True)
            stats.append(run_analysis(varied))

        cc[key] = np.array([s.cc_unclipped for s in stats])
        mape[key] = np.array([s.sim_error for s in stats])

    return SensitivityResult(perturb_pct, cc, mape, nominal)





class MonteCarloResult(NamedTuple):
    nominal: MPCStats
    cc: time_vec
    mape: time_vec
    attempted: int


def monte_carlo_cc(
    run: RunResult,
    n_samples: int = 50,
    delta: float = 0.10,
    segment_delta_frac: float = 0.5,
    max_mape: float = 100.0,
    mape_tolerance: float | None = None,
    clip_params: bool = True,
    seed: int | None = None,
) -> MonteCarloResult:
    """Perturb parameters jointly and replay the same saved VSL policy."""
    nominal = run_analysis(replay(run, until_ff=True))
    limit = (
        max_mape if mape_tolerance is None
        else nominal.sim_error + mape_tolerance
    )
    rng = np.random.default_rng(seed)
    cc_samples: list[float] = []
    mape_samples: list[float] = []

    for _ in tqdm(range(n_samples), desc="Monte Carlo", unit="sample"):
        params = perturb_params(
            run.params, rng, delta, segment_delta_frac, clip_params
        )
        varied = replay(replace(run, params=params), until_ff=True)
        stats = run_analysis(varied)

        if not stats.sim_error < limit:
            continue
        if stats.sim_tt == stats.ff_tt:
            continue

        cc = stats.cc_unclipped
        if np.isfinite(cc):
            cc_samples.append(cc)
            mape_samples.append(stats.sim_error)

    return MonteCarloResult(
        nominal,
        np.asarray(cc_samples, dtype=float),
        np.asarray(mape_samples, dtype=float),
        n_samples,
    )