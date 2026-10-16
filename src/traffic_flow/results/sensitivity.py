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