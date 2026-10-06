"""Typed settings and named choices for the calibration → optimization pipeline.

Enums are string-compatible for existing callers, paths and JSON artifacts.
Literal aliases retain type checking for callers that still use string values.
Dates, paths and numerical tuning values deliberately remain open-ended.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, TypeAlias, TypedDict

import numpy as np

from .paths import DEFAULT_CALIBRATION
from .types import hr, km, space_vec, time_vec, time_space


class _StringChoice(str, Enum):
    # Python 3.10 equivalent of StrEnum's string/path formatting behavior.
    def __str__(self) -> str:
        return self.value


class LossMode(_StringChoice):
    """How the losses across perturbed calibration scenarios are combined."""

    MINMAX = "minmax"
    MEAN = "mean"
    MEAN_PLUS_WORST = "mean_plus_worst"


class InitMode(_StringChoice):
    """Named alternative to a numeric constant-speed MPC initialization."""

    ADAPTIVE = "adaptive"  # Mean initial velocity, not an additional solver mode.


class Study(_StringChoice):
    """Existing result-directory labels, not switches that enable constraints.

    Set the actual bounds/hold durations on MPCConfig. Use None for the
    ordinary result directory.
    """

    SPEED_LB = "speed_lb"
    HOLD_LENGTH = "hold_length"
    SAFETY_SWEEP = "safety_sweep"


class CalSource(_StringChoice):
    """Known on-disk layouts; selecting one does not run calibration.

    Custom source strings remain supported. DYNAMIC additionally
    requires CalRef.interval.
    """

    FIXED_RAMPS = DEFAULT_CALIBRATION
    VARYING_RAMPS = "calibration_static/time_varying_ramping"
    DYNAMIC = "calibration_dynamic"


LossChoice: TypeAlias = LossMode | Literal["minmax", "mean", "mean_plus_worst"]
StudyChoice: TypeAlias = Study | Literal["", "speed_lb", "hold_length", "safety_sweep"] | None
InitValue: TypeAlias = float | InitMode | Literal["adaptive"]
# Tuples are preferred for configurations; lists remain supported for old callers.
InitChoices: TypeAlias = InitValue | tuple[InitValue, ...] | list[InitValue] | None


@dataclass(frozen=True)
class CalRef:
    """Which saved calibration to load—not settings for fitting a new one."""
    source: CalSource | str = CalSource.FIXED_RAMPS
    interval: int | None = None


@dataclass(frozen=True)
class ScenarioConfig:
    """Resolved dataset identity, geometry and timing, without calibration settings.

    Time values are hours; L is segment length in km.
    """
    freeway: str
    date: str
    L: km
    num_segments: int
    time_step: hr
    time_steps: int
    start_time: hr = 0.0


@dataclass(frozen=True)
class SyntheticConfig:
    """Synthetic bottleneck replay settings; times are hours, lengths km.

    Saved CSV policies have no metadata, so these settings must match the run
    that produced them. Defaults describe the existing 10 km demand sweep.
    """
    duration: hr = 2.0
    time_step: hr = 10 / 3600
    L: km = 0.4
    lanes: tuple[float, ...] = (4.0,) * 20 + (2.0,) * 5
    flow_standard: float = 4000.0
    peak_start: hr = 0.055
    peak_duration: hr = 0.5
    initial_speed: float = 90.0

    def __post_init__(self) -> None:
        positive = (self.duration, self.time_step, self.L, self.initial_speed, *self.lanes)
        if not self.lanes or not all(np.isfinite(x) and x > 0 for x in positive):
            raise ValueError("Duration, time step, segment length, initial speed and lanes must be positive and finite")
        if self.time_steps < 1:
            raise ValueError("Duration must cover at least one time step")
        if not np.isfinite(self.flow_standard) or self.flow_standard < 0:
            raise ValueError("Base demand must be finite and nonnegative")
        if not (0 <= self.peak_start <= self.duration and
                0 <= self.peak_duration <= self.duration - self.peak_start):
            raise ValueError("Peak interval must lie within the simulation duration")

    @property
    def num_segments(self) -> int:
        return len(self.lanes)

    @property
    def time_steps(self) -> int:
        return int(self.duration / self.time_step)

    @property
    def total_distance(self) -> km:
        return self.L * self.num_segments


@dataclass(frozen=True)
class MPCConfig:
    """Rolling-horizon settings; horizons/hold_length are steps, speeds km/hr."""
    pred_horizon: int
    control_horizon: int
    hold_length: int = 1

    speed_lb: float = 40.0
    v_fd_penalty: float = 0.1
    control_zone: tuple[int, ...] | None = None
    control_one_segment: int | None = None
    control_changepoints: tuple[int, ...] | None = None
    safety_temporal: float | None = None
    safety_spatial: float | None = None

    warmup_time: int = 0
    initialize_vsl: time_space | None = None
    init_fixed: InitChoices = None

    verbose: bool = False
    tee: bool = False

@dataclass
class RobustOptConfig:
    """
    Configuration for scenario-based robust METANET calibration.

    lam_worst is used only with LossMode.MEAN_PLUS_WORST.
    Pass this as CalibrationConfig.robust_opt; None selects nominal calibration.
    """

    # --- Robust scenario settings ---
    S: int = 25                              # number of scenarios
    bc_noise_percent: float = 10.0           # passed to generate_perturbations(percent_noise=...)
    seed: int = 0                            # RNG seed for perturbations

    # --- Robust objective selection ---
    objective_mode: LossChoice = LossMode.MINMAX
    lam_worst: float = 0.2   

class RampMapping(TypedDict):
    on_ramps: np.ndarray
    off_ramps: np.ndarray

class CalibrationInitialGuess(TypedDict, total=False):
    eta_high: space_vec
    tau: space_vec
    K: space_vec
    rho_crit: space_vec
    v_free: space_vec
    a: space_vec
    beta: space_vec
    r_inflow: space_vec

@dataclass(frozen=True)
class CalibrationConfig:
    """Fit options. robust_opt=None selects the existing nominal fitter."""
    include_ramping: bool = True
    ramp_mapping: RampMapping | None = None
    constraint_tol: float = 1e-12
    robust_opt: RobustOptConfig | None = None

    warmstart: Path | None = None
    prev_param_path: Path | None = None

    time_varying_ramps: bool = False
    fixed_inflows: dict[int, time_vec] | None = None

    use_A_regularizer: bool = False
    lambda_reg: float = 1

    x0: CalibrationInitialGuess | None = None
    tee: bool = True


def default_mpc_config(spec: ScenarioConfig) -> MPCConfig:
    """The current cc_run settings, including its 150 km/hr fallback seed."""
    return MPCConfig(
        pred_horizon=45, control_horizon=5, hold_length=1,
        speed_lb=0, v_fd_penalty=1e14,
        control_zone=tuple(range(2, spec.num_segments)),
        init_fixed=150.0, verbose=True, tee=False,
    )
