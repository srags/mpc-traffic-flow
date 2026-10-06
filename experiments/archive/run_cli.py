"""Command-line CC and safety sweeps using the shared load/optimize/save pipeline."""

import argparse
from dataclasses import fields, replace
from itertools import product
import logging
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traffic_flow import load_params, load_result, load_scenario, optimize, save_result
from traffic_flow.config import CalRef, CalSource, InitMode, MPCConfig, Study, default_mpc_config
from traffic_flow.results.policies import run_dir


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CC or safety-bound optimization; save complete bundles, without plotting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("cc", "safety"), default="cc", help="Workflow to run")
    parser.add_argument("--dataset", default="i24", help="Dataset name")
    parser.add_argument("--date", default="11_30", help="Observation date")
    parser.add_argument("--calibration-source", default="fixed_ramps",
                        help="fixed_ramps, varying_ramps, dynamic, or a relative calibration folder")
    parser.add_argument("--calibration-interval", type=int,
                        help="Saved control_h_<N> parameter interval; required for dynamic")
    parser.add_argument("--study", choices=("none", *(study.value for study in Study)),
                        help="Folder label only; defaults to none for CC, safety_sweep for safety")
    parser.add_argument("--speed-lbs", nargs="+", type=float,
                        help="Minimum speeds to sweep, km/hr; defaults to the CC preset (0)")
    parser.add_argument("--temporal-values", nargs="+", type=float,
                        help="Safety mode: temporal bounds, km/hr per step; default: 0.7")
    parser.add_argument("--spatial-values", nargs="+", type=float,
                        help="Safety mode: spatial bounds, km/hr; default: 25")
    for name, description in (
        ("pred-horizon", "Prediction horizon, in steps"),
        ("control-horizon", "Steps applied before reoptimizing"),
        ("hold-length", "VSL holding-block length, in steps"),
        ("warmup-time", "Initial uncontrolled warm-up, in steps"),
    ):
        parser.add_argument(f"--{name}", type=int, help=f"{description}; otherwise use cc_mpc")
    parser.add_argument("--control-zone", nargs="+", type=int, help="Zero-based controlled segment indices")
    parser.add_argument("--control-one-segment", type=int, help="Further restrict control to one segment")
    parser.add_argument("--control-changepoints", nargs="+", type=int, help="Starts of shared-VSL segment groups")
    parser.add_argument("--init-fixed", nargs="+", metavar="SPEED_OR_ADAPTIVE",
                        help="Ordered fallback speeds/adaptive; 'none' disables fixed fallbacks. "
                             "Default: 150 for CC, adaptive 120 100 80 60 40 for safety")
    parser.add_argument("--warm-start", type=Path,
                        help="VSL .npy, run.npz, or bundle directory; paths relative to the working directory")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=None,
                        help="MPC progress bar; otherwise use cc_mpc (on)")
    parser.add_argument("--tee", action=argparse.BooleanOptionalAction, default=None,
                        help="Detailed solver output; otherwise use cc_mpc (off)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load/validate inputs and print settings/destinations; no solving or saving")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = argument_parser()
    args = parser.parse_args(argv)
    if args.mode == "cc" and (args.temporal_values is not None or args.spatial_values is not None):
        parser.error("--temporal-values and --spatial-values require --mode safety")
    for name in ("dataset", "date"):
        value = getattr(args, name)
        if not value or Path(value).name != value or value in (".", ".."):
            parser.error(f"--{name} must be a dataset/date name, not a path")
    sources = {source.name.lower(): source for source in CalSource}
    source = sources.get(args.calibration_source, args.calibration_source)
    if not str(source) or Path(source).is_absolute() or ".." in Path(source).parts:
        parser.error("--calibration-source must be a relative folder inside the dataset")
    if args.calibration_interval is not None and args.calibration_interval <= 0:
        parser.error("--calibration-interval must be positive")
    if source == CalSource.DYNAMIC and args.calibration_interval is None:
        parser.error("dynamic calibration requires --calibration-interval")
    for name in ("speed_lbs", "temporal_values", "spatial_values"):
        values = getattr(args, name)
        if values is not None and any(not math.isfinite(value) or value < 0 for value in values):
            parser.error(f"--{name.replace('_', '-')} must contain finite, nonnegative values")
    if args.init_fixed is not None:
        try:
            args.init_fixed = (() if args.init_fixed == ["none"] else tuple(
                InitMode.ADAPTIVE if value == "adaptive" else float(value) for value in args.init_fixed
            ))
            if any(value != InitMode.ADAPTIVE and (not math.isfinite(value) or value < 0)
                   for value in args.init_fixed):
                raise ValueError
        except ValueError:
            parser.error("--init-fixed expects nonnegative speeds, adaptive, or none by itself")

    calibration = CalRef(source, args.calibration_interval)
    if args.study is None:
        study = Study.SAFETY_SWEEP if args.mode == "safety" else None
    else:
        study = None if args.study == "none" else Study(args.study)
    logging.getLogger("pyomo.core").setLevel(logging.ERROR)
    scenario = load_scenario(args.dataset, args.date)
    base = default_mpc_config(scenario.spec)
    if args.mode == "safety":
        base = replace(base, init_fixed=(InitMode.ADAPTIVE, 120, 100, 80, 60, 40))
    # Only explicitly supplied flags override the central preset.
    overrides = {
        field.name: getattr(args, field.name) for field in fields(MPCConfig)
        if getattr(args, field.name, None) is not None
    }
    for name in ("control_zone", "control_changepoints"):
        if name in overrides:
            overrides[name] = tuple(overrides[name])
    base = replace(base, **overrides)
    if not 1 <= base.control_horizon <= base.pred_horizon <= scenario.spec.time_steps:
        parser.error("Require 1 <= control-horizon <= pred-horizon <= scenario duration")
    if base.hold_length < 1 or base.warmup_time < 0:
        parser.error("--hold-length must be positive; --warmup-time must be nonnegative")
    segments = (*tuple(base.control_zone or ()), *tuple(base.control_changepoints or ()),
                *((base.control_one_segment,) if base.control_one_segment is not None else ()))
    if any(not 0 <= segment < scenario.spec.num_segments for segment in segments):
        parser.error(f"Segment indices must be between 0 and {scenario.spec.num_segments - 1}")
    if base.control_changepoints is not None and tuple(sorted(set(base.control_changepoints))) != base.control_changepoints:
        parser.error("--control-changepoints must be strictly increasing")

    if args.warm_start is not None:
        path = args.warm_start.expanduser()
        if path.is_dir() or path.name == "run.npz":
            seed = load_result(path if path.is_dir() else path.parent).optimization.vsl
        elif path.suffix == ".npy":
            seed = np.load(path, allow_pickle=False)
        else:
            parser.error("--warm-start must be a .npy file, run.npz, or bundle directory")
        if (seed.ndim != 2 or seed.shape[1] != scenario.spec.num_segments
                or seed.shape[0] < scenario.spec.time_steps or seed.dtype.kind not in "iuf"
                or not np.isfinite(seed).all() or np.any(seed < 0)):
            parser.error("Warm start must have one column per segment, cover the duration, and contain finite nonnegative speeds")
        base = replace(base, initialize_vsl=seed)

    params = load_params(scenario, calibration=calibration)
    speed_lbs = args.speed_lbs if args.speed_lbs is not None else (base.speed_lb,)
    # Preserve safety_run's reversed temporal/spatial order; each pair reuses the same seed.
    temporal = tuple(reversed(args.temporal_values or (0.7,))) if args.mode == "safety" else (None,)
    spatial = tuple(reversed(args.spatial_values or (25.,))) if args.mode == "safety" else (None,)
    print(f"{args.mode}: {args.dataset}/{args.date}, calibration={calibration.source}, "
          f"interval={calibration.interval}, study={study or 'none'}; "
          f"{len(speed_lbs) * len(temporal) * len(spatial)} run(s)")
    print(f"horizons={base.pred_horizon}/{base.control_horizon}, hold={base.hold_length}, "
          f"control_zone={base.control_zone}, init_fixed={base.init_fixed}, "
          f"warm_start={args.warm_start or 'none'}")
    for speed_lb, safety_temporal, safety_spatial in product(speed_lbs, temporal, spatial):
        config = replace(base, speed_lb=speed_lb, safety_temporal=safety_temporal, safety_spatial=safety_spatial)
        print(f"\nSpeed lower bound={speed_lb}, temporal={safety_temporal}, spatial={safety_spatial}")
        if args.dry_run:
            print(run_dir(scenario.spec, calibration=calibration, config=config, study=study))
            continue
        result = optimize(scenario, params, config)
        directory = run_dir(result.scenario.spec, calibration=calibration, config=result.config, study=study)
        save_result(result, directory)


if __name__ == "__main__":
    main()
