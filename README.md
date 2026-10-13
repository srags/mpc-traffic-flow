# Quantifying Controllable Highway Congestion

Code, data and results for [Toward Controllability-Aware Performance Measures: A Case Study on Controllable Highway Congestion](https://arxiv.org/abs/2608.18510v1).

The pipeline fits a METANET model, optimizes variable speed limits (VSLs) with
Pyomo/IPOPT, then compares uncontrolled and controlled traffic:

```text
delay  = baseline TTT - free-flow TTT
CC (%) = (baseline TTT - controlled TTT) / delay * 100
```

TTT is total time spent, in vehicle-hours. [run_analysis](src/traffic_flow/results/analysis.py)
calculates report metrics; its displayed CC is clipped to 0–100%.
Case studies cover a synthetic 10 km bottleneck and ten weekdays of I-24
MOTION data (November 21–December 2, 2022).

## Setup

From the repository root:

```bash
conda env create -f environment-new.yml
conda activate automatic_control_2
make test
```

`environment-new.yml` includes Pyomo, IPOPT, notebook tools and pytest, using
conda-forge builds. The original `environment.yml` is a platform-specific export.
Experiment scripts add `src/` to their import path; for your own scripts or an
interactive session, use `PYTHONPATH=src`.

Paths derive from the location of `traffic_flow/paths.py`; no `MPC_ROOT`
environment variable is required. `data/` is read-only for repository maintenance.

## Main workflow

```python
from traffic_flow import load_scenario, load_params, optimize, save_result
from traffic_flow.config import CalRef, CalSource

calibration = CalRef(CalSource.FIXED_RAMPS)
scenario = load_scenario("i24", "11_30")
params = load_params(scenario, calibration)
config = scenario.spec.to_mpc_config(speed_lb=0, init_fixed=150.0)
result = optimize(scenario, params, config)
path = save_result(result, calibration=calibration, study=None)
```

To fit parameters instead of loading an existing calibration:

```python
from traffic_flow import calibrate
from traffic_flow.config import CalibrationConfig

params = calibrate(scenario, CalibrationConfig(tee=False))
```

`calibrate` returns parameter arrays; it does not save a calibration automatically.
`optimize` returns a complete `RunResult`: scenario, parameters, MPC settings,
policy, and baseline/controlled simulations. Only `save_result` writes that run.
Its `calibration` argument labels the directory; it does not run calibration.

### Inputs and settings

- `Scenario` bundles prepared observations (`TrafficData`) with resolved geometry
  and timing (`ScenarioConfig`). `load_scenario` infers duration and segment count
  from arrays. Its dataset registry supplies segment length and sample period:
  0.4 km and 10 seconds for I-24. The loader currently uses the full corridor/day.
  `ScenarioConfig.start_hour` is the wall-clock origin, defaulting to `7.5`
  (07:30); it does not select a simulation offset or change the input arrays.
- Inputs are `data/i24/i24_<date>/{rho_hat,q_hat,lane_mapping}.npy`.
  Measurements include two boundary columns. Prepared density is per lane;
  prepared flow is across all lanes. Boundary inflow/density use the existing
  two-sample edge-padded average.
- `MetanetParams` contains static segment vectors or time-by-segment arrays.
  `CalRef` selects saved parameter files; `CalibrationConfig` controls a new fit.
- `MPCConfig` controls optimization. Horizons, hold lengths and warm-up are in
  steps; speed bounds are in km/h. Internal time/distance units are hours/km.

Named choices live in [config.py](src/traffic_flow/config.py):

| Choice | Meaning |
|---|---|
| `CalSource.FIXED_RAMPS` | `calibration_static/fixed_ramping` |
| `CalSource.VARYING_RAMPS` | `calibration_static/time_varying_ramping` |
| `CalSource.DYNAMIC` | `calibration_dynamic`; select a block interval with `CalRef(..., interval=90)` |
| `Study.SPEED_LB`, `HOLD_LENGTH`, `SAFETY_SWEEP` | Result-directory labels; set actual constraints separately on `MPCConfig` |
| `InitMode.ADAPTIVE` | Initialize VSL from mean initial velocity |
| `LossMode.MINMAX`, `MEAN`, `MEAN_PLUS_WORST` | Robust calibration objectives, through `RobustOptConfig` |

For example, `CalibrationConfig(robust_opt=RobustOptConfig())` enables robust
calibration, not robust MPC. Custom calibration directory names remain strings.

`scenario.spec.to_mpc_config(...)` supplies experiment defaults: prediction/control
horizons 45/5, hold length 1, speed lower bound 0, controlled segments starting at
index 2, initial VSL 150, progress enabled, solver output disabled. Overrides use
`MPCConfig` field names. Direct `MPCConfig(...)` has different defaults.
The `v_fd_penalty` field is currently unused by the objective.

## Saved runs and lookup

Each NPZ contains observations, geometry, parameters, MPC settings, raw policy,
and both simulation histories/TTTs. Loading needs neither original input files
nor a solver run. Archives use schema version 2 and `allow_pickle=False`.
Version 1 remains readable: its `start_time` offset is converted to
`start_hour = 7.5 + start_time`. Saving replaces the destination atomically.

```text
results/<dataset>/<dataset>_<date>/<calibration.source>/
    [control_h_<interval>/]
    [<study>/]
    run.npz
```

Bracketed path components are optional. `collection_dir` constructs the path.
`find_npzs` lists NPZs directly in that directory, not recursively.
`results_bu/` is a separate historical backup and is not searched.

**The same filename overwrites the previous run**, even if settings differ.
Use `save_result(..., file_name="run2")` to retain another run. Calibration source
and study are directory selectors, not additional provenance fields in the NPZ.

```python
from traffic_flow.results.io import load_npz, load_results

run = load_npz(path)  # A full .npz path, not its containing directory.
matches = load_results(
    "i24", "11_30", calibration=calibration,
    study=None, file_name="run",
    where=lambda config: config.speed_lb == 0,
)
if len(matches) != 1:
    raise ValueError(f"Expected one run, found {len(matches)}")
run = matches[0]
```

`load_one_result` currently returns the first sorted match, not an enforced unique
match, and raises `IndexError` if none exist. Use the explicit check above when
multiple saved configurations are possible.

## Evaluation and reporting conventions

```python
from traffic_flow import evaluate
from traffic_flow.pipeline import replay
from traffic_flow.results.analysis import run_analysis
from traffic_flow.types import LATEST_UPDATE

comparison = evaluate(scenario, params, result.optimization.vsl, until_ff=False)
run = replay(run, until_ff=LATEST_UPDATE)
stats = run_analysis(run)
```

`evaluate` simulates without optimization. Its baseline uses measured origin
inflow (`real_data=True`) and no VSL; the controlled simulation uses the modeled
origin/queue and supplied policy. No VSL uses the simulator's unrestricted speed
proxy, not a 150 km/h cap. `replay` recomputes histories without changing the file.

The report switch is `LATEST_UPDATE` in [types.py](src/traffic_flow/types.py),
currently `False`:

| Mode | Report replay | Free-flow reference / vehicle count |
|---|---|---|
| `False` | Observed horizon only | Mainline arrivals × full-corridor free-flow travel time / mainline arrivals |
| `True` | Continue until drainage tolerance or step limit | Baseline vehicle-km divided by free-flow speeds / mainline and off-ramp departures |

Report scripts pass the switch to `replay`; `run_analysis` reads its imported copy
for metrics. `optimize` and `evaluate` still default to `until_ff=False`. Changing
the switch does not update saved bundles. Restart Python after changing it:
imported booleans are not live configuration objects. For time-varying free-flow
speed, report references use the maximum over time.

For **all** simulator return modes, TTT excludes terminal road occupancy but
includes the full queue history:

```python
TTT = T * (L * (density[:-1].sum(axis=0) @ lanes) + queue.sum())
```

This is the agreed convention, even though historical `main` included the final
density row. Histories still include the terminal state. Drainage adds no new
origin/on-ramp demand, releases VSL, and uses a zero-gradient downstream boundary;
it can reach a maximum step count with a warning.

## Experiment entry points

Edit each script's selected dates/settings before running from the root.
Currently `cc_run.py` selects **11/30**, while `cc_plot.py` selects **11/28**;
align them to report the run you just generated.

| Script | Action |
|---|---|
| `experiments/cc_run.py` | Load observations/parameters, optimize, save NPZ |
| `experiments/cc_plot.py` | Load/replay a bundle, print metrics, save CC and progressive-reveal figures |
| `experiments/table_plot.py` | Replay saved dates/calibrations; generate tables and bar chart |
| `experiments/tsd_plot.py` | Generate multi-day time-space diagrams |
| `experiments/constr_plot.py` | Replay constraint sweeps; generate curves/heatmap |
| `experiments/synthetic_cc_demand.py`, `synthetic_demand_tsd.py` | Replay synthetic CSV policies without optimization |

Reports write to `figs/`. The synthetic scripts currently point at
`results_bu/synthetic_10km/demand`; supplied CSVs are under
`results/synthetic_10km/demand`, so adjust their `POLICY_DIR` before using them.
Archived experiments are historical references, not maintained entry points.
`run_nb.sh` is a personal command list (including Git/cache operations), not a
general notebook runner.

The local [HTML viewer](html/README.md) opens with
`python html/server.py --open`; use `--port 8001` if needed. It reads saved NPZs
without replaying them. Its time-saved curve uses a controlled prefix plus
baseline remainder, not a new simulation at each cutoff, and retains the
demand-based free-flow reference independently of `LATEST_UPDATE`.

## Code organization

| Location | Responsibility |
|---|---|
| `config.py`, `types.py` | Settings, named choices, input/output records and Pyomo typing wrappers |
| `inputs/` | Dataset loading/preparation and synthetic input generation |
| `model/` | Parameter loading/slicing and numerical METANET simulation |
| `solvers/` | Pyomo calibration and MPC |
| `pipeline.py` | In-memory calibration → optimization → evaluation |
| `results/io.py` | Complete-run persistence and filtered lookup |
| `results/analysis.py` | Numerical report metrics |
| `results/plots.py` | Plotting and figure exports |
| `results/synthetic.py` | Historical synthetic-policy replay and figures |
| `paths.py`, `console.py` | Shared path layout and terminal formatting |

## Tests

`make test` runs the small pytest suite: input/simulation parity, both report
conventions, TTT across all simulator APIs, NPZ round trips, atomic-save failure
handling, HTML payloads and mocked MPC retry behavior. These checks do **not**
establish live optimizer convergence. Numerical checks use `rtol=atol=1e-10`;
reference implementations are read from pinned Git commits without switching
branches. Some integration cases require local data/results.

`make test-detailed` currently runs the same tests: its former optional
`test_solver_baseline.py` is no longer present. For the separate, slower read-only
all-bundle audit, see:

```bash
python -B tests/audit_main_parity.py --help
```

That audit compares saved policies against pre/post-update main snapshots;
it does not rerun optimization or regenerate figures.
