# Traffic Flow Scenario Explorer

A lightweight, local viewer for the METANET scenarios stored in the repository's
`results/` directory. It discovers `.npz` bundles saved by `save_result`,
reads their baseline and controlled histories, and displays synchronized time-space diagrams
and an aggregate road view. A continuous moving herringbone pattern shows
velocity by segment, while the fill height shows density; it does not imply
individual vehicle positions or a fixed number of lanes.

Choose a **date**, **calibration**, and **study** separately. The dataset selector
appears when multiple datasets are available; dynamic calibrations also offer
their saved calibration interval. Study-specific controls select hold length,
minimum speed, or temporal and spatial safety bounds. Each selection narrows
the next controls to combinations that actually exist. Numeric choices are
sorted numerically. If several saved files still match, a **Run** selector keeps
all of them accessible instead of silently picking a duplicate.

Dates and constraint values come from each bundle's metadata/config, not its
filename. Calibration source, interval and study come from the existing result
directory layout, since those labels are not recorded in the NPZ schema. This
does not rename, migrate or write any results.

The playback timeline is an optimization cutoff. It starts with the complete
no-control baseline and progressively replaces it with optimized results from
left to right. A black divider marks the optimized history and baseline future.

Check **Include drainage** to replay the selected bundle with `until_ff=True`.
This uses the saved inputs, calibrated parameters and VSL policy; it does not
run MPC or save anything. After the original observation window, new origin and
on-ramp demand stop and VSLs are released, using the project's existing simulator.
The first replay may take a moment; it is cached in memory. Unchecking the box
returns to the saved histories. Bundles already containing drainage are also
supported without replay.

The timeline extends to the later of the baseline and controlled endpoints. A
separate marker shows where drainage begins. Once either recorded trajectory
ends, its remaining heatmap region is grey and contributes no further travel
time. Any simulator drain-limit warning is displayed: a history ending does not
by itself prove the corridor emptied.

The time-saved line grows with the same cutoff. It shows baseline delay minus
the delay of the **displayed mix** (controlled prefix + baseline remainder),
including the origin queue. It starts at zero and ends at the full policy's
savings. The percentage is `time_saved / baseline_delay * 100`, not a percentage
of baseline TTT (no percentage is shown if baseline delay is nonpositive).
The summary card still shows current delay: TTT minus a fixed free-flow TTT.
The curve may temporarily dip or go negative; it is not forced to be monotone
and is **not** a fresh simulation with controls turned off after the cutoff.

Travel time is computed from full-precision saved density/queue histories:
`time_step * sum(segment_length * sum(density * lanes) + queue)`. The viewer
matches the saved TTT: older bundles include terminal density; newer bundles
exclude it. Both include the full queue sum. Each trajectory's terminal
contribution is counted with its own final interval, even when the baseline and
controlled drainage lengths differ. Inconsistent totals are rejected.
Free-flow TTT comes from the shared `run_analysis` function, following the
project's `LATEST_UPDATE` setting: demand-based when false, vehicle-kilometre-based
when true. Restart the server after changing that setting. The drainage checkbox
controls replay only; it does not change the global reporting convention.
The free-flow reference is fixed across the reveal, not recalculated from the
mixed history. Negative delays and temporary decreases in savings are not clipped.
Only the color-display arrays are rounded. Click either heatmap or the line
chart to seek; arrow keys and Home/End also work when a chart is focused.
Playback stops at the end; pressing Play again restarts it.

The frontend is plain HTML, CSS and JavaScript. The Python server imports the
project package, so use the project environment (including NumPy, Pyomo and
tabulate), even though it never calls a solver. There is no Node.js, `npm`, or
`pnpm` setup.

## Open it

On macOS, double-click `run.command`.

Alternatively, run:

```bash
cd html
python3 server.py --open
```

The explorer opens at `http://localhost:8000`. Keep the Terminal window open
while using it; press `Control-C` there to stop it.

Use the project's Python/conda environment. If port 8000 is
already occupied, use `python3 server.py --port 8001 --open`.

The server reads **only current `results/**/*.npz` bundles** using
`load_npz(file_path)` in `src/`; it does not read `data/` or `results_bu/`,
run MPC, or copy/convert results. Simulation is opt-in via **Include drainage**
and stays entirely in memory. Geometry, lane counts and numerical settings
come from each bundle. The wall-clock display uses the saved scenario's
`start_hour` directly (`8` means 08:00 for I-24); the viewer adds no
dataset-specific time offset. Reload the page
to discover new runs. Cached payloads are invalidated when a bundle is replaced.

Bundles can live directly in the calibration/study directory, for example
`results/i24/i24_11_30/calibration_static/fixed_ramping/safety_sweep/run2.npz`.
Custom filenames are included in the selector labels. Invalid archives are
skipped; paths outside `results/` are rejected.
