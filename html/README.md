# Traffic Flow Scenario Explorer

A lightweight, local viewer for the METANET scenarios stored in the repository's
`results/` directory. It discovers `.npz` bundles saved by `save_result`,
reads their baseline and controlled histories, and displays synchronized time-space diagrams
and an aggregate road view. A continuous moving herringbone pattern shows
velocity by segment, while the fill height shows density; it does not imply
individual vehicle positions or a fixed number of lanes.

The playback timeline is an optimization cutoff. It starts with the complete
no-control baseline and progressively replaces it with optimized results from
left to right. A black divider marks the optimized history and baseline future.

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
exclude it. Both include the full queue sum. Any terminal contribution switches
to controlled with the final displayed interval. Inconsistent totals are rejected.
Free-flow TTT uses the current reports' shared `get_ff_tts`:
`sum(inflow) * time_step * sum(segment_length / v_free)`, with the per-segment
maximum over time for dynamic free-flow speeds. It is computed once from the
saved bundle and subtracted at every cutoff, not recalculated from the mixed
history. This uses the current branch's demand-based convention, not upstream's
new vehicle-kilometre calculation. Negative delays are not clipped.
Only the color-display arrays are rounded. Click either heatmap or the line
chart to seek; arrow keys and Home/End also work when a chart is focused.
Playback stops at the end; pressing Play again restarts it.

It uses only plain HTML, CSS, JavaScript, Python, and NumPy. There is no Node.js,
`npm`, or `pnpm` setup.

## Open it

On macOS, double-click `run.command`.

Alternatively, run:

```bash
cd html
python3 server.py --open
```

The explorer opens at `http://localhost:8000`. Keep the Terminal window open
while using it; press `Control-C` there to stop it.

Use the project's Python/conda environment (it needs NumPy). If port 8000 is
already occupied, use `python3 server.py --port 8001 --open`.

The server reads **only current `results/**/*.npz` bundles** using
`load_npz(file_path)` in `src/`; it does not read `data/` or `results_bu/`, rerun simulations,
run MPC, or copy/convert results. Geometry, lane counts and numerical settings
come from each bundle. The I-24 wall-clock display retains the 07:30 start;
datasets without a known clock origin display elapsed time. Reload the page
to discover new runs. Cached payloads are invalidated when a bundle is replaced.

Bundles can live directly in the calibration/study directory, for example
`results/i24/i24_11_30/calibration_static/fixed_ramping/safety_sweep/run2.npz`.
Custom filenames are included in the selector labels. Invalid archives are
skipped; paths outside `results/` are rejected.
