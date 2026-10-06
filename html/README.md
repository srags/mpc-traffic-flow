# Traffic Flow Scenario Explorer

A lightweight, local viewer for the METANET scenarios stored in the repository's
`results/` directory. It discovers `run.npz` bundles saved by `save_result`,
reads their baseline and controlled histories, and displays synchronized time-space diagrams
and an aggregate road view. A continuous moving herringbone pattern shows
velocity by segment, while the fill height shows density; it does not imply
individual vehicle positions or a fixed number of lanes.

The playback timeline is an optimization cutoff. It starts with the complete
no-control baseline and progressively replaces it with optimized results from
left to right. A black divider marks the optimized history and baseline future.

The travel-time line grows with the same cutoff. Its value is the total time
spent in the **displayed mix** (controlled prefix + baseline remainder), including
the origin queue. It starts at the saved baseline total and ends at the saved
controlled total. Savings and percentage reduction update alongside it. The
curve may temporarily rise; it is not forced to be monotone and is **not** a
fresh simulation with controls turned off after the cutoff.

Travel time is computed from full-precision saved density/queue histories:
`time_step * sum(segment_length * sum(density * lanes) + queue)`. The bundle's
terminal history row is included and switches to controlled with the final
displayed interval, keeping both endpoints consistent with the Python reports.
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

The server reads **only current `results/**/run.npz` bundles** using the shared
loader in `src/`; it does not read `data/` or `results_bu/`, rerun simulations,
run MPC, or copy/convert results. Geometry, lane counts and numerical settings
come from each bundle. The I-24 wall-clock display retains the 07:30 start;
datasets without a known clock origin display elapsed time. Reload the page
to discover new runs. Cached payloads are invalidated when a bundle is replaced.
