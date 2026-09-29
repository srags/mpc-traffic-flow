# Traffic Flow Scenario Explorer

A lightweight, local viewer for the METANET scenarios stored in the repository's
`results/` directory. It discovers optimized VSL files, simulates the associated
velocity and density histories, and displays synchronized time-space diagrams
and an aggregate road view. A continuous moving herringbone pattern shows
velocity by segment, while the fill height shows density; it does not imply
individual vehicle positions or a fixed number of lanes.

The playback timeline is an optimization cutoff. It starts with the complete
no-control baseline and progressively replaces it with optimized results from
left to right. A black divider marks the optimized history and baseline future.

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

The server reads from `data/`, `results/`, and `src/` without modifying them.
