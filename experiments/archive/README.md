# Historical drivers

These files were moved out of `src/`; they are retained for reference, not
advertised as working experiment entry points. The synthetic sweep retains its
historical assumptions and stale calls. Shared demand and travel-time helpers
now live in `traffic_flow.inputs.generation` and `traffic_flow.results.analysis`.

`run_controllable_congestion.ipynb` is also archived here. Its supported main
workflow is now `experiments/cc_run.py` (optimize and save) followed by
`experiments/cc_plot.py` (replay and report). The notebook retains exploratory
diagnostics, constant-VSL comparisons and older figure layouts for reference;
it is not maintained as a runnable entry point. Active-source checks exclude it.

Use the active scripts in `experiments/` to reproduce current figures. The
refactor will migrate those scripts onto the shared typed pipeline before
removing duplicated experiment orchestration.
