# Online Mode

Online mode connects a frame source, an Allegro force evaluator, atom-wise OOD
scoring, event buffering, and optional CP2K task submission.

Minimal flow:

1. Load config with `load_config()`.
2. Build an `AllegroRunner(force_evaluator=...)`.
3. Use either `LAMMPSController` or an iterable of `FrameData`.
4. Create `CP2KTaskSubmitter(mode="dry_run" | "local" | "slurm")`.
5. Pass the submitter through `OnlineEventScheduler`.
6. Run `OnlineMonitor.run()`.

`examples/06_online_monitor.py` uses a fake frame source and dry-run CP2K
submission, so it can be run without external executables.
