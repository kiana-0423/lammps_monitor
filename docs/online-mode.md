# Online Mode

Online mode connects a frame source, an Allegro force evaluator, atom-wise OOD
scoring, event buffering, and optional CP2K task submission.

Monitoring runs as a staged cascade by default:

```text
light -> physics -> committee/full
```

`monitor.physics_interval: triggered` upgrades only after a light trigger, and
`monitor.committee_interval: triggered` upgrades only after a physics trigger.
Positive integer intervals are periodic sampling rules and do not require the
previous stage to trigger. `online.monitor_freq > 1` adds periodic forced-full
frames on matching frame indices; with the default `monitor_freq: 1`, frames use
the staged cascade rather than forcing full evaluation every step.

Minimal flow:

1. Load config with `load_config()`.
2. Build an `AllegroRunner.from_config(config)` for real Allegro/NequIP
   inference, or `AllegroRunner(force_evaluator=...)` for tests and custom
   runtimes.
3. Use either `LAMMPSController` or an iterable of `FrameData`.
4. Create `CP2KTaskSubmitter(mode="dry_run" | "local" | "slurm")`.
5. Pass the submitter through `OnlineEventScheduler`.
6. Run `OnlineMonitor.run()`.

`examples/06_online_monitor.py` uses a fake frame source and dry-run CP2K
submission, so it can be run without external executables.

For production inference, set `allegro.deployed_model_paths` after exporting
models and install the optional inference dependencies with
`pip install -e ".[inference]"`.
