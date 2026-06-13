# infrastructure/ — benchmarks, training, data quality (shared)

- **computational_performance_evaluation** — wall-clock for forward + gradient over N (photons) × K (scatter iters), both tracking and calibration modes.
- **train_siren** — SIREN emitter training/validation harness (wraps `lucid-train-siren` / `lucid/siren/validate.py`).
- **data_vs_pred_hit_predictions** — prediction-vs-data hit statistics + Nrays dependence study.
