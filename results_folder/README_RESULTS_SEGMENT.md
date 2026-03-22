## Results (airborne acoustics, current)

Full benchmark details, plots, and per-depth breakdowns: [`results/RESULTS.md`](results/RESULTS.md)

### Classical ML — ExtraTrees on 20 acoustic features

Evaluated on **7 complete plate runs (343 holes)** held out from all training
and feature selection. Grouped holdout — the model never sees the acoustic
signature of a run during training.

| Metric | Value |
|--------|-------|
| MAE | **0.032 mm** |
| R² | 0.975 |
| P90 abs error | 0.071 mm |
| Nested CV MAE (31 runs, OOF) | 0.056 mm |
| Goal < 0.05 mm | ✓ |

### Deep Learning — SpecResNet regression

Evaluated on **2 completely unseen plate runs (98 holes)** — excluded from
every phase including hyperparameter selection.

| Metric | Value |
|--------|-------|
| MAE | **0.002 mm** |
| R² | 0.9999 |
| Step accuracy (±0.05 mm) | 100 % |

The DOE step size is 0.1 mm. Classical MAE = 32 % of one step.
DL MAE on unseen runs = 2 % of one step.

> Structure-borne and fused results will be added as those pipelines
> are completed. See [`results/RESULTS.md`](results/RESULTS.md) for
> the full breakdown including augmented test sets and per-depth analysis.

---

## Tests

```
42 passed in 7.51s — 100 % of test cases pass on a clean install
```

The test suite covers data I/O and segmentation (FLAC + HDF5), all feature
extraction families (including analytical checks for RMS and ZCR), classical
ML training and inference round-trip, and the full fusion layer including
weight computation, uncertainty propagation, and record alignment. Run with:

```bash
pytest -v
```
