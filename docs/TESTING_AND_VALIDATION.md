# Testing & Validation

## Run

```bash
pytest tests/ -q
```

## Result (this machine)

```
90 passed, 1 skipped in ~25s
```

The skipped test is `test_cuda_forward_and_profile` (no CUDA device).

## Coverage

| File | Focus |
|------|-------|
| `tests/test_classical.py` | interpolation, JBU, guided filter, hole filling, boundary refinement, wrapper validation, RGB edge relocation, adaptive JBU |
| `tests/test_deep.py` | SGNet model construction, checkpoint resolution/verification, dataset loaders, degradation, preprocessing, evaluation honesty |
| `tests/test_hybrid.py` | confidence normalization, invalid-deep handling, fusion, consistency, hybrid unavailability, determinism |
| `tests/test_pipeline.py` | cross-branch integration, all classical methods, scale 2/4 |
| `tests/test_app.py` | visualization, export, CLI dispatch, method routing |

## End-to-end checks

- `python -m classical.benchmark --scale 4` — runs all classical methods on a
  synthetic scene and reports metrics.
- `python main.py --rgb ... --depth ... --scale 4 --method classical|hybrid` —
  verified on synthetic data (classical gives RMSE 0 on a piecewise-constant scene).

## Honesty guarantees enforced by tests

- Invalid/non-finite depth is never treated as a measurement.
- A missing deep checkpoint makes `hybrid` report "not evaluated", not silently
  produce classical results.
- `W_c + W_d == 1` after normalization.
- Non-finite deep output is replaced by the classical value during fusion.
