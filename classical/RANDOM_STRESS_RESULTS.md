# Randomized RGB-D stress test

## Coverage fix and rerun

When color guidance leaves too little total weight to normalize, JBU now
falls back to spatial weights over the same valid local samples. The fallback
honors the kernel radius and any protected depth anchor. Pixels with no
eligible local samples still return zero.

The same seeds and settings produced zero constant-depth failures across
182 contract cases, including the original one-white-pixel reproducer.
Across all 1,680 method/input comparisons, coverage increased in 278 and
decreased in none. RMSE decreased in 270, increased in eight, and stayed
within `1e-7` of the original value in the other 1,402.

| 2× case | Method | Mean coverage before → after | Mean RMSE before → after |
| --- | --- | ---: | ---: |
| Aligned RGB | Tuned JBU | 100.00% → 100.00% | 0.1295 → 0.1295 |
| Unrelated RGB | Tuned JBU | 99.35% → 100.00% | 0.7572 → 0.7012 |
| Sparse depth | Tuned JBU | 65.33% → 82.98% | 2.1128 → 1.5895 |

Bilinear still had lower mean RMSE with unrelated RGB at all three scales.
Misleading RGB can still affect depth when the bilateral weights are strong
enough to normalize.

On one 640×480 NYU frame (`nyu_0000`), the 2× tuned JBU kernel produced
identical output before and after the fix. In seven alternating calls per
version after warm-up, median kernel latency rose from 439 to 488 ms and
peak traced allocations rose from 6.90 to 7.92 MiB. This spot check measures
the cost of accumulating spatial weights on that frame.

```powershell
.\venv\Scripts\python.exe -B -m classical.evaluation.random_stress --out classical/results/random-fixed
```

[Post-fix per-scene metrics](results/random-fixed/scenes.csv) ·
[Post-fix uncertainty summary](results/random-fixed/summary.csv) ·
[Post-fix contract checks](results/random-fixed/contracts.csv)

## Original run

Run on 6 November 2026 using Python 3.10.11, NumPy 2.2.6, and OpenCV 5.0.0
with one OpenCV thread. The evaluation kept the existing JBU parameters frozen.

```powershell
.\venv\Scripts\python.exe -B -m classical.evaluation.random_stress
```

The runner used seeds 1000–1019 at 2×, 4×, and 8×. Each seed generated four
conditions at each scale, for 240 paired inputs and 1,680 method/input
comparisons. The HR images ranged from 32 to 64 pixels per side. Each method
received the same LR depth, validity mask, and RGB image for a given input.

| Condition | Reference and degradation |
| --- | --- |
| Aligned RGB | Random overlapping ellipses and sloped depth surfaces, with RGB boundaries at depth boundaries plus RGB noise |
| Unrelated RGB | Same depth and LR measurements as aligned RGB, with independent random RGB pixels |
| Sparse | Same structured depth with unrelated RGB, higher LR noise, approximately 85% random dropout, and a central hole |
| IID depth | Independently sampled depth and RGB pixels, with little spatial structure to guide reconstruction |

All inputs used center subsampling. The three non-sparse conditions used 10%
random dropout before ensuring one valid sample. RMSE includes zero outputs
as errors against the known HR target. Coverage is the fraction of nonzero
outputs. Edge F1 uses the same pixelwise depth-boundary metric as the ablation.
For each input, timing is the median of three timed calls after one warm-up
call. The contract checks used a separate RNG seed, 91234.

## Reconstruction quality

Mean scene RMSE across 20 seeds per row, in the generated depth units.

| Scale | Condition | Nearest | Bilinear | A | B | C | D | E |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2× | Aligned RGB | 1.1804 | 0.4351 | 0.1749 | 0.1754 | 0.1295 | 0.1817 | 0.1799 |
| 2× | Unrelated RGB | 1.1804 | 0.4351 | 0.5845 | 0.5894 | 0.7572 | 0.6321 | 0.6320 |
| 2× | Sparse | 3.1861 | 2.7530 | 1.6057 | 1.6057 | 2.1128 | 1.5634 | 1.5634 |
| 2× | IID depth | 2.4809 | 1.9476 | 2.1124 | 2.1124 | 2.2786 | 2.2540 | 2.2537 |
| 4× | Aligned RGB | 1.2724 | 0.5812 | 0.2230 | 0.2232 | 0.2232 | 0.2870 | 0.2852 |
| 4× | Unrelated RGB | 1.2724 | 0.5812 | 0.7456 | 0.7466 | 0.7466 | 0.6944 | 0.6940 |
| 4× | Sparse | 3.1520 | 2.7140 | 1.5821 | 1.5821 | 1.5821 | 1.5387 | 1.5387 |
| 4× | IID depth | 2.7131 | 2.1514 | 2.1032 | 2.1032 | 2.1032 | 2.1168 | 2.1164 |
| 8× | Aligned RGB | 1.2441 | 0.7873 | 0.4044 | 0.4044 | 0.4053 | 0.4564 | 0.4548 |
| 8× | Unrelated RGB | 1.2441 | 0.7873 | 0.8628 | 0.8631 | 0.9042 | 0.8696 | 0.8687 |
| 8× | Sparse | 3.0468 | 2.5598 | 1.6967 | 1.6967 | 1.6830 | 1.6363 | 1.6363 |
| 8× | IID depth | 2.7363 | 2.2614 | 2.0490 | 2.0490 | 1.9952 | 2.0145 | 2.0140 |

A is JBU, B is original-mask JBU, C is tuned JBU, D is adaptive JBU, and E is
boundary-refined JBU.

- Replacing aligned RGB with unrelated RGB worsened A–E at every scale on
  identical depth inputs. Bilinear beat all five JBU variants in mean RMSE
  with unrelated RGB. The largest A-versus-bilinear gap occurred at 4× on
  seed 1019, where A scored 1.0861 and bilinear scored 0.7129 despite both
  having full coverage.
- Sparse 2× inputs showed the largest coverage difference between A and C.
  Mean coverage was 82.08% for A, 65.33% for C, and 82.91% for D and E.
  Bilinear reached 34.57%.
- IID depth produced Edge F1 above 0.90 for every JBU variant at 8× while RMSE
  remained around 2 depth units. Frequent reference edges can make the
  pixelwise overlap score look strong even when depth values remain far off.

### Paired tuned JBU versus JBU

The differences are C minus A, so negative values favor C. Each overall row
averages the four conditions within a seed before computing variability.
Intervals are 95% percentile bootstraps from 10,000 seed resamples.

| Scale | Condition | Mean RMSE difference | Sample SD | 95% interval |
| --- | --- | ---: | ---: | --- |
| 2× | Aligned RGB | −0.04544 | 0.02246 | [−0.05515, −0.03603] |
| 2× | Unrelated RGB | +0.17272 | 0.04546 | [+0.15359, +0.19289] |
| 2× | Sparse | +0.50712 | 0.15091 | [+0.44333, +0.57240] |
| 2× | IID depth | +0.16619 | 0.01495 | [+0.15977, +0.17258] |
| 2× | Overall | +0.20015 | 0.04590 | [+0.18119, +0.21956] |
| 4× | Overall | +0.00030 | 0.00078 | [0.00000, +0.00066] |
| 8× | Overall | −0.00628 | 0.01680 | [−0.01364, +0.00090] |

At 4×, A and C use the same JBU kernel parameters. A can also fill small holes
before upsampling, which explains how the methods can differ at that scale.
These intervals describe this generator and its chosen mix of stress cases.

## Runtime and input checks

Median / 95th percentile per-scene latency in milliseconds, pooled across the
four conditions at each scale. These measurements apply to the small images
and host used for this run.

| Scale | Bilinear | A | C | E |
| --- | ---: | ---: | ---: | ---: |
| 2× | 0.034 / 0.075 | 3.631 / 8.061 | 4.003 / 7.477 | 4.972 / 10.533 |
| 4× | 0.029 / 0.236 | 3.500 / 21.637 | 3.102 / 19.287 | 4.992 / 29.224 |
| 8× | 0.026 / 0.188 | 3.009 / 7.765 | 3.026 / 8.028 | 4.152 / 10.470 |

All 1,680 quality runs returned finite, nonnegative float32 output of the
expected shape, with positive values within the valid LR depth range to
floating-point tolerance. Repeated calls returned identical arrays and the
inputs were unmodified. Another 175 method/scale/shape combinations exercised
1×, 2×, 3×, 4×, and 8×, including one-row and one-column inputs and outputs
crossing the JBU kernel's 128-row chunk boundary. They passed the checks for
all-invalid masks, masked NaN/Inf/negative/outlier values, and equivalent
uint8 versus normalized float32 RGB.

**Constant-depth coverage failure.** With fully valid constant depth 3.25
and random RGB, tuned JBU at 2× emitted 27 zeros across three tested input
shapes, 8×1, 17×9, and 129×3. A separate high-contrast case reproduced the
failure in all five JBU variants. Give the method a fully valid 1×1 depth
image containing 3.25 and a 2×2 RGB image with one white pixel and three
black pixels. At 2× the JBU variants return 0 at the white pixel and 3.25
elsewhere. Nearest and bilinear preserve 3.25 at all four pixels. The white
pixel is far from the LR averaged RGB color, so its bilateral weight falls
below the kernel's normalization threshold and the output retains its
initialized zero.

The randomized stress run and 41 repository tests completed. The constant-depth
failures are recorded in the contract CSV. The reconstruction implementation
was kept fixed for this experiment.

## Artifacts

- [Per-scene metrics](results/random/scenes.csv)
- [Means, standard deviations, and confidence intervals](results/random/summary.csv)
- [Contract checks and constant-depth failures](results/random/contracts.csv)
- [Example comparison images for each condition and scale](results/random/images)

The runner generates these local artifacts. The inputs are synthetic and do
not measure real camera behavior or Redwood performance.
