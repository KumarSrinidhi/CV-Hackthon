# Real-image depth upsampling

The dataset results below predate the [JBU coverage fix](RANDOM_STRESS_RESULTS.md#coverage-fix-and-rerun).

The evaluation compares nearest, bilinear, and the five frozen JBU variants
at 2×, 4×, and 8×. It uses metric depth and a fixed top-left crop whose
dimensions are multiples of eight. For each scale it takes the center sample
of each LR cell. Each method sees identical RGB, LR depth, and validity
inputs. The experiment measures controlled depth upsampling from real-image
references.

| Dataset | Reference and protocol |
| --- | --- |
| NYU Depth V2 | Aligned Kinect RGB and labeled, inpainted depth. A fixed 10 m ceiling defines reference validity and the PSNR/SSIM range. |
| Middlebury Stereo Eval3 | Public left-view training references at quarter resolution. Convert PFM disparity to meters using `baseline_mm * focal_px / (disparity_px + doffs_px) / 1000`. This run uses a fixed 20 m ceiling and PSNR/SSIM range. |
| KITTI-360 | Rectified left-camera RGB and projected, synchronized Velodyne camera-Z depth. Use the center-lattice points as inputs and score a common holdout that excludes input points from all three scales. Leave PSNR, SSIM, edge F1, and boundary RMSE blank for sparse references. |

Dataset documentation is available from [NYU Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html),
[Middlebury data and calibration](https://vision.middlebury.edu/stereo/data/scenes2014/),
[Middlebury Eval3 downloads](https://vision.middlebury.edu/stereo/submit3/),
[KITTI-360 projection documentation](https://www.cvlibs.net/datasets/kitti-360/documentation.php).

## Metrics

- **RMSE and MAE** use valid reference pixels and count zero predictions as
  errors. Coverage reports the fraction of scored pixels with positive depth.
- **PSNR and SSIM** use the stated fixed depth range. SSIM averages only
  11×11 windows where all reference pixels are valid.
- **Edge F1** uses exact pixel overlap of depth boundaries marked on both
  sides of a jump greater than 10% of the larger adjacent depth. Boundary
  RMSE scores all reference edge pixels. The CSV leaves undefined scores blank.
- **Latency** is the median of three CPU calls after a warm-up, with one
  OpenCV thread. It excludes data loading and metric computation.
- **Memory** is peak traced allocation during a separate inference call,
  measured with `tracemalloc` in MiB. It excludes preloaded inputs and does
  not capture all native allocations or total process RSS.

Summaries average frames within each scene or drive before averaging across
groups. They report sample standard deviation and use a 10,000-resample
percentile bootstrap for 95% intervals when there are at least ten groups
with finite results. Use `--descriptive` when independent groups are unverified.
Compare methods within the same dataset and protocol.

## Middlebury results

The run covered 15 public quarter-resolution training views in 12 scene
groups. The related Motorcycle, Piano, and Playtable views were grouped
before aggregation. These are depth upsampling results on public stereo
references. Middlebury's official leaderboard instead scores stereo
disparity at full resolution.

| Scale | Method | RMSE m | MAE m | PSNR dB | SSIM | Edge F1 | Boundary RMSE m | Latency ms | Peak traced MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2× | Bilinear | 0.0019 | 0.0003 | 83.66 | 1.0000 | 0.599 | 0.0077 | 2.99 | 4.17 |
| 2× | JBU | 0.0015 | 0.0003 | 85.59 | 1.0000 | 0.696 | 0.0065 | 439.78 | 12.14 |
| 4× | Bilinear | 0.0026 | 0.0006 | 80.97 | 0.9999 | 0.360 | 0.0091 | 3.22 | 3.90 |
| 4× | JBU | 0.0019 | 0.0005 | 83.75 | 1.0000 | 0.621 | 0.0072 | 625.05 | 10.88 |
| 8× | Bilinear | 0.0036 | 0.0010 | 78.45 | 0.9998 | 0.109 | 0.0101 | 2.40 | 3.83 |
| 8× | JBU | 0.0026 | 0.0008 | 81.11 | 0.9999 | 0.514 | 0.0079 | 470.51 | 10.56 |

JBU improved mean RMSE and boundary RMSE over bilinear at all three factors.
Among all seven methods, tuned JBU had the lowest mean RMSE at 2×, and
standard JBU had the lowest at 4× and 8×. SSIM rounds to nearly one with
the fixed 20 m range, so it separates these methods less clearly than
edge F1 or boundary RMSE. Latency was measured while the benchmark runs
overlapped and reflects this host and workload.

[Per-view Middlebury metrics](results/real/middlebury/frames.csv) ·
[Middlebury summary and confidence intervals](results/real/middlebury/summary.csv)

## NYU Depth V2 results

The run covered 16 of the 654 standard test frames. The table reports
descriptive frame means, including the mean of the per-frame median
latencies. The dense reference uses the labeled dataset's inpainted depths.

| Scale | Method | RMSE m | MAE m | PSNR dB | SSIM | Edge F1 | Boundary RMSE m | Latency ms | Peak traced MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2× | Bilinear | 0.0464 | 0.0114 | 48.01 | 0.9951 | 0.506 | 0.2984 | 3.19 | 4.18 |
| 2× | JBU | 0.0424 | 0.0116 | 48.76 | 0.9956 | 0.695 | 0.2546 | 471.12 | 12.03 |
| 4× | Bilinear | 0.0657 | 0.0183 | 44.91 | 0.9902 | 0.237 | 0.3704 | 2.59 | 3.90 |
| 4× | JBU | 0.0623 | 0.0190 | 45.44 | 0.9909 | 0.626 | 0.3214 | 500.88 | 10.76 |
| 8× | Bilinear | 0.0999 | 0.0326 | 41.28 | 0.9820 | 0.083 | 0.4529 | 2.59 | 3.83 |
| 8× | JBU | 0.1028 | 0.0355 | 41.12 | 0.9813 | 0.542 | 0.4207 | 479.29 | 10.45 |

At 8×, bilinear had the lowest mean RMSE across the seven methods, while
standard JBU had the highest edge F1. JBU reduced boundary RMSE relative
to bilinear at every factor, while bilinear had lower mean MAE than
standard JBU at all three factors.

[Per-frame NYU metrics](results/real/nyu-v2/frames.csv) ·
[NYU descriptive summary](results/real/nyu-v2/summary.csv)

## KITTI-360 status

The raw-directory loader, calibration and projection, common sparse holdout,
and CSV export passed an end-to-end fixture. No real KITTI-360 frames were
available for this run. The official dataset requires registration.
[KITTI-360 access](https://www.cvlibs.net/datasets/kitti-360/)

## Reproduce

Extract both Middlebury quarter-resolution archives, data and left-view
ground truth, into the same directory. The runner reads `im0.png`,
`disp0GT.pfm` or `disp0.pfm`, and `calib.txt`. Pass the public training folder.

```powershell
.\venv\Scripts\python.exe -B -m classical.evaluation.real_depth middlebury datasets/middlebury-q/MiddEval3/trainingQ --peak-meters 20
```

NYU Depth V2 and preprojected KITTI-360 inputs use an aligned manifest. Paths resolve relative to
the CSV. A 2-D NPY file or single-channel depth image is supported.
`meters_per_unit` is 1 for depth already in meters and 0.001 for millimeters.
Use the original scene ID for NYU and drive ID for KITTI-360 as `group`.

```csv
group,frame,rgb,depth,meters_per_unit
scene-01,0001,rgb/0001.png,depth/0001.npy,1
```

For NYU, export the official held-out indices with aligned `images` and
`depths` from the labeled MAT file, or use the millimeter PNG export's
`val` split. This run used 16 sorted positions from the 654 test frames,
selected by `np.linspace(0, 653, 16, dtype=int)` before scoring. The PNG
export does not supply scene IDs in the filenames, so this run summarizes
the selected frames descriptively. [NYU PNG export documentation](https://docs.ultralytics.com/datasets/depth/nyu-depth-v2)

For KITTI-360, pass the extracted dataset root or a preprojected manifest.
The directory loader reads matched rectified `image_00` PNGs and Velodyne
scans using the official calibration layout. It keeps the nearest positive
camera-Z when points project to the same pixel. For a preprojected manifest,
store depth in the same convention and set unsampled pixels to zero.

```powershell
.\venv\Scripts\python.exe -B -m classical.evaluation.real_depth nyu-v2 datasets/nyu-v2/test-subset.csv --descriptive
.\venv\Scripts\python.exe -B -m classical.evaluation.real_depth kitti360 datasets/kitti360
```

Each run writes `frames.csv`, `summary.csv`, and `protocol.json` under
`results/real/<dataset>/`.
