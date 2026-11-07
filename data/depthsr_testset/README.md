# Depth Super-Resolution Test Dataset (20 samples)

A small synthetic dataset for **manual testing** of depth-guided or RGB-guided
depth super-resolution (depth SR). Each sample gives you exactly the four
inputs your evaluation loop expects:

| Your test input                         | File in this dataset            | Shape / type            |
| --------------------------------------- | ------------------------------- | ----------------------- |
| HR RGB image  `H × W × 3`               | `rgb.png`                       | 512 × 512 × 3, uint8    |
| LR depth map  `(H/s) × (W/s)`           | `lr_depth_{2,4,8}x.npy`         | see table below, float32 |
| Optional validity mask (same LR shape)  | `lr_mask_{2,4,8}x.npy`          | same LR shape, uint8    |
| Optional HR ground-truth depth (metrics)| `depth_hr.npy`                  | 512 × 512, float32      |

Everything was rendered with a vectorized ray tracer (planes, spheres, AABBs,
cylinders) with Lambertian shading + hard shadows, then downsampled with a
true block-mean (area average) to mimic a real depth sensor's pixel aperture.
Masks were generated to look sensor-like: holes concentrate at depth
discontinuities (mixed-pixel / edge failures), plus a small sparse failure
rate and an invalid 1-px border. Background / sky pixels (no surface hit) are
also marked invalid.

There are **20 samples** in two groups of 10:

- `sample_00` … `sample_09` — **indoor rooms** (NYU-style): enclosed room with
  floor, walls, ceiling and furniture (boxes + a sphere). Depth range ~2.4–7.2 m,
  ~85 % of LR pixels valid (rest are sensor holes).
- `sample_10` … `sample_19` — **geometric primitives**: single sphere, three
  spheres, cube, staircase, pillars, sphere pyramid, mixed, grid of cubes,
  concentric cylinders, offset stacked cubes, all on a checkered floor. Depth
  range ~2–10 m, ~35–45 % of LR pixels valid (the rest is open background/sky +
  holes). These stress sharp depth discontinuities and curved surfaces.

## Folder layout

```
depthsr_testset/
├── README.md            (this file)
├── index.csv            (per-sample summary: depth range, valid %)
└── sample_XX/
    ├── rgb.png          HR RGB, 512x512x3 uint8
    ├── depth_hr.npy     HR ground-truth depth, 512x512 float32, meters
    ├── lr_depth_2x.npy  LR depth,  256x256 float32, meters
    ├── lr_mask_2x.npy   LR mask,   256x256 uint8   (1 valid, 0 invalid)
    ├── lr_depth_4x.npy  LR depth,  128x128 float32
    ├── lr_mask_4x.npy   LR mask,   128x128 uint8
    ├── lr_depth_8x.npy  LR depth,   64x64  float32
    ├── lr_mask_8x.npy   LR mask,    64x64  uint8
    └── meta.json        per-sample metadata (camera, shapes, units)
```

## Conventions (read before using)

- **Depth unit:** meters. Positive float = distance along the camera +Z axis
  (pinhole camera at origin, looking down +Z, up = +Y).
- **Depth value `0.0`** means "no depth" — either a sensor hole or background
  with no surface hit. Always consult the mask; do **not** treat `0.0` as a real
  near-range measurement.
- **Mask:** `1 = valid pixel (real depth)`, `0 = invalid (sensor hole OR
  background/sky)`. The LR depth array is already zeroed at every `mask==0`
  pixel, so `lr_depth * mask == lr_depth` always holds.
- **LR downsampling:** block-mean over an `s × s` window of the HR depth
  (area average), which is the physically correct sensor model and produces
  mixed-pixel values at object edges — those edge pixels are then invalidated
  by the mask. Nearest-neighbor / bilinear-downsampled variants are **not**
  included; if your project expects those, re-downsample `depth_hr.npy`
  yourself (see snippet below).
- **Three scales are included per sample** (`2x`, `4x`, `8x`) so you can test
  each scale factor without re-running anything.

## Camera intrinsics (identical for all samples)

Pinhole camera, eye at origin, looking down +Z, +Y up, square 512 × 512 image.

| Parameter   | Value          |
| ----------- | -------------- |
| FOV         | 60° (full)     |
| Focal length| 443.41 px      |
| Principal point | (255.5, 255.5) px |
| Image size  | 512 × 512      |

A 3-D point `p = (x, y, z)` in camera space projects to pixel
`u = fx * x / z + cx`, `v = -fy * y / z + cy` (the minus sign is because image
row 0 is the top of the image while world +Y is up). `f` and `(cx, cy)` are
also stored in each `meta.json` under `camera`.

## Quick loader (Python)

```python
import json, numpy as np
from PIL import Image

def load_sample(folder, scale=4):
    """Returns (rgb_hr, lr_depth, lr_mask, depth_hr, meta)."""
    with open(f"{folder}/meta.json") as f:
        meta = json.load(f)
    rgb = np.asarray(Image.open(f"{folder}/rgb.png"))          # (512,512,3) uint8
    depth_hr = np.load(f"{folder}/depth_hr.npy")               # (512,512) float32, m
    lr_depth = np.load(f"{folder}/lr_depth_{scale}x.npy")      # (512/s,512/s) float32
    lr_mask  = np.load(f"{folder}/lr_mask_{scale}x.npy")       # same LR shape, uint8
    return rgb, lr_depth, lr_mask, depth_hr, meta

# Example: run your SR model on the LR depth, guided by the RGB,
# then evaluate against the HR ground truth on valid pixels only.
rgb, lr_d, lr_m, gt, meta = load_sample("sample_00", scale=4)
pred = your_model(rgb, lr_d, lr_mask=lr_m, scale=4)            # -> (512,512) float32

# Evaluate on HR pixels that were VALID in the LR input (upsample the mask),
# and exclude background (gt == 0).
import numpy as np
hr_valid = np.repeat(np.repeat(lr_m, 4, 0), 4, 1).astype(bool) & (gt > 0)
rmse = float(np.sqrt(((pred[hr_valid] - gt[hr_valid]) ** 2).mean()))
```

## Re-downsampling to a custom scale (if you need a different one)

```python
import numpy as np
def block_mean(arr, s):
    h, w = arr.shape; h2, w2 = h // s, w // s
    return arr[:h2*s, :w2*s].reshape(h2, s, w2, s).mean(axis=(1, 3))
gt = np.load("sample_00/depth_hr.npy")
lr_16x = block_mean(gt, 16)   # (32, 32) float32
```

## How to use this dataset for manual testing

1. Unzip the archive; the top folder is `depthsr_testset/`.
2. Pick a sample folder and a scale (`2x`, `4x`, or `8x`).
3. Feed your model `rgb.png` + `lr_depth_{s}x.npy` (+ `lr_mask_{s}x.npy` if your
   model supports validity masks).
4. Compare the model's HR depth output against `depth_hr.npy` (use the mask to
   exclude invalid/background pixels from the metric, as in the snippet above).
5. `index.csv` lists every sample's depth range and per-scale valid fraction so
   you can pick easy/hard cases (indoor rooms are fully filled and easier;
   primitive scenes have large background areas and sharper edges).

## Notes / limitations

- Synthetic scenes, not real captures — geometry and shading are clean, so this
  dataset is best for **functional / sanity testing** of your pipeline (shapes,
  scaling, masking, metric computation), not for benchmarking against
  real-world noise distributions.
- Shadows are hard (binary) rather than soft; no ambient occlusion.
- LR depth is generated by downsampling the HR depth, so it is geometrically
  consistent with `depth_hr.npy` by construction — do not use this dataset to
  validate depth-RGB *registration*; use it to validate the SR step itself.
