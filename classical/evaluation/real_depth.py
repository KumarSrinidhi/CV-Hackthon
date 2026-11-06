"""Evaluate frozen classical depth upsampling on aligned real RGB-D data."""

import argparse
import json
import platform
import statistics
import tracemalloc
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from data.dataset import manifest_frames, read_rgb
from data.degradation import center_sample, holdout_mask
from data.preprocessing import evaluation_crop
from eval.metrics import quality
from .ablate_jbu import RESULTS, write_csv
from .random_stress import METHODS, predict
from .synthetic_data import SCALES


METRICS = ("rmse_m", "mae_m", "psnr_db", "ssim", "edge_f1", "boundary_rmse_m",
           "coverage", "latency_ms", "peak_traced_mib")
MIDDLEBURY_VARIANTS = {"MotorcycleE": "Motorcycle", "PianoL": "Piano", "PlaytableP": "Playtable"}


def middlebury_frames(root):
    """Read public left-view stereo references and convert disparity to metric depth."""
    scenes = sorted({path.parent for pattern in ("disp0GT.pfm", "disp0.pfm")
                     for path in root.rglob(pattern)})
    for folder in scenes:
        disparity_path = folder / "disp0GT.pfm"
        if not disparity_path.is_file():
            disparity_path = folder / "disp0.pfm"
        disparity = cv2.imread(str(disparity_path), cv2.IMREAD_UNCHANGED)
        if disparity is None or disparity.ndim != 2:
            raise ValueError(f"Could not read grayscale PFM: {disparity_path}")
        calib = dict(line.split("=", 1) for line in (folder / "calib.txt").read_text().splitlines()
                     if "=" in line)
        focal = float(calib["cam0"].strip().lstrip("[").split()[0])
        baseline = float(calib["baseline"]) / 1000
        denominator = disparity + float(calib["doffs"])
        valid = np.isfinite(denominator) & (denominator > 0)
        depth = np.divide(focal * baseline, denominator, out=np.zeros_like(disparity), where=valid)
        # Group perfect/imperfect views of the same underlying scene together.
        group = folder.name.removesuffix("-perfect").removesuffix("-imperfect")
        yield MIDDLEBURY_VARIANTS.get(group, group), folder.name, read_rgb(folder / "im0.png"), depth


def project_velodyne(points, shape, velo_to_rect, projection):
    rect = points @ velo_to_rect[:3, :3].T + velo_to_rect[:3, 3]
    rect = rect[np.isfinite(rect).all(axis=1) & (rect[:, 2] > 0)]
    projected = rect @ projection[:, :3].T + projection[:, 3]
    pixels = np.rint(projected[:, :2] / projected[:, 2, None])
    h, w = shape
    visible = ((pixels[:, 0] >= 0) & (pixels[:, 0] < w)
               & (pixels[:, 1] >= 0) & (pixels[:, 1] < h))
    x, y = pixels[visible].astype(np.intp).T
    depth = np.full(shape, np.inf, dtype=np.float32)
    np.minimum.at(depth, (y, x), rect[visible, 2])
    depth[np.isinf(depth)] = 0
    return depth


def kitti_frames(root):
    calibration = root / "calibration"
    cam_to_velo = np.eye(4)
    cam_to_velo[:3] = np.loadtxt(calibration / "calib_cam_to_velo.txt").reshape(3, 4)
    perspective = dict(line.split(":", 1) for line in
                       (calibration / "perspective.txt").read_text().splitlines() if ":" in line)
    rectification = np.fromstring(perspective["R_rect_00"], sep=" ").reshape(3, 3)
    projection = np.fromstring(perspective["P_rect_00"], sep=" ").reshape(3, 4)
    velo_to_rect = rectification @ np.linalg.inv(cam_to_velo)[:3]
    for rgb_path in sorted(root.glob("data_2d_raw/*/image_00/data_rect/*.png")):
        drive = rgb_path.parents[2].name
        scan = root / "data_3d_raw" / drive / "velodyne_points" / "data" / (rgb_path.stem + ".bin")
        if scan.is_file():
            points = np.fromfile(scan, dtype="<f4").reshape(-1, 4)[:, :3]
            rgb = read_rgb(rgb_path)
            yield drive, rgb_path.stem, rgb, project_velodyne(points, rgb.shape[:2], velo_to_rect, projection)


def evaluate(frames, dataset, repeats, peak, *, methods=METHODS, predictor=predict, on_frame=None):
    sparse = dataset == "kitti360"
    for group, frame, rgb, target in frames:
        try:
            rgb, target, valid = evaluation_crop(rgb, target, max(SCALES), peak)
        except ValueError as error:
            raise ValueError(f"{group}/{frame}: {error}") from error
        h, w = target.shape
        scored = holdout_mask(valid, SCALES) if sparse else valid
        for scale in SCALES:
            lr = center_sample(target, scale)
            lr_valid = lr > 0
            if not lr_valid.any() or not scored.any():
                raise ValueError(f"{group}/{frame} at {scale}×: empty LR input or evaluation support")
            for method in methods:
                output = predictor(method, lr, rgb, lr_valid, scale)
                timings = []
                for _ in range(repeats):
                    start = perf_counter()
                    predictor(method, lr, rgb, lr_valid, scale)
                    timings.append((perf_counter() - start) * 1000)
                tracemalloc.start()
                try:
                    predictor(method, lr, rgb, lr_valid, scale)
                    memory = tracemalloc.get_traced_memory()[1] / 1024**2
                finally:
                    tracemalloc.stop()
                yield dict(dataset=dataset, group=group, frame=frame, scale=scale, method=method,
                           height=h, width=w, lr_valid=int(lr_valid.sum()), scored=int(scored.sum()),
                           **quality(output, target, scored, sparse=sparse, peak=peak),
                           latency_ms=statistics.median(timings), peak_traced_mib=memory)
        if on_frame is not None:
            on_frame(group, frame)


def summarize(rows, *, confidence=True):
    by_group = defaultdict(list)
    for row in rows:
        for metric in METRICS:
            if row[metric] is not None:
                key = row["dataset"], row["scale"], row["method"], metric, row["group"]
                by_group[key].append(row[metric])
    groups = defaultdict(list)
    for (*key, _), values in by_group.items():
        groups[tuple(key)].append(statistics.mean(values))

    rng = np.random.default_rng(2026)
    for (dataset, scale, method, metric), values in sorted(groups.items()):
        n = len(values)
        finite = np.isfinite(values).all()
        sd = statistics.stdev(values) if finite and n > 1 else None
        low = high = None
        if confidence and finite and n >= 10:
            means = np.asarray(values)[rng.integers(n, size=(10000, n))].mean(axis=1)
            low, high = np.quantile(means, [0.025, 0.975])
        yield dict(dataset=dataset, scale=scale, method=method, metric=metric, groups=n,
                   mean=statistics.mean(values), sd=sd, ci95_low=low, ci95_high=high)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("nyu-v2", "kitti360", "middlebury"))
    parser.add_argument("source", type=Path, help="Aligned frame CSV or extracted Middlebury/KITTI-360 directory.")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--peak-meters", type=float,
                        help="Fixed reference depth ceiling and PSNR/SSIM range for dense datasets.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--descriptive", action="store_true",
                        help="Omit confidence intervals when independent groups are unverified.")
    args = parser.parse_args()
    if not args.source.exists():
        parser.error(f"Source does not exist: {args.source}")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    peak = args.peak_meters
    if peak is not None and (not np.isfinite(peak) or peak <= 0):
        parser.error("--peak-meters must be finite and positive")
    if args.dataset == "nyu-v2" and peak is None:
        peak = 10.0
    if args.source.is_file():
        frames = manifest_frames(args.source)
    elif args.dataset == "middlebury":
        frames = middlebury_frames(args.source)
    elif args.dataset == "kitti360":
        frames = kitti_frames(args.source)
    else:
        parser.error("NYU V2 requires an aligned frame CSV")
    cv2.setNumThreads(1)
    rows = list(evaluate(frames, args.dataset, args.repeats, peak))
    if not rows:
        parser.error("Source contained no evaluable frames")
    out = args.out or RESULTS / "real" / args.dataset
    write_csv(out / "frames.csv", rows)
    write_csv(out / "summary.csv", list(summarize(rows, confidence=not args.descriptive)))
    metadata = dict(dataset=args.dataset, source=str(args.source.resolve()), peak_meters=peak,
                    repeats=args.repeats, descriptive=args.descriptive, python=platform.python_version(),
                    numpy=np.__version__, opencv=cv2.__version__, platform=platform.platform())
    (out / "protocol.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} method/frame/scale rows to {out}")


if __name__ == "__main__":
    main()
