"""Run the P4 cross-view robustness audit on matched Middlebury samples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import median, quantiles

import cv2

from data.degradation import DegradationConfig
from data.middlebury_raw import MiddleburyRawDataset
from eval.evaluate import BASELINES, evaluate_methods


class AuditDataset(MiddleburyRawDataset):
    def _load_raw(self, raw_idx):
        name, rgb, depth, valid = super()._load_raw(raw_idx)
        h, w = depth.shape
        y, x = (h - 160) // 2, (w - 160) // 2
        crop = (slice(y, y + 160), slice(x, x + 160))
        return name, rgb[crop], depth[crop], valid[crop]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default="results/submission_audit")
    parser.add_argument("--scales", nargs="+", type=int, choices=(2, 4, 8), default=(2, 4, 8))
    args = parser.parse_args()

    import torch
    from deep.deep_model import resolve_checkpoint

    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    noise = dict(noise_std=0.01, noise_relative=True)
    holes = dict(hole_ratio=0.25, n_block_holes=2)
    conditions = {
        "clean": DegradationConfig(),
        "noise": DegradationConfig(**noise),
        "holes": DegradationConfig(**holes),
        "noise_holes": DegradationConfig(**noise, **holes),
    }
    summary = {
        "protocol": "Middlebury 2006 ThirdSize, view 5, 160x160 centre crops, disparity pixels",
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
        "checkpoints": {},
        "runs": [],
    }
    rows = []
    for scale in args.scales:
        methods = (*BASELINES, "classical")
        if scale != 2:
            checkpoint = resolve_checkpoint(scale)
            with checkpoint.open("rb") as f:
                digest = hashlib.file_digest(f, "sha256").hexdigest()
            summary["checkpoints"][f"x{scale}"] = {"name": checkpoint.name, "sha256": digest}
            methods += ("deep", "hybrid")
        for condition, degradation in conditions.items():
            dataset = AuditDataset(args.root, scale, views=(5,), degradation=degradation)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            result = evaluate_methods(dataset, methods, verbose=False)
            for name, values in result["summary"].items():
                latency = [r["latency_ms"] for r in result["per_image"] if r["method"] == name]
                values["latency_p50_ms"] = median(latency)
                values["latency_p95_ms"] = quantiles(latency, n=20, method="inclusive")[-1] if len(latency) > 1 else latency[0]
            peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
            summary["runs"].append({
                "condition": condition, "meta": result["meta"], "methods": result["summary"],
                "not_evaluated": result["not_evaluated"], "peak_gpu_allocated_mib": peak,
            })
            rows.extend({"scale": scale, "condition": condition, **row} for row in result["per_image"])
            print(f"x{scale} {condition}: {len(rows)} cumulative predictions", flush=True)
            with (output / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            if rows:
                with (output / "per_image.csv").open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0])
                    writer.writeheader()
                    writer.writerows(rows)
    if any(run["not_evaluated"] or any(m["n_failed"] for m in run["methods"].values())
           for run in summary["runs"]):
        raise SystemExit("Audit recorded one or more unavailable methods or sample failures.")


if __name__ == "__main__":
    main()
