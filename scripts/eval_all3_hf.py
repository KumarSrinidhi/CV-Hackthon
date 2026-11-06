# Team Owner: GPU verification run
# Module: scripts/eval_all3_hf.py
# Description: 3-track comparison (classical / deep / hybrid) + baselines on the same
#              HuggingFace NYU-v2 val samples, using the official evaluate_methods harness.
"""Usage:
    python scripts/eval_all3_hf.py --tars datasets/val-000000.tar --max-samples 100
"""
from __future__ import annotations

import argparse
import io
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np

from data.dataset import DepthSRDataset
from data.degradation import DegradationConfig, degrade
from data.preprocessing import to_float_rgb
from eval.evaluate import evaluate_methods, save_results


class HFNYUValDataset(DepthSRDataset):
    """Same sample contract as data.dataset loaders, backed by HF FastDepth val tars."""

    name = "hf_nyu_val"
    unit = "m"
    data_range = 10.0

    def __init__(self, scale: int, degradation=None, indices=None, tar_paths=()):
        super().__init__(scale, degradation, indices)
        self._samples = []
        for tp in tar_paths:
            try:
                with tarfile.open(tp, "r") as tar:
                    for member in tar:
                        if member.name.endswith(".h5"):
                            self._samples.append((tp, member.name))
            except tarfile.ReadError:
                print(f"note: {tp} truncated -> {len(self._samples)} samples so far")
                break

    def _num_raw(self) -> int:
        return len(self._samples)

    def _load_raw(self, raw_idx: int):
        tp, member_name = self._samples[raw_idx]
        with tarfile.open(tp, "r") as tar:
            f = tar.extractfile(tar.getmember(member_name))
            with h5py.File(io.BytesIO(f.read()), "r") as h5:
                rgb = np.transpose(np.array(h5["rgb"]), (1, 2, 0))
                depth = np.array(h5["depth"], dtype=np.float32)
        valid = np.isfinite(depth) & (depth > 0)
        return Path(member_name).stem, to_float_rgb(rgb), depth, valid


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tars", nargs="+", required=True)
    ap.add_argument("--max-samples", type=int, default=100)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--out", default="results/hf_nyu_val_x4_all3")
    ap.add_argument("--methods", nargs="+",
                    default=["nearest", "bilinear", "bicubic", "classical", "deep", "hybrid"])
    a = ap.parse_args(argv)

    ds = HFNYUValDataset(a.scale, DegradationConfig(), tar_paths=a.tars)
    print(f"{ds.name}: {len(ds)} samples available, evaluating up to {a.max_samples}, methods: {a.methods}")
    res = evaluate_methods(ds, a.methods, max_samples=a.max_samples)
    paths = save_results(res, a.out, f"hf_nyu_val_x{a.scale}")
    from eval.evaluate import summary_markdown
    print(summary_markdown(res))
    print("saved:", ", ".join(str(p) for p in paths.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
