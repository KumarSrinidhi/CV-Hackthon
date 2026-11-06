"""Configurable SGNet fine-tuning v2: pluggable losses + degradation curriculum.

Run from repo root: python -m deep.finetune_v2 --help
Extends deep/finetune.py (which is left untouched) with:
  * loss composition  L = lam_d*L_depth + lam_g*L_grad + lam_m*L_multiscale
    (L_depth: masked L1 | smooth L1 | BerHu; L_grad: Sobel-gradient L1 on the
     main output and/or the unsupervised auxiliary gradient output ``out_re``;
     L_multiscale: masked L1 on 1/2 and 1/4 pooled predictions)
  * degradation curriculum (clean / noise / holes / block holes / combined),
    using data.degradation.DegradationConfig so the training operator matches
    the evaluation operator
  * multi-regime validation and checkpoint selection on validation only
FP32 is retained (SGNet Fourier blocks). No new dependencies.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.dataset import NYUv2Dataset
from data.degradation import DegradationConfig
from data.preprocessing import fill_invalid_nearest, minmax_range, minmax_normalize
from deep.deep_model import load_sgnet
from deep.inference import get_device
from deep.sgnet.common import Get_gradient_nopadding_d
from eval.metrics import compute_all_metrics

REGIMES = {
    "clean": DegradationConfig(),
    "noise": DegradationConfig(noise_std=0.02),
    "noise_strong": DegradationConfig(noise_std=0.05),
    "holes": DegradationConfig(hole_ratio=0.1),
    "holes_strong": DegradationConfig(hole_ratio=0.2),
    "block": DegradationConfig(n_block_holes=2),
    "combined": DegradationConfig(noise_std=0.02, hole_ratio=0.1),
}


def prepare(sample):
    mask = sample["lr_valid_mask"]
    lr = fill_invalid_nearest(sample["lr_depth"], mask)
    lo, hi = minmax_range(lr, mask)
    rgb = np.ascontiguousarray(sample["hr_rgb"].transpose(2, 0, 1))
    return rgb, minmax_normalize(lr, lo, hi), lo, hi


def draw_patch(dataset, rng, patch, scale):
    for _ in range(50):
        sample = dataset[int(rng.integers(len(dataset)))]
        rgb, lr, lo, hi = prepare(sample)
        h, w = lr.shape
        side = patch // scale
        if min(h, w) < side:
            raise ValueError(f"Patch {patch} exceeds image dimensions; reduce --patch.")
        y = int(rng.integers(h - side + 1))
        x = int(rng.integers(w - side + 1))
        ys = slice(y * scale, y * scale + patch)
        xs = slice(x * scale, x * scale + patch)
        valid = sample["valid_mask_original"][ys, xs].copy()
        if not valid.any() or hi - lo < 1e-6:
            continue
        target = (sample["depth_original"][ys, xs] - lo) / (hi - lo)
        arrays = [rgb[:, ys, xs], lr[None, y:y + side, x:x + side], target[None], valid[None]]
        if rng.random() < 0.5:
            arrays = [a[..., ::-1] for a in arrays]
        return [torch.from_numpy(np.ascontiguousarray(a)) for a in arrays]
    raise RuntimeError("Could not find a nonconstant valid training patch.")


def forward(model, rgb, lr):
    out = model((rgb, lr))
    if isinstance(out, (tuple, list)):
        return out[0], out[1]
    return out, None


def masked_error(pred, target, mask):
    return (pred - target).abs()[mask]


def depth_loss(kind, pred, target, mask):
    e = masked_error(pred, target, mask)
    if kind == "l1":
        return e.mean()
    if kind == "smooth_l1":
        return F.smooth_l1_loss(pred[mask], target[mask], beta=0.05)
    if kind == "berhu":
        c = 0.2 * float(e.max().detach()) + 1e-8
        ae = e.abs()
        return torch.where(ae <= c, ae, (ae * ae + c * c) / (2 * c)).mean()
    raise ValueError(kind)


_SOBEL = Get_gradient_nopadding_d()


def sobel_gradient(x):
    if _SOBEL.weight_v.device != x.device:
        _SOBEL.to(x.device)
    return _SOBEL(x)


def grad_loss(pred, target, mask):
    g = (sobel_gradient(pred) - sobel_gradient(target)).abs()
    interior = mask[:, :, 1:-1, 1:-1]
    if not interior.any():
        return pred.sum() * 0.0
    return g[:, :, 1:-1, 1:-1][interior].mean()


def pooled_mask_loss(pred, target, mask, factor):
    p = F.avg_pool2d(pred, factor)
    t = F.avg_pool2d(target, factor)
    m = F.avg_pool2d(mask.float(), factor) >= 0.99
    if not m.any():
        return pred.sum() * 0.0
    return (p - t).abs()[m].mean()


def multiscale_loss(pred, target, mask):
    return 0.5 * pooled_mask_loss(pred, target, mask, 2) + 0.5 * pooled_mask_loss(pred, target, mask, 4)


def composite_loss(args, pred, out_re, target, mask):
    losses = {}
    losses["depth"] = depth_loss(args.depth_loss, pred, target, mask)
    total = args.lam_d * losses["depth"]
    if args.lam_g > 0:
        g = grad_loss(pred, target, mask)
        if args.grad_target in ("out_re", "both") and out_re is not None:
            g = g + grad_loss(out_re, target, mask)
        losses["grad"] = g
        total = total + args.lam_g * g
    if args.lam_m > 0:
        losses["multiscale"] = multiscale_loss(pred, target, mask)
        total = total + args.lam_m * losses["multiscale"]
    return total, losses


@torch.no_grad()
def validate(model, dataset, count, device):
    model.eval()
    rows = []
    for i in range(min(count, len(dataset))):
        s = dataset[i]
        rgb, lr, lo, hi = prepare(s)
        pred, _ = forward(model, torch.from_numpy(rgb)[None].to(device),
                          torch.from_numpy(lr)[None, None].to(device))
        pred = pred[0, 0].cpu().numpy() * (hi - lo) + lo
        pred = np.maximum(pred, 0).astype(np.float32)
        if not np.isfinite(pred).all():
            raise RuntimeError("Nonfinite validation prediction.")
        metrics = compute_all_metrics(pred, s["depth_original"],
                                      data_range=s["data_range"], valid=s["valid_mask_original"])
        rows.append(metrics["rmse"])
    value = float(np.mean(rows))
    if not np.isfinite(value):
        raise RuntimeError("Nonfinite validation RMSE.")
    return value


def save_weights(model, path):
    temporary = path.with_suffix(".tmp")
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, temporary)
    temporary.replace(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--scale", type=int, choices=[4, 8, 16], default=4)
    p.add_argument("--patch", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--val-samples", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--schedule", choices=["constant", "cosine"], default="constant")
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    p.add_argument("--scope", choices=["head", "all"], default="head")
    p.add_argument("--depth-loss", choices=["l1", "smooth_l1", "berhu"], default="l1")
    p.add_argument("--lam-d", type=float, default=1.0)
    p.add_argument("--lam-g", type=float, default=0.0)
    p.add_argument("--lam-m", type=float, default=0.0)
    p.add_argument("--grad-target", choices=["main", "out_re", "both"], default="out_re",
                   help="apply the gradient loss to the main output, the auxiliary out_re, or both")
    p.add_argument("--train-regimes", default="clean",
                   help="comma list from: " + ", ".join(REGIMES))
    p.add_argument("--val-regimes", default="clean",
                   help="validation regimes; checkpoint selection uses the mean across them")
    a = p.parse_args(argv)
    if min(a.patch, a.batch_size, a.epochs, a.steps_per_epoch, a.val_samples) <= 0 or a.lr <= 0:
        p.error("Counts, patch size and learning rate must be positive.")
    if a.patch % (a.scale * 8):
        p.error("--patch must be divisible by scale*8 (128 works at x4/x8/x16).")
    for name in a.train_regimes.split(",") + a.val_regimes.split(","):
        if name not in REGIMES:
            p.error(f"Unknown regime '{name}'. Choose from: {', '.join(REGIMES)}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        p.error("Output directory is not empty. Choose a new --out to preserve earlier runs.")
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    device = get_device(a.device)
    train_names = a.train_regimes.split(",")
    val_names = a.val_regimes.split(",")
    train_sets = {n: NYUv2Dataset(a.root, split="train", scale=a.scale, degradation=REGIMES[n])
                  for n in train_names}
    val_sets = {n: NYUv2Dataset(a.root, split="val", scale=a.scale, degradation=REGIMES[n])
                for n in val_names}
    model, info = load_sgnet(a.scale, a.ckpt, device)
    model.requires_grad_(a.scope == "all")
    if a.scope == "head":
        model.last_conv.requires_grad_(True)
    params = [q for q in model.parameters() if q.requires_grad]
    optimizer = torch.optim.Adam(params, lr=a.lr, foreach=False)
    scheduler = None
    if a.schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs * a.steps_per_epoch)
    manifest = {**vars(a), "initial_model": info, "train_unit": train_sets[train_names[0]].unit,
                "val_unit": val_sets[val_names[0]].unit, "train_images": len(train_sets[train_names[0]]),
                "validation_images_per_regime": min(a.val_samples, len(val_sets[val_names[0]])),
                "trainable_parameters": sum(q.numel() for q in params),
                "loss": (f"lam_d={a.lam_d}*{a.depth_loss} + lam_g={a.lam_g}*grad({a.grad_target})"
                         f" + lam_m={a.lam_m}*multiscale"),
                "regime_configs": {n: REGIMES[n].as_dict() for n in set(train_names + val_names)},
                "split_note": "Existing image split; scene separation not verified.",
                "best_selection": f"mean val RMSE across regimes {val_names}; best.pth may retain pretrained weights",
                "torch_version": str(torch.__version__)}
    (out / "config.json").write_text(json.dumps(manifest, indent=2))
    print(f"Device={device}; trainable parameters={manifest['trainable_parameters']:,}", flush=True)
    baseline = {}
    for n in val_names:
        baseline[n] = validate(model, val_sets[n], a.val_samples, device)
        print(f"Pretrained val RMSE [{n}]: {baseline[n]:.6f} m", flush=True)
    best = float(np.mean(list(baseline.values())))
    save_weights(model, out / "best.pth")
    rng = np.random.default_rng(a.seed)
    start = time.perf_counter()
    history = []

    def record(epoch, losses, rmses):
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in losses.items()},
                        **{f"val_rmse_{n}": r for n, r in rmses.items()},
                        "val_rmse_mean": float(np.mean(list(rmses.values()))),
                        "lr": optimizer.param_groups[0]["lr"],
                        "elapsed_s": time.perf_counter() - start,
                        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20
                        if device.type == "cuda" else 0})
        with (out / "history.csv").open("w", newline="") as f:
            keys = []
            for row in history:
                for k in row:
                    if k not in keys:
                        keys.append(k)
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(history)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    record(0, {}, baseline)
    print(f"Pretrained mean val RMSE: {best:.6f} m", flush=True)
    for epoch in range(1, a.epochs + 1):
        model.train()
        totals = {}
        epoch_start = time.perf_counter()
        for step in range(a.steps_per_epoch):
            regime = train_names[int(rng.integers(len(train_names)))]
            samples = [draw_patch(train_sets[regime], rng, a.patch, a.scale) for _ in range(a.batch_size)]
            rgb, lr, target, mask = [torch.stack([s[k] for s in samples]).to(device) for k in range(4)]
            optimizer.zero_grad(set_to_none=True)
            pred, out_re = forward(model, rgb, lr)
            loss, parts = composite_loss(a, pred, out_re, target, mask)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss. Stop and inspect inputs.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, a.clip, error_if_nonfinite=True)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            for k, v in parts.items():
                totals[k] = totals.get(k, 0.0) + float(v.detach())
            if step == 0 or (step + 1) % 25 == 0:
                seconds = (time.perf_counter() - epoch_start) / (step + 1)
                part_s = ", ".join(f"{k}={v/(step+1):.5f}" for k, v in totals.items())
                print(f"Epoch {epoch}/{a.epochs}, step {step+1}/{a.steps_per_epoch} [{regime}]: "
                      f"{part_s}, {seconds:.2f} s/step", flush=True)
        optimizer.zero_grad(set_to_none=True)
        del samples, rgb, lr, target, mask, pred, loss
        rmses = {n: validate(model, val_sets[n], a.val_samples, device) for n in val_names}
        save_weights(model, out / "last.pth")
        mean_rmse = float(np.mean(list(rmses.values())))
        improved = mean_rmse < best
        if improved:
            best = mean_rmse
            save_weights(model, out / "best.pth")
        record(epoch, {k: v / a.steps_per_epoch for k, v in totals.items()}, rmses)
        print(f"Epoch {epoch}: val RMSE " + " ".join(f"[{n}]={r:.6f}" for n, r in rmses.items())
              + f" mean={mean_rmse:.6f}; best={best:.6f}", flush=True)
    best_epoch = min(history, key=lambda r: r["val_rmse_mean"])["epoch"]
    (out / "summary.json").write_text(json.dumps({
        "pretrained_val_rmse_m": history[0]["val_rmse_mean"],
        "pretrained_val_rmse_by_regime": history[0],
        "best_val_rmse_m": best, "best_epoch": best_epoch,
        "last_val_rmse_m": history[-1]["val_rmse_mean"],
        "note": "last.pth is adapted; best.pth retains baseline if adaptation did not improve.",
    }, indent=2))
    print(f"Finished. Weights and logs: {out}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except torch.cuda.OutOfMemoryError:
        raise SystemExit("CUDA out of memory. Close other GPU apps; use --scope head "
                         "--patch 64 --batch-size 1 with a NEW --out directory.")
