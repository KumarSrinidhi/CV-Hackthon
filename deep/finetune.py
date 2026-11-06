"""Patch-based SGNet fine-tuning. Run from repo root: python -m deep.finetune.
Uses the existing NYU loader and bicubic operator. Does not access the test split.
FP32 deliberately retained for SGNet's Fourier blocks. No new dependencies.
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
from data.dataset import NYUv2Dataset
from data.preprocessing import fill_invalid_nearest, minmax_range, minmax_normalize
from deep.deep_model import load_sgnet
from deep.inference import get_device
from eval.metrics import compute_all_metrics


def prepare(sample):
    """Use only LR input to obtain normalization, matching deployed inference."""
    mask = sample['lr_valid_mask']
    lr = fill_invalid_nearest(sample['lr_depth'], mask)
    lo, hi = minmax_range(lr, mask)
    rgb = np.ascontiguousarray(sample['hr_rgb'].transpose(2, 0, 1))
    return rgb, minmax_normalize(lr, lo, hi), lo, hi


def draw_patch(dataset, rng, patch, scale):
    # Degrade the full image BEFORE cropping, preserving the evaluation operator.
    for _ in range(50):
        sample = dataset[int(rng.integers(len(dataset)))]
        rgb, lr, lo, hi = prepare(sample)
        h, w = lr.shape
        side = patch // scale
        if min(h, w) < side:
            raise ValueError(f'Patch {patch} exceeds image dimensions; reduce --patch.')
        y = int(rng.integers(h - side + 1)); x = int(rng.integers(w - side + 1))
        ys = slice(y * scale, y * scale + patch)
        xs = slice(x * scale, x * scale + patch)
        valid = sample['valid_mask_original'][ys, xs].copy()
        if not valid.any() or hi - lo < 1e-6:
            continue
        target = (sample['depth_original'][ys, xs] - lo) / (hi - lo)
        arrays = [rgb[:, ys, xs], lr[None, y:y+side, x:x+side], target[None], valid[None]]
        # Paired augmentations preserve RGB/depth registration.
        if rng.random() < .5:
            arrays = [a[..., ::-1] for a in arrays]
        return [torch.from_numpy(np.ascontiguousarray(a)) for a in arrays]
    raise RuntimeError('Could not find a nonconstant valid training patch.')


def forward(model, rgb, lr):
    out = model((rgb, lr))
    return out[0] if isinstance(out, (tuple, list)) else out


@torch.no_grad()
def validate(model, dataset, count, device):
    """Full-image, fixed subset validation in metres using shared metric code."""
    model.eval()
    rows = []
    for i in range(min(count, len(dataset))):
        s = dataset[i]
        rgb, lr, lo, hi = prepare(s)
        pred = forward(model, torch.from_numpy(rgb)[None].to(device),
                       torch.from_numpy(lr)[None, None].to(device))
        pred = pred[0, 0].cpu().numpy() * (hi - lo) + lo
        pred = np.maximum(pred, 0).astype(np.float32)
        if not np.isfinite(pred).all():
            raise RuntimeError('Nonfinite validation prediction.')
        metrics = compute_all_metrics(pred, s['depth_original'],
                                      data_range=s['data_range'], valid=s['valid_mask_original'])
        rows.append(metrics['rmse'])
    value = float(np.mean(rows))
    if not np.isfinite(value):
        raise RuntimeError('Nonfinite validation RMSE.')
    return value


def save_weights(model, path):
    # Plain state_dict: immediately compatible with existing SGNet wrapper.
    temporary = path.with_suffix('.tmp')
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, temporary)
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default=None)
    p.add_argument('--ckpt', default=None)
    p.add_argument('--scale', type=int, choices=[4, 8, 16], default=4)
    p.add_argument('--patch', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--steps-per-epoch', type=int, default=200)
    p.add_argument('--val-samples', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', default='runs/sgnet_x4_finetune')
    p.add_argument('--scope', choices=['head', 'all'], default='head',
                   help='head: adapt final reconstruction convolutions; all: full fine-tuning')
    a = p.parse_args()
    if min(a.patch, a.batch_size, a.epochs, a.steps_per_epoch, a.val_samples) <= 0 or a.lr <= 0:
        p.error('Counts, patch size and learning rate must be positive.')
    if a.patch % (a.scale * 8):
        p.error('--patch must be divisible by scale*8 (128 works at x4/x8/x16).')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        p.error('Output directory is not empty. Choose a new --out to preserve earlier runs.')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    device = get_device(a.device)
    train = NYUv2Dataset(a.root, split='train', scale=a.scale)
    val = NYUv2Dataset(a.root, split='val', scale=a.scale)
    model, info = load_sgnet(a.scale, a.ckpt, device)
    # load_sgnet is inference-only by default: explicitly restore trainability.
    model.requires_grad_(a.scope == 'all')
    if a.scope == 'head':
        model.last_conv.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=a.lr, foreach=False)
    manifest = {**vars(a), 'initial_model': info, 'train_unit': train.unit,
                'val_unit': val.unit, 'train_images': len(train),
                'validation_images': min(a.val_samples, len(val)),
                'trainable_parameters': sum(p.numel() for p in params),
                'loss': 'masked L1 in valid-LR-normalized depth',
                'split_note': 'Existing image split; scene separation not verified.',
                'best_selection': 'validation RMSE; best.pth may retain pretrained weights',
                'torch_version': str(torch.__version__)}
    (out / 'config.json').write_text(json.dumps(manifest, indent=2))
    print(f'Device={device}; trainable parameters={manifest["trainable_parameters"]:,}', flush=True)
    print('Measuring pretrained validation baseline...', flush=True)
    best = validate(model, val, a.val_samples, device)
    save_weights(model, out / 'best.pth')
    rng = np.random.default_rng(a.seed)
    start = time.perf_counter()
    history = []
    def record(epoch, loss, rmse):
        history.append({'epoch': epoch, 'train_l1': loss, 'val_rmse_m': rmse,
                        'elapsed_s': time.perf_counter()-start,
                        'peak_allocated_mb': torch.cuda.max_memory_allocated(device)/2**20
                        if device.type == 'cuda' else 0})
        with (out / 'history.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader(); writer.writerows(history)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    record(0, '', best)
    print(f'Pretrained val RMSE: {best:.6f} m', flush=True)
    for epoch in range(1, a.epochs+1):
        model.train(); total = 0.
        epoch_start = time.perf_counter()
        for step in range(a.steps_per_epoch):
            samples = [draw_patch(train, rng, a.patch, a.scale) for _ in range(a.batch_size)]
            rgb, lr, target, mask = [torch.stack([s[k] for s in samples]).to(device)
                                     for k in range(4)]
            optimizer.zero_grad(set_to_none=True)
            pred = forward(model, rgb, lr)
            loss = (pred - target).abs()[mask].mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss. Stop and inspect inputs.')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step(); total += loss.item()
            if step == 0 or (step+1) % 25 == 0:
                seconds = (time.perf_counter()-epoch_start)/(step+1)
                print(f'Epoch {epoch}/{a.epochs}, step {step+1}/{a.steps_per_epoch}: '
                      f'L1={loss.item():.6f}, {seconds:.2f} s/step', flush=True)
        # Release gradients before full-resolution validation to reduce VRAM use.
        optimizer.zero_grad(set_to_none=True)
        del samples, rgb, lr, target, mask, pred, loss
        rmse = validate(model, val, a.val_samples, device)
        save_weights(model, out / 'last.pth')
        improved = rmse < best
        if improved:
            best = rmse
            save_weights(model, out / 'best.pth')
        record(epoch, total/a.steps_per_epoch, rmse)
        print(f'Epoch {epoch}: val RMSE={rmse:.6f} m; best={best:.6f} m', flush=True)
    (out / 'summary.json').write_text(json.dumps({
        'pretrained_val_rmse_m': history[0]['val_rmse_m'], 'best_val_rmse_m': best,
        'best_epoch': min(history, key=lambda r:r['val_rmse_m'])['epoch'],
        'last_val_rmse_m': history[-1]['val_rmse_m'],
        'note': 'last.pth is adapted; best.pth retains baseline if adaptation did not improve.'
    }, indent=2))
    print(f'Finished. Weights and logs: {out}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        raise SystemExit('CUDA out of memory. Close other GPU apps; use --scope head '
                         '--patch 64 --batch-size 1 with a NEW --out directory.')
