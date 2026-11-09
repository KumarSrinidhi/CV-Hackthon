# Application Guide (integration, demo, deployment)

## Data flow

```
upload / CLI args
   -> app/validation.py   (RGB, LR depth, mask, scale, method; clear errors)
   -> main.reconstruct    (reads configs/default_config.yaml via app/config.py)
        classical methods -> classical.pipeline.reconstruct
        deep              -> deep.inference.deep_hr_depth
        hybrid            -> hybrid.hybrid_hr_depth(input-adaptive fusion, consistency on)
                             (explicit classical fallback if deep unavailable / scale x2)
   -> app/visualization.py (colormap, boundaries, confidence, error map, side-by-side, PNG/NPY)
   -> CLI files  or  Gradio UI (app/ui.py: process())
```

The application only calls the three frozen wrapper interfaces; it never edits
`deep/`, `classical/`, `hybrid/`, `data/` or `eval/`.

## Input rules

- LR depth: 2-D `.npy` (metres) or an image (use `--depth-scale 0.001` for millimetre PNGs).
- RGB must be exactly `scale` times the LR depth in both dimensions.
- Mask (optional): same shape as the LR depth, non-zero = valid. Invalid depth (<= 0, NaN, Inf) is never a measurement.
- Scales: classical x2/x4/x8/x16; deep and hybrid x4/x8/x16.
- The Gradio UI resizes an RGB upload to the required dimensions when it does not
  match the LR depth; the status panel reports the adjustment. The CLI remains strict
  and reports a geometry error instead.

## Demo sequence

1. `python main.py --rgb rgb.png --depth lr.npy --gt gt.npy --scale 4 --method hybrid --compare --output out/`
   (or `python -m app.ui` and upload the same files).
2. Show `out/comparison_x4.png` (RGB | bicubic | classical | hybrid | GT, shared colour range).
3. Show the printed metrics and `out/hybrid_x4_error.png`.
4. Show the exports: `hybrid_x4.npy` (float32 metres) and `hybrid_x4.png` (16-bit, min-max scaled preview).
5. Show a failure on purpose (wrong scale) to demonstrate the validation message.

## Outputs

`<method>_x<scale>.npy` float32 depth · `.png` 16-bit min-max scaled · `_colormap.png` ·
`_error.png` (with `--gt`) · `comparison_x<scale>.png` (with `--compare`). If hybrid falls back, files
are named after the method that actually ran (`classical_*`) so results are never mislabelled.

## Deployment

```bash
pip install -r requirements.txt gradio
export SGNET_CKPT_DIR=/path/to/checkpoints/sgnet   # official x4/x8/x16 *.pth
python -m app.ui                                   # local: http://127.0.0.1:7860
```

For Hugging Face Spaces use the Gradio SDK with `app.py` containing
`from app.ui import build_app; demo = build_app(); demo.launch()`. Without the checkpoint the app
still runs and shows classical results, labelled as a fallback.

For AWS App Runner use the `Dockerfile` (listens on `0.0.0.0:$PORT`, default 8080) — see the
"Deploy to AWS App Runner" section in the top-level `README.md`.

## Not verified here

The Gradio layout was not launched (no gradio in the build environment) and local deep/hybrid
inference was not run (no torch, checkpoint or dataset). Both are covered by code-level tests only.
