# P4 — Phased Implementation Plan
## Hybrid Depth-Map Super-Resolution

**Team:** Rakshitha · Tanush · Rishita · Kumar  
**Sprint duration:** 20 hours  
**Hardware:** RTX 5050 Laptop GPU (8 GB VRAM), 32 GB RAM  
**Primary objective:** Build, integrate and evaluate a hybrid depth-map super-resolution system using independent deep-learning and classical-computer-vision branches.

---

## 1. Project Goal

Given a high-resolution RGB guidance image and a low-resolution depth map, reconstruct a high-resolution depth map by combining:

1. **Deep-learning branch** — implemented and evaluated by Rakshitha.
2. **Classical CV branch** — implemented by Tanush.
3. **Hybrid fusion branch** — implemented by Rishita.
4. **Application and integration** — implemented by Kumar.

The two reconstruction branches must work independently before they are combined. The final hybrid must be compared fairly against both standalone branches and simple interpolation baselines.

## 2. Team Ownership

| Member | Workstream | Ownership |
|---|---|---|
| **Rakshitha** | Deep Learning + Data + Evaluation | Deep model integration, dataset pipeline, preprocessing, metrics and benchmark execution |
| **Tanush** | Classical Computer Vision | Interpolation baselines, JBU, guided filtering, hole handling and classical pipeline |
| **Rishita** | Hybrid Fusion | Confidence estimation, branch fusion, measurement consistency and boundary refinement |
| **Kumar** | Application + Integration + Deployment | UI, visualization, integration, repository coordination and final documentation |

### 2.1 Rakshitha — Deep Learning, Data and Evaluation

**Deep-learning responsibilities**
- Verify the official SGNet implementation, checkpoint, license and inference instructions.
- Verify the model's actual input/output contract and supported scale factors.
- Run a real forward pass on the RTX 5050 GPU; check latency and peak memory.
- Confirm that the model is intended for depth super-resolution and that its preprocessing matches the project data.
- Evaluate first on datasets/scales supported by the verified implementation and paper.
- Investigate KITTI separately. KITTI sparse LiDAR depth completion is not automatically equivalent to dense depth super-resolution.
- If SGNet is infeasible, time-box investigation of a task-appropriate alternative. Do not add a generic U-Net fallback.

**Data responsibilities**
- Prepare NYU-v2 and Middlebury loaders.
- Investigate KITTI format, alignment and sparse-depth handling if KITTI is included.
- Implement RGB-depth alignment and consistent depth units.
- Implement the exact low-resolution degradation protocol used by experiments.
- Create the original validity mask before any interpolation or hole filling.
- Preserve `depth_original`, `valid_mask_original` and branch-specific working copies.
- Create leakage-safe train/validation/test splits.

**Evaluation responsibilities**
- Implement common metrics and evaluation scripts.
- Ensure all branches use the same held-out samples, scales, depth units and evaluation mask.
- Record inference latency and peak GPU memory.
- Produce the final comparison tables and plots.

**Deliverables**
```text
deep/
  deep_model.py
  inference.py

data/
  dataset.py
  preprocessing.py
  degradation.py

eval/
  metrics.py
  evaluate.py
  benchmark.py
```

Required wrapper:
```python
def deep_hr_depth(lr_depth, hr_rgb, valid_mask, scale=4):
    ...
```

The wrapper must adapt to the verified model contract. Do not assume the pretrained model accepts an extra validity-mask channel.

### 2.2 Tanush — Classical Computer Vision

**Baselines**
- Nearest-neighbor interpolation.
- Bilinear interpolation.
- Bicubic interpolation.

**Classical reconstruction**
- Joint Bilateral Upsampling (JBU).
- Guided Filtering.
- Boundary-aware Joint Bilateral Filtering.
- Conservative depth-hole filling.
- Invalid-depth handling.

**Boundary and artifact handling**
- Use RGB Sobel/Canny edges as guidance, not as ground-truth depth boundaries.
- Use depth discontinuities as primary boundary evidence.
- Avoid transferring RGB texture edges into the reconstructed depth map.
- Avoid smoothing across genuine depth discontinuities.
- Preserve the original depth map and validity mask; do not overwrite them with filled values.

**Evaluation**
- Compare the classical methods on validation data.
- Tune parameters without using test ground truth.
- Select the best-performing classical pipeline.
- Report quality, runtime and failure cases.

**Deliverables**
```text
classical/
  interpolation.py
  jbu.py
  guided_filter.py
  boundary_refinement.py
  hole_filling.py
  pipeline.py
```

Required wrapper:
```python
def classical_hr_depth(lr_depth, hr_rgb, valid_mask, scale=4):
    ...
```

### 2.3 Rishita — Hybrid Fusion

**Branch integration**
- Consume the outputs from Rakshitha and Tanush: deep HR depth `D_d` and classical HR depth `D_c`.
- Do not duplicate either reconstruction branch.
- Validate that both outputs have matching dimensions, units and compatible invalid-value conventions.

**Confidence-guided fusion**
- Estimate branch confidence using available evidence such as depth gradients, RGB/depth edge agreement, original validity information, distance to valid measurements, model uncertainty (if available) and reconstruction residuals.
- Normalize weights so that `W_c + W_d = 1`.
- Fuse:

  \[
  D_f = W_c D_c + W_d D_d
  \]

- Calibrate weights on validation data. Do not present arbitrary hand-picked confidence formulas as calibrated confidence.

**Measurement consistency**
- If the synthetic degradation operator is known, compute:

  \[
  r_{LR} = D_{LR} - S(D_f)
  \]

- Use the residual to correct the fused output.
- For real sensor data with unknown formation/degradation, do not claim exact consistency. Anchor known measurements only where justified.

**Adaptive boundary refinement**
- Use depth evidence as the primary boundary signal.
- Use RGB guidance as supporting evidence.
- Refine likely depth boundaries while avoiding texture copying and oversmoothing.

**Ablations**
1. Classical only.
2. Deep only.
3. Fixed-weight fusion.
4. Confidence-guided fusion.
5. Confidence-guided fusion plus boundary refinement.
6. Final hybrid with consistency correction where the forward operator is known.

**Deliverables**
```text
hybrid/
  confidence.py
  fusion.py
  consistency.py
  boundary_refinement.py
  pipeline.py
```

Required wrapper:
```python
def hybrid_hr_depth(lr_depth, hr_rgb, valid_mask, scale=4):
    ...
```

### 2.4 Kumar — Application, Integration and Deployment

**Application**
- RGB image upload.
- LR depth-map upload.
- Optional validity-mask upload.
- Scale selection.
- Method selection: Classical, Deep or Hybrid.
- Output depth visualization.
- Side-by-side branch comparison.
- Confidence map and error map when applicable.
- PNG and NPY export.
- Optional PLY export if time permits.

**Integration**
- Create the application entry point and connect the stable branch interfaces.
- Own application configuration and coordinate dependency-file changes.
- Add input validation and clear error messages.
- Run end-to-end tests.
- Optimize memory and inference flow where necessary.

**Delivery**
- README and setup instructions.
- Architecture and implementation documentation.
- Demonstration workflow and screenshots.
- Deployment instructions and final presentation support.

**Deliverables**
```text
app/
  ui.py
  visualization.py

configs/
docs/
main.py
README.md
```

---

## 3. Phased Execution Plan

The sprint is divided into seven phases. Each phase has a goal, owners, concrete tasks and an exit condition. Do not move to integration based only on code being present; the relevant phase must pass its exit condition.

### Phase 0 — Preparation and Feasibility Checks
**Timing:** Before the 20-hour sprint

**Rakshitha**
- Locate the official deep-model repository and checkpoint.
- Check license, dependencies, expected inputs, supported scales and dataset protocol.
- Verify dataset availability and access.
- Prepare a minimal GPU inference test.

**Tanush**
- Confirm classical method dependencies and implementation choices.
- Prepare small RGB/depth test pairs and expected input conventions.

**Rishita**
- Review the fusion design and agree on the input/output contract.
- Prepare synthetic/mock branch outputs for early development.

**Kumar**
- Create the repository, folder structure and individual branches.
- Add a minimal application skeleton.
- Add the initial README and environment setup instructions.

**Phase exit condition**
- Dataset availability is known.
- Deep-model feasibility has a clear status: verified, at risk or blocked.
- Shared interfaces and folder ownership are agreed.
- Everyone can run the project environment.

### Phase 1 — Environment, Contracts and Baselines
**Timing:** Hours 0–1

**All members**
- Verify Python environment and dependencies.
- Confirm common data types, tensor/array layouts, image dimensions and depth units.
- Freeze function signatures and error conventions.
- Agree on supported scale factors for the first end-to-end demo.

**Rakshitha**
- Run GPU availability and actual model forward-pass smoke tests.
- Confirm dataset loader assumptions.

**Tanush**
- Implement or verify nearest, bilinear and bicubic baselines.

**Rishita**
- Create fusion module skeleton and mock prediction fixtures.

**Kumar**
- Verify repository branches and launch the UI skeleton.

**Phase exit condition**
- All members can run their own module.
- Interface contract is written down.
- Baseline interpolation produces an output with the expected dimensions.

### Phase 2 — Independent Branch Development
**Timing:** Hours 1–6

**Rakshitha**
- Build dataset loading, preprocessing and degradation pipeline.
- Integrate the selected deep model.
- Run inference on a small sample.
- Record output shape, depth units, runtime and GPU memory.
- If the primary model fails the feasibility gate, investigate the pre-agreed alternative within a strict time limit.

**Tanush**
- Implement JBU and Guided Filtering.
- Implement invalid-depth handling and conservative hole filling.
- Implement boundary-aware refinement.
- Compare methods on a small validation subset.

**Rishita**
- Implement confidence estimation and fusion using mock branch predictions.
- Add shape/unit checks and invalid-value handling.
- Write unit tests for normalized weights and fusion behavior.

**Kumar**
- Build upload controls, method selection and output visualization using mock outputs.
- Keep the UI independent of model internals.

**Phase exit condition**
- Classical branch returns an HR depth map.
- Deep branch either returns a verified HR depth map or has a documented blocker and fallback decision.
- Fusion works with mock outputs.
- UI can display mock outputs.

### Phase 3 — Branch Freeze and Interface Handoff
**Timing:** Hours 6–10

**Rakshitha**
- Deliver the deep wrapper and its preprocessing requirements.
- Deliver dataset and evaluation interfaces.
- Document model limitations and supported scales.

**Tanush**
- Deliver the selected classical pipeline and wrapper.
- Document parameters and known failure cases.

**Rishita**
- Replace mock predictions with actual branch outputs as they become available.
- Validate shape, unit and mask compatibility.
- Run first hybrid inference.

**Kumar**
- Connect the branch wrappers to the application through the agreed interfaces.
- Avoid editing deep/classical/hybrid implementation files.

**Phase exit condition**
- Both branch interfaces are stable, or a blocker is explicitly recorded.
- Hybrid accepts real branch outputs where available.
- Application can run at least one complete inference path.

### Phase 4 — Hybrid Refinement and End-to-End Integration
**Timing:** Hours 10–14

**Rakshitha**
- Run the common evaluation pipeline for deep and classical branches.
- Verify evaluation masks and units.
- Produce initial standalone benchmark results.

**Tanush**
- Tune classical parameters using validation data.
- Fix visible artifacts and runtime bottlenecks.

**Rishita**
- Calibrate confidence weights on validation data.
- Add boundary refinement.
- Add measurement-consistency correction only when the forward degradation operator is known.
- Run the planned ablations.

**Kumar**
- Complete end-to-end application integration.
- Add result export and error handling.
- Test the full user workflow.

**Phase exit condition**
- Classical, deep and hybrid paths run end to end, or any unavailable path is clearly documented.
- Initial metrics are generated on the same evaluation samples.
- No critical interface mismatch remains.

### Phase 5 — Evaluation, Optimization and Reliability
**Timing:** Hours 14–17

**Rakshitha**
- Produce final metrics and benchmark tables.
- Check repeatability and report inference latency/peak memory.
- Ensure no test-set tuning has occurred.

**Tanush**
- Freeze the best classical configuration.
- Document failure cases and parameter settings.

**Rishita**
- Compare hybrid against both standalone branches.
- Remove fusion components that do not improve results or introduce artifacts.
- Freeze the final fusion configuration.

**Kumar**
- Test application behavior with valid and invalid inputs.
- Check export files and clean startup.
- Address integration bugs.

**Phase exit condition**
- Final configurations are frozen.
- Results are reproducible.
- Any claimed hybrid improvement is supported by the evaluation.

### Phase 6 — Final Freeze and Demonstration
**Timing:** Hours 17–20

**All members**
- Stop adding new algorithms and features.
- Fix only critical bugs.
- Verify the final demo inputs and outputs.
- Confirm the repository can be set up from the README.

**Rakshitha**
- Finalize model, dataset and evaluation documentation.

**Tanush**
- Finalize classical method documentation and results.

**Rishita**
- Finalize fusion design, ablations and limitations.

**Kumar**
- Freeze the integrated application.
- Finalize README, architecture diagram, screenshots and demonstration sequence.

**Phase exit condition**
- A clean end-to-end demonstration is ready.
- Code, configuration and documentation are committed.
- Results and limitations are accurately reported.

---

## 4. Final System Architecture

```text
                 INPUT
      ┌───────────────────────────┐
      │ HR RGB + LR Depth + Mask  │
      └─────────────┬─────────────┘
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
┌──────────────────┐  ┌──────────────────┐
│ Classical Branch │  │   Deep Branch    │
│     Tanush       │  │    Rakshitha     │
└────────┬─────────┘  └────────┬─────────┘
         │                     │
         ▼                     ▼
   Classical HR           Deep HR
    Depth D_c              Depth D_d
         │                     │
         └──────────┬──────────┘
                    ▼
          ┌─────────────────────┐
          │   Hybrid Fusion     │
          │      Rishita        │
          │                     │
          │ Confidence Fusion   │
          │ Consistency Check   │
          │ Boundary Refinement │
          └──────────┬──────────┘
                     ▼
          ┌─────────────────────┐
          │ Final HR Depth Map  │
          └──────────┬──────────┘
                     ▼
          ┌─────────────────────┐
          │ UI / Visualization  │
          │ Kumar               │
          └─────────────────────┘
```

## 5. Dataset and Model Strategy

| Dataset | Deep branch | Classical branch | Notes |
|---|---|---|---|
| NYU-v2 | Primary evaluation candidate for SGNet, subject to implementation verification | Evaluate all selected classical methods | Dense RGB-D setting |
| Middlebury | Evaluate if supported by the verified model/checkpoint | Evaluate all selected classical methods | High-resolution depth evaluation |
| KITTI | Separate compatibility investigation; do not assume SGNet support | Sparse-depth-aware processing if included | Sparse LiDAR depth completion differs from dense depth SR |

**Important model distinction:** SGNet and SGSNet are different models. Do not select SGSNet as if it were SGNet. A sparse-depth completion model may be relevant to KITTI, but it is not automatically a drop-in replacement for a depth super-resolution model.

## 6. Shared Interface Contract

Freeze these interfaces before parallel development:

```python
def deep_hr_depth(lr_depth, hr_rgb, valid_mask, scale=4):
    # Return HR depth using the verified deep model.
    ...

def classical_hr_depth(lr_depth, hr_rgb, valid_mask, scale=4):
    # Return HR depth using the selected classical pipeline.
    ...

def hybrid_hr_depth(lr_depth, hr_rgb, valid_mask, scale=4):
    # Return HR depth by combining the two branch outputs.
    ...
```

**Common conventions to document and freeze**
- RGB channel order and numeric range.
- Depth array shape and data type.
- Depth units and invalid-value representation.
- Validity-mask shape and meaning.
- Scale-factor definition and output dimensions.
- Whether each wrapper returns a NumPy array or tensor.
- Device handling and error behavior.

The deep wrapper may internally use a different input contract. The public wrapper above is an adapter, not a claim about the model's native inputs.

---

## 7. Repository Structure and File Ownership

```text
P4-Depth-SuperResolution/
├── deep/                       # Rakshitha
│   ├── deep_model.py
│   └── inference.py
├── data/                       # Rakshitha
│   ├── dataset.py
│   ├── preprocessing.py
│   └── degradation.py
├── eval/                       # Rakshitha
│   ├── metrics.py
│   ├── evaluate.py
│   └── benchmark.py
├── classical/                  # Tanush
│   ├── interpolation.py
│   ├── jbu.py
│   ├── guided_filter.py
│   ├── boundary_refinement.py
│   ├── hole_filling.py
│   └── pipeline.py
├── hybrid/                     # Rishita
│   ├── confidence.py
│   ├── fusion.py
│   ├── consistency.py
│   ├── boundary_refinement.py
│   └── pipeline.py
├── app/                        # Kumar
│   ├── ui.py
│   └── visualization.py
├── configs/                    # Kumar coordinates
├── docs/                       # Kumar coordinates
├── tests/                      # Each owner tests their own modules
├── main.py                     # Kumar coordinates
├── requirements.txt            # Kumar coordinates changes
└── README.md                   # Kumar coordinates
```

## 8. Git Workflow — Conflict Prevention

### 8.1 Branch naming

Create one long-lived feature branch per member:

- `feature/deep-data-rakshitha`
- `feature/classical-tanush`
- `feature/hybrid-rishita`
- `feature/app-kumar`

No one commits directly to `main`.

### 8.2 Folder ownership

- Rakshitha: `deep/`, `data/`, `eval/`
- Tanush: `classical/`
- Rishita: `hybrid/`
- Kumar: `app/`, `configs/`, `docs/`, `main.py`, `README.md`

Do not edit another member's owned files without an explicit handoff.

### 8.3 Shared-file policy

The following files are common conflict points:
- `requirements.txt`
- `main.py`
- `README.md`
- Shared configuration files
- Shared interface definitions

Kumar coordinates changes to these files. Members send dependency or integration requests to Kumar rather than editing the same files in parallel.

### 8.4 Interface-first development

- Write the function signatures and data conventions before implementation.
- Do not change a public function signature without informing all affected members.
- If a contract change is unavoidable, propose it first and update the contract in one coordinated commit.

### 8.5 Mock-first parallel work

- Rishita develops and tests fusion with deterministic mock branch outputs.
- Kumar develops and tests the UI with mock outputs.
- Rakshitha and Tanush work independently on their branches.
- Replace mocks with real outputs only after the branch owners confirm their wrappers are stable.

### 8.6 Pull-request process

1. Commit only files owned by the member.
2. Push the member's feature branch.
3. Open a pull request into `main`.
4. Include a short summary and test evidence.
5. Request review from the affected owner(s).
6. Merge only after tests pass and the interface is confirmed.

Kumar coordinates integration order, but each owner remains responsible for testing their own code.

### 8.7 Safe synchronization

- Pull the latest `main` into the feature branch before starting a new work session and before opening a pull request.
- Do not rebase or force-push a branch another member is using.
- Avoid simultaneous edits to shared files.
- Keep commits small and focused.
- Do not merge unfinished code simply to satisfy the schedule.

---

## 9. Evaluation Protocol

For a fair comparison:

1. Use the same held-out scenes for all methods.
2. Use the same scale factor and depth units.
3. Use the same evaluation mask.
4. Tune parameters and confidence weights on validation data only.
5. Keep test data untouched until final evaluation.
6. Report unsupported datasets/scales as not evaluated rather than implying compatibility.

**Methods to compare**
- Nearest-neighbor
- Bilinear
- Bicubic
- Best classical pipeline
- Deep model
- Fixed-weight fusion
- Confidence-guided hybrid
- Final hybrid with applicable refinement stages

**Metrics**
- RMSE and MAE for depth error.
- PSNR and SSIM with a fixed, documented depth range.
- Boundary RMSE around ground-truth depth discontinuities.
- Edge-preservation metric.
- Inference latency.
- Peak GPU memory.

RGB edges must not be used as ground-truth depth boundaries.

---

## 10. Completion Checklist

### Rakshitha
- [ ] Deep model repository/checkpoint and license verified.
- [ ] Actual GPU inference tested.
- [ ] Dataset loaders and preprocessing completed.
- [ ] Original validity masks preserved.
- [ ] Common metrics and benchmark scripts completed.
- [ ] Deep branch results documented.

### Tanush
- [ ] Interpolation baselines completed.
- [ ] JBU and Guided Filtering completed.
- [ ] Hole and invalid-depth handling completed.
- [ ] Classical pipeline wrapper tested.
- [ ] Best classical configuration selected on validation data.
- [ ] Classical results documented.

### Rishita
- [ ] Fusion works with mock outputs.
- [ ] Fusion accepts real branch outputs.
- [ ] Confidence weights are normalized.
- [ ] Boundary refinement tested.
- [ ] Consistency correction used only with a known forward operator.
- [ ] Ablation results documented.

### Kumar
- [ ] Repository and branches created.
- [ ] UI works with mock outputs.
- [ ] All branch interfaces integrated.
- [ ] Input validation and exports tested.
- [ ] README and setup instructions completed.
- [ ] Final demonstration tested from a clean run.

### Final project
- [ ] Classical, deep and hybrid methods compared fairly.
- [ ] Same test samples and evaluation protocol used.
- [ ] Limitations and unsupported configurations documented.
- [ ] No unsupported claims about KITTI compatibility.
- [ ] No direct commits to `main`.
- [ ] Final end-to-end demonstration works.

---

## 11. Non-Negotiable Technical Rules

1. Generate and preserve the original validity mask before interpolation or hole filling.
2. Keep original depth data separate from branch-specific filled or filtered copies.
3. Do not assume SGNet's native input interface; verify it from the implementation.
4. Do not claim SGNet supports KITTI without testing and evidence.
5. Do not use test ground truth to tune confidence weights or classical parameters.
6. Do not claim exact measurement consistency when the forward degradation operator is unknown.
7. Do not add U-Net as a default fallback.
8. Do not add extra models or optional features until both primary branches work.
9. Do not let multiple members edit the same files in parallel.
10. Prioritize a working, evaluated hybrid system over feature count.

**Final execution principle:** Rakshitha and Tanush first produce two independently testable reconstruction branches. Rishita combines their verified outputs. Kumar integrates the stable interfaces into the application. The team advances phase by phase using the exit conditions above.
