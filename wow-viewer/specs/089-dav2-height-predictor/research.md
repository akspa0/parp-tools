# Research: 089 — DA-V2-Small LoRA Height Predictor with Cross-Tile Consistency

**Phase 0 output. Companion to `plan.md`.**
**Date**: 2026-07-03.

This file resolves the planning unknowns that were still implicit in `spec.md` and `plan.md`: backbone choice, multi-channel adaptation strategy, cross-tile consistency strategy, determinism scope, V22 input contract details, and RunPod packaging boundaries.

---

## 1. Backbone and Fine-Tune Strategy

### Decision

Use the Hugging Face `depth-anything/Depth-Anything-V2-Small-hf` checkpoint as the canonical V23 encoder surface, loaded through `transformers.AutoModelForDepthEstimation`, with LoRA-r16 adapters on the transformer attention projections and a fully replaced first patch-embed conv for the V23 channel contract.

### Rationale

- The HF model card exposes a supported `transformers` path for `AutoImageProcessor` and `AutoModelForDepthEstimation`, which fits the repo's Python-first training surface and avoids a custom upstream repo fork.
- The model card reports **24.8M parameters**, which is the only DepthAnything-V2 size in the family that comfortably fits the spec's training/inference envelope after adding LoRA, a fresh decoder head, GPCT batching pressure, and CAI inference overhead.
- The model is already packaged as a DPT-style depth-estimation model with a DINOv2 backbone, so the V23 head can consume a conventional feature pyramid instead of reverse-engineering a new intermediate contract.
- Replacing the first patch-embed conv is the smallest explicit seam for adapting from RGB-only input to the V23 multi-channel tensor while keeping the rest of the pretrained encoder frozen.

### Alternatives Considered

- **DepthAnything-V2 Base/Large**: rejected for Phase 1 because the extra encoder size makes the 24 GB RunPod envelope tighter before any V23-specific head/loss costs are added.
- **Metric3D / ZoeDepth / MiDaS family**: rejected because the spec is already anchored to DepthAnything-V2's affine-invariant training behavior and existing HF packaging.
- **Train the entire encoder unfrozen**: rejected because LoRA is the bounded, reversible adaptation surface that fits the repo's one-model-one-signal discipline better than full-encoder fine-tuning.

---

## 2. Cross-Tile Consistency Strategy

### Decision

Use **PRO's Grouped Patch Consistency Training (GPCT)** during training and **PatchFusion's Consistency-Aware Inference (CAI)** during inference. They solve different halves of the seam problem and do not conflict.

### Rationale

- PRO specifically addresses tiled high-resolution depth prediction by jointly processing overlapping patches and penalizing disagreement in the overlap regions within one step.
- PatchFusion specifically addresses inference-time stitching by averaging overlapping shifted predictions instead of trusting a single independently-scaled tile pass.
- GPCT reduces the model's tendency to invent different local affine solutions per crop; CAI removes the residual boundary mismatch that remains after training.
- This split matches the user's constraint exactly: solve the seam structurally, not by seed selection.

### Alternatives Considered

- **CAI only**: rejected because it cleans up inference seams but leaves the model under-trained for overlap agreement.
- **GPCT only**: rejected because deterministic tile-wise inference can still show residual seams without an averaging/stitching pass.
- **Whole-map inference only**: rejected because it violates the 24 GB training envelope and the 6 GB inference target.

---

## 3. Loss Stack and Hallucination Control

### Decision

Keep the V23 loss stack at four primary terms plus one input regularizer:

1. affine-invariant `Lssi`
2. gradient-matching `Lgm`
3. Spatial Distance Constraint (SDC)
4. GPCT overlap-consistency
5. Bias-Free Masking as an input transform

Do **not** add spectral/fractal losses in V23 Phase 1.

### Rationale

- `Lssi` is the core scale/shift-invariant depth signal and is the right base loss for the "per-tile affine ambiguity" part of the problem.
- `Lgm` preserves slope and edge transitions in terrain without forcing a second output head.
- DepthAnything-AC's SDC is the smallest explicit geometric term that pushes patch-level relative structure without reopening the paused fractal-loss lane.
- Bias-Free Masking is a regularizer, not a second supervision target; it fits the repo's single-signal model rule.
- Keeping V23 free of spectral/fractal terms avoids re-opening Spec 068 inside a new plan. That work stays paused unless V23 underperforms after its own real-data proof.

### Alternatives Considered

- **Add Spec 068 spectral loss now**: rejected because it re-expands scope before the DA-V2 + GPCT + CAI route is validated.
- **Predict normals/liquid jointly to regularize height**: rejected because it breaks the one-model-one-signal rule.
- **Use only L1/L2 on metric height**: rejected because it does not address the affine ambiguity or hallucination modes the spec is explicitly trying to avoid.

---

## 4. V22 Input Contract and Cross-Build Tileset Identity

### Decision

Keep the default V23 input tensor at **15 channels** exactly as the spec defines, and use a **single union-top-K prune table per training run** across all selected V22 builds.

### Rationale

- A union-top-K table gives one stable checkpoint contract across `0_5_3_3368`, `3_3_5_12340`, and `4_0_0_11927`; that matters more than preserving per-build local ids.
- The V22 store already exposes build-wide `tilesets/tileset_paths` and `mcly_tileset_ids`, so deriving a union-top-K table is a bounded preprocessing step rather than a new dataset format.
- Keeping the channel contract fixed across builds makes checkpoint reuse, determinism proofs, and RunPod packaging simpler.
- The degraded input modes remain explicit opt-outs, not alternate defaults.

### Alternatives Considered

- **Per-build prune tables**: rejected because they complicate checkpoint portability and require extra runtime conditioning just to interpret the same one-hot channels.
- **Raw `mcly_texture_ids` without pruning**: rejected because they are tile-local and not stable asset identities.
- **Drop tileset channels entirely**: rejected because terrain texture identity is one of the few structured non-RGB cues V23 has without WDL priors.

---

## 5. Determinism Scope

### Decision

Define V23 determinism at three nested scopes:

1. **Per-call inference determinism**: identical input, checkpoint, CUDA arch, and software stack produce bit-identical output.
2. **Per-run training determinism**: identical config, checkpoint init, data order, CUDA arch, and software stack produce bit-identical weights.
3. **Cross-environment reproducibility**: only promised when the Pod image, PyTorch stack, CUDA arch, and checkpoint metadata all match.

### Rationale

- The spec's bit-identical promise is realistic only when the hardware/software envelope is pinned. The checkpoint metadata must therefore record commit SHA, image, library versions, seed, input mode, and data hashes.
- This aligns with the spec's actual assumption language: cross-arch reproducibility is a non-goal.
- It keeps the determinism claim strict enough to be testable without making impossible promises across mixed GPU families.

### Alternatives Considered

- **Promise cross-arch bitwise identity**: rejected as unrealistic and outside the spec's stated non-goals.
- **Treat inference as deterministic but training as only statistically reproducible**: rejected because FR-019 explicitly requires a bitwise re-run proof.

---

## 6. RunPod Packaging Boundary

### Decision

Treat V23 as a **Spec 079 consumer**, not a second RunPod integration owner. The V23 bundle carries only:

- Python source under `src/harvester/v23/`
- V23 scripts
- V23 tests and pod helpers
- a bounded V22 Zarr subset
- manifests / metadata / lockfiles

It must not carry staged clients, WoWArchive sources, or any path that resolves into `output/tmp/wowarchive-clients/`.

### Rationale

- Spec 079 already captures the Pod creation, transfer, and bootstrap patterns. Re-deriving them in 089 would create a second owner.
- V23's only project-specific bundle work is deciding which derived artifacts need to ship together for training.
- Keeping the bundle derived-data-only preserves the repo's BYOD policy and avoids turning V23 into a data distribution surface.

### Alternatives Considered

- **Let V23 own a custom Pod bootstrap flow**: rejected because Spec 079 already exists precisely to prevent that duplication.
- **Ship full staged clients for convenience**: rejected by policy and by the spec's own FR-016/SC-007 bundle constraints.

---

## 7. Operational Caveats Found During Planning

- The local Spec Kit PowerShell routing was stale: `.specify/feature.json` pointed at spec 056 while work had moved to 089. This is a workflow blocker, not a model-design blocker, and must be corrected in the same pass as the planning artifacts.
- The repo branch is `v0.5.0-prerelease`, not a Spec Kit feature branch. Planning must therefore route by explicit feature-directory pinning instead of branch-name discovery.
- The local V23 Phase 0 code slice is only source-applied. `uv sync`, import smoke, and pytest remain the gate before any Phase 1 implementation task can be considered active.

---

## 8. Primary Sources Used

- Hugging Face model card: `https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf`
- PatchFusion paper: `https://arxiv.org/abs/2312.02284`
- PRO paper: `https://arxiv.org/abs/2503.22351`
- DepthAnything-AC paper: `https://arxiv.org/abs/2507.01634`

---

## 9. Resolved Research Outcomes

| Topic | Decision |
|---|---|
| Backbone | HF `Depth-Anything-V2-Small-hf` via `transformers` |
| Multi-channel adaptation | Replace first patch-embed conv; freeze base encoder; LoRA on attention projections |
| Cross-tile consistency | GPCT at train time + CAI at inference time |
| Hallucination control | `Lssi + Lgm + SDC + GPCT`, with Bias-Free Masking; no spectral loss in V23 v1 |
| Tileset identity | One union-top-K prune table per training run |
| Determinism scope | Same CUDA arch + same software stack + recorded metadata |
| RunPod boundary | Reuse Spec 079; bundle derived data only |

*End of research. Next: `data-model.md`, `contracts/`, and `quickstart.md` are the Phase 1 planning outputs.*
