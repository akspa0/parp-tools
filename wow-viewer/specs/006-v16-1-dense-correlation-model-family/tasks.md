# Tasks: V16.1 Dense Correlation Model Family

**Plan**: `006-v16-1-dense-correlation-model-family/plan.md`

---

## Phase 1: Contract

- [x] **1.1** Create the V16.1 spec pack
  - add `spec.md`
  - add `plan.md`
  - add `tasks.md`
- [x] **1.2** Update continuity files
  - record V16.1 as the next terrain-model lane
  - record that V16 remains the baseline/reference path

## Phase 1A: Curation Layer

- [x] **1A.1** Create a separate reusable V16 curation layer
  - build manifests between Zarr stores and trainers
  - do not hide the rule set only inside one training loop
- [x] **1A.2** Add a normal-oriented curation profile
  - reject blank/low-signal tiles
  - check minimap-vs-normal edge agreement
- [x] **1A.3** Wire V16.1 trainer/dataset manifest consumption
  - `--curation-manifest`
  - filtered train/val dataset entry selection
- [x] **1A.4** Run focused proof
  - manifest builder writes kept/rejected outputs
  - normal smoke run completes through the curated tile set
- [x] **1A.5** Add multi-process curation execution
  - `--workers`
  - `--chunk-size`
  - visible chunk progress

## Phase 2: Normal First

- [x] **2.1** Extract V16.1-safe trainer helpers from `train_v16.py`
  - keep shared utilities only
  - no shared model/loss contract
- [x] **2.2** Create `data-harvester/scripts/train_v16_1_normal.py`
  - minimap -> normals only
  - include object/liquid-aware loss gating
  - dedicated checkpoints and validation artifacts
- [x] **2.3** Run 1-epoch CPU smoke for `3_3_5_12340`
  - verify run root exists
  - verify best/last checkpoints exist
- [x] **2.4** Preserve V16 runtime training seam
  - `torch.compile`
  - worker auto-resolution
  - prefetch / persistent workers
  - gradient accumulation
- [x] **2.5** Write optimized operator launch commands
  - curated normal-training command
  - VRAM fallback ladder

## Phase 3: Height Follow-On

- [x] **3.1** Create `data-harvester/scripts/train_v16_1_height.py`
  - minimap -> height only
- [x] **3.2** Run 1-epoch CPU smoke for `3_3_5_12340`
  - verify height-only validation panels
- [ ] **3.3** Write a short note on what the normal lane teaches the height lane

## Phase 4: Liquid Family

- [x] **4.1** Create `train_v16_1_liquid.py`
  - minimap -> liquid footprint / placement
  - minimap -> liquid type
- [x] **4.2** Define the initial liquid-type label contract
  - document class set
  - document fallback/degraded cases

## Phase 5: Texture Decomposition/Recomposition

- [x] **5.1** Audit existing D1 work for migration
  - `scripts/train_d1.py`
  - `src/harvester/d1_model.py`
  - `src/harvester/dataset.py`
  - record what survives into V16.1
- [x] **5.2** Create `train_v16_1_texcomp.py`
  - minimap -> `mcly_texture_ids`
  - minimap -> `alpha_256` / `mcly_layer_mask`
- [x] **5.3** Add recomposition validation output
  - predicted decomposition
  - recomposed terrain-only review image
- [x] **5.4** Migrate trainer inputs to V16 Zarr-quality signals
  - `alpha_256`
  - `mcly_texture_ids`
  - `mcly_layer_mask`
  - object-mask-derived loss gating

## Phase 6: Remaining Target Trainers

- [x] **6.1** Create `train_v16_1_holes.py`

## Phase 7: Shared Object Loss Gating

- [x] **7.1** Define shared object-mask gating utility/contract
- [x] **7.2** Apply gating to height, normal, liquid, and texture-decomposition trainers

## Phase 8: Stitched Inference

- [x] **8.1** Define checkpoint manifest / CLI contract
- [x] **8.2** Implement first stitched inference smoke path
