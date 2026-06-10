# V9 Native-Signal Training Plan

## Intent

- keep the active `v7.7` run only as the last major proof on the old raster-heavy contract
- pivot the next real architecture branch to `v9` immediately
- treat `v9` as the first branch whose training contract is grounded in native terrain data instead of exported image interpretations of that data
- reduce `v7.8` to optional bounded sanity-check work only, not the main execution lane

## Why Pivot To V9 Now

- `v7.7` already changed several controlling variables at once: unclamped train-time heads, higher residual amplitude, delayed bounds pressure, detail head, brush-aware sampling, corrected spatial augmentation, and earlier GAN experiments
- the repeat floor and repeated late-stage saturation suggest the current contract is the ceiling, not just the optimizer or scheduler
- the current trainer already proves the immediate bottleneck is not just tuning; it is a representation and supervision problem
- the old `v7.x` assumption was too image-centric:
  - the real terrain truth is native ADT/WDL data
  - minimaps are one useful conditioning signal, not the canonical source of terrain truth
  - rasterizing the truth first and then training the model against those rasterizations adds avoidable distortion
- if the contract is wrong, continuing to optimize inside that contract is the wrong main investment

## Current Diagnosis

- the native ADT terrain lattice is `257x257` per tile, while the current trainer supervises `512x512` rasters derived from exported PNG heightmaps
- WDL prior is much coarser (`17x17` outer grid) and currently enters as an upsampled image-like prior
- the trainer mixes real signals and derived helper signals aggressively; some of those helpers may help, while others may be broad enough to become anti-signal
- the repeat plateau around roughly `0.05` suggests a structural ceiling rather than a one-off scheduler mistake
- likely ceiling contributors:
  - target-resolution mismatch between native terrain and rasterized supervision
  - minimap ambiguity that no amount of plain image loss can resolve
  - loss composition that rewards smooth agreement more than structural correctness late in training
  - mask breadth and noisy auxiliary context
  - treating helper images as if they were more native than the terrain data they were derived from

## V9 Scope

### Goal

- stop treating terrain primarily as exported images plus helper PNGs
- move to a native-signal sample contract built from terrain JSON/native terrain fields and cached tensor artifacts
- make the model consume the original terrain truth directly, with minimaps and other visual surfaces acting as guidance rather than canonical targets

### Naming Recommendation

- reserve `v9` for this branch
- do not keep the next real architecture under the `v7.x` naming line, because the contract itself is changing rather than just the model

### V9 Data Contract

- source from harvested tile JSON and native terrain fields first
- retain archive-backed extraction in the exporter/harvester layer, not inside the training loop
- convert native terrain into cached tensors or shard files before training

- target sample contents:
  - native ADT height lattice at `257x257`
  - native local-detail residual target derived from that lattice
  - native WDL grid at `17x17`
  - optional coarse intermediate terrain targets such as `129x129`, `65x65`, or pooled forms derived from the ADT lattice
  - masks and context from terrain/object/liquid metadata as arrays, not as ad hoc image assets where avoidable
  - minimap remains an image input because that signal is genuinely visual
  - normals can be recomputed from height where that is more trustworthy than reading an exported PNG

### V9 Pipeline Architecture

- preferred ownership split:
  1. native terrain JSON reader layer
  2. sample harvester or tensor-cache builder layer
  3. cached tensor dataset layer
  4. training layer

- do not parse MPQ/CASC/loose client assets directly in `Dataset.__getitem__`
- do not regenerate heavy derived signals during every epoch
- do not make the training loop responsible for mount/staging/archive resolution logic
- use the already-harvested dataset JSON as the immediate truth surface so `v9` can start now without reopening exporter requirements

### V9 Model Direction

- move away from one raster-only output head
- preferred direction:
  - coarse global terrain branch supervised directly against native WDL/pooled ADT targets
  - mid-resolution terrain branch supervised against pooled ADT terrain
  - high-resolution residual/detail branch supervised against native ADT residuals
  - minimap and optional normal/albedo encoders as visual guidance, not as the only source of truth

- optional architecture work for `v9`:
  - explicit cross-scale fusion instead of one-head-per-task competition
  - low-cost attention only where cross-scale routing matters
  - tile-edge or neighbor-context handling based on native terrain adjacency rather than only raster border losses

### V9 Native-Signal Priorities

- first ownership slices:
  1. define a cached tensor schema from the existing harvested JSON tiles
  2. build a converter that emits native arrays from current harvested tiles without archive reads in the training loop
  3. add a `v9` trainer path that consumes cached tensors and supervises `257` plus `17` directly
  4. add multi-resolution residual supervision from the same native sample, not from exported helper PNGs
  5. only after those land, consider archive-refresh tooling for broader corpus regeneration

### V9 Performance And Ada Optimization Work

- keep what is already helping on Ada:
  - CUDA
  - bfloat16 AMP
  - TF32
  - cuDNN benchmark

- add or test next once the `v9` path is runnable:
  1. `torch.compile` for the model forward and loss path if the current PyTorch build is stable on this environment
  2. channels-last memory format for the convolution-heavy parts of the model
  3. fused `AdamW` if supported by the installed PyTorch/CUDA stack
  4. persistent worker and pinned-memory tuning after the tensor-cache contract is stable

- what is probably not the first lever:
  - FlashAttention2: useful only if we add substantial attention modules with real attention tensors; not a generic speedup for a conv-heavy model
  - Triton custom kernels: only worth it after profiling shows a concrete hotspot PyTorch/cuDNN is not already handling well
  - `accelerate`: useful for orchestration and multi-device management, not a direct single-GPU performance cure

### V9 Validation Gates

- require each `v9` slice to answer one narrow structural question
- success gates:
  1. the model is supervised primarily against native terrain truth rather than PNG derivatives
  2. coarse branch follows WDL and pooled ADT structure more faithfully than `v7.7`
  3. high-resolution residual branch improves preview structure without collapsing into the same smooth floor
  4. the tensor-cache pipeline is deterministic and cheap enough for repeated training runs

## Optional V7.8 Work

- `v7.8` is no longer the main branch
- if used at all, it should answer only bounded legacy questions while `v9` is being implemented
- allowed `v7.8` tasks:
  1. signal ablations over the existing input contract
  2. cheap Ada profiling such as `torch.compile`, channels-last, and fused optimizer checks
  3. one lightweight attention experiment behind a flag if profiling suggests a routing bottleneck

- `v7.8` should not become the new main architecture lane
- do not spend weeks polishing the old raster-heavy contract if `v9` is already in motion

## Immediate Build Order

### Track A - While V7.7 Trains

1. define the `v9` tensor shard schema from the existing dataset JSON
2. build the converter that emits native arrays from current harvested tiles
3. define the base `v9` model contract around `257` and `17`
4. keep `v7.7` running only as a comparison baseline

### Track B - First Runnable V9 Slice

1. add a `v9` dataset loader that reads cached tensors instead of PNG-backed dataset assets
2. add a `v9` trainer path with coarse plus mid plus residual supervision
3. wire minimap as visual guidance rather than primary truth source
4. launch the first reduced-data `v9` pilot against the same held-out tiles used for `v7.7`

### Track C - After First Runnable V9 Proof

1. profile Ada-specific performance and add `torch.compile` or channels-last or fused optimizer where stable
2. decide whether any attention block earns its cost based on cross-scale fusion needs
3. only then revisit archive-refresh tooling for broader corpus regeneration

## Decision Gates

- stay on `v9` as the main line unless one of these turns out false:
  1. existing harvested JSON lacks the native terrain truth needed to emit reliable tensor shards
  2. native `257` plus `17` supervision still reproduces the same floor with no structural gain
  3. the actual bottleneck turns out to be purely runtime performance, not representation

## Recommendation

- `v9` is now the main execution lane
- `v7.7` is the last major run on the old image-heavy contract
- `v7.8` is optional and bounded only
- do not put direct archive reads inside the live trainer; put them in a harvester or tensor-cache builder that can stage/archive data deterministically
- start implementing `v9` now from the already-harvested dataset JSON, because that is the shortest path to a model grounded in the original terrain truth instead of interpolated visual interpretations of it