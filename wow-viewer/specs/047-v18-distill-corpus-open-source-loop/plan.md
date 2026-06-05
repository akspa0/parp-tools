# Implementation Plan: V18 Focused Two-Build Minimap-to-Terrain Loop

**Branch**: `047-v18-distill-corpus-open-source-loop` | **Date**: 2026-06-04 | **Spec**: [`wow-viewer/specs/047-v18-distill-corpus-open-source-loop/spec.md`](spec.md)

**Input**: Feature specification from [`wow-viewer/specs/047-v18-distill-corpus-open-source-loop/spec.md`](spec.md)

## Summary

The active plan is intentionally narrow:

1. keep the V18 corpus focused on `0_5_3_3368` and `3_3_5_12340`,
2. use only `minimap_rgb` as the model input,
3. train height with plain L1 supervision,
4. train normals with plain masked cosine supervision,
5. ignore renderer-truth, object-mask, roof-mask, and extra loss surfaces for
   active signoff.

This plan does not reopen distillation, open-source release, or capture-truth
work.

## Technical Context

- **Languages**: Python 3.11+ via `uv`; existing C# corpus tooling remains in
  place but is not the active proof owner.
- **Primary Surfaces**:
  - `wow-viewer/data-harvester/scripts/build_focused_two_build_corpus.py`
  - `wow-viewer/data-harvester/scripts/train_v16_1_height.py`
  - `wow-viewer/data-harvester/scripts/train_v16_1_normal.py`
  - `wow-viewer/data-harvester/scripts/train_v16_1_common.py`
  - `wow-viewer/data-harvester/src/harvester/v16_1_dataset.py`
- **Storage**:
  - `wow-viewer/output/datasets/v18/*.zarr`
  - `wow-viewer/models/v18/height/runs/`
  - `wow-viewer/models/v18/normal/runs/`
- **Trusted Clients**:
  - `output/tmp/wowarchive-clients/0_5_3_3368/`
  - `output/tmp/wowarchive-clients/3_3_5_12340/`

## Constitution Check

- **Repo independence**: pass. Active work stays under `wow-viewer/`.
- **Real-data validation**: required. Signoff uses focused stores plus recorded
  training evidence.
- **No untrusted client paths**: pass. Only staged clients under
  `output/tmp/wowarchive-clients/`.
- **One phase at a time**: pass. Focused stores first, then bounded height and
  normal training.
- **No architecture drift**: pass. Existing V16.1 / V18 models remain the
  implementation owner.
- **No speculative losses**: pass. Active losses are plain height L1 and plain
  normal cosine.

## Project Structure

```text
wow-viewer/
├── data-harvester/
│   ├── scripts/
│   │   ├── build_focused_two_build_corpus.py
│   │   ├── train_v16_1_height.py
│   │   ├── train_v16_1_normal.py
│   │   └── train_v16_1_common.py
│   └── src/harvester/
│       └── v16_1_dataset.py
├── docs/architecture/
│   └── v18-distill-corpus-open-source-loop-2026-06-04.md
├── output/datasets/
│   └── v18/
└── specs/047-v18-distill-corpus-open-source-loop/
    ├── spec.md
    ├── plan.md
    └── tasks.md
```

## Phase A1 - Focused Two-Build Corpus

**Goal**: keep the active corpus bounded to `0_5_3_3368` and `3_3_5_12340`.

1. Confirm the two staged client roots exist.
2. Run the focused build / validation path only for those two builds.
3. Record the exact store roots and validation outputs.

**Validation**:

- Two focused stores exist.
- Required arrays for minimap, height, and normals validate cleanly.

## Phase A2 - Simplified Height Surface

**Goal**: prove the height lane works from minimap only.

1. Keep the height input contract at `minimap_rgb -> height_257`.
2. Keep the height loss plain `L1(pred_height, target_height)`.
3. Record run config, seed, previews, and best checkpoint.

**Validation**:

- A bounded height run completes.
- Evidence shows the minimap-only input contract explicitly.

## Phase A3 - Simplified Normal Surface

**Goal**: prove the normal lane works from minimap only.

1. Keep the normal input contract at `minimap_rgb -> normal_xyz`.
2. Keep the normal loss to masked cosine against `normal_mask`.
3. Record run config, seed, previews, and best checkpoint.

**Validation**:

- A bounded normal run completes.
- Evidence shows the minimap-only input contract explicitly.

## Deferred Scope

These are not part of the active implementation plan:

- renderer-truth capture as training truth,
- object/roof/liquid-driven loss weighting,
- synthesized-input generation,
- teacher distillation,
- open-source student-model release.
