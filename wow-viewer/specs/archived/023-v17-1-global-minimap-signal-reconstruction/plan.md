# Implementation Plan: V17.1 Global Minimap-Signal Reconstruction Contract

**Branch**: `023-v17-1-global-minimap-signal-reconstruction` | **Date**: 2026-05-24 | **Spec**: `specs/023-v17-1-global-minimap-signal-reconstruction/spec.md`

## Summary

Ratify and implement a strict V17.1 workflow where normals training stays `minimap -> normals` with height as supervisor-only guidance, and MdxViewer precise-object capture is driven by curated manifest tiles instead of full-map brute force.

## Technical Context

**Language/Version**: Python 3.11+, C# .NET 10

**Primary Dependencies**: torch, numpy, pyarrow, zarr

**Storage**: `wow-viewer/output/datasets/v16/*.zarr`, curated manifest parquet/json, capture roots under `output/tmp/`

**Testing**: bounded manifest-stub generation, bounded renderer-truth patching, 1-epoch trainer contract sanity

**Target Platform**: Windows + CUDA for training, Windows for MdxViewer capture workflows

## Project Structure

```
wow-viewer/specs/023-v17-1-global-minimap-signal-reconstruction/
├── spec.md
├── plan.md
└── tasks.md

wow-viewer/data-harvester/scripts/
├── build_v16_dataset.py              # manifest-driven stub generation + patch orchestration
├── patch_v16_renderer_truth.py       # manifest-scoped patching + completion ledger
└── train_v16_1_common.py             # contract logging + supervisor-only preview clarity

wow-viewer/data-harvester/src/harvester/
└── v16_1_dataset.py                  # curation metadata and gating hooks
```

## Implementation Phases

### Phase 1: Manifest-Driven Capture Targeting

1. Add `--curation-manifest` support to `generate-viewer-stubs` so only manifest-selected tiles get JSON stubs.
2. Emit a manifest capture ledger (requested tiles + pending status) for downstream capture accounting.

### Phase 2: Manifest-Scoped Renderer-Truth Patching

3. Add `--curation-manifest` support to `patch-renderer-truth` orchestration and patch script.
4. Patch only manifest-targeted tiles and emit completion evidence: captured/partial/missing/skipped counts keyed to manifest scope.

### Phase 3: Normals-Training Contract Clarity

5. Rename misleading preview label `refined_gt` to an explicit teacher/refiner label.
6. Persist/print explicit normal contract fields so run intent is auditable.

### Phase 4: Curation/Mismatch Gate Tightening (Bounded)

7. Add bounded curation-threshold gates for V16.1 dataset loading using manifest metadata (terrain validity + minimap usefulness + whiteplate).
8. Surface selected/rejected counts in run evidence for reproducibility.

### Phase 5: Validation

9. Run bounded manifest-stub generation and verify reduced tile count against manifest.
10. Run bounded renderer-truth patch and verify manifest completion ledger.
11. Run 1-epoch training sanity and verify contract logs + clearer preview labeling.
