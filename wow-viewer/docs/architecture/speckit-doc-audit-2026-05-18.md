# Spec Kit Documentation Audit — 2026-05-18

## Scope

All architecture/spec documents under `wow-viewer/docs/architecture/` plus subdirectories. Classified against current code in `wow-viewer/src/`, `wow-viewer/tools/`, `wow-viewer/data-harvester/scripts/`, and `wow-viewer/data-harvester/src/`.

## Audit Table

| doc_path | status | evidence | action |
|----------|--------|----------|--------|
| `alpha-mcnk-flags-and-metadata-plan.md` | partial | Height bug fixed. `AlphaWdtReader.cs` and `AlphaWdtWriter.cs` exist. `AlphaMcnkHeader` record type NOT standalone. Phases 2-4 not done. | Keep open. Create `AlphaMcnkHeader` record type for Phase 1. |
| `alpha-placement-coordinate-transforms-2026-05-09.md` | implemented | Research doc verified via Ghidra. Transforms implemented in `AlphaWdtReader`, `AlphaWdtWriter`, `AlphaToLkConverter`, `LkToAlphaConverter`. | Archive as reference. |
| `alpha-wdt-ghidra-research-2026-05-10.md` | implemented | Research-only. Findings reflected in existing reader/writer behavior. All referenced files exist. | Archive as reference. |
| `audio-engine-plan-2026-04-21.md` | partial | `AlphaAreaAudioCatalog*.cs` files exist (4 files). MCSE reader, MIDI/DLS, audio runtime, audio backend missing. No `Audio` namespace in `Core.Runtime`. | Keep open. Next: MCSE reader + area-audio inspect proof. |
| `game-viewer-host-plan-2026-05-13.md` | partial | `WowViewer.App` has 35+ files. `Core.Runtime` has World/ and M2/. Terrain shaders, full session, game manager shell not done. | Keep open as sub-plan. |
| `low-resolution-world-image-alignment-plan-2026-04-22.md` | planned | No code. Pure design doc. No `world-image-align` or `world-image-cut-tiles` commands exist. | Keep open. Low priority. |
| `m2-native-client-research-2026-03-31.md` | implemented | Raw Ghidra evidence. Findings consumed by `m2/implementation-contract.md`. `WowViewer.Core.M2/` (11 files), `Core.IO.M2/` (6 files), `Core.Runtime.M2/` (26 files) all exist. | Archive as reference. |
| `multi-model-terrain-reconstruction-2026-05-16.md` | planned | Draft spec. Depends on object segmentation model (does NOT exist). Four-model chain not implemented. | Keep open. Blocked on object segmentation. |
| `pm4-llm-evidence-export-2026-05-21.md` | implemented | `MdxViewer` PM4 workbench now exposes selected-region peer summaries and writes JSON/Markdown/SVG evidence bundles from the visible overlay. | Keep as active compatibility note. |
| `v10-stage2-terrain-synth-architecture-2026-04-27.md` | stale | Superseded by V14/V15/V16. `train_v10_stage2_terrain_synth.py` exists but is legacy. | Mark historical. Point to V16. |
| `v11-terrain-model-architecture-2026-05-02.md` | stale | Superseded by V14/V15/V16. `train_v11.py` and `infer_v11.py` exist but are legacy. | Mark historical. Point to V16. |
| `v14-model-and-refactor-plan-2026-05-06.md` | partial | Core philosophy doc. Gaps it identified (synthetic minimap compositor, Harvester CLI, data-harvester/) ALL NOW EXIST. Code audit tables stale. Full V14 multi-model pipeline not fully landed; V16 is active. | Update code audit tables. Keep as architectural reference. |
| `v15-terrain-model-spec-2026-05-15.md` | implemented | All referenced code exists: `train_v15.py`, `v15_model.py`, `v15_dataset.py`. Architecture matches landed code. | Keep as active spec. Superseded by V16 for production. |
| `v16-harvest-recovery-plan-2026-05-17.md` | partial | `build_v16_dataset.py` and `backfill_v16_resume_state.py` exist. Phase 1-2 completion unclear from doc alone. Phase 3 is config change. | Keep open. Validate Phase 1-2 landing. |
| `v16-terrain-model-spec-2026-05-16.md` | implemented | All referenced code exists: `build_v16_dataset.py`, `v16_model.py`, `v16_dataset.py`, `train_v16.py`, `infer_v16.py`, etc. Pipeline functional. | Keep as active production spec. |
| `viewer-legacy-cutover-boundary-2026-04-17.md` | implemented | `WowViewer.App` exists (35+ files). Boundary enforced by AGENTS.md. MdxViewer treated as compatibility-only. | Keep as active policy doc. |
| `wdt-format-notes-2026-04-17.md` | implemented | All 7 referenced files exist. Code matches doc description. | Archive as reference. |
| `wmo-m2-mdx-converter-port-plan-2026-05-11.md` | partial | WMO converters landed: `WmoV17ToV14Converter.cs`, `WmoV14ToV17Converter.cs` with tests. M2<->MDX landed: `M2ToMdxConverter.cs`, `MdxToM2Converter.cs` with tests. WMO has real-data proof (311 WMOs, 0 failures). M2<->MDX structural only. | Keep open. Next: broader M2 animation/material coverage. |
| `wow-engine-editor-and-interop-plan-2026-05-14.md` | planned | Pure design doc. Import/export shell, DBC editor, Museums profile — none exist. | Keep open as long-range direction. |
| `wow-engine-modernization-plan-2026-05-14.md` | partial | Core/IO/Runtime/App all exist. OpenGL backend works. Vulkan, WebGL, Museums, engine-neutral BASE layer not implemented. | Keep open as top-level direction. |
| `wow-viewer-full-porting-roadmap.md` | partial | Phase A (terrain types) DONE. Phase B (harvest pipeline) DONE. Phase C (converters) partially landed. Phase D (renderers) and Phase E (viewer) mostly not done. | Keep open as master tracker. |
| `wow-viewer-library-completeness-plan-2026-05-06.md` | partial | Many format readers done. Many renderers still missing (MdxRenderer, TerrainRenderer, WmoRenderer, LiquidRenderer). Converters partially done. | Keep open as gap tracker. |

## Subdirectories

| doc_path | status | evidence | action |
|----------|--------|----------|--------|
| `ab-session-a-2026-04-01/` | implemented | M2 investigation session. Findings consumed by `m2/` and landed M2 runtime code. | Archive as historical evidence. |
| `game-viewer-plan-pack-2026-05-14/` | planned | 49 micro-plan files (GV-00 through GV-26). None have direct implementation yet. | Keep open as execution skeleton. |
| `m2/` | implemented | Canonical M2 docs. All code targets exist: 43 files across Core.M2, Core.IO.M2, Core.Runtime.M2. | Keep as active implementation reference. |

## Summary

| Classification | Count |
|---------------|-------|
| implemented | 10 |
| partial | 11 |
| planned | 4 |
| stale | 2 |
| **Total** | **27** |

## Key Findings

1. **Two docs stale**: `v10-stage2-terrain-synth-architecture-2026-04-27.md` and `v11-terrain-model-architecture-2026-05-02.md` — superseded by V16.
2. **One doc has stale code audit tables**: `v14-model-and-refactor-plan-2026-05-06.md` lists gaps that are now closed.
3. **Audio furthest behind**: Only 4 audio files exist. MCSE reader, MIDI, audio runtime missing.
4. **Vulkan/WebGL do not exist**: Only OpenGL (Silk.NET) implemented.
5. **Game-viewer-plan-pack is largest planned surface**: 49 micro-plans with zero implementation.
6. **M2 subsystem well-implemented**: 43 files matching implementation contract.
7. **PM4 evidence export now has a dedicated note**: the selected-region / JSON+Markdown+SVG compatibility surface is documented instead of living only in viewer code.
8. **V16 pipeline is most current ML path**: All scripts and model files form a coherent pipeline.

## Highest-Value Active Lane

**`v16-inference-output-contract-and-patch-pipeline`** — the inference→patch pipeline has partial implementation (`infer_v16.py` + `terrain-patch-adt`) but the v16 spec identifies three missing ergonomic wrappers:

1. Direct `.pred.zarr` consumption without per-tile summary staging
2. One-shot "infer + patch + optional alpha conversion" pipeline command
3. Direct ADT liquid chunk patching (`MH2O`/`MCLQ` write integration)

This lane directly unblocks the terrain-model training→inference→ADT-patching loop and has the highest ROI for immediate Spec Kit investment.

## Commands Used

```powershell
# Build verification
dotnet build wow-viewer/WowViewer.slnx -c Debug

# File existence checks via glob/grep
# Checked: AlphaWdtReader.cs, AlphaWdtWriter.cs, infer_v16.py, train_v16.py,
#          v16_model.py, v16_dataset.py, TerrainPatchAdtCommand.cs,
#          WowViewer.Tool.Harvest, WowViewer.App (35+ files),
#          WowViewer.Core.M2/ (11 files), Core.IO.M2/ (6 files),
#          Core.Runtime.M2/ (26 files)
```
