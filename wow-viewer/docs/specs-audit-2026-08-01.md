# Spec Audit — 2026-08-01 (Finalized Backlog)

**Decision**: The existing code, the viewer, and the v50 dataset are **feature-complete** as of
the synthesized-minimap lighting fixes (v0.5.2). This audit finalizes the backlog to current
reality: most of the historical draft specs (V10–V25 model eras, fractal/brush lanes, audio,
ONNX, RunPod, UE bridge, UI split micro-specs) are **archived**. Only the specs that describe
work still on the active viewer/v50 path remain.

## Direction

**Lift off from current code.** The existing code, viewer, and v50 dataset are the base; new
speckit plans are written fresh from what exists today. The specs below are the only ones that
describe genuinely in-flight work worth keeping. Everything else is historical.

## KEEP — current in-flight (viewer + v50 + format work)

| Spec | Why it stays |
|------|--------------|
| 009-full-project-reimplementation-spec | Master design reference (2,650-line); never archive |
| 046-pm4-asset-matching | Active — PM4 identity matching lane |
| 065-pm4-correlation-to-world-assets | Active — PM4 surface correlation |
| 069-viewer-ui-overhaul | Active — viewer shell (Phase 15) |
| 080-wow-ui-consolidation | Active — UI consolidation target |
| 104-legacy-m2-rendering | Active — 1.0.0–3.0.0 M2 gap (known issue) |
| 105-format-version-profiles | Needed by the M2 version-dispatch work |
| 106-native-daynight-lighting | Viewer lighting fidelity |
| 107-lighting-quick-inspection | Active — lighting controls |
| 108-image-wdl-prior | Active — v50 WDL prior lane |
| 109-v50-clean-room-audit | Active — dataset provenance audit |
| 110-viewer-stabilization | Viewer stability surface |
| 111-minimap-lighting-calibration | v50 synthetic minimap calibration |
| 112-v50-height-model | v50 height-first model |
| 114-direct-terrain-reconstruction | v50 direct minimap→terrain |
| 115-terrain-feature-classifier | v50 deconfounding |
| 117-wdl-lattice-prior | v50 coarse prior |
| 118-object-occlusion-masks | v50 object deconfounding |
| 123-real-wdl-detailer | v50 real WDL prior + detailer |
| 124-legacy-detangle-runpod | v50 legacy Python detangle + RunPod tooling |

## ARCHIVE — old/irrelevant backlog (not current reality)

### V10–V25 model-era specs (superseded by the v50 pipeline)

066-v19-height-regressor · 067-v20-multimodal-terrain-intent · 068-fractal-aware-height-loss ·
068-onnx-feasibility · 074-alpha-brush-library · 075-scar-mask-segmentation ·
076-full-map-fractal-brush-library · 086-v22-consolidated-dataset · 087-v22-asset-library-payloads ·
088-v22-enrichment-from-v18 · 089-dav2-height-predictor · 092-heightmap-pattern-miner ·
094-wdl-prior-v24 · 095-learned-minimap-cleaner · 096-v24-minimap-deploy · 097-v18-to-wdl-adt ·
098-v24-lattice-reconstruction · 099-stage-a-full-retrain · 100-patchgan-wdl-discriminator ·
101-v241-dav2-model · 102-v25-terrain-convergence · 103-image-only-reconstruction ·
113-minimap-superres

### Format/archive/dataset side-lanes (done or dead)

001-v18-dataset-spec · 012-real-validation-batch-extraction · 014-terrain-mcal-rendering-parity ·
024-v18-canvas-paste-refinement-layer · 025-object-roof-mask-library-and-minimap-sieve ·
029-wmo-minimap-signal · 042-zarr-first-mpq-fallback-data-source · 044-viewer-shell-usability ·
049-viewer-ui-consolidation · 060-ui-cleanup-and-migration-notes · 061-weak-signal-terrain-restoration ·
062-weak-signal-tile-patcher · 063-pm4-collision-algorithm · 064-blank-map-generation ·
091-raw-audio-unswizzle · 093-render-performance-liquid-audit · 116-relational-terrain-layers ·
121-v7-wdl-height · 122-dataset-curation

### Research consumed elsewhere (archive as evidence)

030-wmo-render-pass-architecture · 031-terrain-cell-awareness · 032-native-renderer-parity ·
038-m2-301-renderer-perf-research · 040-mh2o-mclq-liquid-type-determination

### UI/micro-specs folded into 069/080 (archive)

070-map-workbench-window · 071-left-right-sidebar-split · 073a-toolbar-leftsidebar-dedup ·
073b-tools-tab-converters · 077-ui-fix-and-bar-layout · 078-m2-runtime-animation-diagnosis ·
079-runpod-integration-guide · 090-viewer-memory-profiler · 055-unreal-engine-bridge ·
057-client-archive-version-selector

### Legacy/one-off (archive)

001-precise-m2-masks · 045-scene-graph-workbench · 053-m2-animation-pose-farm ·
054-pm4-camera-window-cache · 058-pm4-scene-graph-semantics-and-panel · 077-minimap-deconstruction-engine

## Net result

- **KEEP**: 20 specs (the active viewer/v50 surface + master reference)
- **ARCHIVE**: ~52 specs (old model eras, dead lanes, folded UI micro-specs)
- **plans/**: both `wow-viewer/plans/` (roof-capture historical) and repo-root `plans/`
  (`wmo-overlay-synthesized-minimap-plan.md`) stay as-is — small and accurate.

## Execution

```powershell
cd i:\parp\parp-tools\wow-viewer\specs
# Archive the ~52 specs above. Example (edit list as needed):
$toArchive = @(
  "001-precise-m2-masks","001-v18-dataset-spec","012-real-validation-batch-extraction",
  "014-terrain-mcal-rendering-parity","024-v18-canvas-paste-refinement-layer",
  "025-object-roof-mask-library-and-minimap-sieve","029-wmo-minimap-signal",
  "030-wmo-render-pass-architecture","031-terrain-cell-awareness","032-native-renderer-parity",
  "038-m2-301-renderer-perf-research","040-mh2o-mclq-liquid-type-determination",
  "042-zarr-first-mpq-fallback-data-source","044-viewer-shell-usability",
  "045-scene-graph-workbench","049-viewer-ui-consolidation","053-m2-animation-pose-farm",
  "054-pm4-camera-window-cache","055-unreal-engine-bridge","057-client-archive-version-selector",
  "058-pm4-scene-graph-semantics-and-panel","060-ui-cleanup-and-migration-notes",
  "061-weak-signal-terrain-restoration","062-weak-signal-tile-patcher","063-pm4-collision-algorithm",
  "064-blank-map-generation","066-v19-height-regressor","067-v20-multimodal-terrain-intent",
  "068-fractal-aware-height-loss","068-onnx-feasibility","070-map-workbench-window",
  "071-left-right-sidebar-split","073a-toolbar-leftsidebar-dedup","073b-tools-tab-converters",
  "074-alpha-brush-library","075-scar-mask-segmentation","076-full-map-fractal-brush-library",
  "077-minimap-deconstruction-engine","077-ui-fix-and-bar-layout","078-m2-runtime-animation-diagnosis",
  "079-runpod-integration-guide","086-v22-consolidated-dataset","087-v22-asset-library-payloads",
  "088-v22-enrichment-from-v18","089-dav2-height-predictor","090-viewer-memory-profiler",
  "091-raw-audio-unswizzle","092-heightmap-pattern-miner","093-render-performance-liquid-audit",
  "094-wdl-prior-v24","095-learned-minimap-cleaner","096-v24-minimap-deploy","097-v18-to-wdl-adt",
  "098-v24-lattice-reconstruction","099-stage-a-full-retrain","100-patchgan-wdl-discriminator",
  "101-v241-dav2-model","102-v25-terrain-convergence","103-image-only-reconstruction",
  "113-minimap-superres","116-relational-terrain-layers","121-v7-wdl-height","122-dataset-curation"
)
foreach ($s in $toArchive) { if (Test-Path $s) { git mv "$s" "archived/$s" } }
```

After the moves, `specs/` contains only the 20 KEEP specs plus `archived/`. Update
`specs/archived/ARCHIVED.md` with a one-line reason per newly archived spec.
