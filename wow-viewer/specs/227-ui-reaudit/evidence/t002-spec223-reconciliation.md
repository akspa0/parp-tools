# T002 Receipt — Spec 223 Disposition Reconciliation

**Date**: 2026-09-06
**Task**: `T002` in [tasks.md](../tasks.md)
**Scope**: documentation-only comparison of Spec 223 inventory v1 to present source routes and
shared content bodies. No viewer source, data authority, or runtime behavior changed.

## Files changed

| File | Change |
|---|---|
| `specs/227-ui-reaudit/surface-inventory-v2.md` | Added section E, recording every v1 retire/merge/keep/move disposition and its current source state. |
| `specs/227-ui-reaudit/evidence/t002-spec223-reconciliation.md` | This receipt. |
| `specs/227-ui-reaudit/tasks.md` | Recorded the T002 receipt. |
| `memory-bank/activeContext.md` | Advanced the dashboard to the operator screenshot matrix handoff. |
| `memory-bank/progress.md` | Updated the current Spec 227 handoff. |

## Verification

| Command | Exit status | Result |
|---|---:|---|
| `(Get-Content wow-viewer/specs/223-ui-consolidation-audit/surface-inventory.md).Count; Get-Content wow-viewer/specs/223-ui-consolidation-audit/surface-inventory.md \| Select-Object -Skip 110 -First 260` | 0 | Confirmed the complete v1 inventory is 112 lines and reviewed its final operator decisions. |
| `rg -n -g "*.cs" "Draw(WorldOverview|MapDiscovery|PhaseLayer|ChunkClipboard|ModelInfo|Camera|Mcnk|AdtChunk|TerrainChunkHover|TerrainAnalysis|SynthesizedMinimap|CameraPath|CaptureAutomation|Perf|RenderQuality|Pm4|LogViewer|MlTraining|WdlPreview|MapPreview|SceneHoverAsset|UniqueIdArchaeology|WeakSignal)" wow-viewer/src/viewer/WoWViewer` | 0 | Located current source hosts and delegates for the v1 disposition families. |
| `rg -n -g "*.cs" "GetBottomTabLabels|GetArchaeologyWorkbenchLabels|DrawMinimapWindow|DrawUtilitiesMinimap|DrawUnifiedInspectorContent|DrawTerrainControlsAdjustmentWeakSignalContent|DrawTerrainLabSubTab" wow-viewer/src/viewer/WoWViewer` | 0 | Confirmed the current visible workbench roots and duplicate-family anchors. |

## Criterion-to-evidence

| Task criterion | Evidence | Result |
|---|---|---|
| Compare every Spec 223 retire/merge disposition against current source. | [inventory v2 section E](../surface-inventory-v2.md#e-spec-223-retiremerge-disposition-reconciliation) enumerates all v1 disposition rows, including legacy exceptions and later operator decisions. | Met for source comparison. |
| Record unresolved rows rather than treating old intent as completion. | Section E marks unresolved target decisions, blocked Cartography parity, and pending visual/input checks. | Met. |
| Establish a safe source-edit target. | The full inventory still lacks screenshots and interaction evidence. | **Not met by design**; T003 and T004 remain open, so source consolidation is still blocked. |
