# Research: Source Decomposition — God-Class Split

**Date**: 2026-09-06
**Scope**: planning evidence only; no product source was changed.

## Baseline observations

| Area | Measured source | Result | Consequence |
|---|---|---:|---|
| `WorldScene` | `src/viewer/WoWViewer/Terrain/WorldScene.cs` | 16,915 lines | Feature ownership is not locally readable. |
| `ViewerApp` main declaration | `src/viewer/WoWViewer/ViewerApp.cs` | 16,740 lines | Bootstrap and feature ownership are conflated. |
| `ViewerApp` partials | 25 `ViewerApp_*.cs` files | 25,511 lines total | A partial-file split did not create a separate state boundary. |
| Largest partials | `ViewerApp_Sidebars.cs`, `ViewerApp_Pm4Utilities.cs`, `ViewerApp_CaptureAutomation.cs` | 6,135 / 4,450 / 2,220 lines | Do not extend these files; they are migration sources, not service owners. |
| Other present budget breaches | `VlmDatasetExporter.cs`, `WmoRenderer.cs`, `ModelRenderer.cs`, `WarcraftNetM2Adapter.cs`, `StandardTerrainAdapter.cs` | 3,131–5,364 lines | Record them as a later queue; do not fold them into the first extraction. |

## Decisions

### D1 — Extract cohesive ownership, never redistribute a partial class

An extraction moves the feature state, operations, and tests to an owned class. `WorldScene` or
`ViewerApp` retains only the smallest composition/delegation seam needed to call it. A new
`ViewerApp_*.cs` partial is never an acceptable result.

**Rationale**: partials share the same private state and force readers to reconstruct the full
class before making a safe change.

**Alternatives rejected**: split only by file (already failed); rewrite either god class in one
pass (not characterizable or reversible); and change UI while extracting (violates the
behavior-preserving constraint and blocks on Spec 227).

### D2 — Start with WorldScene selection only after the Spec 227 inventory gate

The first candidate is the hover/click ray-selection region currently rooted at
`WorldScene.cs` around `UpdateHoveredAssetInfo`, `TryPickSceneObjectByRay`, and their helper
methods. It is a cohesive algorithmic area with existing public entry points, but its Inspector
presentation and canonical UI home remain owned by Spec 227. The first viewer-route implementation
task is blocked until Spec 227 T004 records that gate.

**Rationale**: it reduces `WorldScene` without choosing or moving a UI surface.

**Alternatives deferred**: capture automation (after Spec 223's live capture retest and the Spec
227 UI-home decision); Sidebars, Editor, and PM4 utilities (all current UI-authority candidates).

### D3 — The selection service receives snapshots, never `WorldScene`

The extracted service receives explicit input records and returns a selection result. The viewer
adapter constructs inputs from currently resident scene data and applies the returned selection.
The service must not retain or reference `WorldScene`, OpenGL objects, ImGui state, or a parent
delegate.

### D4 — A budget is enforced incrementally, with an honest legacy ledger

Every new owned source file is kept below approximately 2,000 lines. No extraction may increase
any existing over-budget file. Existing breaches are recorded in this plan and become separately
scheduled migration sources; they are not falsely described as meeting the new budget already.

### D5 — Each extraction has three separate proof layers

1. A before/after line-count and dependency receipt;
2. focused automated tests plus solution build/test; and
3. an operator smoke of the moved surface when the feature has an interactive route.

Compilation and unit tests prove only source contracts, not rendering, picking, input, capture, or
performance behavior.

## Discovery commands and results

| Command | Exit | Result |
|---|---:|---|
| `(Get-Content src/viewer/WoWViewer/Terrain/WorldScene.cs).Count` | 0 | `16915` |
| `(Get-Content src/viewer/WoWViewer/ViewerApp.cs).Count` | 0 | `16740` |
| `(Get-ChildItem src/viewer/WoWViewer -Filter 'ViewerApp_*.cs' \| ForEach-Object { (Get-Content $_.FullName).Count } \| Measure-Object -Sum).Sum` | 0 | `25511` |
| `rg -n --glob 'WorldScene.cs' "UpdateHoveredAssetInfo|TryPickSceneObjectByRay|CollectSceneObjectPickHits" src/viewer/WoWViewer/Terrain` | 0 | Located the first selection candidate and its current composition methods. |
