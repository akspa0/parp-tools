# Plan — Epic 251 Viewer UX, Shell & Code Health

**Status**: Implementation approach **not yet selected** (operator directive 2026-09-23: triage first).

## Dependencies between backlog items

```text
U-10 sidebar standard (was 227-T004 inventory gate) ──> U-01 extraction ──> U-30 WoW shell
U-10 ──> U-11/U-13/U-14/U-15 dedupe work
U-18, U-21 independent small fixes
```

The 223 → 227 → 231 chain was three successive consolidation attempts, each superseding the last's
unfinished part; U-10–U-19 are its single merged residue.

## Design documents adopted by reference

| Item | Adopted design |
|---|---|
| U-01 | [228 plan](../archived/228-source-decomposition/plan.md) · [research](../archived/228-source-decomposition/research.md) |
| U-02 | [224 spec](../archived/224-speckit-governance/spec.md); rules live in [AGENTS.md](../../../AGENTS.md) §9 |
| U-03 | [213 spec](../archived/213-mcp-tooling-harness/spec.md) · [178 spec](../archived/178-mcp-automation-surface/spec.md) |
| U-10–U-12 | [227 plan](../archived/227-ui-reaudit/plan.md) · [inventory v2](../archived/227-ui-reaudit/surface-inventory-v2.md) (the §11 enforcement artifact) |
| U-13–U-16 | [223 plan](../archived/223-ui-consolidation-audit/plan.md) · [surface inventory](../archived/223-ui-consolidation-audit/surface-inventory.md) |
| U-17, U-18 | [231 plan](../archived/231-editor-archaeology-ui-overhaul/plan.md) |
| U-30 | [229 spec](../archived/229-wow-shell-keybind-profiles/spec.md) |
| U-31 | [225 spec](../archived/225-overhead-ortho-view/spec.md) |
| U-32 | [212 plan](../archived/212-spatial-ui-shell/plan.md) US6 |

## Standing constraints

AGENTS.md §10 (no new `WorldScene`/`ViewerApp` members, ~2,000-line budget, owned services) and §11
(`SharedUiWidgets`, one authoritative home per surface, inventory row per new surface).

## Approach — U-01 (operator P1, 2026-09-23; pending approval of the re-ordering)

One extraction per change, each independently buildable and revertible. Order by context removed:

| Step | Extract | From | Approx. lines out | Target | Operator smoke |
|---|---|---|---|---|---|
| E1 | PM4 overlay: load, tile objects, colouring, OBJ export, placement-match state | `WorldScene.cs` | ~5,500 | owned `Pm4OverlayScene` (+ smaller helpers, each <2,000) under `Terrain/Pm4/` | load a PM4 map; overlay, colours, selection, OBJ export unchanged |
| E2 | `Render()` → frame orchestrator + per-pass classes (terrain, WMO, MDX/doodad, liquid, transparent, overlays) | `WorldScene.cs` | ~1,800 | `Terrain/Passes/*` | fly a legacy map + a modern map; frame counters unchanged |
| E3 | `DrawMenuBar`, map/WMO converter dialogs, world-objects panel | `ViewerApp.cs` | ~2,150 | owned UI services under `Workbench/` | every menu item + both converter dialogs open and run |
| E4 | Selection / hover / pick (228's pure Core selection service first) | both | ~2,800 | Core selection service + viewer adapter | click/hover selection on terrain, WMO, doodad, PM4 |

After E1–E4 both files would sit near 8,000–9,000 lines. Remaining work continues the same way
(settings, terrain/tile management) as later steps; each needs its own operator-visible smoke.

Mechanics per step: move code verbatim first (no logic edits in the same change); replace the
god-class members with one field + delegating calls; build; run the full test suite; operator smoke.

**Spec-sync 2026-09-25 (E1 as built).** E1 moved ~8,850 lines (not ~5,500: the measured map missed
the PM4 structs/records declared before and after the class). `WorldScene.cs` 17,175 → 8,326.
Shape: `Pm4OverlayScene` is one class over three partial files by responsibility (state/load/cache,
selection, reports) so every moved body stayed verbatim; pure static code went to five separate
static classes; world-scene state is read only through `IPm4OverlayHost`. Callers use
`WorldScene.Pm4Overlay.X` rather than ~88 pass-through members on `WorldScene`. The PM4 draw blocks
inside `Render()` stay there (qualified) and move with E2. Receipt:
[evidence/u01-e1-pm4-extraction-2026-09-25.md](evidence/u01-e1-pm4-extraction-2026-09-25.md).
R-10 (Epic 249) lands **before** E2, so the performance fix is not entangled with a move of the WMO pass.

## ViewerApp extraction technique (U-01, 2026-09-25)

`ViewerApp` is one partial class whose fields are shared by every feature, and ImGui code passes many
of them by `ref`. Each extraction therefore:

1. Picks a feature cluster with a Roslyn member map (all partial files) plus a closure: members used
   only by the cluster move with it, including the fields that are its own state.
2. Moves the members verbatim into `internal sealed class <Feature>Service` under
   `Workbench/Services/<Feature>/`, namespace `WoWViewer` (same as `ViewerApp`, so every type name in
   moved code resolves exactly as before).
3. Reaches remaining app state only through `IViewerAppHost` (`Workbench/Services/IViewerAppHost.cs`),
   implemented explicitly by `ViewerApp`. Mutable fields are exposed as `ref`-returning properties, so
   `ImGui.Checkbox(..., ref _flag)` still works. The service re-declares each used member as a private
   bridge with its old name, so no moved body changes.
4. `ViewerApp` keeps one field per service, built in the `ViewerApp()` constructor (composition root);
   remaining references are rewritten to `_service.Member` by a syntax-aware pass.
5. Only declaration visibility changes (`private` → `internal`) where the split needs it.
6. Receipt per step: line-multiset audit (only intended lines differ), full-solution build,
   warnings and test failure set compared with the step's base.
