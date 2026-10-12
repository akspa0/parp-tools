# Spec 220: Plan — WMO Doodad Placement Editing, Custom Doodad Sets & WMO Writing

## Architecture

Editing composes on the **existing V17 read model** and the **existing V14 writer**; the viewer is a thin picking/gizmo layer.

```
Viewer (WoWViewer)                     Core.Editor                        Core.IO
──────────────────                     ───────────                        ────────
Spec 211 picking ──► WmoDoodadEditor ──► WmoDoodadEditOperations ──► WmoV14ToV17Converter.WmoV14Data
gizmo overlay UI      (session state,      (move/rotate/scale/          (single editable model,
                       undo stack)          add/delete/duplicate,         no forks)
                                            set authoring)                      │
                                                                                ▼
                                            WmoDoodadSetAuthoring ──► WmoV17ToV14Converter.WriteWmoV14
                                                                          (V14 save, first target)
```

### Key decisions
1. **Single source of truth**: `WmoV14Data.DoodadDefs` / `DoodadSets` / `DoodadNamesRaw` are mutated in place by Core.Editor operations. `WmoRenderer` re-derives `_doodadInstances` from the model via the existing `LoadActiveDoodadSet` path (exposed as a public "reload doodads" hook) — no second placement list.
2. **MODN name table**: additions reuse an existing entry when the model path already exists (binary search over the null-separated blob); only new names append. Name index is a byte offset — preserved exactly.
3. **MODS ranges**: new sets append a MODS record whose range is computed from the def indices assigned to it. When assignments are non-contiguous the operation fails with a named reason unless the set is flagged to allow it (matches how the reader model represents ranges today — ranges are `(StartIndex, Count)`).
4. **Save path**: `WmoV17ToV14Converter` already round-trips V17→V14 including doodad chunks. Phase 1 verifies that claim with the US5 round-trip suite **before** any editing UI exists, so the writer's correctness is proven on unmodified data first.
5. **Undo**: each operation is a self-describing command (before/after snapshots of the affected MODD/MODS records only, not the whole model) pushed to a per-WMO undo stack in Core.Editor.

### File changes
| Area | File | Change |
|---|---|---|
| Editor ops | `src/core/WowViewer.Core.Editor/Operations/WmoDoodadEditOperations.cs` | NEW — move/rotate/scale/duplicate/delete/add placement + set authoring, all returning undoable commands |
| Undo | `src/core/WowViewer.Core.Editor/WmoDoodadUndoStack.cs` | NEW — command stack, per-WMO session |
| Writer | `src/core/WowViewer.Core.IO/Converters/WmoV17ToV14Converter.cs` | Verify/extend only where round-trip tests find gaps (MODD/MODN/MODS rewrite on count change) |
| Renderer hook | `src/viewer/WoWViewer/Rendering/WmoRenderer.cs` | Public `ReloadDoodadsFromModel()` re-running `LoadActiveDoodadSet` after edits |
| Save plumbing | `src/viewer/WoWViewer/Rendering/WmoRenderer.cs` | Expose the loaded `WmoV14Data` + source path/version for save (read-only accessor) |
| UI | `src/viewer/WoWViewer/ViewerApp_Editor.cs` | Doodad edit panel: gizmo-driven transform fields, add/duplicate/delete, set authoring, Save (preflight + diff summary) |
| Tests | `tests/WowViewer.Core.Tests/Editor/WmoDoodadEditOperationsTests.cs` | NEW — ops + undo + round-trip |

### Phases
- **Phase 0 — Round-trip gate (blocking)**: real V14 WMO → V17 → V14 → byte/field diff. Fix writer gaps only if the gate fails. No UI.
- **Phase 1 — Core edit operations**: placement transform/duplicate/delete + undo; tests green.
- **Phase 2 — Set authoring**: MODS create/assign + MODN reuse; tests green.
- **Phase 3 — Renderer hook + live edit**: reload doodads, transform applies live via picking (Spec 211 selection feeds the editor).
- **Phase 4 — Add-from-model + Save UI**: model resolution gate, versioned save with preflight, provenance sidecar, operator visual gate.

### Risks
- V14 MODS/MODD stride mismatches if the converter model was normalized on read (e.g. padding). Phase 0 is the detector.
- MODS ranges assume contiguity in some downstream consumers; authoring must preserve the invariant or fail loudly.
- Group-file re-emission must be byte-for-byte from source; never regenerate groups from V17 in this spec.
