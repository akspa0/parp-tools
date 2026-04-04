# wow-viewer Editor Transition Plan

## Apr 04, 2026 - MdxViewer dirty-map save packaging now groups staged moves by ADT source

- extended the first `MdxViewer` save consumer beyond one current selection state:
  - staged translation-only MDDF and MODF placement moves now persist across selection changes
  - pending moves are grouped by source ADT and can be saved per source or with `Save All Pending`
  - the `Publish` workspace now exposes the grouped dirty-source queue instead of leaving save packaging only in the selected-object controls
- proof captured so far:
  - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing warnings
- important boundaries:
  - this is still translation-only placement persistence for existing ADT entries only
  - no automated tests were added for this slice
  - no real-data interactive workflow signoff was captured yet for grouped save packaging
  - add/remove placement support, terrain writer ownership, and full packaged map-save workflow are still open

## Apr 03, 2026 - Editor direction reset

- status: active continuity plan
- current state:
  - the first viewer/editor workspace shell is now landed in `gillijimproject_refactor/src/MdxViewer`
  - the first save-capable object move transaction is now landed in `wow-viewer`
  - the first `MdxViewer` consumer wiring for selected existing ADT object moves is now landed on top of that shared seam
  - the broader editor-transition plan is still open because editor save is still translation-only and does not yet cover add/remove, dirty-map flow, terrain writes, or full save packaging
- goal:
  - treat the current feature request as the start of `wow-viewer` becoming the modern WoW map viewer-editor, not only a viewer or inspection shell
  - keep the next chats on narrow editor slices instead of mixing terrain editing, PM4 evidence, object persistence, UI reorganization, and file writing into one unstable task

## Apr 03, 2026 - First workspace shell landed in MdxViewer

- the editor UI organization bucket is no longer prompt-only
- landed in the live `MdxViewer` UI:
  - `Viewer` vs `Editor` workspace mode in the menu and toolbar
  - editor task routing for `Terrain`, `Objects`, `PM4 Evidence`, `Inspect`, and `Publish`
  - editor-mode navigator routing on the left sidebar
  - editor-mode inspector routing on the right sidebar
  - status-bar affordances for workspace mode, active task, current target, and explicit save boundary
- important boundaries:
  - this is still UI-shell regrouping over existing `MdxViewer` services, not shared save ownership
  - `Objects` still reuses the mixed legacy world-object surface as a first regrouping step
  - there is now narrow UI wiring for selected existing ADT object moves only
  - there is still no dirty-map pipeline, aggregated save queue, add/remove object support, or terrain writer path
- proof captured so far:
  - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing warnings
  - no runtime signoff was captured yet for the new workspace flow

## Apr 03, 2026 - First save-capable object move transaction landed in wow-viewer

- landed shared save ownership for one narrow object-edit milestone in `wow-viewer`:
  - `AdtPlacementEditTransaction` now models translation-only moves for existing ADT placements
  - `AdtPlacementWriter` now patches `MDDF` and `MODF` entries in place and translates `MODF` bounds with moved WMOs
- proof captured so far:
  - focused synthetic roundtrip tests for both placement families
  - real-data roundtrip test against `development_0_0_obj0.adt`
  - `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter "AdtPlacementReaderTests|AdtPlacementWriterTests"` passed
- important boundaries:
  - this only supports moving existing placements; it does not add or remove entries
  - string-table rebuilds for new model paths are still out of scope
  - there is still no dirty-map state, terrain-write ownership, or end-to-end editor save workflow in `MdxViewer`

## Apr 04, 2026 - First MdxViewer selected-placement save consumer landed

- landed the first active-viewer UI consumer over the shared `wow-viewer` object-move seam:
  - selected existing ADT MDDF and MODF placements can now be translated from the `Objects` workspace
  - the moved preview updates live `WorldScene` placement state and cached tile placement data instead of only changing status text
  - save target handling now distinguishes between writable loose sources and explicit user-chosen output `.adt` paths
- proof captured so far:
  - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing warnings
- important boundaries:
  - this is still one-placement-at-a-time translation-only save wiring
  - no aggregated dirty-map flow, add/remove placement support, terrain writer path, or packaged map-save workflow landed in this slice

## Confirmed starting inputs

- current `MdxViewer` already has real editor-facing seams worth preserving as reference inputs:
  - chunk clipboard copy or paste with terrain, alpha, shadow, and texture options
  - alpha-mask import or export and terrain image workflows
  - PM4 workbench overlay or selection or correlation panels
  - saved PM4 object-match selections persisted in viewer settings
- current `wow-viewer` already has shared seams relevant to editor migration:
  - ADT top-level summary, texture, placement, and MCAL decode ownership
  - PM4 placement and `MPRL` summary or correlation ownership
  - inspect or converter hosts that can become dual-surface editor consumers later

## New prompt surface added in this slice

- `.github/prompts/wow-viewer-editor-plan-set.prompt.md`
- `.github/prompts/wow-viewer-map-editing-foundation-plan.prompt.md`
- `.github/prompts/wow-viewer-editor-ui-surface-plan.prompt.md`
- matching Codex mirrors under `.codex/prompts/`

## What each prompt is for

- `wow-viewer-editor-plan-set`:
  - route broad editor-transition asks to the right next focused prompt
- `wow-viewer-map-editing-foundation-plan`:
  - plan the first real terrain or object editing or dirty-map or save pipeline boundary as an implementation-ready build plan with exact slice order, file scope, and validation
- `wow-viewer-editor-ui-surface-plan`:
  - plan viewer vs editor workspaces, panel presets, and editor affordances as an implementation-ready UI build plan with exact slice scope, dependencies, and validation
- `wow-viewer-cli-gui-surface-plan`:
  - plan dual-surface workflows as an implementation-ready build plan with exact slice scope, shared-service ownership, and proof targets
- `wow-viewer-tool-migration-sequence-plan`:
  - plan the migration order as an implementation-ready phase queue with the exact first slice to build now and its validation path

## Prompt output rule

- these editor prompts are not meant to stop at architecture commentary
- their outputs should be the implementation plans the next chat can build from directly
- every plan response should name the first slice to implement now, the exact repo or file scope, the proof target, and the immediate follow-up prompt or implementation seam

## First editor problem buckets

### 1. PM4-assisted terrain and object editing foundation

- use `MPRL` points as evidence for terrain conform where PM4 objects pierce the ADT mesh
- support saving the chosen PM4-backed object selections as real map content, not only viewer settings
- support saving moved objects on maps the user edits
- keep terrain-write and object-write ownership in shared `wow-viewer` seams rather than UI-only code

### 2. Editor UI organization

- reorganize the current tool sprawl into explicit viewer and editor workspaces or panel presets
- keep chunk clipboard, alpha-mask tools, PM4 workbench, and related editor features grouped by task instead of spread across debug-style menus
- make dirty-state, selection target, and save scope obvious once true editing starts
- first scaffolding slice is now landed in `MdxViewer`; follow-up UI work should focus on extracting mixed legacy object panels and wiring real dirty/save contracts once the shared foundation exists

## Guardrails

- do not claim editor closure from persisted PM4 match selections alone; that is evidence or preference persistence, not map writing
- do not claim terrain conform from `MPRL` as native truth; treat it as evidence-guided editing unless the format or runtime proof says more
- do not let panel code become the long-term owner of map writing or object persistence
- do not treat viewer-mode and editor-mode reorganization as a reason to fork apps; prefer one app with workspace profiles or presets

## Recommended continuation order

1. extend the shared object persistence seam from translation-only moves into the next editor-needed operation, likely add/remove or chosen-object persistence
2. capture real-data interactive proof for the new grouped `MdxViewer` dirty-source queue instead of treating the build as runtime closure
3. keep terrain-write ownership separate until object-save flow and save packaging are honest end to end

## Proof boundary

- this continuity slice now covers workflow structure, one landed `MdxViewer` UI-scaffolding slice, one landed `wow-viewer` object-move transaction seam, and one build-validated grouped dirty-source save queue in `MdxViewer`
- no full `wow-viewer` editor runtime, terrain writer path, add/remove object persistence, or fully runtime-validated packaged map-save workflow is landed yet
- do not archive this plan until the save-capable editor foundation slices are also closed