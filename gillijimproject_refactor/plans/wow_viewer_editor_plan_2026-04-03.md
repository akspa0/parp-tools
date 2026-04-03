# wow-viewer Editor Transition Plan

## Apr 03, 2026 - Editor direction reset

- status: active continuity plan
- goal:
  - treat the current feature request as the start of `wow-viewer` becoming the modern WoW map viewer-editor, not only a viewer or inspection shell
  - keep the next chats on narrow editor slices instead of mixing terrain editing, PM4 evidence, object persistence, UI reorganization, and file writing into one unstable task

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

## Guardrails

- do not claim editor closure from persisted PM4 match selections alone; that is evidence or preference persistence, not map writing
- do not claim terrain conform from `MPRL` as native truth; treat it as evidence-guided editing unless the format or runtime proof says more
- do not let panel code become the long-term owner of map writing or object persistence
- do not treat viewer-mode and editor-mode reorganization as a reason to fork apps; prefer one app with workspace profiles or presets

## Recommended continuation order

1. plan the first saved-map milestone with `wow-viewer-map-editing-foundation-plan.prompt.md`
2. plan editor vs viewer workspaces with `wow-viewer-editor-ui-surface-plan.prompt.md`
3. after those boundaries are concrete, use the existing shared-I/O and PM4 implementation prompts for the first actual library slices

## Proof boundary

- this continuity slice adds workflow assets and planning structure only
- no `wow-viewer` editor runtime, map save pipeline, or UI reorganization code is landed yet