# Spec 231 — Editor & Archaeology Workspace UI Overhaul

Status: **Draft** (spec + plan + tasks authored 2026-09-07; implementation deferred to a fresh
session per operator instruction)
Owner epic: UI & Approachability (227 · 229 · 225 · 212 · 223 · **231**)
Supersedes: the unimplemented workspace-reorganization portions of [227](../227-ui-reaudit/spec.md)
("sane Editor tabs" remains unexecuted there); 227's inventory-gate process (T003/T004) still
applies and is folded into Phase 0 of this spec's plan.

## Operator-originated amendment (verbatim, 2026-09-07, later same session) — Tools curation & removal pass

> "the whole editor panel is a mess. the data i/o page is probably the cleanest and best example,
> along with the Quick panel, which is the best example of how the ui should have a good mix of
> simple tools in one location, the rest of the ui should follow suit, and keep a shared ui
> library in mind, no one-off weird things.
> To that end, we have weird terrain tools and selection tools and the old weak signal amplifier
> and various other things that should be removed that either do not work, never worked, were
> never fully wired, or do not fit the current ui and need to be rewritten from the ground up.
> This in turn is meant to act as a minor refactor of the ViewerApp *.cs files to get them under
> control and ensure only the best functionality that works, is retained in the ui."

Amendment scope (operator-directed, this session):
1. **UI model**: Editor → Data I/O page and the Quick panel are the canonical pattern — simple
   tools in one location, SharedUiWidgets primitives, no one-off styling. Remaining Editor/
   Archaeology pages converge on this pattern.
2. **Removal pass**: terrain tools, selection tools, the old weak-signal amplifier, and other
   surfaces that do not work, never worked, or were never fully wired are **removed**, not
   preserved. "Preserve working format readers" (AGENTS.md §4) still applies — this pass removes
   UI surfaces, not MPQ/ADT/WMO/M2 readers.
3. **ViewerApp refactor**: the removals act as a size/wiring reduction across the `ViewerApp_*.cs`
   partials; dead state fields, handlers, and draw methods are deleted with their surfaces.

## Operator-originated requirement (verbatim, 2026-09-07)

> "the ui is STILL atrocious and relaly not functional nor useful in anything other than the
> Viewer profile. The Editor and Archeology bits need some serious deduplication and help, and
> complete reorganization. I cannot figure out where things are, so how do you think end users
> will fare? I doubt any will enjoy using the program."

> "use speckit and plan the ui overhaul perfectly, please ... we will implement changes in a
> fresh chat"

Earlier same-session directives that are already implemented and must be preserved by this
overhaul: wireframe-on-selection instead of bounding boxes; semi-transparent white terrain
wireframe; toolbar `Anim` toggle (default on); hovered-WMO doodad-set combo; PM4 OBJ export must
not freeze the render thread.

## Problem statement (measured, 2026-09-07 source survey)

An operator cannot predict where a command lives. Concrete duplication found in
`src/viewer/WoWViewer/`:

| Duplicated surface | Sites | Locations |
|---|---|---|
| "Dump PM4 Objects JSON" + "Export PM4 OBJ Set" buttons | 5 | ViewerApp_Pm4Utilities.cs ~378, ~448, ~1201, ~1782, ~3900 |
| PM4/WMO correlation panel (near-identical code) | 2 | ViewerApp_Sidebars.cs ~801–901; ViewerApp_Pm4Utilities.cs ~1863–2007 |
| "Clipboard + Save" section | 2 | ViewerApp_Sidebars.cs ~1868, ~5581 |
| WMO doodad-set combo | 4 | ViewerApp_Sidebars.cs ~2281, ~3039, ~5283; toolbar (Spec 231 interim) |
| Editor functions hosted inside Archaeology tab | 2 pages | Editor sub-pages "Tasks & Workspace" and "Imports & Exports" are drawn by `DrawArchaeologyEditorTasksSubTab` / `DrawArchaeologyEditorImportsSubTab` |

Additionally, the Editor tab's Imports & Exports page rendered five unrelated tool groups
expanded at once (fixed interim on 2026-09-07 by defaulting them collapsed; the structural fix —
one page per task — is this spec).

## User scenarios

### US1 — Find an export without hunting
An operator who has never opened the Editor workspace selects a PM4 object and wants an OBJ set
on disk. They open the Editor tab, see a page named **Data I/O**, and every export/import in the
program is there, grouped, each group collapsed. They find "PM4 OBJ Set" in one navigation.

**Done when**: every export/import action in the program is reachable from the single Data I/O
page, and no other panel embeds an export button (cross-links only).

### US2 — Edit a placement without archaeology noise
An operator wants to nudge a PM4 object and reconcile it against a WMO. They open the Editor
tab's **Placement & Objects** page and find transform controls, match suggestions, reconcile, and
collection tools in one place — not scattered across an "Archaeology" tab and a "workbench"
selection panel.

**Done when**: placement/object editing tools have one home; the Archaeology tab contains no
editor task pages.

### US3 — Analyze in Archaeology without editor clutter
An operator doing weak-signal stratigraphy opens the Archaeology tab and sees only analysis
pages. Nothing asks them about converters, imports, or exports.

**Done when**: Archaeology tab = analysis pages only (weak signal/stratigraphy, unique-id
timeline, layers/provenance, playback/capture, PM4 analysis, cartography).

### US4 — Long operations never freeze the UI
Any command that can take more than about a second (PM4 OBJ export and its peers) runs off the
render thread with immediate status feedback and a re-entrancy guard, as implemented for PM4 OBJ
export on 2026-09-07.

**Done when**: no export/report button on a reorganized page blocks the frame loop.

## Functional requirements

- **FR-1 Single authoritative home.** Every action surface listed in the problem-statement table
  has exactly one draw site after this spec. Grep for each action label returns one UI draw
  location.
- **FR-2 Editor information architecture.** The Editor tab exposes exactly four named task pages:
  **Placement & Objects**, **Terrain Tools**, **Data I/O**, **Converters**. Page names describe
  the operator's task, not internal component names.
- **FR-3 Archaeology scope.** The Archaeology tab contains analysis pages only; the editor task
  and import/export pages currently hosted there are removed (their content moves to Editor).
- **FR-4 Reachability.** Any action on a reorganized workspace is reachable in at most two
  navigations from its workspace tab (tab → page; page sections are collapsed headers).
- **FR-5 Cross-links, not embeds.** Where a context needs to send the operator to a tool that
  lives elsewhere (e.g. Inspector → export), it renders a navigation link that opens the owning
  page with the current selection preserved — never a re-embedded copy of the controls.
- **FR-6 Fast-access toolbar contract.** The editor toolbar controls added 2026-09-07 (Terrain
  WF, M2/WMO WF, Anim, hovered-WMO doodad set) keep their toolbar homes; the full doodad-set
  combo stays on the owning object page. Toolbar and page must not disagree.
- **FR-7 Background execution contract.** Any reorganized command that can exceed ~1 second runs
  off the render thread with immediate status text, completion/error status, and a re-entrancy
  guard (pattern already proven by `ExportPm4ObjectsObjSet`).
- **FR-8 Owned service classes (Spec 228).** Reorganized page content is extracted into owned
  workbench service classes receiving state via constructors/parameters. No new `ViewerApp_*.cs`
  partial files; `WorldScene`/`ViewerApp` gain no new feature members; no source file exceeds
  ~2,000 lines without a same-change split.
- **FR-9 Inventory row per moved surface (Spec 223 FR-9 / 227).** Every moved, merged, or
  removed surface gets an inventory v3 row in the same change, recording its single home.

## Out of scope

- Viewer profile behavior (its sidebar was de-congested 2026-09-07 and operator retests it
  separately).
- Renderer behavior of any kind; visual styling themes beyond the existing SharedUiWidgets
  standard (Spec 227).
- New features; this spec moves and deduplicates existing capability only.
- Keybind profiles (Spec 229) — but the action inventory produced here feeds it.

## Assumptions

- The four-tab shell (Quick / Inspector / Editor / Archaeology) remains; this spec reorganizes
  what is inside the Editor and Archaeology tabs and removes duplication, not the shell itself.
- The operator walks a navigation smoke test at the end (US1–US3 task tests) — that receipt
  closes the spec alongside build/test receipts.

## Success criteria

- **SC-1**: For each surface in the problem-statement table, a source search for its action
  label(s) finds exactly one UI draw site (receipt lists the grep + count per surface).
- **SC-2**: Task-based navigation test passes: from a cold start, an operator reaches PM4 OBJ
  export, WMO doodad-set switching, and terrain layer import in ≤2 navigations each, using only
  page names (recorded as the operator smoke receipt).
- **SC-3**: The Editor tab's default view shows only collapsed sections and fits 1600×900
  without scrolling.
- **SC-4**: `dotnet build` and `dotnet test` for the solution are clean at each phase gate.
- **SC-5**: No duplicated code blocks remain from the correlation-panel pair (the surviving
  implementation is referenced by both former entry points' replacement links).
