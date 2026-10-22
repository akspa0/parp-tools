# Tasks: v0.7.0 UI Presets and the Explore Museum

**Spec**: [`spec.md`](spec.md) · **Plan**: [`plan.md`](plan.md) ·
**Visual analysis**: [`visual-analysis.md`](visual-analysis.md) · **Created**: 2026-10-21

> The original preset/Explore phases remain open. A dated operator amendment and a bounded shell
> correction are tracked at the end of this file; build/reference checks do not close visual gates.

## P0 — Version and guardrails

- [ ] **T001** Bump `wow-viewer/eng/Version.props` to `0.7.0` (`VersionPrefix`, `AssemblyVersion`,
  `FileVersion`, `InformationalVersion`).
- [ ] **T002** Add a preset-ownership test that reads the §4 table and fails when a surface is drawn
  outside its owning preset.
- [ ] **T003** Add a shell-visibility test that fails when a hidden part still consumes layout
  space.
- [ ] **T004** Record the v0.6.1 frame-time baseline (UI shown and hidden) at 1080p and 4K for the
  SC-006 gate.

## P1 — Preset model

- [ ] **T101** Introduce `PresetDefinition` (sidebar sections, bottom-bar pages, chat tool groups,
  Options groups) and the four definitions from plan §4.
- [ ] **T102** Replace the destination `switch` in `NavigatorPanelService.DrawContent()` with the
  current preset's sidebar section list.
- [ ] **T103** Replace `WorkbenchTab` with `Preset` (Viewer, Editor, Archeology, Explore) and update
  every reference.
- [ ] **T104** Migrate the persisted layout document to v4 with the preset and page selections.
- [ ] **T105** Preserve each preset's page selection across switches (FR-007).

## P2 — Bottom bar as preset switcher

- [ ] **T201** Draw four preset buttons with icons and one-line tooltips.
- [ ] **T202** Move Options out of the preset row into its own affordance.
- [ ] **T203** Move the FPS readout from the bottom bar to the status bar.
- [ ] **T204** Register the new labels in the UI control registry and regenerate the reference.

## P3 — Editing containment

- [ ] **T301** Move Cartography export actions to Editor > Data I/O, leaving an `Open in Editor`
  link at the old location for one release.
- [ ] **T302** Move Scene Tools editing actions to Editor > Terrain Tools / Placement & Objects.
- [ ] **T303** Restrict Options > Simulation and Options > Grids to Editor.
- [ ] **T304** Delete the right sidebar "Context" prose block; the Inspector shows the data.
- [ ] **T305** Walk Viewer and Archeology and confirm zero editing controls (SC-002).

## P4 — Explore

- [ ] **T401** Add the Explore preset.
- [ ] **T402** Default Explore to the taxi/route presentation when the map has taxi data.
- [ ] **T403** Fall back to the museum asset browser when the map has no taxi data.
- [ ] **T404** Hide both sidebars by default in Explore.
- [ ] **T405** Add the compact inspection overlay for terrain, WMO, M2, PM4, liquid and spawns.
- [ ] **T406** Add asset search in the chat tools column that flies the camera to the result.
- [ ] **T407** Confirm no editing action is reachable in Explore (FR-011).

## P5 — Shell visibility

- [ ] **T501** Add `ShellVisibility` with one flag per part (left sidebar, right sidebar, minimap,
  chat, status bar, bottom bar, tooltips).
- [ ] **T502** Wire `ShellVisibility` through `UiLayout` so freed space returns to the scene.
- [ ] **T503** Add Options > Shell with one checkbox per flag.
- [ ] **T504** Persist the flags in layout v4.
- [ ] **T505** Confirm every hidden part is restorable from a visible control and a shortcut.

## P6 — Text salad removal

- [ ] **T601** Remove every leading paragraph from every sidebar section.
- [ ] **T602** Remove every leading paragraph from every Options group.
- [ ] **T603** Move the removed text to tooltips and the generated UI reference.
- [ ] **T604** Regenerate the reference and pass `Verify-UiReference.ps1`.

## P7 — Visual pass

- [ ] **T701** Apply the early-WoW proportions: scene first, minimap and chat as the only persistent
  chrome.
- [ ] **T702** Verify style metrics scale with the font at 1080p, 1440p and 4K.
- [ ] **T703** Verify the popup host is input-transparent (`NoInputs`) and the camera works with a
  popup open.
- [ ] **T704** Verify the minimap is a standalone framed map with no controls and no text.

## P8 — Verification

- [ ] **T801** Build: `dotnet build wow-viewer/WowViewer.slnx -c Debug` → 0 errors.
- [ ] **T802** Preset ownership test → 0 violations.
- [ ] **T803** Shell visibility test → 0 violations.
- [ ] **T804** `Verify-UiReference.ps1` → PASSED.
- [ ] **T805** Frame-time comparison against the T004 baseline → ≤ 1 ms difference.
- [ ] **T806** Operator preset walk → 0 editing controls in Viewer/Archeology.
- [ ] **T807** Operator Explore load → taxi/route presentation shown.
- [ ] **T808** Operator shell-visibility round-trip → choice survives restart.

## Known gaps at plan time

- The taxi/route presentation has not been evaluated as a full-scene default. If it is not ready,
  T403's museum asset browser is the documented fallback and T402 becomes a follow-up.
- The preset-ownership test needs a machine-readable form of the §4 table; T002 decides whether that
  is a C# table or a JSON artifact.
- No frame-time baseline exists yet; T004 must run before any UI work so SC-006 is measurable.

## Operator amendment — 2026-10-21: spellbook and compact shell

- [x] **T901** Initial Spellbook prototype and compact chat/minimap slice (historical only; the
  operator later rejected and removed the Spellbook). Receipt: [`evidence/ui-shell-correction-2026-10-21.md`](evidence/ui-shell-correction-2026-10-21.md).
- [x] **T902** Add the missing `Minimap` Options control registry row, regenerate the UI reference,
  and pass the reference verification plus solution build. Receipt: same as T901.
- [ ] **T903** Operator visual/runtime walkthrough against the supplied early-WoW reference; confirm
  minimap/chat proportions and that direct icon actions reach useful destinations. Build output is not
  visual or interaction proof.
- [ ] **T904** Deliver the requested 0.5.3 map-converter/DBC automation as its separately owned
  converter work; not part of this shell receipt.

## Operator amendment — 2026-10-21: match supplied anatomy layout and close spawn preview

Source/static completion for T905–T910 and T912 is receipted in
[`evidence/action-bars-and-popup-windows-2026-10-22.md`](evidence/action-bars-and-popup-windows-2026-10-22.md).
T903/T911 remain the operator-owned visual/interaction gates, including title-bar close and spawn-load
dismissal in a running client. The receipt covers the restored Maps/File Browser/Diagnostics routes,
shared icon primitive, full-name tooltips, chat log width and Taxi/Capture separation; it does not
claim the full preset/Explore implementation complete.

- [x] **T905** Stop drawing the WDL map-preview window once its show flag is false, so map load,
  Cancel, and the close control dismiss it. Receipt: [`evidence/action-bars-and-popup-windows-2026-10-22.md`](evidence/action-bars-and-popup-windows-2026-10-22.md).
- [x] **T906** Bump layout persistence and reset stale oversized widths; default to narrow visible
  icon bars and keep their full data panels closed until an icon is clicked.
- [x] **T907** Make minimap and chat rectangles independent of sidebar widths: map upper-right, chat
  lower-left, scene full-width behind overlays.
- [x] **T908** Replace the text-only destination row with a compact, usable action strip. Each control
  must invoke a real existing action; no Spellbook surface is permitted.
- [x] **T909** Update UI control inventory/reference, build, and add geometry checks/tests for the
  scene/minimap/chat layout. Keep runtime visual acceptance operator-owned.
- [x] **T910** Replace full-height left/right data panels with compact anchored icon bars; each icon
  opens one small, independently closeable window backed by its existing content owner.
- [ ] **T911** Verify toolbar actions, popup open/close behavior, title-bar dismissal, and scene input;
  inventory/document every visible icon and popup control.
- [x] **T912** Remove the rejected Spellbook window, service, and launchers; regenerate the reference
  and confirm the application builds. Receipt: [`evidence/action-bars-and-popup-windows-2026-10-22.md`](evidence/action-bars-and-popup-windows-2026-10-22.md).
