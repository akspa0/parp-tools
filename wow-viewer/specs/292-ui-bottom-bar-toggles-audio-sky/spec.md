# Spec 292 — Bottom-Bar Toggles, Mode Sidebars, Audio Panel & Sky Controls

**Status:** Original scope implemented 2026-11-03; bottom-toolbar correction implemented 2026-11-05. Real-client visual confirmation remains owed.
**Origin:** Operator request 2026-11-03 (wording preserved below). Follows Spec 291.

## Operator requirements (verbatim intent)

1. Skybox control in the UI, under Workspace options. Operator answer: **full** — source, time-of-day, colors, on/off, and star/moon/cloud toggles.
2. Quick-toggle checkboxes return to the bottom bar. Operator answer: slim bar of toggles
   (Sky, Fog, Water, Doodads, WMOs, Terrain, PM4, Grid, Audio mute) **plus** Viewer/Editor/Archaeology/Options mode switches that open sidebars.
3. All lower bottom-bar *panels* become sidebar panels: **Viewer → left, Editor → right, Archaeology → right, Options → left.**
4. **Pages**: remove entirely if it points to nothing. **Spawns** → right sidebar.
5. **Profile buttons**: remove entirely; each option lives in exactly one panel and persists on its own.
6. **Audio**: one left-sidebar panel for all audio (master/music/ambience/SFX volumes, mute, etc.), easy to find.
7. **Panel behavior defects**: sub-tabs that point to nothing, and drop-downs/collapsibles that expand to nothing and scroll the panel (losing place), are defects to fix.

## Acceptance criteria

| ID | Criterion |
|---|---|
| AC-1 | Bottom bar shows the 9 toggle checkboxes + 4 mode switches; each toggle changes the scene immediately and persists across restarts. |
| AC-2 | Viewer and Options open in the left dock; Editor, Archaeology, Spawns open in the right dock. No floating bottom-bar windows remain. |
| AC-3 | Pages is removed (or, if inventory finds real content, that content is re-homed and listed in the receipt). |
| AC-4 | No Profile buttons anywhere in the UI. |
| AC-5 | Left-dock **Audio** panel exposes every audio setting currently scattered elsewhere; the old locations are removed (one authoritative home, Spec 223 FR-3). |
| AC-6 | Left-dock Workspace/Sky section: on/off, source (Auto-from-lighting / Procedural / Sky model), time-of-day slider + lock, zenith/horizon/fog colors, stars/moon/clouds checkboxes. |
| AC-7 | Expanding/collapsing any section or switching a sub-tab does not reset the panel's scroll position; every sub-tab has content. |
| AC-8 | `ui-controls.json`/`UI-REFERENCE.md` updated; `Verify-UiReference.ps1` clean. |

Criteria naming visible/runtime behavior need operator screenshots or a real-client run as proof; build/tests alone do not satisfy them (§9.2).

## Out of scope

Renderer sky rewrite, new audio engine features, PM4/export work.

## Open item for operator

Sky "source = Sky model" assumes MPQ sky M2/MDX models can be listed and rendered. Phase 1 inventory verifies this; if unsupported, that option ships disabled and is reported.

## Amendment 2026-11-05 — Correct the bottom toolbar controls

**Operator direction (latest):** “Remove Sky/Fog; add A1–A7 buttons that select one Alpha channel at a time and turn the overlay on. Keep L0–L7 visibility toggles, the other scene toggles, and both opacity sliders.” Follow-up: “auto-adjust the number of layers and alpha masks to show, based on the actual terrain loaded”; “We have MCLQ/MH2O AND WL* data for 'water', but it's not always water, it's sometimes slime and lava. We have no separate controls for different liquidtypes”; “why the fuck is there a mute button on this bar and the bottom bar? have mute turned on by default.”

This supersedes the preceding toolbar interpretation. The bottom toolbar shows per-liquid-family toggles (Water, Ocean, Magma, Slime), Doodads, WMOs, Terrain, PM4, Grid, and Mute; L0–Ln visibility toggles sized to the resident terrain's actual maximum layer count; one A1–An alpha selector per non-base layer; and separate M2/MDX and WMO opacity sliders. Selecting A<n> selects that renderer alpha channel and enables the alpha overlay; clicking the active A<n> again turns the overlay off. Sky/Fog and Texture Masks are not bottom-toolbar controls. The Workspace sky controls and Layers sidebar remain in their existing homes. Persist the alpha overlay-enabled flag and selected channel along with the existing toolbar settings.

Liquid families are filtered in both terrain liquid sources (MCLQ/MH2O and WL*) and WMO MLIQ groups. The status bar no longer duplicates the Mute control; the bottom toolbar is the single quick Mute toggle, and audio starts muted by default.

The UI registry/reference, focused footer tests, and solution build must validate the corrected control set. Real-client visual and restart/persistence proof remains operator-owned under §9.2.

## Amendment 2026-11-04 — operator-reported flicker defect (post-landing): ROOT CAUSE FOUND

The operator reported the whole UI strobing/flickering and being unresponsive, beginning around
commit `d346cf24` (this spec's landing), with two crashes-to-desktop. Windows Event Log (.NET
Runtime 1026, 2026-11-04 01:03 and 01:31) captured both crashes: unhandled access violations at
`ImGuiNative.igEndMenu` (from `MainMenuBarService.DrawContent`) and `ImGuiNative.igBeginPopupModal`
(from `ImGuiPathPicker.Draw`) — **ImGui window-stack corruption**, not two unrelated bugs.

**Root cause:** the ImGui frame was split across callbacks — `_imGui.Update()` (ImGui `NewFrame`)
ran in `ViewerApp.OnUpdate` while `_imGui.Render()` (EndFrame) ran in `ViewerApp.OnRender`. Commit
`2b3762d3` set independent 60 Hz schedules for both callbacks (`FramesPerSecond`/`UpdatesPerSecond`),
so Silk.NET invokes them on different cycles; widget code then regularly ran outside a begun frame.
Survivable corruption rendered widgets into wrong windows — the strobing, unusable UI — and the
corrupted stack eventually crashed in random ImGui entry points. This also explains why pacing
Render inside OnRender (this session's earlier cadence attempts) amplified the flicker: it
desynchronized Render from Update further.

**Fixes in code this date:**
1. **ImGui frame bound to OnRender** (`ViewerApp`): `_imGui.Update((float)dt)` moved from OnUpdate
   to immediately before the UI block in OnRender, with `DrawUI` + `ImGuiPathPicker.Draw` wrapped
   in try/finally around `_imGui.Render()` — NewFrame/EndFrame are now always paired inside one
   callback, and a throwing widget closes the frame instead of terminating the process.
2. **Presentation cadence** (`ViewerApp.OnRender`): unconditional Stopwatch-targeted even ~60 FPS
   cadence (hybrid 1 ms sleep + short spin, no dt gate, no catch-up bursts) + `timeBeginPeriod(1)`
   on Windows; keeps swaps clear of the DWM composition race when VSync is not pacing.
3. **Per-frame I/O removed** (`ViewerUiManager`): `OpenLeftDock` made idempotent (was re-entered
   per frame by `DrawDiagnosticsWindows` while a diagnostics flag was set); `SaveLayout` skips
   byte-identical rewrites.
4. **Temporary diagnostic** (`UiOscilloscope`, on by default, `PARP_UI_OSCILLOSCOPE=0` opts out):
   per-frame shell-state/rect/draw-data logging to `output/settings/ui-oscilloscope.log`; remove
   once the operator confirms the fix.

Verification: solution build 0 errors; 22/22 focused UI tests pass. Runtime flicker/crash
resolution is operator-owned (§9.2): rebuild, launch, exercise the File menu and panels, confirm
stable and interactive; the oscilloscope log verifies cadence and flags any residual per-frame
toggle. Residual uncertainty: if strobing persists on this build, the log discriminates the
remaining cause (content-level toggle vs scene-side) in the same run.
