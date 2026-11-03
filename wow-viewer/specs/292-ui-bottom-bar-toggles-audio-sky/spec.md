# Spec 292 — Bottom-Bar Toggles, Mode Sidebars, Audio Panel & Sky Controls

**Status:** DRAFT — awaiting operator approval (AGENTS.md §9.1). No code written.
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

