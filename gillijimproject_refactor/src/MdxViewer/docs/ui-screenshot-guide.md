# MdxViewer UI Screenshot Guide

Use this guide to capture consistent screenshots for README/release pages and first-time-user docs.

## Output Folder

Drop captured images into:

- `src/MdxViewer/docs/screenshots/`

Recommended naming:

- `01-start-open-game-folder.png`
- `02-build-selection-dialog.png`
- `03-world-map-loaded.png`
- `04-pm4-tooltip-hover.png`
- `05-pm4-workbench-selection.png`
- `06-pm4-workbench-correlation.png`
- `07-minimap-fullscreen.png`
- `08-render-quality-window.png`

## Capture Baseline

- Start from a fresh app launch.
- Use the default fixed-sidebar layout (do not enable dock panels for baseline screenshots).
- Load the same map/area each time when possible for consistency.
- Keep UI scale/font consistent between captures.
- Avoid debug clutter unless the screenshot is specifically for a debug feature.

## Required Core Shots

1. Startup and onboarding
- Show the top bar and the `Open Game Folder...` action path.
- Goal: make it obvious that users should open a game path first.

2. Build selection dialog
- Show explicit build selection before MPQ load.
- Goal: remove confusion around direct single-file opening.

3. First world scene loaded
- Left sidebar map browser visible.
- Right sidebar inspector visible.
- Goal: communicate the default sidebar-first layout.

4. PM4 WoW-styled tooltip
- Hover PM4-rich geometry with tooltip visible.
- Goal: highlight better PM4 data display in tooltips.

5. PM4 Workbench - Selection tab
- A selected PM4 object with match list visible.
- Goal: teach object inspection flow.

6. PM4 Workbench - Correlation tab
- Correlation candidates visible.
- Goal: teach PM4-to-placement investigation flow.

7. Minimap fullscreen mode
- Show marker and navigation context.
- Goal: demonstrate map navigation.

8. Optional advanced windows
- Render Quality, Log Viewer, Perf, or exporter dialogs.
- Goal: document power-user tools without replacing core onboarding shots.

## Selection Rules For README Hero Image

Pick one image that best communicates:

- a loaded world scene
- clear UI readability
- visible PM4 tooltip/workbench relevance
- minimal visual clutter

If multiple candidates are strong, prefer the image that most clearly shows "this is a world/map viewer with PM4 inspection" at a glance.

## Notes

- If you provide a batch of screenshots, we can pick the best hero image and wire it into README/release notes in a follow-up pass.
- Keep raw source screenshots if post-cropping is needed later.

## M2 Regression Loop

Use the in-app capture automation path for renderer debugging instead of one-off manual screenshots.

- Save stable camera shot points from `Tools -> Capture Automation...` for the exact asset family or world viewpoint under investigation.
- Prefer `Capture Selected (No UI)` or `Capture Filtered Set (No UI)` for before/after comparisons so ImGui chrome does not pollute the evidence.
- Keep the same map, build, FOV, yaw, pitch, and camera position across captures.
- Record the status-bar WoW coordinates and facing when you find a broken scene during a flythrough so the same viewpoint can be reconstructed later.
- Compare the resulting PNGs under `output/captures/<map>/<build>/...` with probe output from `AssetProbe` or adapter/runtime diagnostics.
- For broad world regressions, do not limit the batch to one tree family. Keep scene bookmarks for any broken object cluster found while flying maps such as Azeroth or Kalimdor.
- For tree regressions, keep at least one near shot for foliage cards and one medium shot for silhouette/alpha sorting.
