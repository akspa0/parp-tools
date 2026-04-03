# M2 Capture Workflow

Use this workflow for repeatable M2 renderer debugging in the live viewer.

## Purpose

The goal is to keep a stable before/after evidence loop for regressions such as:

- broken tree foliage or trunk materials
- wrong alpha cutout or transparency ordering
- missing creature clothing or baked textures
- animation regressions
- reflective surfaces rendering incorrectly
- any world scene where flying the continent reveals a bad object family or a one-off broken placement

## Storage Paths

- Shot-point settings file: `output/settings/camera_shot_points.json`
- Default capture output: `output/captures/<map>/<build>/...`

Both paths are relative to the app base directory.

## Recommended Shot Set

For each problematic asset family or world scene, keep these viewpoints:

1. Near material shot
- Close enough to inspect texture routing, alpha cutout, and wrap artifacts.

2. Medium silhouette shot
- Far enough to inspect canopy shape, sorting, and overdraw artifacts.

3. Side-angle shot
- Useful for alpha-card stretching, billboard-like errors, and reflective/transparency mistakes.

4. Animation shot
- For creatures or animated props, use a viewpoint where limb deformation and attachments are obvious.

5. Scene bookmark shot
- Use the exact world scene where the regression was discovered during a flythrough, even if multiple object families are visible in the frame.

## Viewer Loop

1. Open the target map/build in MdxViewer.
2. Move the camera to the exact viewpoint.
3. Open `Tools -> Capture Automation...`.
4. Save the shot point with a stable name.
5. Repeat for the rest of the viewpoints.
6. Use `Capture Filtered Set (No UI)` to generate the baseline batch.
7. Make the renderer change.
8. Re-run the same filtered no-UI batch.
9. Compare the new PNGs against the baseline and probe output.

When logging or sharing a scene, record the status-bar values together:

- WoW X/Y/Z
- Facing
- map/build

That gives a stable scene bookmark even when the problem was found by free-flying across a continent rather than by loading one standalone model.

## Naming Rules

Use names that stay stable across code changes.

Examples:

- `winterspring_tree02_near_foliage`
- `winterspring_tree02_mid_silhouette`
- `azshara_tree03_branch_split`
- `creature_clothing_front`
- `creature_clothing_side`
- `azeroth_elwynn_tower_tree_cluster`
- `azeroth_flythrough_bad_canopy_scene`

Avoid timestamps in the shot-point name itself. The capture system already timestamps each PNG file.

## Probe Pairing

Use capture evidence together with probe evidence, not as a replacement for it.

- If the probe shows wrong texture IDs, the bug is likely still in adapter/runtime metadata.
- If the probe shows correct textures but the live capture still looks wrong, the bug is likely in renderer sampling, alpha, UV handling, or ordering.

## Shot-Point Template

Use the checked-in example file in `docs/screenshots/m2-shot-points.template.json` as the structure reference for the live settings file.

## Future Outfit Workflow

The current viewer already has a meaningful seam for an eventual Armory-style outfitter:

- `ReplaceableTextureResolver` loads `CreatureDisplayInfo`, `CreatureDisplayInfoExtra`, and `ItemDisplayInfo`
- item model textures, armor-region textures, and weapon replaceable texture slots already exist in the active texture-resolution path

That means future outfit assembly should build on the existing DBC/display pipeline rather than adding a disconnected one-off preview system.