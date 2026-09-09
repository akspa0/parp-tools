# Operator Quickstart — Spec 232 Gates

These are validation witnesses, not implementation substitutes. Configure the approved client root and target build at runtime; do not hardcode it in settings or source.

## Cell alignment (SC-1 / T013)

1. Load Azeroth with a DeadminesInstance layer at the recorded 90° clockwise tile transform.
2. In Map Layers, use the cell X/Y controls to align the Moonbrook roadway with the buildings.
3. Capture the view and record the exact tile/cell offsets, rotation, mirrors, map/build and client-root fingerprint. Confirm a border-crossing nudge does not leave a gap.

## Persistence and locks (SC-2 / SC-3 / T022)

1. Save the layer project, then restart the viewer and reload the same base map.
2. Confirm channel gates, offsets, cell offsets, rotation origin, mirrors, placements and locks match before the restart.
3. Attempt a transform/channel edit while locked; it must be rejected. Unlock explicitly, edit, save and confirm the JSON change under `output/projects/cartography/`.

## Seam repair (T015d)

1. Rotate DeadminesInstance by one quarter turn.
2. Capture coastline and roadway across several MCNK boundaries, including a cell fine-tune.
3. Confirm continuous shared edges with no patchwork. Record build, client root, screenshot and the focused test/build commands in the T015 receipt.

