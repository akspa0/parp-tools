# Quickstart: Precise Object Selection

All four stories are viewer-interactive; none has a CLI surface. Verification is by loading a real scene
and clicking, per this project's standard that UI/rendering features must be exercised in the running app.

## Phase 1 (US2) — triangle-precise PM4 picking

1. Launch the viewer, load a build with PM4 data, enable the PM4 overlay.
2. Find two PM4 objects whose bounding boxes overlap but whose drawn surfaces do not.
3. Click a point inside the overlapping-box region that lies on only one object's actual surface.
4. **Expect**: that object is selected, not the other (SC-002). Before this phase, either could win.

## Phase 2 (US1) — triangle-precise picking for regular objects

1. Load a scene containing a non-convex object — an L-shaped building, a sparse foliage model, or an
   animated creature with an extended limb.
2. Click inside its bounding volume but visibly outside the model itself.
3. **Expect**: it is not selected; whatever is genuinely behind that empty space is (SC-001).
4. Click directly on the visible surface. **Expect**: it is selected.
5. Find an object whose asset failed to load. **Expect**: its selection behavior is exactly as before this
   feature — the bounding-volume fallback (SC-005).

## Phase 3 (US3) — confirmed-match library

1. Select a PM4 object and a real placement you are confident are the same object.
2. Confirm the match, entering a reason.
3. Restart the viewer, select that PM4 object again. **Expect**: its confirmed match is shown, not
   "unknown" (SC-007).
4. Retract it. **Expect**: the retraction is recorded and the object returns to unconfirmed, with the
   original confirmation still in history (SC-008).

## Phase 4 (US4) — world-space cursor

1. Move the mouse across varied terrain (a slope, a cliff edge).
2. **Expect**: a marker tracks the terrain point under the cursor every frame.
3. Position the camera so the tracked point is behind a hill. **Expect**: the marker is hidden by the
   terrain, not drawn on top of it (SC-003).
4. Point at open sky. **Expect**: no marker, not a marker parked at a fixed distance.
