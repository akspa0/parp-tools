# Feature Specification: Camera Capture Paths

**Feature Branch**: `144-camera-capture-paths`
**Created**: 2026-08-11
**Status**: Implementing
**Input**: Restore capture automation in the left sidebar and author/play back map-bound M2-style camera paths.

## User Scenarios

### User Story 1 - Author a camera path in the world (P1)

With a world loaded, the user can record the current camera as ordered key points, adjust their timing, and preview the resulting spline while the world remains the active scene.

**Independent Test**: Add three camera keys, press Play, and observe the camera move through all three points without leaving the loaded map.

### User Story 2 - Save and reopen a path (P1)

The user can save an authored path as a readable project document and as a native camera-only M2, then load it again with map/build provenance preserved.

**Independent Test**: Save a path, reload it, and verify the key count, map binding, duration, and camera positions are unchanged.

### User Story 3 - Drive capture automation from a path (P1)

The user can reach the existing still/video capture automation from the left sidebar and start playback with video recording or queue captures at authored keys.

**Independent Test**: Start path playback plus video recording, then stop both; the capture output is produced by the existing framebuffer/ffmpeg route and the camera path remains reusable.

### User Story 4 - Reuse existing M2 camera assets on a map (P2)

The user can import a readable M2 camera asset, bind its sampled tracks to the current map, and play the imported path through the world.

**Independent Test**: Import a camera-only M2 with at least one camera track and verify that sampled keys appear and can be played on the matching active map.

## Edge Cases

- Playback refuses a path whose map binding does not match the active world, instead of silently moving the camera in the wrong scene.
- A path with fewer than two keys can be saved and inspected but cannot be played as a moving spline.
- An M2 camera with no animated tracks imports as one static key.
- Missing or malformed path files leave the current path untouched and report a readable error.
- Native M2 export uses sampled linear keys; the project document remains the authoritative authored spline.

## Requirements

- **FR-001**: The left sidebar MUST expose a tabbed Capture category containing Capture Automation and Camera Path tabs.
- **FR-002**: Users MUST be able to add, select, delete, and retime camera key points from the active world camera.
- **FR-003**: Playback MUST evaluate position, target direction, and field of view over the authored duration and update the viewer camera.
- **FR-003a**: The active path MUST be optionally visible in the 3D world as a batched editor overlay with key markers and target guides.
- **FR-004**: Playback MUST validate the path map binding against the active map before starting.
- **FR-005**: The existing capture queue and video recorder MUST remain the capture implementation used by path actions.
- **FR-006**: Path documents MUST include map name, client build, duration, interpolation intent, and ordered key data.
- **FR-007**: The viewer MUST support importing readable M2 camera tracks and exporting a camera-only native M2 representation.
- **FR-008**: The authored JSON and `.m2.json` sidecar MUST preserve camera, target, FOV, roll, timing, map, and build data. The native classic camera-only M2 MUST preserve the interoperable position, target, and roll tracks; its static binary FOV is the first-key baseline because the classic camera layout has no animated FOV track.
- **FR-009**: Focused tests MUST cover interpolation, map binding, and native M2 round-trip camera data.

## Key Entities

- **Camera path**: A map-bound authored sequence with ordered camera and target keyframes.
- **Camera keyframe**: Time, position, target, field of view, and roll values for one path point.
- **Capture automation session**: Existing still/video queue state driven by a path without replacing its output route.
- **M2 camera asset**: A readable native camera track imported into the authored path representation.

## Success Criteria

- **SC-001**: A user can reach Capture Automation and Camera Path from the left sidebar in one click each.
- **SC-002**: A three-key path starts, runs, and stops without changing the active map or losing the authored keys.
- **SC-002a**: The user can see the authored spline and key/target markers over the active world before recording.
- **SC-003**: Save/reload preserves all authored keys and map/build metadata.
- **SC-004**: A native M2 exported by the viewer is readable by the existing M2 reader and exposes the same camera key timing within the chosen sampling resolution.
- **SC-005**: Path-driven video uses the existing capture output and does not require a second capture pipeline.

## Assumptions

- The active map name and selected client build are the binding authority; a native M2 file alone does not contain a reliable map identity.
- The project JSON is the lossless authored representation; native M2 export is an interoperability artifact.
- Camera roll is preserved in data/export even though the current free camera has no roll axis; position, target, and field of view are applied during playback.
