# Feature Specification: Camera Capture Paths

**Feature Branch**: `144-camera-capture-paths`
**Created**: 2026-08-11
**Status**: Implementing
**Input**: Restore capture automation as a stable Utilities panel and author/play back map-bound M2-style camera paths.

## User Scenarios

### User Story 1 - Author a camera path in the world (P1)

With a world loaded, the user can record the current camera as ordered key points, adjust their timing, and preview the resulting spline while the world remains the active scene.

**Independent Test**: Add three camera keys, press Play, and observe the camera move through all three points without leaving the loaded map.

### User Story 2 - Save and reopen a path (P1)

The user can save an authored path as a readable project document and as a native camera-only M2, then load it again with map/build provenance preserved.

**Independent Test**: Save a path, reload it, and verify the key count, map binding, duration, and camera positions are unchanged.

### User Story 3 - Drive capture automation from a path (P1)

The user can reach the existing still/video capture automation from Tools > Utilities > Capture, outside the left sidebar, and start playback with video recording or queue captures at authored keys.

**Independent Test**: Start path playback plus video recording, then stop both; the capture output is produced by the existing framebuffer/ffmpeg route and the camera path remains reusable.

### User Story 3a - Warm a capture path before recording (P1)

Before video capture or queued key stills begin, the user can ask the viewer to sample the authored path, retain the bounded terrain-tile footprint, and warm the existing MDX/WMO/doodad/material queues. Capture starts only after the pinned tiles and queued world assets have remained ready for a short stability window.

**Independent Test**: Enable path preload, start a path video, and verify the status reaches “Path preload ready” before the camera begins moving; moving through the path does not trigger first-use tile/object loads for the warmed footprint.

### User Story 4 - Reuse existing M2 camera assets on a map (P2)

The user can import a readable M2 camera asset, bind its sampled tracks to the current map, and play the imported path through the world.

**Independent Test**: Import a camera-only M2 with at least one camera track and verify that sampled keys appear and can be played on the matching active map.

### User Story 5 - Reuse cameras from the loaded client and keep shots inside the world (P1)

The user can select an M2 or MDX camera asset in the loaded client's file browser, import a chosen camera/sequence into the active map, scrub its timeline, and optionally constrain playback against loaded terrain height and resident WMO placement bounds.

**Independent Test**: Select a client camera asset, import it, scrub to a middle key, enable terrain/WMO collision, and play the path through the active scene without the camera entering loaded world bounds.

## Edge Cases

- Playback refuses a path whose map binding does not match the active world, instead of silently moving the camera in the wrong scene.
- A path with fewer than two keys can be saved and inspected but cannot be played as a moving spline.
- An M2 camera with no animated tracks imports as one static key.
- Missing or malformed path files leave the current path untouched and report a readable error.
- Native M2 export uses sampled linear keys; the project document remains the authoritative authored spline.

## Requirements

- **FR-001**: The viewer MUST expose a stable Capture panel outside the left sidebar, reachable from Tools > Utilities, containing Capture Automation and Camera Path tabs.
- **FR-002**: Users MUST be able to add, select, delete, and retime camera key points from the active world camera.
- **FR-003**: Playback MUST evaluate position, target direction, and field of view over the authored duration and update the viewer camera.
- **FR-003a**: The active path MUST be optionally visible in the 3D world as a batched editor overlay with key markers and target guides.
- **FR-004**: Playback MUST validate the path map binding against the active map before starting.
- **FR-005**: The existing capture queue and video recorder MUST remain the capture implementation used by path actions.
- **FR-006**: Path documents MUST include map name, client build, duration, interpolation intent, and ordered key data.
- **FR-007**: The viewer MUST support importing readable M2 camera tracks and exporting a camera-only native M2 representation.
- **FR-008**: The authored JSON and `.m2.json` sidecar MUST preserve camera, target, FOV, roll, timing, map, and build data. The native classic camera-only M2 MUST preserve the interoperable position, target, and roll tracks; its static binary FOV is the first-key baseline because the classic camera layout has no animated FOV track.
- **FR-009**: Focused tests MUST cover interpolation, map binding, and native M2 round-trip camera data.
- **FR-010**: Path-driven playback, video, and queued key captures MUST support an opt-in bounded preload lease that samples the path, pins only its available terrain tiles, closes gaps between samples in tile space, queues the placements in those tiles through the existing asset manager, and waits for terrain/object readiness before motion or capture starts.
- **FR-011**: Releasing or completing path capture MUST clear the preload lease so ordinary camera-driven AOI eviction resumes; preload MUST NOT imply full-map residency.
- **FR-012**: The Camera Path panel MUST import `.m2` and binary `.mdx` camera assets selected from the loaded client file browser through the active data source, without requiring extraction to a loose filesystem path.
- **FR-013**: Client camera import MUST expose camera index, sequence index, and bounded sample interval, and MUST use the existing M2/MDX readers and track samplers.
- **FR-014**: The path editor MUST provide a timeline playhead and MUST support opt-in collision resolution during scrubbing and playback. Terrain collision MUST use the loaded terrain heightfield; WMO collision MUST use conservative resident placement bounds and MUST be labeled as such.
- **FR-015**: The Camera Path panel MUST provide an opt-in keyboard-authoring mode. While enabled, WASD movement remains available and the user MUST be able to add, update, select, delete, retime the selected key, move the playhead, play/pause, save JSON, export native M2, and adjust camera roll without relying on mouse controls. The keymap MUST be visible in the panel and MUST not fire while a text field is being edited.
- **FR-016**: Camera-path JSON and native-M2 sidecars MUST serialize world-space `Position` and `Target` vector components as numeric values, not empty objects. Playback and free-camera view construction MUST apply the authored roll around the camera forward axis.
- **FR-017**: When importing a client M2 or MDX FlyBy camera, the viewer MUST resolve the matching `CinematicCamera.dbc` row by model path, derive its ADT tile with the active WoW map projection, and transform every local position and target key by the DBC origin and facing exactly once. If no matching row is available, the viewer MUST leave the decoded track unchanged and report that placement metadata was unavailable.

## Key Entities

- **Camera path**: A map-bound authored sequence with ordered camera and target keyframes.
- **Camera keyframe**: Time, position, target, field of view, and roll values for one path point.
- **Capture automation session**: Existing still/video queue state driven by a path without replacing its output route.
- **M2 camera asset**: A readable native camera track imported into the authored path representation.

## Success Criteria

- **SC-001**: A user can reach the Capture panel from Tools > Utilities and reach Capture Automation and Camera Path from its tabs without opening the left sidebar.
- **SC-002**: A three-key path starts, runs, and stops without changing the active map or losing the authored keys.
- **SC-002a**: The user can see the authored spline and key/target markers over the active world before recording.
- **SC-003**: Save/reload preserves all authored keys and map/build metadata.
- **SC-004**: A native M2 exported by the viewer is readable by the existing M2 reader and exposes the same camera key timing within the chosen sampling resolution.
- **SC-005**: Path-driven video uses the existing capture output and does not require a second capture pipeline.
- **SC-006**: With preload enabled, path playback/capture reports a bounded swept tile footprint and a ready gate before motion or recording; after playback/capture, the lease is released and the normal AOI stream remains active.
- **SC-007**: A selected client `.m2` or `.mdx` camera imports into the path editor with its decoded camera tracks and a usable timeline; no loose extraction is required.
- **SC-008**: With collision enabled, a path sample crossing loaded terrain is lifted above the terrain height, and a swept sample entering a resident WMO exterior-bounds volume from outside is stopped before the volume; paths already inside a WMO remain playable.
- **SC-009**: Importing the built-in Undead FlyBy resolves its `CinematicCamera.dbc` origin to ADT tile `(28,28)`, stores world-space keys, and does not double-translate the path on subsequent playback or save/load.

## Assumptions

- The active map name and selected client build are the binding authority; a native M2 file alone does not contain a reliable map identity.
- Classic FlyBy assets contain local camera coordinates; `CinematicCamera.dbc` is the authority for their map-space origin and facing. `CinematicSequences.dbc` links cinematic sequences to camera IDs but does not replace the camera-origin lookup.
- The project JSON is the lossless authored representation; native M2 export is an interoperability artifact.
- Camera roll is stored in degrees in authored JSON and sidecars, converted to the native track's radians at export, and applied around the camera forward axis during playback and view construction.
