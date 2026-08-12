# Tasks: Camera Capture Paths

- [x] T001 Record existing capture and M2 camera seams in `spec.md` and `research.md`.
- [x] T002 Add the reusable map-bound M2-style path model/evaluator in `src/core/WowViewer.Core.Runtime/M2/M2CameraPath.cs`.
- [x] T003 Add camera-only native M2 export in `src/core/WowViewer.Core.Runtime/M2/M2CameraPathWriter.cs`.
- [x] T004 Add focused interpolation/import/export tests in `tests/WowViewer.Core.Tests/M2CameraPathTests.cs`.
- [x] T005 Add the Utilities Capture panel tabs, path editor, and batched 3D overlay in `src/viewer/WoWViewer/ViewerApp_CameraPaths.cs` and `ViewerApp_Sidebars.cs`; keep capture out of the left sidebar.
- [x] T006 Drive playback and path capture through the existing `OnUpdate`, capture queue, and video recorder.
- [x] T007 Build, update memory-bank continuity, and commit the bounded slice.
- [x] T008 Add bounded path-footprint preload for path video and queued key captures using the existing TerrainManager and WorldAssetManager queues; keep the lease capture-scoped and outside full-map residency.
- [ ] T009 User-run real-client validation: warm a multi-key path, confirm the ready gate precedes recording, and compare frame stability/object pending counts before and after preload.
- [x] T010 Add loaded-client M2/MDX camera import through the file-browser selection and shared format readers, with camera/sequence/sample controls.
- [x] T011 Add a timeline playhead and playhead-based key authoring/retiming controls.
- [x] T012 Add persisted opt-in terrain-height and conservative resident-WMO-bounds collision resolution for path scrubbing/playback.
- [x] T013 Add MDX camera sampling proof, build/test the viewer slice, and update continuity docs.
- [ ] T014 User-run real-client validation: select built-in client cameras, verify imported tracks, scrub/play with collision enabled, and confirm capture output follows the constrained path.
