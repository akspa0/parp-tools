# Research: Camera Capture Paths

- Existing capture automation is implemented in `ViewerApp_CaptureAutomation.cs`: shot-point JSON, still capture queue, ffmpeg video recording, and startup capture arguments already exist.
- The existing UI exposes that route only through `Tools > Utilities > Capture`; the modern left sidebar currently owns world overview/file discovery and has no Capture category.
- `M2ModelReader` and `M2TrackSampler` already read camera position, target, roll, and modern field-of-view tracks. `M2CameraPathOverlayBuilder` proves sampled camera-only M2 visualization, but no world playback or writer exists.
- The authored project document is required because native M2 has no reliable map/build identity and the viewer's spline intent is richer than a linear sampled track. Native M2 export will therefore be an interoperability artifact with a sidecar.
- The current free camera has no roll axis, so playback applies position, target, and FOV while retaining roll in the project/native track data. The existing capture queue can consume path keys as still captures, and the existing ffmpeg recorder consumes the continuous path video action.
