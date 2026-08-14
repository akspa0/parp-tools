# Quickstart: Artifact World Simulator Runtime

## Developer proof

From PowerShell 7:

```powershell
dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --no-restore
dotnet build I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/WoWViewer.CrossPlatform.csproj -c Debug --no-restore
```

Then, after the focused slice is integrated:

```powershell
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
dotnet test I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
```

## User-owned client proof

1. Open a configured client root in the viewer and record the client build/profile shown by the
   application. Do not paste a machine-local root into source or portable documentation.
2. Load a map with known MCSE and/or area music data.
3. Open the audio inspection surface and capture one table/snapshot showing tile/chunk, raw XYZ,
   world XYZ, SoundEntries/path, source/read/decode/backend stages, distance, and terminal state.
4. Test one known WAV/OGG/MP3 resource and one MIDI/DLS resource if present. Report whether the
   failure is path, read, decode, backend, range, mute, or unsupported format.
5. For performance, run the same fixed camera/path in a dense WMO and a terrain-heavy area. Record
   residency reasons, WMO-internal doodad attribution, draw calls, and FPS. This is runtime proof,
   not something a build can establish.

## Stop conditions

- Stop before changing MCSE coordinate conversion if raw/transformed diagnostics are not both
  visible.
- Stop before adding a MIDI/DLS library until its native dependencies, soundbank handling, and
  distribution/licensing constraints are documented.
- Stop before replacing tile/doodad selection until a fixed capture shows which lease or culling
  stage releases the wrong near-field content.
