# 223-T609 video-capture release hardening — 2026-09-08

## Scope

This receipt records the source and package-preflight repair for the Playback & Capture video
route. It does **not** close 223-T609: recording a real viewer scene, stopping it, and opening the
resulting MP4/MOV remain operator-owned runtime evidence.

## Files changed

- `src/viewer/WoWViewer/Capture/VideoEncoderExecutableResolver.cs`
- `src/viewer/WoWViewer/Capture/ffmpeg/win-x64/README.md`
- `src/viewer/WoWViewer/ViewerApp_CaptureAutomation.cs`
- `src/viewer/WoWViewer/WoWViewer.csproj`
- `tests/WowViewer.Core.Tests/VideoEncoderExecutableResolverTests.cs`
- `tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj`

## Root cause and repair

The viewer used the bare executable name `ffmpeg`; no `ffmpeg.exe` was present beside its Debug or
Release executable and no publish target copied one. Video capture therefore depended on the
developer machine's PATH. This workstation happened to resolve it through the Chocolatey shim,
which is not a release contract.

The repair resolves a compatible `ffmpeg.exe` beside `ParpToolsWoWViewer.exe` before a configured
path or PATH fallback. The project copies an operator-supplied x64 binary from
`Capture/ffmpeg/win-x64/ffmpeg.exe` to build and publish roots when present; the binary remains
ignored and is deliberately not supplied by this repository. The README requires the release
operator to choose the source and ship its licence/notices. Capture Automation now displays the
resolved source, has **Verify ffmpeg** to check `libx264`, accepts quoted configured paths, and
reports a missing encoder or output-directory failure without escaping the render loop.

## Verification

| Criterion | Evidence | Result |
|---|---|---|
| App-local encoder wins over PATH; explicit paths are normalized | `VideoEncoderExecutableResolverTests` | 6/6 passed |
| Windows viewer compiles with the new capture route and optional build targets | `dotnet build wow-viewer/WowViewer.slnx -c Debug --no-restore` | exit 0; 281 existing warnings; 0 errors |
| The configured codec can run on this workstation | `ffmpeg -hide_banner -h encoder=libx264` | exit 0; reports `Encoder libx264` |
| Release package does not silently depend on this workstation's PATH | `dotnet publish src/viewer/WoWViewer/WoWViewer.csproj -c Release --no-restore -p:PublishSingleFile=false` exited 0 and emitted the explicit missing-encoder warning; no `ffmpeg.exe` was present in its publish root | source/package gate passed; binary still operator-supplied |
| Displayed controls start, stop, and produce a playable video | Not run: this requires a shipped binary plus a real rendered viewer scene and output-file playback | **open operator gate** |

## Required operator release witness

1. Place the chosen, licence-reviewed x64 `ffmpeg.exe` at
   `src/viewer/WoWViewer/Capture/ffmpeg/win-x64/ffmpeg.exe`.
2. Publish the viewer and confirm `ffmpeg.exe` is beside `ParpToolsWoWViewer.exe`.
3. In **Archaeology > Playback & Capture > Capture Automation**, click **Verify ffmpeg** and
   record the success message.
4. Record and stop at least several seconds with UI and scene-only modes, then open both generated
   files. Repeat **Play + Video** for a two-key camera path.
5. Record the exact encoder version, encoder licence/notices included in the package, client root,
   build, map, output paths, and playback results before checking 223-T609 or 223-T610.
