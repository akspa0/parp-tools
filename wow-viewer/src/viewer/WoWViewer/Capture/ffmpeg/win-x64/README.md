# Optional Windows video encoder

Video recording uses an external `ffmpeg.exe` with the `libx264` encoder. The viewer first looks
for that executable beside `ParpToolsWoWViewer.exe`, then falls back to the path configured in
Capture Automation, then to `ffmpeg` on `PATH`.

For a distributable Windows build, place the selected, compatible x64 `ffmpeg.exe` in this
directory before building or publishing. The project copies it to the viewer's output root as
`ffmpeg.exe` when present. The repository intentionally does not contain a third-party encoder
binary: its source, licence obligations, notices, and redistribution terms must be chosen and
included by the release operator.

Run **Verify ffmpeg** in Archaeology > Playback & Capture > Capture Automation before release.
It verifies that the executable starts and exposes `libx264`; it does not replace the required
real-viewer recording and playback check.
