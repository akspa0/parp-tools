# Quickstart: World Context And Lighting Parity

This document is the validation handoff after implementation. It is intentionally PowerShell-ready.
The agent may run focused builds/tests, but the user runs real-client viewer and GPU captures.

## Focused checks

From `I:/parp/parp-tools/wow-viewer`:

```powershell
dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --no-restore
dotnet build I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug --no-restore -p:BaseOutputPath=I:/parp/parp-tools/output/build/world-context-lighting/ -v minimal
```

The focused tests must cover at least: standard and Alpha MCNK IDs, coordinate-to-chunk selection,
map mismatch diagnostics, DBD logical-column selection, WMO evidence unavailable fallback, stable
WMO candidate ordering, camera-head snapshot reuse, and lighting fallback attribution.

## User-run real-client matrix

Use a configured approved client root such as `H:\CLIENTS\...` or another explicitly selected root;
do not bake the path into code. The exact profile-render command should be copied from the current
viewer capture tooling after implementation because its options are profile-specific.

For each selected early, 1.x/3.x, and 4.x client:

1. Capture an outdoor camera point over a known ADT chunk and record raw area ID, map ID,
   `ZoneText`, `SubzoneText`, table build/locale, and coordinate source.
2. Capture a WMO entrance, interior, overlapping group, and exit. Record WMO identity, group,
   WMOAreaID evidence, selected source, fallback reason, and AreaName transitions.
3. Toggle camera `PlayerHead` and explicit `Museum` modes and verify the saved/restored eye state.
4. Capture one WMO interior, one WMO exterior, and one model-heavy scene with lighting diagnostics.
5. Compare p95 frame-stage timings to the flat baseline; do not use a visual impression as proof.

## Acceptance evidence

The run is acceptable only when valid ADT IDs resolve, `SubzoneText` follows the expected leaf-or-zone
fallback, invalid values explain themselves, WMO context is deterministic, camera consumers share one
frame state, lighting inputs are attributable, and the
performance report stays within the spec budget or documents an approved exception. Native OpenGL
crashes, process exits before terminal JSON, or missing diagnostics are failures rather than passing
fallbacks.
