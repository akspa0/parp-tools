# Quickstart: Spec 149 Validation

This document is for implementation follow-through. It does not claim that the real client or audio
backend has been proven.

## Focused source checks

From `I:/parp/parp-tools`:

```powershell
dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter 'FullyQualifiedName~Pm4Region|FullyQualifiedName~AudioTrigger|FullyQualifiedName~WorldAudioRuntime' --no-restore --verbosity minimal
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --no-restore --verbosity minimal
```

The exact filter may be narrowed to the final test class names after implementation. A passing build or
focused test is source proof only; it is not visual, streaming, or audible proof.

## User-run PM4 proof

1. Configure the viewer with the approved client root and record the client build/fingerprint.
2. Load a PM4-backed map with at least two non-empty regions.
3. Open PM4 Workbench, verify the region rows and decoded totals, and double-click each row.
4. Confirm the camera frames the selected region, the normal AOI/residency path updates, and no whole-map
   load is requested.
5. Hover PM4 geometry and verify the tooltip contains decoded facts only; no matching/correlation UI or
   fields remain.

## User-run area overlay proof

1. Load a configured map with at least two resident AreaTable values and leave `Show Area Boundaries`
   disabled; confirm no area lines, pins, or labels are submitted.
2. Enable `Show Area Boundaries` in the spatial investigation panel. Confirm resolved `Zone:` and
   `Subzone:` labels appear over resident chunk footprints with distinct colors.
3. Stream or move far enough to change tile residency. Confirm stale footprint cells disappear and the
   overlay remains bounded to resident chunks rather than filling unloaded map space.
4. If unresolved area chunks are reported, confirm they remain diagnostic-only and do not receive guessed
   names or boundaries. Record the configured client root, build, and proof level.

## User-run audio proof

1. Load a configured 0.5.3 map/session containing MCNK liquid/environment data and record that MCSE is
   absent or not populated. Open the audio inspector and confirm the MCNK-derived rows are present.
2. Load a later-build map/session containing resident MCSE records and an applicable area audio
   assignment. Confirm MCSE rows are additive rather than replacing the MCNK rows.
3. Inspect at least one MCSE diagnostic and confirm raw/local coordinates are shown separately from the
   normalized tile/chunk-aware world position; verify the latter, not the raw value, drives range.
4. Confirm every listed world-trigger control is off and active world sources are zero.
5. Move through a liquid/environment emitter and area boundary while controls remain off; confirm
   silence and visible disabled diagnostics for MCNK, MCSE, and area rows.
6. Enable one supported trigger, verify one source/diagnostic transition, then disable it and verify it
   stops without restarting on subsequent camera updates.
7. Exercise unresolved liquid-to-SoundEntries mappings, unsupported formats, and unavailable-backend
   rows without expecting guessed playback or invented asset pairing.

## Deferred game-mode gate

Do not use this feature's region-focus camera as evidence for player-height, walking/running speed,
jumping, collision, or game-mode behavior. Those require a separate feature and acceptance plan.
