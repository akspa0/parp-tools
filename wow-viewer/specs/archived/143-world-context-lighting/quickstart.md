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

The current bounded implementation proof is narrower: the pure `AreaDisplayTextResolver` tests pass,
the LIT coordinate/source tests pass, and the isolated viewer build passes. The DBCD-backed AreaTable
fixture and real-client `SubzoneText` proof remain open until the next validation slice.

The LIT spatial contract is now explicit: list-header positions are fixed-point client XZY values.
Decode them by dividing by 36, swap the file Y/Z components into semantic game XYZ, then convert to
renderer coordinates with `renderer=(mapOrigin-gameY, mapOrigin-gameX, gameZ)`. The viewer's LIT
tooltip reports raw XZY, decoded WoW coordinates, and renderer coordinates separately. The same
conversion is used by LIT minimap markers and camera focus; source/build proof is automated, while
real-client marker placement remains user-owned.

LIT source selection is map-folder based rather than filename based. The viewer keeps the known
`lights.lit`, `areatest.lit`, and `light.lit` probes, then adds every `.lit` file directly under
`World\\<map>` or `World\\Maps\\<map>` from the active data-source index. The Lighting page and LIT
investigation panel expose the discovered files in a switcher; selecting one reparses that profile,
including when the currently selected profile failed to parse.

When the active map has no usable map-scoped Light DBC profile—or no Light DBC provider is available—
the viewer loads the first available LIT profile automatically and enables the LIT lighting/fog
override by default. The user can still turn the override off; LIT marker visualization remains
explicitly opt-in. If no LIT profile exists either, the status reports that the automatic fallback
has no source rather than claiming lighting parity.

The pre-alpha version-2 `areatest.lit` compatibility proof can be reproduced with the inspect tool
against the approved 0.5.3 client root:

```powershell
dotnet wow-viewer/tools/inspect/WowViewer.Tool.Inspect/bin/Debug/net10.0/WowViewer.Tool.Inspect.dll `
  lit profile `
  --archive-root H:\\053-client `
  --virtual-path World\\Maps\\Azeroth\\areatest.lit
```

Expected decoded shape: version `0x00000002`, raw light count `-1`, track count `9`, group stride
`0x1484`, selected light `Global Light`, and a primary `Partial` group. This is library/tool proof
against the named client build; viewer appearance and lighting parity remain user-run validation.

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
