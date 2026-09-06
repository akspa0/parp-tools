# Spec 223 Phase 6 validation and acceptance remediation — 2026-09-06

## Scope

T601–T608: active taxi simulation, interactive camera playback, terrain Inspector/pinning,
four-tab navigation, utility migration, shared widgets, profile preservation and user guide.
The remediation adds source-only corrections for fog-editor ownership, WMO-only global placement
admission, and Playback & Capture discoverability. No format reader/parser behavior changes.

## Automated checks

- `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter FullyQualifiedName~TaxiRideSimulationPolicyTests` — **passed: 2/2**. Confirms an active ride route bypasses taxi presentation/selection filters while inactive routes retain their existing gates.
- `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter FullyQualifiedName~CameraPathBindingPolicyTests` — **passed: 3/3**. Confirms path/build formatting normalization permits equivalent identities and that genuinely changed or missing playback provenance is rejected.
- `dotnet build wow-viewer/WowViewer.slnx -c Debug --no-restore` — **passed: 0 errors, 305 existing warnings**. The warnings include the existing `Snappier` `NU1903` advisory and pre-existing analyzer/compiler warnings; no new build error was introduced.
- `dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug --no-restore` after the final Quick-profile preservation adjustment — **passed: 0 errors, 265 existing warnings**.
- `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --no-restore --filter "FullyQualifiedName~CameraHudTransformTests|FullyQualifiedName~WdtSummaryReaderTests|FullyQualifiedName~WmoAdmissionTallyTests|FullyQualifiedName~WorldObjectVisibilityCollectorTests"` — **passed: 32/32** (3 camera-space HUD contract tests + 29 WDT/WMO policy tests). Covers camera-space transform/projection contracts, WDT WMO-based classification, and generic WMO visibility/admission policy, not the GL-backed `WorldScene` render path.
- `dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug --no-restore -p:OutputPath=bin/remediation/` — **passed: 0 errors**. An isolated output directory was required because the running viewer locked the normal Debug output. Existing warnings, including `NU1903`, remain.

### Source review

- Taxi pose simulation now has a dedicated active-ride route override (`TaxiRideSimulationPolicy`), distinct from route visibility/selection; it remains independent of mount model availability.
- Interactive path playback begins immediately and streams progressively. Video and queued captures still use bounded preload readiness gates. Map/build identity uses a common normalization policy and remains checked during playback/preload.
- Inspector terrain context is source-scoped and resolves resident chunk data at action/draw time. A ground click pins only when no selectable object wins the click; pin expiry on eviction/source change falls through to hover/camera/world overview.
- The right workbench exposes only Quick, Inspector, Editor and Archaeology. Scene and Utilities remain compatibility routing identifiers, with Scene mapped to Inspector pages and Utilities mapped to Quick's shared utility dispatcher. Quick is retained when changing workspace profiles.

### Acceptance-remediation source review

- All accessible Fog Start/Fog End editors now call the shared authoritative control, which reads and
  writes `WorldScene`'s user override rather than repopulating sliders from render-time
  `TerrainLighting` values.
- WMO-only adapters move their WDT global MODF instances to the external WMO collector, the same
  normal admission path that does not require terrain-tile residency. Existing WDT/MWMO/MODF readers
  remain untouched.
- Archaeology > Playback & Capture now renders the canonical Capture Automation / Camera Path panel
  and presents **Apply playback to next capture**. Scene-only recording no longer changes Tab/UI-chrome
  state automatically; a recording stops archeology playback only if that recording started it.
- `CameraHudRig3D` is decorative geometry only. It now hides with Tab, but it has no panel surface,
  hit testing, or pointer routing; Spec 212 tasks 212-T101 through 212-T403 remain open.

Source/build/test evidence does not prove real-client motion, visual layout, WMO rendering, slider
input, ffmpeg invocation/output, capture timing, audio, or interactive 3D HUD behavior.

## Operator acceptance feedback and required walkthrough

The operator ran an acceptance walkthrough and reported these failures before the source remediation:

- Fog Start and Fog End rubberbanded instead of retaining adjustments.
- WMO-only maps did not render their global WMO.
- The workbench remained unacceptable as a 2D UI and did not provide the requested interactive 3D HUD.
- Archaeology > Playback & Capture showed text referring to video, playback, and apply behavior without
  exposing the corresponding record, path-playback, or apply actions.

The following is therefore a **retest**, not an unrun first walkthrough. It remains operator-owned.

Launch from PowerShell 7 with the chosen client configured in the viewer:

```powershell
dotnet run --project I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug
```

1. Record client root, exact build and map. Load a terrain-backed world. Without selecting a
   model, inspect the camera/hovered chunk and area name. Click bare terrain, move the mouse and
   camera, and verify the pin remains the target. Frame/copy it, unload its tile, and change map:
   stale data must never remain selected. Repeat on standalone terrain if used.
2. Verify layer names/flags/effects/alpha, holes/elevation, shadow/MCCV, liquid details and tile
   placement counts. Toggle each MCNK overlay; with Impassable enabled, check diagonal markers.
3. Search and frame a WMO/MDX in Inspector > Placements. Check LOD & Budget. Select a WMO and
   change its doodad set; return to terrain. Confirm all context actions still work.
4. Set Fog End and Fog Start from every exposed route (Quick, Settings, terrain controls and lighting).
   Drag both values while global/DBC/LIT lighting changes; values must hold instead of rubberbanding.
   Then set time of day, camera speed, render quality and audio in Quick. Switch Viewer, Editor and
   Archaeology profiles while Quick remains active. Repeat at compact sidebar width; tabs, help
   popups and action buttons must remain reachable.
5. Walk every Phase 6 inventory replacement, especially Minimap, Log, Performance, Taxi,
   Capture, Asset Catalog, Runtime Stats, Lighting and Audio. Test menu/keyboard shortcuts and
   restore settings saved with an old Scene/Utilities/Experimental destination. Repeat relevant
   menu routes in the legacy non-tab shell.
6. Select a taxi route and attach the ride camera. Clear route selection, select another route,
   hide route lines/mount actors and verify the active ride keeps moving. Test delayed/missing
   mount assets, detach, camera-path takeover and switching worlds.
7. Add two separated camera keys. Play during streaming: time must advance immediately. Start
   video/queued stills with preload enabled: capture must wait for residency. Stop/cancel and
   verify residency releases. Equivalent identity formatting may play; a different map/build
   must be rejected. Change worlds during playback and pending warmup.
8. Load a known WMO-only map and confirm that its global WMO appears, can be framed, and remains
   subject to ordinary frustum/object filters. Repeat a terrain-backed map regression check.
9. In Archaeology > Playback & Capture, confirm **Camera Paths & Video Capture** is reachable.
   In Capture Automation, configure a known working `ffmpeg` executable, record several seconds with
   and without UI, stop, and open the resulting output file. In Camera Path, verify **Play**,
   **Play + Video**, and **Stop**. Verify **Apply playback to next capture** appears and affects only
   the next queued capture. Record client root, build, map, ffmpeg version, configured output path,
   and output-file result.
10. Confirm the camera rig hides with Tab but is visually and behaviorally decorative. Do not treat it
    as acceptance for the interactive spatial UI; that work is owned by Spec 212.

These checks require the operator. Build/tests provide no visual, FPS, WMO, slider-input, ffmpeg,
audible, capture, or interactive-spatial-UI proof.
