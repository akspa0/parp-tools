# Research: PM4 Region Navigation and Audio Trigger Controls

## Existing ownership and evidence

### PM4 region data

- `src/viewer/WoWViewer/Terrain/WorldScene.cs` already stores resident PM4 objects in
  `_pm4TileObjects` and indexes individual objects in `_pm4ObjectLookup`.
- `Pm4OverlayObject.MshdRegionId` is already carried into viewer object/debug/export state. The current
  `GetPm4VisibleOverlaySummary()` aggregates region IDs, object counts, tile counts, and type buckets.
- `Pm4SelectedObjectRegionInfo` and `TryGetSelectedPm4RegionInfo()` already prove that region peers can
  be derived from the selected object's region without external asset matching. The missing navigation
  data is a region-wide finite bounds/surface-total snapshot and an explicit focus operation.
- `src/core/WowViewer.Core.PM4/Models/Pm4RegionObjectModels.cs` defines region/object/sub-object records
  from the decoded PM4 grouping work. The viewer should reuse the established region identity and
  coordinate transforms rather than create a second PM4 parser.
- The PM4 workbench in `ViewerApp_Pm4Utilities.cs` currently includes Overlay, Selection, and
  Correlation tabs. Selection also exposes WMO matching, shape search, saved match rows, match details,
  and match suggestions. `ViewerApp.cs`, `ViewerApp_Sidebars.cs`, `ViewerApp_Workspaces.cs`, and
  `Workbench/WorkbenchNavigator.cs` retain additional tab/state references.
- The viewport hover path in `ViewerApp.cs` calls `DrawHoveredPm4MatchCandidates`; removing only the
  visible text would leave matching work active on hover. The existing scene-outliner tooltip in
  `ViewerApp_Pm4Utilities.cs` already has factual CK24/type/object/MSLK/MSHD/surface/bounds fields and
  is the better shape for the replacement tooltip.
- `ViewerApp_Pm4Utilities.cs` already has `FocusCameraOnBounds` and existing “Frame All Parts” bounds
  union logic. Region navigation should reuse those camera semantics rather than add a second framing
  algorithm in `ViewerApp.cs`.

### PM4 retirement boundary

The requested cleanup is a user-facing UI retirement, not an unconditional deletion of every historical
matching class. `Pm4WmoGroupMatchService`, object-match models, and research reports must first be
searched for non-UI callers. If they are orphaned, their viewer state, persistence, and dead services
can be removed in the same implementation phase; if an offline/export tool still owns them, leave that
owner intact and remove only the workbench presentation.

### Audio trigger data and current behavior

- `src/core/WowViewer.Core/Audio/AudioTriggerDiagnostic.cs` already distinguishes `Mcse`, `ZoneMusic`,
  and `AreaAmbience` and has terminal states for unresolved, missing, decode, backend, range, muted,
  ready, active, and stopped decisions.
- `AlphaTerrainAdapter.LoadTileWithPlacements()` reads the Alpha 0.5.3 MCNK sound records into
  `TileLoadResult.SoundEmitters` through `AdtMcseReader.ReadAlpha053Mcnk()`. The current hand-off
  forwards only this MCSE list; `TerrainChunkData.McnkFlags` and `LiquidChunkData.Type` are decoded for
  rendering but are not projected into audio candidates.
- `WorldAudioRuntime` owns resident tile emitter registration, diagnostic refresh, OpenAL source
  lifecycle, area music resolution, and explicit SoundEntries preview. `Update()` currently calls
  `TryStartEmitter()` for every newly in-range emitter and `UpdateAreaMusic()` can start a looping area
  source when the active area changes.
- `ViewerApp_Audio.cs` currently renders only resident MCSE diagnostics and exposes master/emitter gain,
  mute through other UI surfaces, and deliberate SoundEntries preview. It has no per-trigger enablement
  state and its current audio wording still describes automatic MCSE playback.
- `WorldScene` already forwards audio runtime state and preserves the packed Zone/SubZone lookup used by
  the status bar. The new control API must be forwarded through this facade rather than having ImGui own
  source state.
- `AlphaTerrainAdapter.ConvertSoundPosition()` and the standard adapter's equivalent apply the global
  `MapOrigin - raw` conversion directly. The screenshot evidence shows Alpha MCSE raw values near
  `(3, -3, 62.5)` being reported as approximately `(17069.7, 17063.7, 62.5)` for tile `(31,31)` and
  therefore compared against the wrong world frame. The audio contract must compose the MCSE local
  position with the owning tile/chunk origin before range checks and preserve both coordinate forms.
- `LiquidChunkData` documents Alpha 0.5.3 liquid family selection from MCNK flag bits `2..5`; standard
  loading also retains raw MCNK flags and resolves MH2O liquid IDs into a basic family. This is enough
  to establish the producer seam, but client-proven SoundEntries selection for each legacy flag/liquid
  family remains an open mapping task rather than a reason to omit the rows.
- Area identity is build-specific and must stay that way at every consumer. Alpha 0.5.x MCNK values
  address `AreaTable` through packed `AreaNumber` words (`high16=Zone`, `low16=SubZone`) and
  `ParentAreaNum`; 3.3.5+ MCNK values address the table through direct `ID`/`ParentAreaID` values.
  The status bar, resident area overlay, and area-audio catalog now receive the same explicit layout
  contract, so modern direct IDs cannot be captured by an Alpha-style `AreaNumber` alias.
- The music audit found that `AreaTable.MIDIAmbience` is joined correctly to `AreaMIDIAmbiences` by
  `AlphaAreaAudioCatalogReader`, preserving day/night sequence paths, shared DLS path, and volume. The
  runtime deliberately reports those MIDI/DLS bindings as unsupported rather than guessing a PCM path.
- `AreaTable.ZoneMusic` is not yet resolved correctly. `libs/wowdev/WoWDBDefs/definitions/ZoneMusic.dbd`
  defines the Alpha 0.5.3 row with `MusicFile[2]`, scheduler fields, and `Sounds<32>[2]` typed as
  `SoundEntries::ID`; the PDB/SQL evidence shows row 1 selecting SoundEntries 2523 for day and 2533
  for night. The current runtime instead assigns `binding.Area.ZoneMusicId` directly to
  `soundEntryId`, so it is treating the ZoneMusic row ID as a SoundEntries ID.
- The current runtime also does not select `MIDIAmbienceUnderwater`, does not load a `ZoneMusic` row
  model, and reduces ZoneMusic playback to one looping OpenAL source. This is not equivalent to the
  client's `ZoneMusicIdle`/`PlayMusic` scheduling path until the row indirection and scheduler contract
  are implemented or explicitly reported as unsupported.

### Legacy MCNK liquid/audio implementation evidence

- The shared `McnkFlagDecoder` is the single raw-flag owner: `0x04` is river/water, `0x08` is ocean,
  `0x10` is magma, and `0x20` is slime. The viewer now projects a resident MCNK liquid candidate
  from those flags or from decoded MCLQ/MH2O liquid presence even when the map contains no MCSE rows.
- The Alpha MCLQ path now preserves the 81 packed vertex words/heights and 64 tile flags. The prior
  Alpha adapter flattened the payload to a uniform surface, which was inconsistent with the existing
  repository extractors' 8-byte vertex records and is a likely cause of the reported 0.5.3 liquid
  rendering corruption. This is repository-lineage evidence, not a new native-client proof claim.
- `SoundWaterType.dbd` is available for both `0.5.3.3368` and `3.3.5.12340` with `SoundType`,
  `SoundSubtype`, and a typed `SoundEntries::ID` field. The runtime now loads the exact active-build
  table and resolves MCNK candidates through `(liquid family, subtype)`; it never invents a
  `SoundEntries` ID. The current `0x04 -> subtype 4`, `0x08 -> subtype 8`, and magma/slime/water
  `-> subtype 0` selection is a conservative implementation inference from the decoded flag names
  and DBD subtype values. It remains visible as diagnostic metadata until direct native callback
  evidence proves the mapping.
- MCNK candidates are resident and inspectable but world-trigger playback is default-off. Missing
  `SoundWaterType` rows remain unresolved rather than starting a guessed resource; explicit
  SoundEntries preview remains independent.

### Streaming/WMO smoothness guard

- Camera rotation no longer invalidates the terrain residency lease. It is a frustum concern, not a
  reason to rebuild the detailed/retained tile sets, so mouse-look does not churn tile placement and
  WMO admission.
- Terrain unload now has one extra camera-centered hysteresis ring, capped by the existing maximum
  retained radius. The parsed tile cache was already persistent; retaining the GPU mesh/placement
  lease across one boundary is the part that prevents a neighbor/interior from disappearing while its
  replacement is still loading.
- WMO portal traversal remains fail-open for correctness: groups whose transformed bounds are in the
  camera frustum are unioned back into the renderer's runtime-visible group set, and the scene graph
  does not apply a second portal cull. This keeps connected/interior groups from becoming spotty when
  portal evidence or clip volumes disagree.

## Decisions

1. **Aggregate MSHD regions from resident decoded objects.** The first implementation lists the current
   camera/residency-visible PM4 set, and states that scope in the UI. It does not load every PM4 tile
   merely to populate the panel. Region rows are deterministic by `MshdRegionId` and include union bounds
   plus surface totals derived from existing object records. A region is not presented as a proven external
   object identity or coordinate frame.
2. **Focus with the existing viewer camera helper.** The region model supplies a finite center/bounds
   target; `ViewerApp_Pm4Utilities.FocusCameraOnBounds` applies the established safe framing semantics
   and the normal terrain/PM4 AOI update runs. This keeps camera mutation out of core PM4 code and avoids
   a guessed placement transform.
3. **Make matching absence explicit.** Remove correlation/matching controls and wording from PM4
   presentation surfaces. Preserve non-UI research code only when a caller audit demonstrates a current
   owner.
4. **Separate trigger enablement from mute/gain.** Add a default-off master world-trigger gate and
   stable per-instance enablement keyed by emitter instance or area-trigger identity. Diagnostics are
   still built while disabled, with a user-disabled terminal state or equivalent explicit detail.
5. **Gate both spatial emitters and area music.** The automatic start points are separate today, so both
   must consult the same world-trigger policy. Explicit SoundEntries preview remains independent.
6. **Treat legacy terrain audio as a first-class producer.** MCNK flags plus decoded liquid presence/type
   produce inspectable environmental/water candidates even when the 0.5.3 map has no MCSE records.
   Later-build MCSE data is additive; it does not replace or suppress those candidates without proven
   client identity.
7. **Normalize before admission.** MCSE raw/local positions remain in diagnostics, but only the
   tile/chunk-normalized renderer position may drive distance, `InRange`, and OpenAL placement.
8. **Keep audio enumeration bounded.** List all resident MCNK-derived candidates, resident MCSE instances,
   and the applicable current-area trigger represented by proven metadata. Do not pretend that an
   unloaded map-wide trigger catalog is available.
9. **Keep music table identities separate.** Resolve `AreaTable.ZoneMusic` as
   `AreaTable reference -> ZoneMusic row -> day/night SoundEntries IDs`; never use the ZoneMusic row ID
   as a SoundEntries ID. Preserve the MIDI/DLS row-level pairing and keep underwater selection explicit.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Region rows become stale after PM4 AOI changes | State the current visible/resident scope, refresh the snapshot from `WorldScene`, and clear selected region when its source data disappears. |
| Large region bounds produce a bad camera placement | Validate finite bounds and centralize framing/clamping in the viewer camera path. |
| Matching code has hidden export/research consumers | Perform a caller audit before deletion; retire only viewer state/UI when ownership is uncertain. |
| Per-trigger audio state leaks across map replacement | Reset the master gate and instance map when the runtime/session is configured or disposed. |
| A disabled trigger is mistaken for unresolved data | Preserve diagnostic stage fields and add explicit user-disabled control state. |
| ZoneMusic and MCSE rows use different identities | Use a typed trigger kind plus instance key; never key only on SoundEntries ID. |
| MCSE coordinates remain in the raw MCNK frame | Require a normalization test using tile/chunk origin and expose raw plus normalized values in diagnostics. |
| 0.5.3 has no MCSE rows | Build MCNK environmental/liquid candidates from decoded terrain chunks and keep unresolved mappings visible. |
| Later MCSE and MCNK rows are double-counted or silently collapsed | Keep source-kind identity separate until a client-proven merge key exists. |
| AreaTable.ZoneMusic is mistaken for a SoundEntries ID | Add a build-aware ZoneMusic reader/model and test row 1 -> SoundEntries 2523/2533 before runtime playback. |
| Underwater area ambience silently uses normal ambience | Carry an explicit underwater state/selection input; otherwise report the normal-only limitation. |

## Open proof gates (not blockers for planning)

- The exact real-client fixture and configured client build for multi-region camera proof remain user-run.
- Audible proof remains gated by the current OpenAL/backend and supported file formats.
- MIDI/DLS playback and native MCSE callback installation remain unsupported/unproven and are not
  inferred from this UI/control work.
