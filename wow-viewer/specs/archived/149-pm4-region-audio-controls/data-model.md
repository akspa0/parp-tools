# Data Model: PM4 Region Navigation and Audio Trigger Controls

## Pm4RegionNavigationItem

One presentation-ready region assembled from the existing resident PM4 overlay objects.

| Field | Type | Rule |
|---|---|---|
| `RegionId` | `uint` | Stable decoded `MSHD.Field04` identity. |
| `ObjectCount` | `int` | Number of resident PM4 object parts in the region. |
| `TileCount` | `int` | Number of distinct resident tile coordinates represented. |
| `SurfaceCount` | `int` | Sum of decoded object surface counts; never inferred from a match. |
| `BoundsMin/BoundsMax` | `Vector3` | Union of transformed object bounds; finite when `IsAvailable`. |
| `Center` | `Vector3` | Safe center derived from bounds, not an external placement. |
| `IsEmptyStub` | `bool` | Identifies the known empty-stub policy such as region 1. |
| `Availability` | enum | `Unavailable`, `Resident`, `Pending`, or `Stale`. |
| `IsSelected` | `bool` | Viewer selection state only; not persisted. |

Rows are sorted by `RegionId` and de-duplicated by that ID. Empty stubs are either shown as disabled
rows or filtered by one explicit policy; they never yield a focus request.

## Pm4RegionFocusRequest

An ephemeral request passed from the region list to the authoritative camera path.

| Field | Type | Rule |
|---|---|---|
| `RegionId` | `uint` | Must match the selected navigation item. |
| `TargetCenter` | `Vector3` | Must be finite and derived from decoded bounds. |
| `TargetBounds` | min/max vectors | Used to choose framing distance and vertical offset. |
| `SuggestedPosition` | `Vector3` | Finite, clamped viewer camera position. |
| `ResidencyHint` | tile set/summary | Advisory only; normal AOI streaming remains authoritative. |

Invalid or stale requests are rejected with a visible status message and do not mutate the camera.

## AudioTriggerInstance

One controllable world trigger in the bounded interactive set. The source may be a legacy MCNK
environment/liquid record, an MCSE record, or an area/music catalog record.

| Field | Type | Rule |
|---|---|---|
| `Kind` | `AudioTriggerKind` | `Mcnk`, `Mcse`, `ZoneMusic`, or `AreaAmbience`. |
| `InstanceKey` | typed string/record | MCNK uses tile/chunk plus source flag/liquid variant; MCSE uses tile/chunk/record index; area uses area context, day/night, and selected entry identity. |
| `SourceContext` | record | MCNK flags/liquid identity plus tile/chunk, MCSE raw/local plus normalized position, or packed Zone/SubZone/area text. |
| `SoundPointId/SoundNameId` | `uint` | Preserved when the source has those fields. |
| `SoundEntryId` | `uint?` | Resolved catalog identity when proven. |
| `McnkFlags` | `uint?` | Raw MCNK environmental/liquid flags when the source is MCNK-derived. |
| `LiquidTypeId` | `ushort?` | Raw liquid type identity when the decoded source provides it. |
| `LiquidFamily` | enum/string | Derived water/ocean/magma/slime or another proven client family; never substitutes for raw identity. |
| `RawPosition` | `Vector3?` | Original source coordinates, retained for diagnosis. |
| `NormalizedWorldPosition` | `Vector3?` | Renderer-space position after applying the owning tile/chunk origin and axis convention. |
| `ZoneMusicRowId` | `int?` | `AreaTable.ZoneMusic` reference, not a SoundEntries identity. |
| `DaySoundEntryId/NightSoundEntryId` | `int?` | Values read from the selected `ZoneMusic.Sounds[2]` row. |
| `DayMusicFile/NightMusicFile` | `string?` | Authored ZoneMusic file fields, retained as metadata and never used to bypass SoundEntries resolution. |
| `CandidateVirtualPaths` | list | Existing DBC/DB2-declared candidates only. |
| `ResolutionState` | enum | Read/decode/backend stages from existing diagnostics. |
| `Enablement` | enum | `DisabledByDefault`, `Enabled`, or `BlockedByMaster`. |
| `PlaybackState` | enum | `Stopped`, `Ready`, `Active`, or explicit failure state. |

The `InstanceKey` prevents two emitters sharing one SoundEntries row from sharing enablement or source
lifecycle accidentally.

## MCNK and MCSE spatial/audio inputs

The terrain adapter already exposes `TerrainChunkData.McnkFlags` and `LiquidChunkData.Type` for the
rendering path. Audio candidate production must consume the same decoded chunk records rather than
re-reading MCNK bytes in the UI or inventing a parallel liquid parser.

For legacy Alpha data, MCNK flag bits identify environmental/liquid families and inline MCLQ data
identifies the liquid-bearing chunk. A candidate may be diagnostic-only when the client-backed mapping
from that flag/liquid identity to a SoundEntries row has not been proven. The row still appears in the
bounded trigger list and records the unresolved mapping reason.

For MCSE, the `RawPosition` is the decoded record value. In the observed 0.5.3 path it is local to the
owning MCNK/tile and MUST NOT be passed directly to the global `MapOrigin - raw` transform. The runtime
derives `NormalizedWorldPosition` by composing the owning tile/chunk origin with the local offset using
the existing renderer axis convention. Range checks, OpenAL source positions, and diagnostics that say
`WorldPosition` use the normalized value; raw/local coordinates remain visible beside it.

If both MCNK and MCSE data exist for a later build, both source records remain independently inspectable
until a client-proven identity establishes a safe merge. Matching by SoundEntries ID, position alone, or
liquid family alone is not sufficient to collapse them.

For area music, `ZoneMusicRowId` and the selected day/night SoundEntries ID are separate fields. The
reader must resolve the row before the runtime asks the SoundEntries catalog for a file. The
`AreaMIDIAmbiences` row remains a separate MIDI/DLS binding; its DLS file belongs to that row and is not
selected by matching filenames or SoundEntries IDs. Underwater ambience selection is an explicit input,
not an implicit alias for the normal row.

## AudioTriggerEnablement

Runtime-owned control state, reset at world/session initialization and map/client replacement.

```text
WorldTriggersEnabled: false                 # master gate, default
EnabledInstances: Set<AudioTriggerInstanceKey>
```

Effective start permission is:

```text
WorldTriggersEnabled && EnabledInstances.Contains(instanceKey)
```

Mute and gain affect output level but do not grant start permission. Explicit SoundEntries preview has
its own source and does not enter `EnabledInstances`.

## State transitions

```text
new world/session
  -> all world triggers DisabledByDefault

user enables master + row
  -> eligible trigger may resolve/start once

resident/in-range update while enabled
  -> update existing source; never duplicate

user disables row OR master
  -> stop owned source; retain diagnostic; block restart

tile unload / area change
  -> stop/release sources according to existing lifecycle; row may return disabled

map/client replacement / dispose
  -> clear sources, diagnostics, and enablement; next session starts default-off
```
