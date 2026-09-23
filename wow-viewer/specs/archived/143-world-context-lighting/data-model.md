# Data Model: World Context And Lighting Parity

The model is runtime-only and profile-scoped. Values are immutable snapshots where possible so the
status bar, visibility, fog, and renderer consume one same-frame answer.

## `WorldContextSnapshot`

| Field | Type | Meaning |
|---|---|---|
| `FrameId` | `long` | Render/frame sequence that produced the snapshot. |
| `Camera` | `CameraHeadState` | Eye position, orientation, mode, and explicit offset. |
| `Map` | `WorldMapIdentity` | Active map ID/name and build/profile identity. |
| `Terrain` | `TerrainAreaContext` | ADT source, raw area ID, chunk/tile, table result, and reason. |
| `Wmo` | `WmoAreaContext` | WMO/group candidates, containment, raw ID evidence, and result. |
| `SelectedArea` | `AreaResolution` | Deterministic WMO-first or ADT-fallback selection. |
| `Display` | `AreaDisplayText` | Native-style `ZoneText` and `SubzoneText` results. |
| `Diagnostics` | `WorldContextDiagnostics` | Counts, confidence, source fields, and unresolved reasons. |

## `TerrainAreaContext`

Contains the camera sample coordinates, coordinate-space/profile name, resolved tile/chunk key,
raw MCNK area ID, AreaTable lookup key, map-validation result, and `ResolutionReason`.
`ResolutionReason` is a closed set such as `Resolved`, `NoTerrainChunk`, `MissingAreaId`,
`AreaRowMissing`, `MapMismatch`, `MissingLocalizedName`, or `MalformedSource`.

## `WmoAreaContext`

Contains WMO identity/path or asset key, group index/name, containment method, candidate count,
distance/bounds or portal confidence as applicable, `WmoAreaIdEvidence`, `AreaResolution`, and a
fallback reason. Candidate ordering is deterministic: stronger containment/source confidence first,
then smaller volume/distance, then stable asset/group key.

## `WmoAreaIdEvidence`

| Field | Type | Meaning |
|---|---|---|
| `RawValue` | `uint?` | Decoded client value, if a profile-proven field exists. |
| `SourceChunk` | string | Exact chunk/table/source identifier. |
| `SourceOffset` | `int?` | Payload offset when applicable. |
| `Profile` | string | Build/layout profile that authorizes the decode. |
| `Confidence` | enum | `Observed`, `Validated`, `UnavailableForProfile`, `Malformed`. |
| `EvidenceNote` | string | Human-readable bounded explanation. |

No WMO area value is valid without a non-empty source and profile. `UnavailableForProfile` is a
valid result and causes ADT fallback.

## `AreaResolution`

Contains raw ID, canonical row key, localized display name, parent chain, table build/locale,
logical ID/name/map/parent column names, and `ResolutionReason`. A row may be found while map
validation fails; the result must retain both facts.

## `AreaDisplayText`

| Field | Type | Meaning |
|---|---|---|
| `ZoneText` | `string?` | Resolved parent/zone display name. |
| `SubzoneText` | `string?` | Resolved leaf/subzone display name, or the deterministic zone fallback when no leaf exists. |
| `Source` | enum | `Wmo`, `Adt`, `ParentFallback`, or `Unresolved`. |
| `Reason` | enum | Explicit resolution/fallback reason. |

`SubzoneText` is a UI-facing derived result. It is not a replacement for raw IDs or provenance and
must never be used to infer WMOAreaID or ADT area ownership.

## `CameraHeadState`

Contains `Position`, `Yaw`, `Pitch`, `Mode` (`PlayerHead`, `Museum`, or existing explicit mode),
`HeadOffset`, and a serialization version. `EyePosition` is derived from the explicit mode and
offset once per frame. Hidden offsets are invalid.

## `LightingSelection`

Contains asset kind (`Wmo` or `M2`), active client build/profile, directional input, ambient input,
vertex/baked/lightmap inputs, local-light set or absence reason, fog input, shader/effect route,
fallback state, and diagnostics. Every contribution identifies its source (`SceneLight`, `WmoRoot`,
`WmoGroup`, `VertexColor`, `Lightmap`, `M2Track`, `ProfileDefault`, or equivalent).

## Invariants

1. A displayed non-Unknown name has a raw ID, table source, and successful row lookup.
2. `SubzoneText` and `ZoneText` are derived from the selected context and never become area-ID inputs.
3. WMO context never silently replaces ADT context; the selected source is explicit.
4. Camera context and render view use the same `CameraHeadState.FrameId`.
5. A shader route cannot claim BLS/native parity without evidence for the active profile.
6. Unknown, unavailable, malformed, and map-mismatch results remain distinguishable in diagnostics.
7. Snapshot creation performs no full-map scan and does not parse DBC/WMO data on the render loop.
