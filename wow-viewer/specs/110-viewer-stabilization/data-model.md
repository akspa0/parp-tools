# Data Model: Viewer Stabilization

## FogRange

| Field | Meaning | Validation |
|---|---|---|
| `Start` | Distance at which fog begins | finite, `0 <= Start < End` |
| `End` | Distance at which fog completes | finite, positive, at least minimum span beyond start |
| `Source` | `Fallback`, `LightingRecommendation`, or `UserOverride` | always displayed with active range |
| `Adjusted` | Indicates normalization changed the requested/reported values | supports concise diagnostic text |

**Transitions**:

1. Map load applies fallback/default range.
2. LIT/DBC evaluation supplies a recommendation.
3. A user override supersedes the recommendation until reset.
4. Any invalid candidate normalizes to a visible fallback before render submission.

## M2RenderCapability

| Field | Meaning |
|---|---|
| `SourceIdentity` | Asset path and detected format/version profile |
| `ReaderOutcome` | Native reader success or precise failure |
| `RenderDataOutcome` | Embedded division/skin/section availability |
| `RendererRoute` | Native M2 only, or no renderer with diagnostic |
| `Diagnostic` | Actionable reason for unsupported/failure state |

## LitMapInspectionState

| Field | Meaning | Validation |
|---|---|---|
| `MarkersVisible` | Whether LIT markers are drawn on minimap surfaces | opt-in; requests a lazy LIT load only when enabled |
| `SelectedLightIndex` | Selected entry in the loaded LIT source | `-1` or index in the current source |
| `Navigable` | Entry has a non-default finite renderer position | default/invalid entries remain listable but are not marked or focused |
| `Position` / `Extent` | Existing entry location and radius/dropoff shown to the user | read-only source data; no lighting mutation |

**Transitions**:

1. Enabling markers attempts the existing lazy LIT load.
2. Selecting a marker or list row updates the one scene-level selected index.
3. Double-clicking a navigable row frames a camera point above the existing LIT location.
4. Reloading the LIT source clears the selection; the overlay never writes lighting/fog state.

## SynthesizedMinimapExport

| Field | Meaning | Validation |
|---|---|---|
| `ClientRoot` | User-selected client source | existing directory; never persisted as a portable hard-coded path |
| `MapName` | Client map directory name | non-empty and has a readable terrain WDT |
| `TimeOfDay` | Canonical selected clock time | exact `HH:mm`; parsed from `HHmm`, `HH:mm`, or legacy decimal hours |
| `TimeOfDayHours` | Compatibility/projection form of selected time | finite in `[0, 24)`; derived from the exact minute and normalized to `[0, 1)` |
| `EmitTiles` / `EmitWholeMap` | Requested outputs | at least one target is true |
| `LightingSource` | `WhiteTopEdge` | attached to every manifest |
| `LightingEvidence` | `minimap_white_light_not_lit_data` | LIT colors/fog/native direction are excluded from minimap RGB |
| `TileResults` | One result per occupied tile | source coordinate, output path, error/skip reason |
| `StitchedBounds` | Inclusive tile-coordinate bounds for the combined image | emitted only when whole-map stitching succeeds |

**Transitions**:

1. Tools > Export prepopulates client root and map from the active viewer session when available.
2. The user starts the in-repository Harvest command after selecting at least one output target.
3. The command resolves white north/top-edge terrain lighting once, independent of LIT/native-world-light data.
4. Each readable tile is composed and saved independently; the optional combined image stitches only
   successful tile outputs and leaves missing coordinates transparent.
5. The final manifest records successes, skips, failures, provenance, and output dimensions.

## MinimapLightingProvenance

| Field | Meaning | Validation |
|---|---|---|
| `ContractVersion` | Version of the inference sidecar | present on every emitted record |
| `InferenceStatus` | `baked_tint_likely`, `baked_mcsh_likely`, combined, unclassified, or an explicit not-evaluated reason | never silently defaults to a lighting claim |
| `TintRgb` / `TintStrength` / `TintFit` | Robust authored-minimap vs neutral-terrain colour-ratio evidence | populated only with a complete terrain-material baseline |
| `McshDarkeningCorrelation` | Correlation of authored residual darkening with decoded MCSH | evidence only; MCSH remains an independent target |
| `EstimatedTimeOfDayHours` / `TimeOfDayConfidence` | Optional closest global-clear LIT chroma bucket | inference, never a historical-capture assertion |
| `TimeOfDayEvidence` / `TimeOfDayCandidateSource` | Why a bucket exists and which build-local LIT profile supplied candidates | explicit even when no bucket exists |
| `MtexTexturePayloadState` | `complete_name_aligned`, unavailable, or incomplete-not-serialized | no sidecar may shift a texture into another MTEX index |
| `MtexTextureFallbacks` | Original MTEX ID/path, decoded proxy path, and resolution kind | only a verified same-stem `_s.blp` RGB proxy; never material-parity proof |

**Transitions**:

1. Decode authored minimap RGB only when the client provides it.
2. Build a neutral material/MCAL baseline from every referenced decoded texture; incomplete texture
   tables emit an explicit not-evaluated status.
3. Infer tint and optional MCSH-baked evidence without modifying the raw MCSH target.
4. Compare tint chroma to build-local global-clear LIT samples; attach only a conservative candidate
   bucket with its evidence label.
5. Serialize the same sidecar in NPZ and raw V22 streams for downstream consumers.

## ToolSurface

| Field | Meaning |
|---|---|
| `MenuLabel` | User-visible item |
| `Owner` | Current viewer, Inspect, Converter, or removed |
| `Availability` | Ready, dependency missing, or intentionally absent |
| `Replacement` | Current supported destination when one exists |

## ConversionCapabilityProfile

| Field | Meaning |
|---|---|
| `SourceFormat` / `TargetFormat` | One explicit direction |
| `PreservedFeatures` | Data known to survive conversion |
| `KnownLoss` | Target-format or implementation limits |
| `EvidenceLevel` | Fixture-proven, real-client-proven, or unsupported |
| `RuntimeUse` | Always `ExportOnly` for M2→MDX |

## TerrainTextureFallbackResolution

| Field | Meaning |
|---|---|
| `TextureId` / `RequestedPath` | Authoritative MTEX identity that failed to decode |
| `ResolvedPath` | Successfully decoded RGB-proxy BLP; never rewrites MTEX |
| `ResolutionKind` | `specular_companion_rgb_proxy` or `related_diffuse_rgb_proxy` |

Related diffuse candidates are ordinary `.blp` names scanned across the loaded archive/listfile
catalog. Exact or strongly similar basenames rank before shared directory-theme tokens, so moved
assets can recover while unrelated names remain ineligible. The policy supplies candidates only;
the consumer must still prove decode success before recording a resolution.
