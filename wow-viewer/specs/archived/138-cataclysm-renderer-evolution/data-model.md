# Spec 138 Data Model — Source Profiles and Terrain Capabilities

## `ClientSourceProfile`

Represents one configured client or remote source selection. It is runtime input and must not
contain a repository-specific default path.

| Field | Type | Rule |
|---|---|---|
| `profileId` | string | Stable local identifier; unique within a run |
| `eraBand` | enum | `alpha-mpq`, `classic-mpq`, `casc-6x`, `casc-7x`, `casc-modern`, or `unknown` |
| `buildIdentity` | object | Product, build number, executable/version evidence, and fingerprint |
| `sourceKind` | enum | `loose`, `mpq`, `casc-casclib`, `casc-tactsharp`, or `remote-cdn` |
| `configuredRoot` | string or null | Runtime path only; never serialized into portable source code |
| `product` | string or null | CASC/TACT product code when applicable |
| `locale` | string or null | Locale selected for the source |
| `installTag` | string or null | Install-tag selection when the source supports it |
| `adapterId` | string | Concrete adapter selected after probing |
| `adapterVersion` | string | Dependency or pinned commit/package version |
| `listfileSource` | object | Path/commit plus verified/community status |
| `capabilities` | `SourceCapabilities` | Proven capabilities only |
| `probe` | `SourceProbeResult` | Probe status, errors, and content hash |

## `SourceCapabilities`

Capabilities describe what the selected source can actually do, not what the library claims in
general.

- `canReadByVirtualPath`
- `canReadByFileDataId`
- `canEnumerateKnownFiles`
- `canReadLocalStorage`
- `canReadCdnStructuredStorage`
- `canReadOnline`
- `supportsLocaleSelection`
- `supportsInstallTagSelection`
- `supportsEncryptedProducts`
- `supportsMpqWrappers`

## `TerrainFormatProfile`

Build-scoped format and ownership facts consumed by terrain loading:

- `adtOwnership`: `monolithic`, `split`, `mixed`, or `unknown`
- `heightChunk`: `MCVT` or another verified build-specific source
- `normalChunk`: `MCNR` or `derived`
- `vertexColorChunk`: `MCCV` or `absent`
- `vertexLightChunk`: `MCLV` or `absent`
- `shadowChunk`: `MCSH` or `absent`
- `explicitUvChunk`: `MCTV` or `absent`
- `materialChunk`: `MCMT` or `absent`
- `terrainLayerCapacity`: integer with evidence location
- `liquidSource`: `MCLQ`, `MH2O`, split liquid file, or `absent`
- `objectSource`: monolithic object chunks, split object file, or `absent`
- `minimapResolution`: 256, 512, 1024, or `unknown`

Unknown fields remain unknown. They are not mapped to a guessed later format.

## `TerrainCapabilitySet`

The renderer-facing capability set is derived from `TerrainFormatProfile` and source probes. It
controls optional paths such as baked vertex lighting, baked terrain shadows, split-file reads,
liquids, material IDs, and object placement. Every optional path has an explicit fallback to the
base terrain path.

## `SourceProbeResult`

Contains:

- `status`: `passed`, `failed`, or `partial`
- `testedVirtualPaths`
- `testedFileDataIds`
- `missingCapabilities`
- `errors`
- `contentProbeSha256`
- `timestampUtc`

A `failed` result cannot be used for terrain rendering. A `partial` result can only be used for
capabilities explicitly marked as passed.

## Relationships

```text
ClientSourceProfile
  ├── SourceCapabilities
  ├── SourceProbeResult
  └── TerrainFormatProfile
        └── TerrainCapabilitySet -> terrain reader / compositor / renderer
```
