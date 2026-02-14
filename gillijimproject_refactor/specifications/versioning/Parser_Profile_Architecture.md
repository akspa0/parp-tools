# Parser Profile Architecture (Build-Level Dispatch)

## Goal
Define a build-level parser architecture so ADT/WMO/MDX readers can diverge safely per build, especially in unstable eras (`0.8+`).

---

## 1) Core Principles

1. Build-level dispatch, not expansion-level dispatch.
2. Stable baseline profile: `0.6.x/0.7.x`.
3. Post-0.7 profiles are isolated until proven schema-compatible.
4. Unknown/unstable fields must be surfaced via diagnostics; never silently coerced.

---

## 2) Profile Registry

Introduce a registry keyed by parsed build number (`major.minor.patch.build`):

```text
FormatProfileRegistry
  - ResolveAdtProfile(build)
  - ResolveWmoProfile(build)
  - ResolveMdxProfile(build)
```

Resolution behavior:
- Exact-build profile if present.
- Build-range profile if explicitly defined.
- Otherwise: fallback profile with strict fail-safe parsing + high-visibility warnings.

---

## 3) ADT Profile Contract

```text
IAdtProfile
  BuildRange
  RootChunkPolicy
    - RequireStrictTokenOrder
    - UseMhdrOffsetsOnly
  McinPolicy
    - EntrySize
    - OffsetFieldOffset
  McnkPolicy
    - RequiredSubchunks
    - HeaderFieldMap
  MclqPolicy
    - LayerStride
    - SampleStride
    - HeightLaneOffset
    - TileFlagsOffset
    - FlowBlockPolicy
  PlacementPolicy
    - MddfRecordSize
    - ModfRecordSize
    - NameIdIndirectionMode
  Mh2oPolicy
    - Enabled
    - DetectionMode
```

Implementation targets:
- `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`

---

## 4) WMO Profile Contract

```text
IWmoProfile
  BuildRange
  RootChunkPolicy
    - RequiredRootChunks
    - OptionalRootChunkGates
  GroupChunkPolicy
    - RequiredGroupChunks
    - OptionalGroupChunkGates
  PlacementPolicy
    - MODD/MODF interpretation profile
  LiquidPolicy
    - MLIQ interpretation profile
```

Implementation targets:
- `src/WoWMapConverter/WoWMapConverter.Core/Converters/WmoV14ToV17Converter.cs`
- `src/MdxViewer/Rendering/WmoRenderer.cs`

---

## 5) MDX Profile Contract

```text
IMdxProfile
  BuildRange
  GeometryPolicy
    - GEOS/vertex layout assumptions
  MaterialPolicy
    - layer/filter/flags interpretation
  AnimationPolicy
    - sequence/keyframe chunk requirements
    - compression/rotation format policy
  TexturePolicy
    - replaceable/UV/wrap interpretation
```

Implementation targets:
- `src/MdxViewer/Formats/Mdx/*`
- `src/MdxViewer/Rendering/MdxRenderer.cs`
- `src/MdxViewer/Rendering/MdxAnimator.cs`

---

## 6) Parser Execution Pattern

All parsers follow this pattern:

1. Resolve profile from build metadata.
2. Validate required chunk contracts up front.
3. Parse with profile field maps/record sizes.
4. On violation:
   - increment per-tile/per-file diagnostics,
   - skip unsafe section,
   - continue with bounded degradation where possible.
5. Emit one summary warning per file/tile family (avoid log spam).

---

## 7) Diagnostics Contract

Minimum structured counters:
- `InvalidChunkSignatureCount`
- `InvalidChunkSizeCount`
- `MissingRequiredChunkCount`
- `UnknownFieldUsageCount`
- `UnsupportedProfileFallbackCount`

Log format should include:
- build id
- profile id
- file path
- chunk family
- counter summary

---

## 8) Initial Profile Set

Phase 1:
- `AdtProfile_060_070_Baseline`
- `AdtProfile_080_3734`
- `AdtProfile_091_3810`
- `AdtProfile_090x_Unknown` (strict, warning-heavy until validated)

Phase 2:
- WMO/MDX equivalents for same build families.

---

## 9) Migration Plan (Code)

1. Add profile types + registry (new files under `src/MdxViewer/Formats/Profiles/`).
2. Refactor `StandardTerrainAdapter` methods to accept `IAdtProfile`.
3. Move hardcoded constants (record sizes/offset assumptions) into profiles.
4. Add profile id + build id to terrain logs.
5. Repeat for WMO and MDX pipelines.

---

## 10) Definition of Done

- Build resolver deterministically maps all test builds to explicit profiles.
- 0.6/0.7 maps parse with baseline profile unchanged.
- 0.8/0.9 maps no longer use implicit assumptions from baseline profile.
- Unknown post-0.7 fields are visible in diagnostics and actionable.
