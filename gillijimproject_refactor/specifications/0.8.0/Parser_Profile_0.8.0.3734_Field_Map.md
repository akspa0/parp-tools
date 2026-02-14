# 0.8.0.3734 Profile Field Map (Implementation-Ready)

## Objective
Translate `0.8.0.3734` Ghidra findings into concrete profile fields usable in parser code.

---

## 1) `AdtProfile_080_3734`

```text
ProfileId: AdtProfile_080_3734
BuildRange: [0.8.0.3734, 0.8.0.3734]

RootChunkPolicy:
  RequireStrictTokenOrder: true
  UseMhdrOffsetsOnly: true

McinPolicy:
  EntrySize: unresolved in this pass
  OffsetFieldOffset: unresolved in this pass

McnkPolicy:
  RequiredSubchunks:
    - MCVT
    - MCNR
    - MCLY
    - MCRF
    - MCSH
    - MCAL
    - MCLQ
    - MCSE
  HeaderFieldMap:
    OfsMCVT: 0x14
    OfsMCNR: 0x18
    OfsMCLY: 0x1C
    OfsMCRF: 0x20
    OfsMCAL: 0x24
    OfsMCSH: 0x2C
    OfsMCSE: 0x58
    OfsMCLQ: 0x60

MclqPolicy:
  LayerStride: 0x2D4
  SampleStride: 0x08
  SampleBaseOffset: 0x08
  TileFlagsOffset: 0x290
  TailScalarOffset: 0x2D0
  LayerSlotMaskBits: [0x04, 0x08, 0x10, 0x20]

PlacementPolicy:
  MddfRecordSize: 0x24
  ModfRecordSize: 0x40
  NameIdIndirectionMode: NameIdToXidToStringBlock (not contradicted in this pass)

Mh2oPolicy:
  Enabled: false (no proof in this pass)
  DetectionMode: None
```

### Guardrails to enforce
- Validate `MVER` -> `MHDR` order and root token checks from MHDR-derived offsets.
- Hard-fail required MCNK subchunk token mismatches in strict mode.
- Enforce MCLQ fixed slot count (max 4) and fixed layer stride `0x2D4`.
- Keep placement size divisibility checks for `MDDF=0x24`, `MODF=0x40`.

---

## 2) `WmoProfile_080_3734`

```text
ProfileId: WmoProfile_080_3734
BuildRange: [0.8.0.3734, 0.8.0.3734]

RootChunkPolicy:
  RequiredRootChunks:
    - MVER(version=0x10)
    - MOHD
    - MOTX
    - MOMT
    - MOGN
    - MOGI
    - MOPV
    - MOPT
    - MOPR
    - MOLT
    - MODS
    - MODN
    - MODD
    - MFOG
  OptionalRootChunkGates:
    - MCVP (optional trailing)

RootRecordDivisors:
  MOMT: 0x40
  MOGI: 0x20
  MOPV: 0x0C
  MOPT: 0x14
  MOPR: 0x08
  MOLT: 0x30
  MODS: 0x20
  MODD: 0x28
  MFOG: 0x30
  MCVP: 0x10 (optional)

GroupChunkPolicy:
  RequiredGroupChunks:
    - MVER(version=0x10)
    - MOGP
    - MOPY
    - MOVI
    - MOVT
    - MONR
    - MOTV
    - MOBA
  GroupRecordDivisors:
    MOPY: 0x04
    MOVI: 0x02
    MOVT: 0x0C
    MONR: 0x0C
    MOTV: 0x08
    MOBA: 0x20
  OptionalGroupChunkGates:
    - MOLR: flags & 0x00000200
    - MODR: flags & 0x00000800
    - MOBN/MOBR: flags & 0x00000001
    - MPBV/MPBP/MPBI/MPBG: flags & 0x00000400
    - MOCV: flags & 0x00000004
    - MLIQ: flags & 0x00001000

LiquidPolicy:
  MLIQInterpretationProfile:
    SampleStride: 0x08
    HeaderWordsRead: param[2..9]
    SampleBaseOffset: +0x26 bytes from MLIQ chunk start
    SecondaryMaskBase: sampleBase + (xVerts * yVerts * 8)
```

### Guardrails to enforce
- Enforce strict root and required group chunk order.
- Apply optional chunk parsing only when corresponding group flags are present.
- Validate `MLIQ` sample/mask spans against chunk bounds before pointer use.

---

## 3) `MdxProfile_080_3734_Provisional`

```text
ProfileId: MdxProfile_080_3734_Provisional
BuildRange: [0.8.0.3734, 0.8.0.3734]

GeometryPolicy:
  RequiresMdlxMagic: true
  GeosetSectionSeek: GEOS
  GeosetRequiredSubsequence:
    - VRTX
    - NRMS
    - UVAS
    - PTYP
    - PCNT
    - PVTX
    - GNDX
    - MTGC
    - MATS
    - BIDX
    - BWGT
  EnforceNormalsMatchVertices: true
  ExpectedUvasSetCount: 1

MaterialPolicy:
  TextureSectionSeek: TEXS
  TextureRecordSize: 0x10C
  TextureSectionSizeStrict: true
  ExpectedTextureCount: 1

AnimationPolicy:
  CompressionRotationPolicy: provisional
  TopLevelChunkOrder: unresolved in this pass

TexturePolicy:
  Section: TEXS
  RecordSize: 0x10C
  ReplaceableUvWrapPolicy: provisional
```

### Guardrails to enforce
- Hard-fail on missing `MDLX` magic in strict binary profile mode.
- Enforce exact `TEXS` size-to-record relation and legacy count expectation (`1`) for this build path.
- Enforce geoset subsequence and vertex/normal count consistency checks.

---

## 4) Registry dispatch rules

```text
ResolveAdtProfile(build):
  if build == 0.8.0.3734 -> AdtProfile_080_3734
  else if build.major == 0 && build.minor == 8 -> AdtProfile_080x_Unknown

ResolveWmoProfile(build):
  if build == 0.8.0.3734 -> WmoProfile_080_3734
  else if build.major == 0 && build.minor == 8 -> WmoProfile_080x_Unknown

ResolveMdxProfile(build):
  if build == 0.8.0.3734 -> MdxProfile_080_3734_Provisional
  else if build.major == 0 && build.minor == 8 -> MdxProfile_080x_Unknown
```

---

## 5) Diagnostics contract for this build

Emit:
- `InvalidChunkSignatureCount`
- `InvalidChunkSizeCount`
- `MissingRequiredChunkCount`
- `UnknownFieldUsageCount`
- `UnsupportedProfileFallbackCount`

Log context:
- `build=0.8.0.3734`
- `profileId`
- `filePath`
- `chunkFamily=ADT|WMO|MDX`
