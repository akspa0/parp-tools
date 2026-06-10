# 0.9.1.3810 Profile Field Map (Implementation-Ready)

## Objective
Translate binary findings into concrete profile fields for code implementation.

---

## 1) `AdtProfile_091_3810`

```text
ProfileId: AdtProfile_091_3810
BuildRange: [0.9.1.3810, 0.9.1.3810]

RootChunkPolicy:
  RequireStrictTokenOrder: true
  UseMhdrOffsetsOnly: true

McinPolicy:
  EntrySize: 0x10
  OffsetFieldOffset: 0x00

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
    LayerCount: 0x0C
    DoodadRefCount: 0x10
    OfsMCVT: 0x14
    OfsMCNR: 0x18
    OfsMCLY: 0x1C
    OfsMCRF: 0x20
    OfsMCAL: 0x24
    OfsMCSH: 0x2C
    MapObjRefCount: 0x38
    OfsMCSE: 0x58
    OfsMCLQ: 0x60

MclqPolicy:
  LayerStride: 0x324
  SampleStride: 0x08
  HeightLaneOffset: 0x04
  TileFlagsOffset: 0x290
  FlowBlockPolicy: DualFlowSupported

PlacementPolicy:
  MddfRecordSize: 0x24
  ModfRecordSize: 0x40
  NameIdIndirectionMode: NameIdToXidToStringBlock

Mh2oPolicy:
  Enabled: false
  DetectionMode: None
```

### Mandatory parser guardrails
- Reject MCIN entries that are out-of-file before dereference.
- Reject MCNK where `off + 8 + size` exceeds file bounds (no truncation parse).
- Skip invalid placement records cheaply; increment diagnostics counters.

---

## 2) `WmoProfile_091_3810`

```text
ProfileId: WmoProfile_091_3810
BuildRange: [0.9.1.3810, 0.9.1.3810]

RootChunkPolicy:
  RequiredRootChunks:
    - MVER(version=0x11)
    - MOHD
    - MOTX
    - MOMT
    - MOGN
    - MOGI
    - MOSB
    - MOPV
    - MOPT
    - MOPR
    - MOVV
    - MOVB
    - MOLT
    - MODS
    - MODN
    - MODD
    - MFOG
  OptionalRootChunkGates:
    - MCVP (optional trailing)

GroupChunkPolicy:
  RequiredGroupChunks:
    - MOPY (elemSize=0x02)
    - MOVI (elemSize=0x02)
    - MOVT (elemSize=0x0C)
    - MONR (elemSize=0x0C)
    - MOTV (elemSize=0x08)
    - MOBA (elemSize=0x18)
  OptionalGroupChunkGates:
    - MOLR: flags & 0x00000200
    - MODR: flags & 0x00000800
    - MOBN/MOBR: flags & 0x00000001
    - MPBV/MPBP/MPBI/MPBG: flags & 0x00000400
    - MOCV: flags & 0x00000004
    - MLIQ: flags & 0x00001000
    - MORI/MORB: flags & 0x00020000

LiquidPolicy:
  MLIQInterpretationProfile:
    SampleStride: 0x08
    ModeBitMask: firstMaskByte & 0x04
    DecodeModes:
      - UInt32-lane mode
      - UInt16+UInt16+UInt32 mode
```

### Current confidence
- Root order/record divisors: high.
- Group chunk gates/order: high.
- MLIQ header field naming semantics: medium-high.

---

## 3) `MdxProfile_091_3810`

```text
ProfileId: MdxProfile_091_3810
BuildRange: [0.9.1.3810, 0.9.1.3810]

GeometryPolicy:
  RequiresMdlxMagic: true
  GeosetSectionSeek: GEOS
  GeosetHardFailIfMissing: false

MaterialPolicy:
  TextureSectionSeek: TEXS
  TextureRecordSize: 0x10C
  TextureSectionSizeStrict: true

AnimationPolicy:
  ModelGlobalsSeek: MODL
  UsesModelFlagByteAt: +0x174
  CompressionRotationPolicy: provisional

TexturePolicy:
  Section: TEXS
  RecordFieldSemantics: partial
  ReplaceableUvWrapPolicy: provisional

EffectPolicy:
  LightSectionSeek: LITE
  CameraSectionSeek: CAMS
  RibbonSectionReaderPresent: true
```

### Upgrade criteria to non-provisional
- Extract explicit MDX chunk requirement order from top-level binary dispatcher.
- Confirm sequence/keyframe compression handling fields and required chunks.

---

## 4) Registry Dispatch Rules

```text
ResolveAdtProfile(build):
  if build == 0.9.1.3810 -> AdtProfile_091_3810
  else if build.major == 0 && build.minor == 9 -> AdtProfile_090x_Unknown
  else -> existing family rules

ResolveWmoProfile(build):
  if build == 0.9.1.3810 -> WmoProfile_091_3810
  else if build.major == 0 && build.minor == 9 -> WmoProfile_090x_Unknown

ResolveMdxProfile(build):
  if build == 0.9.1.3810 -> MdxProfile_091_3810
  else if build.major == 0 && build.minor == 9 -> MdxProfile_090x_Unknown
```

---

## 5) Diagnostics Contract for this Profile

Emit per file/tile:
- `InvalidChunkSignatureCount`
- `InvalidChunkSizeCount`
- `MissingRequiredChunkCount`
- `UnknownFieldUsageCount`
- `UnsupportedProfileFallbackCount`

Log key:
- `build=0.9.1.3810`
- `profileId`
- `filePath`
- `chunkFamily`

---

## 6) Open Items (explicit)

1. WMO `MLIQ` header field naming at `+0xF0..+0x10C` (material/type/depth naming) — pending consumer-path proof.
2. Full MDX chunk-order contract for 0.9.1 top-level dispatcher — pending dedicated parser walk.
3. Validate if any map subset in 0.9.1 still exposes MH2O-like signatures in non-primary paths.
