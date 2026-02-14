# 0.9.0.3807 Profile Field Map (Implementation-Ready)

## Objective
Translate `0.9.0.3807` Ghidra findings into concrete profile fields usable in parser code.

---

## 1) `AdtProfile_090_3807`

```text
ProfileId: AdtProfile_090_3807
BuildRange: [0.9.0.3807, 0.9.0.3807]

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
  Note: unresolved secondary-path proof still open
```

### Guardrails to enforce
- MCIN offset bounds checks before dereference.
- MCNK subchunk token hard-fail in strict profile mode.
- Placement count and record-size divisibility checks (`0x24`, `0x40`).
- MCLQ slot count limited to 4 in native-equivalent behavior.

---

## 2) `WmoProfile_090_3807`

```text
ProfileId: WmoProfile_090_3807
BuildRange: [0.9.0.3807, 0.9.0.3807]

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

RootRecordDivisors:
  MOMT: 0x40
  MOGI: 0x20
  MOPV: 0x0C
  MOPT: 0x14
  MOPR: 0x08
  MOVV: 0x0C
  MOVB: 0x04
  MOLT: 0x30
  MODS: 0x20
  MODD: 0x28
  MFOG: 0x30
  MCVP: 0x10 (optional)

GroupChunkPolicy:
  RequiredGroupChunks:
    - MVER(version=0x11)
    - MOGP
    - core streams implied by group setup chain
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
    HeaderWordsRead: param[2..9]
    SampleBaseOffset: +0x26 bytes from MLIQ chunk start
    SecondaryMaskBase: sampleBase + (dimA*dimB*8)
```

### Guardrails to enforce
- Strict root order check for all required chunks.
- Strict optional block sequence for each enabled gate.
- Validate `MLIQ` derived sample/mask extents against chunk size.

---

## 3) `MdxProfile_090_3807_Provisional`

```text
ProfileId: MdxProfile_090_3807_Provisional
BuildRange: [0.9.0.3807, 0.9.0.3807]

GeometryPolicy:
  RequiresMdlxMagic: true
  GeosetSectionSeek: GEOS
  GeosetHardFailIfMissing: false (provisional)

MaterialPolicy:
  MaterialSectionSeek: MTLS
  TextureSectionSeek: TEXS
  TextureRecordSize: 0x10C
  TextureSectionSizeStrict: true

AnimationPolicy:
  SequenceSectionSeek: SEQS
  GeosetAnimSectionSeek: GEOA
  PivotSectionSeek: PIVT
  HitShapeSectionSeek: HTST
  CompressionRotationPolicy: provisional

TexturePolicy:
  Section: TEXS
  RecordSize: 0x10C
  ReplaceableUvWrapPolicy: provisional

EffectPolicy:
  LightSectionSeek: LITE
  CameraSectionSeek: CAMS
  RibbonSectionSeek: RIBB
  AttachmentSectionSeek: ATCH
  ParticleSectionSeek: PRE2

CollisionPolicy:
  CollisionSectionSeek: CLID
```

### Dispatcher notes
- Top-level section dispatch function: `FUN_0042a6a0`.
- Two execution paths exist (feature-flag gated); both must remain profile-compatible.

---

## 4) Registry dispatch rules

```text
ResolveAdtProfile(build):
  if build == 0.9.0.3807 -> AdtProfile_090_3807
  else if build.major == 0 && build.minor == 9 -> AdtProfile_090x_Unknown

ResolveWmoProfile(build):
  if build == 0.9.0.3807 -> WmoProfile_090_3807
  else if build.major == 0 && build.minor == 9 -> WmoProfile_090x_Unknown

ResolveMdxProfile(build):
  if build == 0.9.0.3807 -> MdxProfile_090_3807_Provisional
  else if build.major == 0 && build.minor == 9 -> MdxProfile_090x_Unknown
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
- `build=0.9.0.3807`
- `profileId`
- `filePath`
- `chunkFamily=ADT|WMO|MDX`
