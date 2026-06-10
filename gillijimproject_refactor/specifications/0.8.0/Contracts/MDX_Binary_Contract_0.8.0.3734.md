# MDX Binary Contract — 0.8.0.3734 (Ghidra)

## Scope
Binary model-loading contract evidence for WoW.exe build `0.8.0.3734`, focusing on geometry/material section rules that can break parsers.

---

## 1) Function Map

- `0x00422620` — model async completion path (`MDLX` gate + parse dispatch)
- `0x006bbd10` — world/model MDX section parser (`TEXS`, `GEOS`, geoset chunk chain)
- `0x0044e380` — geoset parse core (`VRTX`, `NRMS`, `UVAS` and count checks)
- `0x0044ea20` — geoset continuation (`GNDX`, `MTGC`, `MATS`, `BIDX`, `BWGT`)

Confidence: **High** for listed function roles.

---

## 2) Loader Entry and Magic Contract (M1)

From `FUN_00422620` and `FUN_006bbd10`:

- Requires top-level binary magic `MDLX` (`0x584C444D` in little-endian compare path).
- On mismatch, parser hard-asserts/fails.

### Evidence snippet
```text
if (*fileData != 0x584C444D) assert;  // 'MDLX'
```

Confidence: **High**

---

## 3) `TEXS` and `GEOS` Contract (M2/M3)

From `FUN_006bbd10`:

### `TEXS`
- Finds `TEXS` (`0x53584554`).
- `sectionBytes / 0x10C` is computed as texture count.
- Hard expectation in this path: `numTextures == 1`.
- Also enforces exact size relation: `sectionBytes == numTextures * 0x10C`.

### `GEOS`
- Finds `GEOS` (`0x534F4547`).
- Expects first nested geoset token `VRTX`.
- Vertex count drives allocation and copied as `count*3` float-like components.
- Expects `NRMS` next and enforces `numNormals == numVertices`.
- Expects `UVAS` and in this path enforces count `== 1`.
- Then expects strict sequence:
  - `PTYP` -> `PCNT` -> `PVTX`

Confidence: **High** for these checks.

---

## 4) Geoset Tail Contract (M4)

From `FUN_0044ea20`:

If not in alternate compatibility path (`param_2 & 0x100 == 0`), requires sequence:

1. `GNDX` (`0x58444E47`)
2. `MTGC` (`0x4347544D`)
3. `MATS` (`0x5354414D`)
4. `BIDX` (`0x58444942`)
5. `BWGT` (`0x54475742`)

Also performs size-driven allocations and exact copy loops for each section payload.

Confidence: **High**

---

## 5) Implementation-Ready `IMdxProfile` Seeds

```text
ProfileId: MdxProfile_080_3734
BuildRange: [0.8.0.3734, 0.8.0.3734]

GeometryPolicy:
  RequiresMdlxMagic: true
  GeosetSectionSeek: GEOS
  GeosetRequiredSubsequence:
    [VRTX, NRMS, UVAS, PTYP, PCNT, PVTX, GNDX, MTGC, MATS, BIDX, BWGT]
  EnforceNormalsMatchVertices: true
  ExpectedUvasSetCount: 1

MaterialPolicy:
  TextureSectionSeek: TEXS
  TextureRecordSize: 0x10C
  TextureSectionSizeStrict: true
  ExpectedTextureCount: 1

AnimationPolicy:
  Provisional: true   // not fully extracted in this pass
```

---

## 6) Open Unknowns

1. Full top-level required/optional MDX chunk order from dispatcher under `0x00421700`.
2. Exact animation sequence/keyframe compression semantics for this build.
3. Whether non-world model load paths relax `ExpectedTextureCount == 1`.

Impact severity:
- (1) parser break risk
- (2) animation correctness risk
- (3) compatibility risk
