# PM4 Naming Drift Analysis — Code vs. wowdev.wiki

**Created**: 2026-05-20
**Source**: Comparison of `wow-viewer/src/core/WowViewer.Core.PM4/Models/Pm4ResearchChunkModels.cs` against `https://wowdev.wiki/PM4` and `https://wowdev.wiki/PD4` (retrieved 2026-05-20)

---

## Summary

Our codebase has assigned semantic names to fields that wowdev.wiki deliberately left as `_0xNN` offsets. Some of these names are well-earned from empirical analysis (e.g., `CK24`, `MSPI_first_index`). Others are speculative guesses that hardened into "truth" through repetition — particularly `MdosIndex` (MSUR._0x18) which points to MSCN, not MDOS.

**The wowdev.wiki documentation is intentionally conservative.** Fields are named `_0xNN` when their semantics are unknown. Our code replaced these with invented names, then those names were used in analyzers, tests, and documentation until they felt canonical. This is the hallucination path the user identified.

---

## Chunk-by-Chunk Drift

### MSHD

| Offset | wowdev.wiki | Our Code | Drift |
|--------|-------------|----------|-------|
| 0x00 | `_0x00` | `Field00` | OK — we don't claim to know it |
| 0x04 | `_0x04` | `Field04` | OK |
| 0x08 | `_0x08` | `Field08` | OK |
| 0x0C-1C | `_0x0c[5]` — "Always 0 in version_48, likely placeholders" | `Field0C` through `Field1C` | OK |

**Status**: No naming drift. We correctly kept these as unknown.

---

### MSLK

| Offset | wowdev.wiki | Our Code | Drift | Risk |
|--------|-------------|----------|-------|------|
| 0x00 | `_0x00` — "flags? seen: &1; &2; &4; &8; &16" | `TypeFlags` | **Invented name** | Low — "flags" is in the wiki, our name is close |
| 0x01 | `_0x01` — "0…11-ish; position in some sequence? index into something? not MDB*." | `Subtype` | **Invented name** | **HIGH** — wiki explicitly says "index into something?" not "subtype" |
| 0x02 | `_0x02` — "Always 0 in version_48, likely padding" | `Padding` | OK |
| 0x04 | `_0x04` — "An index somewhere" | `GroupObjectId` | **Invented name** | **HIGH** — wiki says "An index somewhere", we invented "GroupObjectId" |
| 0x08 | `MSPI_first_index` | `MspiFirstIndex` | OK (casing only) |
| 0x0B | `MSPI_index_count` | `MspiIndexCount` | OK (casing only) |
| 0x0C | `_0x0c` — "Always 0xffffffff in version_48" | `LinkId` | **Invented name** | **HIGH** — wiki says always 0xFFFFFFFF, we named it "LinkId" based on our tile-coordinate decoding |
| 0x10 | `msur_index` | `RefIndex` | **Renamed** | **HIGH** — wiki calls it `msur_index`, we renamed to `RefIndex` because we found it doesn't always point to MSUR |
| 0x12 | `_0x12` — "Always 0x8000 in version_48" | `SystemFlag` | **Invented name** | Medium — 0x8000 is confirmed, name is speculative |

**Critical drift**: `msur_index` → `RefIndex`. The wiki says this is a `msur_index`. Our code renamed it because ~3.6% of entries don't fit MSUR. The wiki name may be wrong too (or our fit test is too strict), but we should acknowledge the original name.

---

### MSVT

| Offset | wowdev.wiki | Our Code | Drift |
|--------|-------------|----------|-------|
| (array) | `msvt[]` — "C3Vector; t ≠ tangents. vt = vertices?" | `Msvt` (as `IReadOnlyList<Vector3>`) | OK |

**Note**: wowdev.wiki documents the YXZ coordinate swap and the world-position formula:
```
worldPos.y = 17066.666 - position.y;
worldPos.x = 17066.666 - position.x;
worldPos.z = position.z / 36.0f;
```
Our code uses a more general axis convention detection system (`DetectAxisConventionBySurfaceNormals`) rather than hardcoding this formula. This may be correct for multi-expansion support, but we should verify we're not missing the canonical transform.

---

### MSVI

| Offset | wowdev.wiki | Our Code | Drift |
|--------|-------------|----------|-------|
| (array) | `msv_indices[]` — "index into MSVT" | `Msvi` (as `IReadOnlyList<uint>`) | OK |

---

### MSUR

| Offset | wowdev.wiki | Our Code | Drift | Risk |
|--------|-------------|----------|-------|------|
| 0x00 | `_0x00` — "earlier documentation has this as bitmask32 flags" | `GroupKey` | **Invented name** | Medium — "bitmask32 flags" is the wiki's note, we called it "GroupKey" |
| 0x01 | `_0x01` — "count of indices in MSVI" | `IndexCount` | OK — matches wiki semantics |
| 0x02 | `_0x02` | `AttributeMask` | **Invented name** | High — no wiki guidance, name is pure speculation |
| 0x03 | `_0x03` — "Always 0 in version_48, likely padding" | `Padding` | OK |
| 0x04-0x0F | `float _0x04; float _0x08; float _0x0c` | `Vector3 Normal` | **Invented name** | Medium — wiki lists 3 separate floats, we bundle as "Normal" |
| 0x10 | `float _0x10` | `Height` / `PlaneDistance` | **Invented name** | Medium — wiki lists as unnamed float |
| 0x14 | `MSVI_first_index` | `MsviFirstIndex` | OK |
| 0x18 | `_0x18` | `MdosIndex` | **WRONG NAME** | **CRITICAL** — wiki says `_0x18`, we named it "MdosIndex" implying it points to MDOS chunk. It actually points to MSCN. |
| 0x1C | `_0x1c` | `PackedParams` | **Invented name** | Medium — wiki says nothing about this field |

**Critical drift**: `MSUR._0x18` → `MdosIndex`. This field is an index into **MSCN** (scene nodes), NOT into **MDOS** (destructible object states). The name "MdosIndex" is a collision with the MDOS chunk abbreviation. This is the single most dangerous naming error in the codebase.

---

### MSCN

| Offset | wowdev.wiki | Our Code | Drift |
|--------|-------------|----------|-------|
| (array) | `mscn[]` — "n ≠ normals. Seen to have one entry while MSPV and MSLK has none." | `Mscn` (as `IReadOnlyList<Vector3>`) | OK |

**Note**: wowdev.wiki says "Not related to MSPV and MSLK" — this contradicts our codebase which uses MSLK to link MSCN indirectly via MSUR._0x18. The wiki may be wrong here, or the relationship is more subtle.

---

### MPRL

| Offset | wowdev.wiki | Our Code | Drift | Risk |
|--------|-------------|----------|-------|------|
| 0x00 | `_0x00` — "Always 0 in version_??" | `Unk00` | OK — we acknowledge unknown |
| 0x02 | `_0x02` — "Always -1 in version_??" | `Unk02` | OK |
| 0x04 | `_0x04` | `Unk04` | OK — we don't claim to know |
| 0x06 | `_0x06` | `Unk06` | OK |
| 0x08-0x13 | `C3Vector position` | `Position` | OK |
| 0x14 | `_0x14` | `Unk14` | OK |
| 0x16 | `_0x16` | `Unk16` | OK |

**Status**: MPRL naming is clean — we correctly kept the unknowns as `_0xNN` aliases.

---

### MPRR

| Offset | wowdev.wiki | Our Code | Drift | Risk |
|--------|-------------|----------|-------|------|
| 0x00 | `_0x00` | `Value1` | OK — generic but not misleading |
| 0x02 | `_0x02` | `Value2` | OK |

---

### MDBH / MDBI / MDBF / MDOS / MDSF

| Chunk | wowdev.wiki | Our Code | Drift |
|-------|-------------|----------|-------|
| MDBH | `m_destructible_building_count` | `DestructibleBuildingCount` | OK |
| MDBI | `m_destructible_building_index` | `DestructibleBuildingIndex` | OK |
| MDBF | `m_destructible_building_filename[]` | `Filename` | OK |
| MDOS | `m_destructible_building_index` + `destruction_state` | `DestructibleBuildingIndex` + `DestructionState` | OK |
| MDSF | `msur_index` + `mdos_index` | `MsurIndex` + `MdosIndex` | OK |

**Status**: Destructible chunks are clean. The `MDSF.mdos_index` → `MdosIndex` is correct here (it actually points to MDOS).

---

## Critical Naming Errors

### 1. `MSUR._0x18` → `MdosIndex` (WRONG)

**The field is NOT an index into MDOS.** It is an index into **MSCN** (scene nodes / exterior vertices).

**Evidence**:
- `Pm4PlacementMath.cs:685` reads `surface.MdosIndex` and uses it to index into `exteriorVertices` (the MSCN list)
- `WorldScene.cs:5196-5204` uses `surface.MdosIndex` to access `pm4.KnownChunks.Mscn`
- The Pm4ResearchMscnAnalyzer documents: "MSUR.MdosIndex is the main bridge into MSCN scene-node data"

**Impact**: Every developer reading the code will assume this field points to the MDOS chunk. It does not. The name must change.

**Correct name**: `_0x18` (wowdev.wiki) or `MscnIndex` (our semantic name, if we want to be explicit).

### 2. `MSLK._0x10` → `RefIndex` (renamed from wiki's `msur_index`)

**The wiki calls this `msur_index`.** We renamed it to `RefIndex` because ~3.6% of entries don't fit MSUR.

**The wiki may be wrong** — or our fit test is too strict. But we should:
1. Keep our `RefIndex` name (it's more accurate based on our analysis)
2. Add a comment: "wowdev.wiki calls this `msur_index`; renamed to `RefIndex` because not all entries index into MSUR"

### 3. `MSLK._0x04` → `GroupObjectId` (invented)

**The wiki says "An index somewhere."** We invented "GroupObjectId" based on our union-find grouping analysis.

**The name is reasonable** but speculative. Add a comment: "wowdev.wiki calls this `_0x04`; our name is based on empirical grouping behavior."

---

## Recommended Actions

1. **Immediate**: Rename `MdosIndex` → `MscnIndex` in `Pm4MsurEntry` and all downstream code. This is a correctness issue, not just style.
2. **Immediate**: Add `// wowdev.wiki: _0x18` comments to the renamed field to preserve the wiki mapping.
3. **Short-term**: Add `// wowdev.wiki: _0x04` / `// wowdev.wiki: _0x00` etc. comments to all fields where we've invented names.
4. **Short-term**: Audit whether the MSVT YXZ formula from wowdev.wiki matches our `DetectAxisConventionBySurfaceNormals()` output.
5. **Document**: This drift analysis should be part of the PM4 research output to prevent future hallucination hardening.

---

## The Hallucination Path

The user identified the pattern: "I didn't know how to work AI yet." The drift happened through:

1. **Initial read**: Code was written with correct wowdev.wiki names (`_0x18`, `msur_index`, etc.)
2. **Empirical analysis**: CSV spreadsheets and manual inspection revealed patterns (CK24 grouping, cross-tile objects, RefIndex mismatches)
3. **Naming invention**: Analysts (human or AI) gave invented names to fields based on observed behavior (`GroupObjectId`, `MdosIndex`, `RefIndex`, etc.)
4. **Hardening**: Invented names were used in analyzers, tests, documentation, and specs until they felt canonical
5. **Propagation**: New code and docs referenced the invented names without checking against wowdev.wiki

The fix is not to revert to wowdev.wiki names everywhere — some of our names are genuinely more informative. The fix is to **document the mapping** and **correct the wrong ones** (especially `MdosIndex`).
