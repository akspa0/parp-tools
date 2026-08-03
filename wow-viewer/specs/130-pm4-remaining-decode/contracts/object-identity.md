# Contract: Object Identity Table

**Phase**: 4 | **Satisfies**: FR-003, FR-006, FR-007, FR-011, SC-002, SC-003

**Path**: `output/pm4-decode/object-identity.json`

**Stability**: this is the load-bearing contract of the feature. Spec 129's row layout is
object-primary, so a change here invalidates a built dataset. Breaking changes require a
`schemaVersion` bump and a note in the epic.

## Consumers

| consumer | uses |
|---|---|
| Spec 129 (zarr dataset) | `objects[]` as the row axis; `assignments[]` to attach surface data to rows |
| Spec 128 (matching) | `Pm4ObjectSegmentBuilder` re-keyed onto `objectId` in Phase 4 |
| Viewer (Phase 5) | `assignments[]` for pick→object; `objects[].tileCoordinates` for cross-tile selection |

## Schema

```jsonc
{
  "schemaVersion": 1,
  "generatedUtc": "2026-08-03T00:00:00Z",
  "inputDirectory": "test_data/development/World/Maps/development",
  "corpusSignature": "test_data/development/World/Maps/development@616",
  "ruleId": "G3",
  "ruleDescription": "surface -> MSLK entries -> GroupObjectId -> sibling surfaces",
  "fileCount": 616,

  "coverage": {
    "surfacesTotal": 0,
    "surfacesAssigned": 0,
    "surfacesUngrouped": 0,
    "surfacesSentinelExcluded": 0,
    "objectCount": 0,
    "crossTileObjectCount": 0
  },

  "objects": [
    {
      "objectId": "pm4obj-0123456789abcdef",
      "canonicalKey": "rule=G3|group=12345",
      "surfaceCount": 0,
      "totalIndexCount": 0,
      "tileCoordinates": ["0_0", "0_1"],
      "confidence": "Low",
      "flags": ["MultipleLinkGroupIds"]
    }
  ],

  "assignments": [
    {
      "sourcePath": "development_00_00.pm4",
      "surfaceIndex": 0,
      "objectId": "pm4obj-0123456789abcdef",
      "status": "Assigned",
      "confidence": "Low",
      "reason": null
    }
  ],

  "notes": []
}
```

## The object id is tile-independent

`objectId` contains no tile coordinate. This is the point.

The viewer's current key is `(tileX, tileY, ck24, objectPart)` with a group key of
`(tileX, tileY, ck24)` (`WorldScene.cs:1133-1134`). Both embed the tile, which makes FR-006 —
"object selection MUST include parts of the object residing in other tiles" — **unimplementable
regardless of decode quality**. 266 of 1,229 CK24 keys (21.6%) genuinely span 2–8 tiles, so this is
not a rare case.

Cross-tile membership is expressed by an object having more than one entry in `tileCoordinates`, not
by any merging step at read time.

**Not to be confused with**: `Pm4PlacementMath.BuildMergedGroupMap`, which unions groups in
*adjacent* tiles when their connector keys are geometrically close. That is a rendering convenience.
It merges things that touch and cannot merge parts that are separated — it is not identity and this
table is not built on it.

## Every surface has exactly one row

`assignments[]` covers every MSUR record in every non-empty file. This makes FR-003 and SC-003
structural rather than a matter of discipline: a surface cannot be silently dropped, because a
missing row is a schema violation.

| `status` | meaning | `objectId` | `reason` |
|---|---|---|---|
| `Assigned` | membership determined | non-null | null |
| `Ungrouped` | membership genuinely undetermined | **null** | **required** |
| `SentinelExcluded` | key is a known null sentinel, not an unsolved case | **null** | **required** |

`Ungrouped` and `SentinelExcluded` are separate on purpose. `CK24 = 0` spans 291 tiles and is a null
key — a solved problem. Collapsing it into `Ungrouped` would hide a solved problem inside an unsolved
one and inflate the apparent size of the remaining work.

## Sentinel policy — explicit, not incidental

Named and reported, never emergent from iteration order:

- **`CK24 = 0`** → `SentinelExcluded`. It is a null key, not an object spanning 291 tiles.
- **Zero-CK24 fallback**: `Pm4ObjectSegmentBuilder` currently groups these on
  `(GroupKey, AttributeMask)` and flags `ZeroCk24Seed`. That behaviour is **preserved** and surfaced
  as a named policy with its own confidence, not reinvented differently here.
- **`MSLK.LinkId` sentinel** (`0xFFFF` high half, tile coords in the low half — 1,273,335 entries,
  already verified) is a tile reference. It is never an object id.

## Determinism

Required, and the Phase 4 gate:

- `objects[]` sorted by `objectId`
- `assignments[]` sorted by `(sourcePath, surfaceIndex)`
- `tileCoordinates[]` sorted lexically
- `flags[]` sorted

Re-running on an unchanged corpus produces a **byte-identical** file. Spec 129 may cache on
`corpusSignature` + `ruleId` only because of this.

## Object id derivation

```text
canonicalKey = "rule=<ruleId>|group=<rule's canonical group key>"
objectId     = "pm4obj-" + lowercase hex of the first 8 bytes of SHA-256(canonicalKey)
```

Matching the existing `pm4seg-` convention in `Pm4ObjectSegmentBuilder.BuildSegmentId`, which uses
the same 8-byte SHA-256 prefix.

`canonicalKey` is retained in the output. An opaque hash with no way back to what produced it is a
debugging dead end, and `ruleId` is part of the key because **an object id is only meaningful
relative to the rule that minted it** — two rules may legitimately disagree about what one object is.

## Confidence

FR-007. Every assignment and every object carries confidence. An object's confidence is the weakest
among its assignments — an object is not more trustworthy than its shakiest member.
