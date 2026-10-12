# Contract: Pick Hit Result

Shape of one resolved pick, as produced by the scene's pick methods and consumed by
`ViewerApp_ClickSelection.cs`'s existing candidate list.

A precise (mesh-accurate) hit:

```json
{
  "objectRef": { "objectType": "Wmo", "objectIndex": 412 },
  "testKind": "Precise",
  "distance": 34.71,
  "worldPoint": { "x": 1204.5, "y": -822.0, "z": 61.2 }
}
```

A fallback (bounding-volume) hit, for an object whose mesh geometry is unavailable:

```json
{
  "objectRef": { "objectType": "Mdx", "objectIndex": 1899 },
  "testKind": "BoundingVolume",
  "distance": 51.03,
  "worldPoint": { "x": 1180.2, "y": -790.4, "z": 58.8 }
}
```

**Contract rules**:

- `testKind` is always present. A consumer must be able to distinguish a mesh-accurate hit from an
  approximate one — this is spec.md's Hit Result entity requirement, and it is what makes FR-002's
  fallback observable rather than invisible.
- A `BoundingVolume` result is a **normal** outcome, not a degraded error state: unloaded models,
  failed loads, and structurally unsupported model kinds (research.md §3) all legitimately produce it.
- Results are ordered by `distance` ascending; the existing multi-candidate disambiguation overlay
  (FR-005) consumes that ordering unchanged. Mixing `Precise` and `BoundingVolume` results in one ordered
  list is expected and correct — they are compared by distance, not by kind.
