# PM4 Assembly Relationships

_Last updated: 2025-08-08_

## 2025-08-19 Rewrite Preface

Updated to reflect current guidance:

- **Per-tile processing (Confirmed)**: Assemble one PM4 tile at a time; do not unify tiles into a global scene.
- **Hierarchical containers (Confirmed)**: Identify containers via `MSLK.MspiFirstIndex = -1` and traverse to geometry-bearing links.
- **Placement link (Confirmed)**: `MPRL.Unknown4` equals `MSLK.ParentIndex`.
- **MPRR (Confirmed)**: `Value1 = 65535` acts as property separators, not building boundaries.

See unified errata: [PM4-Errata.md](PM4-Errata.md)

---

## 1. Key Chunks & Fields

| Chunk | Field | Size | Purpose |
|-------|-------|------|---------|
| **MSUR** | `IndexCount` | varies | Diagnostic only; useful for visualization, not an object identifier. |
| **MSUR** | `SurfaceKey` | uint32 | Surface identifier; semantics dataset-dependent and under investigation. |
| **MSUR** | Attributes/Flags | various | Attribute masks/flags exist; exact semantics under investigation. |
| **MSLK** | `ParentIndex` | uint32 | Links to placement (`MPRL.Unknown4`). Container nodes have `MspiFirstIndex = -1`. |
| **MSLK** | `MspiFirstIndex` / `MspiIndexCount` | int32 / uint32 | Geometry slice indices (signed first index; `-1` = container). |
| **MPRL** | `Unknown4` | uint32 | Placement → geometry mapping (`ParentIndex`). |
| **MPRR** | `Value1` | uint16 | Sentinel `65535` marks property separators. |

Note: Container traversal is driven by `MSLK.MspiFirstIndex = -1`; avoid relying on unverified coordinate heuristics.

---

## 2. Proven Grouping Strategies

### 2.1 Container Traversal (recommended)
1. **Per-tile processing**: Isolate each PM4 tile
2. **Container traversal**: Use `MSLK` container nodes (`MspiFirstIndex = -1`) to drive assembly
3. **Placement mapping**: `MPRL.Unknown4 ↔ MSLK.ParentIndex`
4. **Faces from MSUR → MSVI**; treat `MSUR.IndexCount` as diagnostic

### 2.2 Index-Count Strategy (legacy research)
* Group by `MSUR.IndexCount`.
* Useful for visualising large vs small slices; not a full object.

### 2.3 Parent-Index Hierarchy
* Start with `MSLK.ParentIndex` → `MPRL.Unknown4` mapping.
* Recursively include child `MSLK` where `MspiFirstIndex != -1`.
* Use as supplemental to container traversal; still diagnostic.

---

## Analyzer sample: MSLK geometry signatures

The following is a representative slice from the analyzer output `project_output/mscn_analysis_20250819_022908/mscn_analysis_20250819_022800_20250819_022908/objects/mslk_geom_signatures.csv` around the current working window. It shows the distribution of MSLK geometry signature IDs and their counts. Note: container nodes (`MSLK.MspiFirstIndex = -1`) generally do not contribute to geometry signature counts; inclusion depends on the analyzer's signature definition.

Legend:
- `signature`: Analyzer-computed identifier for an MSLK geometry signature pattern
- `count`: Number of MSLK entries with that signature in this session

```csv
signature,count
379,92
166,90
171,90
314,89
66,89
656,89
103,89
74,88
141,88
61,88
265,88
313,88
155,87
501,87
239,87
```

Note: This sample is provided to illustrate the heavy-tailed distribution typical of geometry-bearing `MSLK` signatures. Use it alongside container traversal and the `MPRL.Unknown4 ↔ MSLK.ParentIndex` link when reasoning about assembly.

---

## [Deprecated] Cross-Tile Vertex Resolution
* Avoid unifying tiles into a global vertex pool.
* Process one tile at a time; treat cross-tile references as non-rendering metadata unless proven otherwise.

---

## 4. Implementation Notes
* `ContainerTraversalAssembler`: implements container traversal; uses `MSLK` container detection and `MPRL` mapping.
* `MsurIndexCountAssembler`: keeps IndexCount grouping for diagnostic purposes.
* All assemblers **skip triangles containing invalid vertex indices** to avoid (0,0,0) phantom vertices.
* OBJ exporter inverts X by default; use `--legacy-obj-parity` to disable.

---

## 5. CLI Cheatsheet
```
# Per-object (recommended per tile)
dotnet run --project src/PM4NextExporter -- <mytile.pm4> --assembly container-traversal

# IndexCount diagnostic view
dotnet run --project src/PM4NextExporter -- <mytile.pm4> --assembly msur-indexcount
```

---

## 6. TODO
* Verify container traversal on additional tiles.
* Snapshot tests for container traversal grouping vs legacy exporter SHA.
* Document MSLK container-node detection logic once implemented.
