# PM4 ➜ OBJ Recovery Pipeline

A detailed, step-by-step process to reliably reconstruct every object/component from PM4 tiles, verify chunk integrity, and produce clean OBJ outputs with full traceability.

---

## 0. Goals
* **Data-safe**: Never discard genuine geometry – preserve every triangle.
* **Object-faithful**: Group faces by original `ParentIndex` and MSLK link graph so each OBJ sub-object corresponds to a real game object.
* **Transparent**: Write an exhaustive log file (`pipeline.log`) that documents every stage, timing, counts, and anomalies.
* **Modular**: Each stage can be run in isolation (e.g. `--audit-only`).

---

## 1. Chunk Audit  (`Step A`)
| Task | Output |
|------|--------|
| Iterate all chunks (MSVI, MSUR, MSCN, MSLK, MPRR, etc.) | Counts, byte sizes |
| Validate expected sizes/alignments | Warnings on mismatch |
| Detect missing critical chunks | Abort export if fatal |

`audit.log` section **[CHUNK AUDIT]** lists each file, chunk ids, entry counts.

---

## 2. Object-Graph Build  (`Step B`)
1. Build in-memory map:<br>`ParentIndex → List<SurfaceGroup>`  (per tile)
2. Follow **MSLK** links to merge cross-tile surfaces that form one logical object.
3. Produce metrics:
   * Total ParentIndex values
   * Connected components in link graph
   * Orphan surfaces (no ParentIndex / links)

Logged under **[OBJECT GRAPH]**.

---

## 3. Geometry Extraction  (`Step C`)
For each resolved object/component:
1. Collect unique triangle indices from its surfaces.
2. Skip triangles where any two indices are identical (degenerate).
3. Skip exact duplicate triangles (same 3 indices irrespective of winding). **No vertex removal**.
4. Record counts (raw vs kept) for log.

---

## 4. OBJ Writing  (`Step D`)
* **Per-tile file**: `per_tile_objects/<tile>.obj`.
* Inside each file:
  ```
  o object_<ParentIndex>
  v ...
  f ...
  ```
* Optional **global** merged OBJ for fast preview.

---

## 5. Logging  (`Step E`)
A lightweight `ILogger` streams every `Console.WriteLine` to `pipeline.log` plus timestamps.  File is appended for multi-tile runs.

Log Sections:
1. `[CHUNK AUDIT]`
2. `[OBJECT GRAPH]`
3. `[GEOMETRY]` – per object counts
4. `[OBJ WRITE]` – file paths & timings

---

## 6. CLI Flags  (`Step F`)
Flag | Description | Default
-----|-------------|--------
`--audit-only` | Run Steps A–B, no OBJ output | Off
`--verbose-log` | Enable `pipeline.log` (mirrors console) | On
`--per-object`  | Write `o` groups per resolved object | On
`--global-obj`  | Also write merged OBJ in `global_unified/` | On
`explore <path>` | Prototype automated data-mining harness; outputs per-chunk and cross-chunk reports | n/a

---

## 7. Automated Exploratory Harness (prototype)

The harness (`explore` command) is an **offline analysis** tool intended to surface hidden structures across all PM4 chunks.

Output directory tree:

* `summary.txt` – Top-level list of the most promising grouping keys across *all* chunks.
* `<ChunkType>_summary.txt` – Per-chunk field distributions (groups, min/avg/max counts).
* `cross_chunk_links.txt` – Shared value correlations between fields of *different* chunks (potential foreign-key relationships).

Known limitations (v0):
* Only single-field groupings analysed (multi-field combos TBD).
* OBJ export for correlated groups not wired yet.
* Correlation file capped to 20 shared values per field-pair for readability.

Run via:
```bash
 dotnet run --project src/PM4Rebuilder/PM4Rebuilder.csproj -- explore <pm4_file_or_dir>
```

---

## 8. Future Enhancements
* Material (`.mtl`) export tagging surfaces/groups.
* Parallel tile processing with thread-safe logging.
* Validation suite comparing face/vertex counts against original game client.
