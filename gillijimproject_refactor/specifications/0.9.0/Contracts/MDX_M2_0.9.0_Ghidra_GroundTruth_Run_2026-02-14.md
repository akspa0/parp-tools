# MDX/M2 0.9.0 Ghidra Ground-Truth Run — 2026-02-14

## Run metadata
- Method: live Ghidra decompilation + string evidence extraction
- Target binary fingerprint found in strings: `WoW [Release Assertions Enabled] Build 3807 (Aug 15 2004 15:24:42)`
- Goal alignment: routing, `SEQS` layout, `GEOS` payload semantics, render suppression gates

## Confirmed findings (binary-proven)

### 1) Container routing and precedence
**Confidence: High**

- `FUN_00783070` classifies by extension:
  - returns `0` for `*.mdl` (`l/L`)
  - returns `1` for `*.mdx` (`x/X`)
  - returns fallback default for others
- `FUN_00783550` behavior:
  - if classifier returns `2` => `FUN_00784890` fatal (`"Unrecognized file extension: %s"`)
  - if classifier returns `1` => binary parse path `FUN_007837b0`
  - else => non-binary path `FUN_00783680`
- Binary parser `FUN_007837b0` enforces root magic `0x584C444D` (`MDLX`) before section walk.

**Decision:** in this build, extension influences path selection first, and MDLX magic is then strictly enforced on binary path.

### 2) Chunk dispatch map
**Confidence: High**

`FUN_00784200` dispatches FourCC tags to section handlers (including unknown-tag warning + skip):

- `SEQS` -> `FUN_007a9c70`
- `GEOS` -> `FUN_007a2a00`
- `MTLS` -> `FUN_007a7440`
- `TEXS` -> `FUN_007a8df0`
- `BONE` -> `FUN_0079eb30`
- `GEOA` -> `FUN_007a2280`
- `PRE2` -> `FUN_00795700`
- `RIBB` -> `FUN_007903f0`
- `HELP` -> `FUN_0079bb60`
- `ATCH` -> `FUN_0079ac10`
- `PIVT` -> `FUN_0079a340`
- `HTST` -> `FUN_0078aa10`
- `CLID` -> `FUN_00789d10`
- default -> warning string `Warning: Unknown section tag...`, skip payload via `FUN_007f4260`

### 3) `SEQS` structure expectations
**Confidence: High**

From `FUN_007a9c70`:

- Required section size relation: `(sectionSize - 4) == count * 0x8C`
- Violation path: callback error `"Invalid SEQX section detected ..."` and hard fail
- Per-record fields actively read:
  - `+0x50` start
  - `+0x54` end
  - `+0x58` float
  - `+0x5C` flags/unk
  - `+0x60..0x6B` float3
  - `+0x6C..0x77` float3
  - `+0x78` float
  - `+0x7C` value
  - `+0x80` value
  - `+0x84` value
  - `+0x88` value
- Array resized against model state at `+0x3b4/+0x3b8` with strict guarded reads.

**Decision:** 0x8C-record `SEQS` with leading count is mandatory for this loader path.

### 4) `GEOS` payload semantics
**Confidence: High**

From `FUN_007a2a00` + `FUN_007a2f10` + `FUN_007a2bc0`:

- `GEOS` starts with geoset count (`u32`), then two-pass parse:
  1. Header/count pass (`FUN_007a2f10`)
  2. Payload stream pass (`FUN_007a2bc0`)
- Overrun gate in both passes:
  - condition compares consumed bytes vs section size
  - failure path emits `"Geoset section overran read buffer."` via `FUN_00784810`
- Header pass behavior includes:
  - geoset id/token write at `+0xC8`
  - bounds + radius reads (`+0xCC..+0xE4`), with radius-adjusted mins
  - material/token fields written (`+0xFC` etc.)
  - flags field write at `+0x100`
  - vertex-driven allocation/resizing of multiple vectors (positions, normals, UV-associated tables)
- Payload pass streams per-vertex data with guarded index checks, then matrix/group records, then alignment.

**Decision:** parser interprets this as structured element streams (not a single global "bytes only" policy), with strict per-stage bounds enforcement.

### 5) Render suppression gates (what is currently proven)
**Confidence: Medium**

Parser-level suppressors (proven):
- Any `GEOS` overrun -> hard fail (`"Geoset section overran read buffer."`) and model path failure.
- Any `SEQS` stride mismatch -> hard fail (`"Invalid SEQX ..."`).

Render-time gate anchors (string-proven, xref not resolved in this run):
- `"AddGeosetToScene: geoShared->materialId (%d) >= numMaterials (%d)"`
- `"File contains no geosets, lights, sound emitters, attachments, cameras, particle emitters, or ribbon emitters."`
- `"geoShared->numVertices == 4"`
- `"srcGeoset->batches[0].m_count == srcGeoset->numPrimitiveIndices"`
- `"srcGeoset->batches[0].m_primType == GxPrim_Triangles"`
- `"srcGeoset->numBatches == 1"`

**Decision:** invisible-model outcomes are strongly consistent with material-id bounds and geoset batch invariants, in addition to parser hard-fail gates.

### 6) Renderer function localization (`ModelRender.cpp`)
**Confidence: High**

Concrete renderer-side functions are now bound to `ModelRender.cpp` (`008225bc`):

- `FUN_0043cb90`
  - assertions at lines `0xaf7`, `0xaf8`, `0xb07`
  - validates model/shared, expands render-side containers, creates/attaches material, hands off to `FUN_0043cea0`
- `FUN_0043cea0`
  - assertion at line `0xacd` with `geoShared` null gate
  - copies geoset-shared streams under repeated index bounds checks
- adjacent module helpers: `FUN_0043a680`, `FUN_0043ae70`, `FUN_0043d3a0`, `FUN_0043b6e0`, `FUN_0043b8b0`

Exact material-id literal branch closure:
- `00822668` (`AddGeosetToScene: geoShared->materialId (%d) >= numMaterials (%d)`) is used in `FUN_004349b0`.
- Disassembly proof point: `00434a37: PUSH 0x822668`.
- Caller linkage: `FUN_004348a0` iterates geosets and calls `FUN_004349b0`.

**Decision:** renderer-side geoset/material assembly and material-id bound gate are concretely localized and branch-proven.

## Evidence ledger

### Decompiled functions
- `FUN_00783070` (extension classifier)
- `FUN_00783550` (mode switch + parse path routing)
- `FUN_007837b0` (MDLX magic + section walk + overrun guards)
- `FUN_00784200` (section dispatch)
- `FUN_00784890` (unknown extension fatal)
- `FUN_00783a80`, `FUN_00783b30` (MDLX payload loaders with magic check)
- `FUN_007a9c70` (`SEQS`)
- `FUN_007a2a00`, `FUN_007a2f10`, `FUN_007a2bc0` (`GEOS`)

### High-signal strings
- `00822668`: AddGeosetToScene material bound check
- `00823168`: "File contains no geosets..."
- `0082274c`: `geoShared->numVertices == 4`
- `00822b50`: batch count/index consistency assertion
- `00822b90`: triangle prim-type assertion
- `00822bc8`: single-batch assertion

## Patch-direction implications (for 0.9.0 tooling)
- Prefer strict `SEQS` count+0x8C validation for this profile.
- Preserve two-pass `GEOS` interpretation and bounds logic; avoid generic payload heuristics.
- Treat material/geoset consistency checks as first-class pre-render validation (not optional warnings).
- Keep unknown-tag behavior as warn+skip only when section lengths are valid.

## Remaining closure items (next Ghidra pass)
1. Capture exact condition values at failure sites (`materialId`, `numMaterials`, vertex/index counts) across representative assets.
2. Run per-asset stepping (kelthuzad + newer M2-style cases + control) and fill runtime evidence table rows.
3. Merge runtime rows into a finalized Part 09 evidence artifact.

## Status summary
- Q1 (routing): answered
- Q2 (`SEQS` layout): answered
- Q3 (`GEOS` payload semantics): answered
- Q4 (render suppression): answered (renderer module/functions + exact material-id literal branch proven)

## Incremental run artifacts
- See staged outputs in `specifications/0.9.0/Contracts/runs/2026-02-14/`:
  - `RUN_INDEX.md`
  - `MDX_M2_0.9.0_RenderPath_Part_01_Anchors.md`
  - `MDX_M2_0.9.0_RenderPath_Part_02_Candidate_Functions.md`
  - `MDX_M2_0.9.0_RenderPath_Part_03_Cluster_Decompile.md`
  - `MDX_M2_0.9.0_RenderPath_Part_04_ModelRender_Function_Lock.md`
  - `MDX_M2_0.9.0_RenderPath_Part_05_Parser_Render_Contract_Crosscheck.md`
  - `MDX_M2_0.9.0_RenderPath_Part_06_MaterialGate_Literal_Closure.md`
  - `MDX_M2_0.9.0_RenderPath_Part_07_MaterialGate_Operand_Mapping.md`
  - `MDX_M2_0.9.0_RenderPath_Part_08_Runtime_Evidence_Table_Template.md`
  - `MDX_M2_0.9.0_RenderPath_Part_09_Ghidra_Value_Source_Map.md`

- Renderer contract addendum:
  - `specifications/0.9.0/Contracts/MDX_M2_0.9.0_Renderer_Addendum_2026-02-14.md`