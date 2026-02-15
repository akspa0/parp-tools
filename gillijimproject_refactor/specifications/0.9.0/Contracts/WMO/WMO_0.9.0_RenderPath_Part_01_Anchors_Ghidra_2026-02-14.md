# WMO 0.9.0 Render Path — Part 01 (Anchors, Ghidra Closure)

Date: 2026-02-14  
Binary: WoW.exe (build lineage consistent with 0.9.0/3807 track)

## Locked function anchors

### 1) Group render entry (opaque)
- **Function:** `FUN_006dd850`
- **Address:** `0x006dd850`
- **Role:** Main WMO group batch render path for opaque/int batches; validates batch counters and dispatches per-batch draw state.
- **Key literals in function:**
  - `group->transBatchCount == 0` (`0x008958bc`) used at `0x006dd8a8`
  - `group->intBatchCount == 0` (`0x008958a0`) used at `0x006dd8cc`

### 2) Group render entry (transparent / alternate pass)
- **Function:** `FUN_006ddc00`
- **Address:** `0x006ddc00`
- **Role:** Alternate group draw path (blend/state variant), shares batch walk and material state programming pattern.

### 3) Group stream/light linkage gate
- **Function:** `FUN_006cee60`
- **Address:** `0x006cee60`
- **Role:** Validates map-object-def-group linked lists before downstream render usage; includes light-link list gate.
- **Key literal in function:**
  - `mapObjDefGroup->lightLinkList.Head() == 0` (`0x008942e8`) pushed at `0x006cef14`

### 4) WMO group parse handoff to optional chunk parser
- **Function:** `FUN_006e86f0`
- **Address:** `0x006e86f0`
- **Role:** Parses core group chunks and calls optional chunk parser (`FUN_006e8960`) for MLIQ/MOCV/etc.

### 5) WMO optional chunk parser (includes MLIQ copy)
- **Function:** `FUN_006e8960`
- **Address:** `0x006e8960`
- **Role:** Parses optional chunks, validates chunk tokens, and copies MLIQ header/payload pointers to group struct fields.
- **MLIQ literal gate:** `pIffChunk->token == 'MLIQ'` (`0x00896b48`) at compare site `0x006e8ba9`.

### 6) Liquid mesh build path
- **Function:** `FUN_006df070`
- **Address:** `0x006df070`
- **Role:** Builds liquid vertex stream and index stream, then calls `FUN_006dedc0` for index generation and bound checks.
- **Calls:** `FUN_006dedc0` at `0x006df227`.

### 7) Liquid index/gate helper (layout/bounds)
- **Function:** `FUN_006dedc0`
- **Address:** `0x006dedc0`
- **Role:** Generates liquid triangle index list from per-cell flags and enforces index-domain bound checks.
- **Key literal in function:**
  - `(idxBase[i] - vtxSub) < (uint) (group->liquidVerts.x * group->liquidVerts.y)` (`0x008958f8`) pushed at `0x006def1c`

### 8) Liquid draw dispatcher
- **Function:** `FUN_006def50`
- **Address:** `0x006def50`
- **Role:** Selects liquid type and dispatches draw implementation (`FUN_006df070` / `FUN_006df2f0` / `FUN_006df610`).
- **Calls:** `FUN_006cb570` (liquid type selection) at `0x006def5f`.

## One-hop adjacency map (caller/callee)

### `FUN_006dd850`
- **Caller(s):** `FUN_006dbdd0` (xref from `0x006dbdd6` via function table assignment)
- **Callee(s):** `FUN_006dd660`, `FUN_006dd5d0`, `FUN_006ddb30`, graphics state/buffer calls

### `FUN_006cee60`
- **Caller(s):** `FUN_006d4eb0` (`0x006d4f83`)
- **Callee(s):** `FUN_006685d0` (assert/log), `FUN_006d3f20`, `FUN_004ab2e0`

### `FUN_006e8960`
- **Caller(s):** `FUN_006e86f0` (`0x006e88bd`)
- **Callee(s):** `FUN_006685d0` (token gates), `FUN_006bbf60`

### `FUN_006df070`
- **Caller(s):** `FUN_006def50` (`0x006df227`)
- **Callee(s):** `FUN_006dedc0`, `FUN_005b9300`, `FUN_005b90e0`, `FUN_005b9130`

### `FUN_006dedc0`
- **Caller(s):** `FUN_006df070` (`0x006df227`), `FUN_006df2f0` (`0x006df513`), `FUN_006df610` (`0x006df77c`)
- **Callee(s):** `FUN_006685d0` (bounds assert)

### `FUN_006def50`
- **Caller(s):** (entry reached from higher-level map-object liquid render pass; direct xref not required for closure)
- **Callee(s):** `FUN_006cb570`, `FUN_006df070`, `FUN_006df2f0`, `FUN_006df610`

## Confidence
- Group render anchor set: **High** (`FUN_006dd850` / `FUN_006ddc00`)
- Group light-link gate anchor: **High** (`FUN_006cee60`)
- Liquid build/draw anchors: **High** (`FUN_006df070`, `FUN_006dedc0`, `FUN_006def50`)
- Optional-chunk/MLIQ copy anchor: **High** (`FUN_006e8960`)
