# WoW Alpha 0.6.0.3592 — ADT Format (Definitive)

**Source of truth**: Ghidra analysis of `WoWClient.exe` (Alpha 0.6.0 build 3592), captured in the task reports in this folder.

This document consolidates the **ADT-relevant** determinations into a single spec suitable for implementation (read/write) without re-hunting through individual reports.

## Scope

This is an **ADT terrain tile** format spec as evidenced by the 0.6.0 client:

- Tile naming + how tiles are located (via WDT `MAIN` and per-tile `.adt` files)
- Chunk FourCC byte order
- `MCNK` header layout used by the client
- `MCVT` (heights) and `MCNR` (normals) payload layout (non-interleaved)

Items **not fully specified** here (because they were not established in the provided task reports): full ADT top-level chunk order (e.g., presence/shape of `MHDR`, `MCIN`, etc.) and detailed semantics for some less-discussed `MCNK` sub-chunks.

## Key determinations (one-page summary)

- **Tiles are separate ADT files** (LK-style naming): `MapName_XX_YY.adt`.
- **Chunk FourCC tag bytes are reversed on disk**: e.g. `MCNK` appears as `KNCM` in the file; the client compares against 32-bit constants like `0x4D434E4B`.
- **`MCNK` header is Alpha-style** (very similar to 0.5.3): fixed offset table pointing to sub-chunks.
- **Vertex payloads are non-interleaved** (Alpha-style): `MCVT` and `MCNR` are stored as contiguous arrays.
- **`MCLQ` in 0.6.0 is a normal chunk** inside `MCNK`: the client validates the `'MCLQ'` token and uses payload at `+8`.

## 1) Tile discovery + file naming

### 1.1 ADT file naming

The client constructs per-tile ADT paths using the literal format string:

- Raw report string: `"%s\%s_%d_%d.adt"` (string at `0x008af7f4`)
  - Interpreted as a Win32 path join (single `\` path separator).

Evidence: [task-01-wdt-adt-format-detection.md](task-01-wdt-adt-format-detection.md)

### 1.2 WDT role in 0.6.0

The WDT contains the `MAIN` tile index, and the client loads separate `.adt` files for actual tile data.

Evidence: [task-01-wdt-adt-format-detection.md](task-01-wdt-adt-format-detection.md)

Notes:

- The client reads `MAIN` as **0x8000 bytes**.
- The report concludes this matches **64×64 entries × 8 bytes** (Alpha-style MAIN entry sizing).

## 2) Chunk FourCC byte order (critical)

### Determination

In 0.6.0 ADT/MCNK parsing, the client treats chunk tags as **32-bit integers** and compares them against constants like `0x4D434E4B`.

Because the file is read on a little-endian CPU, this corresponds to **reversed ASCII bytes on disk**.

Example (`MCNK`):

| Concept | Bytes | 32-bit value (little-endian) |
|---|---|---:|
| Canonical tag | `MCNK` | — |
| Bytes as stored in file | `KNCM` (`4B 4E 43 4D`) | `0x4D434E4B` |

This matches what we see in Ghidra: `FUN_006a6d00` checks the first dword of the MCNK chunk against `0x4D434E4B`.

Evidence: decompile of `FUN_006a6d00` (Ghidra).

- `0x4d434e4b` → `'MCNK'`
- `0x4d435654` → `'MCVT'`
- `0x4d434e52` → `'MCNR'`
- `0x4d434c59` → `'MCLY'`
- `0x4d435246` → `'MCRF'`
- `0x4d43414c` → `'MCAL'`
- `0x4d435348` → `'MCSH'`
- `0x4d435345` → `'MCSE'`
- `0x4d434c51` → `'MCLQ'`

Implementation consequence:

- If you read 4 tag bytes from disk and want the human-readable FourCC, reverse the 4 bytes (`KNCM` → `MCNK`).
- Alternatively, if you read as `uint32` little-endian, compare against constants like `0x4D434E4B`.


## 3) `MCNK` chunk structure

The 0.6.0 client validates and consumes `MCNK` using:

- `FUN_006a6710` — `MCNK` load (referenced in reports)
- `FUN_006a6d00` — `MCNK` validation + sub-chunk pointer derivation

Evidence: [task-05-mcnk-header-layout.md](task-05-mcnk-header-layout.md)

### 3.1 `MCNK` chunk header vs `MCNK` *header struct*

The reports indicate a standard WoW chunk wrapper:

- `MCNK` chunk starts with an **8-byte chunk header**: `token (4) + size (4)`
- The client then treats `MCNK_HEADER*` as `mcnk_chunk_start + 8`

Evidence excerpt: “Store MCNK header pointer: `*(int *)(param_1 + 0xee4) = iVar1 + 8;`”

### 3.2 Sub-chunk offset table (within `MCNK_HEADER`)

The `MCNK_HEADER` contains offsets to sub-chunks at fixed positions:

| `MCNK_HEADER` offset | Sub-chunk | FourCC |
|---:|---|---|
| `0x14` | `MCVT` | `0x4d435654` |
| `0x18` | `MCNR` | `0x4d434e52` |
| `0x1c` | `MCLY` | `0x4d434c59` |
| `0x20` | `MCRF` | `0x4d435246` |
| `0x24` | `MCAL` | `0x4d43414c` |
| `0x2c` | `MCSH` | `0x4d435348` |
| `0x58` | `MCSE` | `0x4d435345` |
| `0x60` | `MCLQ` | `0x4d434c51` |

Evidence: [task-05-mcnk-header-layout.md](task-05-mcnk-header-layout.md)

### 3.3 Header field mapping (partial)

The following fields were mapped in the report (offsets relative to `MCNK_HEADER`, i.e. after the 8-byte chunk header):

| Offset | Size | Field | Notes |
|---:|---:|---|---|
| `0x00` | 4 | `Flags` | Header flags |
| `0x04` | 4 | `IndexX` | Chunk/tile index X |
| `0x08` | 4 | `IndexY` | Chunk/tile index Y |
| `0x10` | 4 | `LayerCount` | Number of texture layers |
| `0x14` | 4 | `McvtOffset` | Offset to `MCVT` |
| `0x18` | 4 | `McnrOffset` | Offset to `MCNR` |
| `0x1c` | 4 | `MclyOffset` | Offset to `MCLY` |
| `0x20` | 4 | `McrfOffset` | Offset to `MCRF` |
| `0x24` | 4 | `McalOffset` | Offset to `MCAL` |
| `0x2c` | 4 | `McshOffset` | Offset to `MCSH` |
| `0x30` | 4 | (unknown) | Unidentified |
| `0x34` | 4 | `Flags2` | “Additional flags” in report |
| `0x38` | 4 | (unknown) | Unidentified |
| `0x3c` | 2 | `HoleCount` | 16-bit field |
| `0x58` | 4 | `McseOffset` | Offset to `MCSE` |
| `0x60` | 4 | `MclqOffset` | Offset to `MCLQ` |
| `0x68` | 4 | `PosX` | float |
| `0x6c` | 4 | `PosY` | float |
| `0x70` | 4 | `PosZ` | float |

Evidence: [task-05-mcnk-header-layout.md](task-05-mcnk-header-layout.md)

### 3.4 How sub-chunk offsets are interpreted

From the validation patterns, offsets in the `MCNK_HEADER` are treated as absolute offsets from a stable base pointer for the `MCNK` chunk.

The task report strongly suggests they are relative to the **start of the `MCNK` chunk** (where the `'MCNK'` token lives), which matches the usual WoW convention.

So:

```text
MCNK chunk:
  +0x00 token 'MCNK'
  +0x04 size
  +0x08 MCNK_HEADER begins

MCVT chunk address = mcnk_chunk_start + McvtOffset
MCVT data address  = (mcnk_chunk_start + McvtOffset) + 8
```

This matches the report’s “`+ 8`” adjustment when converting a chunk base pointer into a header/payload pointer, and comparisons like:

- `*(int *)(iVar1 + *(int *)(iVar1 + 0x1c)) == 'MCVT'`

Evidence: [task-05-mcnk-header-layout.md](task-05-mcnk-header-layout.md)

### 3.5 `MCLQ` (liquid) storage and parsing (0.6.0 client)

This section documents what the **0.6.0 client actually expects** for `MCLQ` inside `MCNK`.

#### 3.5.1 `MCLQ` is a normal sub-chunk with an 8-byte header

The client validates that the `MCNK_HEADER.ofsLiquid` (offset `0x60`) points to a normal chunk header.

In code it compares the dword against `0x4D434C51` (canonical `MCLQ`). On disk the 4 bytes would therefore appear reversed as `QLCM`.

- `*(mcnkChunkStart + ofsLiquid) == 'MCLQ'`

Then it stores the payload pointer as:

- `mclqPayload = (mcnkChunkStart + ofsLiquid) + 8`

This is visible directly in `FUN_006a6d00`:

- Token check: `... + *(... + 0x60) == 0x4d434c51` (`'MCLQ'`)
- Payload pointer: `*(int *)(param_1 + 0xf08) = *(... + 0x60) + 8 + base`

Evidence: decompile of `FUN_006a6d00` (Ghidra).

#### 3.5.2 `MCLQ` payload is a packed list of liquid “instances”

The client parses liquid data in `FUN_006a6960`.

Key behavior:

- The presence bits live in `MCNK_HEADER.Flags` (first `uint32` of the header).
- The code iterates over four bits in this order: `0x04`, `0x08`, `0x10`, `0x20`.
- For each bit that is set, the parser **consumes one instance** from the payload and advances its pointer.
- If a bit is not set, it does **not** advance the payload pointer.

This means the payload is **packed**: it contains exactly `N = popcount(flags & 0x3C)` instances, stored in ascending bit order.

Practical consequence for decoding:

- `MCLQ.chunkSize` should be consistent with `N * 0x2D4` (plus any optional padding/alignment your file writer might add).

Evidence: decompile of `FUN_006a6960` (Ghidra).

#### 3.5.3 Instance layout and size (derived from pointer math)

In `FUN_006a6960`, the instance pointer (`puVar4`) is a `uint32*` into the payload. The function uses these fixed offsets:

- `puVar4[0]` and `puVar4[1]` are copied into the runtime liquid object (very likely the `minHeight`/`maxHeight` floats).
- `puVar4 + 2` is stored as a pointer to the per-vertex/plane data.
- `puVar4 + 0xA4` is stored as a pointer to the 8×8 tile flags area.
- `*(puVar4 + 0xB4)` is copied as one additional `uint32` value.
- The next instance begins at `puVar4 + 0xB5`.

Converted to byte offsets (relative to start of one instance):

| Byte offset | Size | Meaning | Evidence |
|---:|---:|---|---|
| `0x000` | 4 | value0 (likely `minHeight` float) | `*(iVar2+4) = *puVar4` |
| `0x004` | 4 | value1 (likely `maxHeight` float) | `*(iVar2+8) = puVar4[1]` |
| `0x008` | `0x288` (648) | vertex/entry array | `ptr = puVar4 + 2` and next pointer is at `+0x290` |
| `0x290` | 64 | 8×8 tile flags (64 bytes) | `ptr = puVar4 + 0xA4` (`0xA4*4 = 0x290`) |
| `0x2D0` | 4 | trailing value | `*(iVar2+0x14) = *(puVar4+0xB4)` |
| `0x2D4` | — | start of next instance | `puVar4 += 0xB5` (`0xB5*4 = 0x2D4`) |

So the **instance size is `0x2D4` bytes (724)**.

The `0x288` (648) byte region between `0x008` and `0x290` strongly suggests **81 entries × 8 bytes** (since `81*8 = 648`), matching the common “9×9 liquid vertices” pattern.

Unknowns (not established by the current Ghidra pass):

- Exact meaning of the 8-byte per-entry format in the `0x008..0x28F` region (the client keeps it as a pointer).
- The meaning of the trailing 4-byte value at `0x2D0`.

Evidence: decompile of `FUN_006a6960` (Ghidra) and its constant offsets.

## 4) `MCVT` (heights) and `MCNR` (normals) payload layout

### 4.1 Non-interleaved vertex layout (high confidence)

The client reads `MCVT` and `MCNR` as separate contiguous arrays (Alpha-style), **not** the later interleaved pattern.

Evidence: [task-02-mcvt-mcnr-vertex-layout.md](task-02-mcvt-mcnr-vertex-layout.md)

### 4.2 `MCVT` payload

- Total vertices per chunk: **145**
  - 9×9 “outer” grid = 81
  - 8×8 “inner” grid = 64
- Storage: **145 floats**, read sequentially
- Total payload size: **580 bytes** (= 145 × 4)

Evidence: [task-02-mcvt-mcnr-vertex-layout.md](task-02-mcvt-mcnr-vertex-layout.md)

### 4.3 `MCNR` payload

- Total normals per chunk: **145**
- Storage: **145 × (3 signed bytes)**, read sequentially
- Total payload size: **435 bytes** (= 145 × 3)

The client converts these bytes to floats with a constant scale (`_DAT_0081cd3c`), but the constant’s value wasn’t captured in the task report.

Evidence: [task-02-mcvt-mcnr-vertex-layout.md](task-02-mcvt-mcnr-vertex-layout.md)

## 5) MPQ extraction notes (only what impacts ADT reading)

If you are reading `.adt` tiles from MPQs in 0.6.0, the client’s decompression behavior matters.

Evidence: [MPQ/README.md](MPQ/README.md), [MPQ/summary-and-recommendations.md](MPQ/summary-and-recommendations.md)

Key points:

- Compressed blocks start with a **compression type mask byte** (e.g. `0x08` for PKWARE DCL).
- For PKWARE DCL (`0x08`), **there are no PKWARE header bytes** in WoW 0.6.0 payloads.
  - Dictionary size is derived from compressed payload length:
    - `< 0x600` → 1024
    - `< 0xC00` → 2048
    - else → 4096
- Observed block flags `0x80000200` indicate **compressed, not encrypted**.

## 6) Known unknowns / follow-ups

These details are not established by the current task set and should be treated as TODOs if you want a fully “from-byte-0” ADT spec:

- Confirm ADT top-level chunk sequence and presence of `MHDR`/`MCIN`/etc.
- Confirm exact meaning of unknown `MCNK_HEADER` fields (e.g. offsets `0x30`, `0x38`).
- Confirm whether any additional `MCNK` sub-chunks are validated/used beyond those listed.

## References (task reports)

- Tile split + naming: [task-01-wdt-adt-format-detection.md](task-01-wdt-adt-format-detection.md)
- Heights/normals layout: [task-02-mcvt-mcnr-vertex-layout.md](task-02-mcvt-mcnr-vertex-layout.md)
- MCNK header: [task-05-mcnk-header-layout.md](task-05-mcnk-header-layout.md)
- FourCC order: [task-06-chunk-fourcc-byte-order.md](task-06-chunk-fourcc-byte-order.md)
- MPQ decompression behavior: [MPQ/README.md](MPQ/README.md)
