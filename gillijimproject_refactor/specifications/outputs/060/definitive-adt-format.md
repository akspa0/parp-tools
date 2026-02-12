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
- **Chunk FourCCs are forward on disk** (LK-style): `MCNK` is literally `'MCNK'` in-file, not reversed.
- **`MCNK` header is Alpha-style** (very similar to 0.5.3): fixed offset table pointing to sub-chunks.
- **Vertex payloads are non-interleaved** (Alpha-style): `MCVT` and `MCNR` are stored as contiguous arrays.

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

All chunk FourCCs are stored **forward** on disk (LK-style).

Evidence: [task-06-chunk-fourcc-byte-order.md](task-06-chunk-fourcc-byte-order.md)

Concrete proof point: the client compares 32-bit values against:

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

- When reading chunk tags from file, treat them as literal strings (no Alpha-era reversed-FourCC handling for 0.6.0).

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
