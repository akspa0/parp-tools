# ADT Write Order (WotLK path)

- MVER (file version)
- MHDR
- MCIN (chunk index)
- MTEX (textures)
- MMDX/MMID (doodad names + indices)
- MWMO/MWID (wmo names + indices)
- MDDF (doodad placements)
- MODF (wmo placements)
- MH2O (top-level water table; per-chunk entries/layers)
- MTXF (texture flags; optional — emit if needed/present)
- MCNK × 256 (per chunk)
- MFBO (only if `MHDR.flags & 1` indicates presence)

Per-MCNK subchunks (explicit):
- Required: `MCVT` (heights), `MCNR` (normals), `MCLY` (texture layers), `MCRF` (refs), `MCAL` (alpha maps)
- Optional: `MCSH` (holes/shadows), `MCCV` (vertex colors)
- Forbidden in outputs: `MCLQ` (legacy liquid). All legacy liquid-related offsets/flags in `MCNK` must be zero. Water is represented via `MH2O` only.

Notes:
- Recompute all offsets after assembly; 4-byte align each block/chunk.
- Follow `lib/gillijimproject/wowfiles/lichking/AdtLk.cpp` as ordering reference.
- Enforce MH2O-only policy: never emit `MCLQ`. If reading Alpha data, convert to MH2O layers and zero legacy fields.

## Required Tables (must exist even if empty)

- MVER (emit correct version)
- MHDR (header with all offsets set correctly)
- MCIN (256 entries; points to each MCNK)
- MTEX (can be empty but chunk must exist with valid size/offset)
- MMDX/MMID and MWMO/MWID (names + indices; emit empty tables when none)
- MDDF/MODF (placements; emit empty tables when none)
- MH2O (emit a valid, possibly empty MH2O with consistent offsets; water via MH2O only)
- All 256 MCNK blocks (emit required subchunks per spec; optional subchunks only when data exists; legacy `MCLQ` must not be present)
- MTXF (optional: include only when texture flags are used/present)
- MFBO (conditional: include only when `MHDR.flags & 1`)

Important:
- Other engines/viewers expect these tables; omission breaks loading. Emit zero-sized/empty chunks with correct offsets instead of skipping.
