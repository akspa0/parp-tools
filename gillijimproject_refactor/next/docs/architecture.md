# Architecture (Next)

- `Domain/` — Value objects and models (Alpha WDT/ADT, LK ADT)
- `IO/` — Alpha readers (WDT/ADT)
- `Transform/` — Alpha → LK pipeline and mappers
- `Services/` — AreaId translation, UniqueID analysis, Report writing
- `Adapters/WarcraftNet/` — LK ADT writer (Warcraft.NET)
- `Adapters/Dbcd/` — DBCD access and helpers

## Invariants

- FourCC forward in-memory; reversed on-disk (writer boundary)
- Include MFBO/MTXF when present, update MHDR offsets
- Write MCLQ last within MCNK; omit MH2O when empty

## Data Flow

Alpha WDT/ADT → AlphaReader → AlphaToLkConverter (+ AreaIdTranslator) → AdtLk → WarcraftNetAdtWriter → LK ADT v18

### Liquids: Alpha MCLQ → MH2O
- Location: `next/src/GillijimProject.Next.Core/IO/AlphaMclqExtractor.cs` (implements `IAlphaLiquidsExtractor`)
- Flow: `AlphaMclqExtractor` → `LiquidsConverter.MclqToMh2o` → `AdtLk.Mh2oByChunk[256]`
- Extraction:
  - Read `MCIN`, iterate `MCNK`, parse 128-byte MCNK header (`ofsLiquid`, `sizeLiquid`).
  - Try offset origins for `MCLQ` start: header end (dataStart + 128), data start (after fourcc/size), chunk begin (including fourcc/size).
  - Read CRange (min/max). Choose vertex layout by MCNK flags:
    - Water: 81 × (depth, flow0, flow1, filler, height)
    - Ocean: 81 × (depth, foam, wet, filler) — heights inferred from `heightMin`
    - Magma: 81 × (u16 s, u16 t, height) — UV values ignored downstream
  - Read tiles 8×8: low nibble → `MclqLiquidType`, high nibble → `MclqTileFlags`.
- Behavior:
  - Defensive bounds checks for all reads; per-chunk failures return null (no throw).
  - Unknown low-nibble tile types are normalized to `None`.
  - `sizeLiquid` is treated as a hint; actual reads are constrained to the MCNK bounds.
- Limitations:
  - UV/flow vectors are currently not used in MH2O generation (LVF 1/3 deferred).
  - Some Alpha variants may require heuristic fallbacks; validation/logging will be expanded.

## Alpha WDT Reverse Writer

- Location: `next/src/GillijimProject.Next.Core/IO/AlphaWdtWriter.cs`
- Purpose: Generate an Alpha-era WDT that embeds raw LK ADT payloads.
- Chunk order (on disk): `MVER`, `MPHD`, `MAIN`, `MDNM` (optional), `MONM` (optional), `MODF` (optional when WMO-based), followed by raw ADT payloads.
- FourCCs are written reversed on disk (e.g., `MVER` -> `REVM`).
- `MVER`: 4-byte version `18`.
- `MPHD` (16-byte data):
  - At data offset +4: absolute file offset to the `MDNM` chunk header (0 when omitted).
  - At data offset +12: absolute file offset to the `MONM` chunk header (0 when omitted).
  - At data offset +8: `2` when WMO-based; `0` otherwise.
- `MAIN`: 64x64 grid, 16 bytes per entry (total 65536 bytes). Each cell:
  - Bytes 0..3: absolute file offset to the embedded tile's `MHDR` within the WDT file.
  - Bytes 4..7: payload size in bytes (entire root ADT file length).
  - Bytes 8..15: reserved (zeros).
- Embedding strategy: Append each LK root ADT (`<map>_x_y.adt`) to the end of the WDT, scanning for the `MHDR` marker (`RDHM` on disk) to compute per-tile absolute offsets. Only root ADTs are used; texture/object ADTs are ignored.
- Padding: Each chunk's data is padded to even size (add one 0 byte when needed).
