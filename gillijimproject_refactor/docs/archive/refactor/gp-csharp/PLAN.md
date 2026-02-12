# Gillijim Project – 1:1 C# Refactor Plan (net9.0)

## Goal
- Full 1:1, dependency-free C# port of the original lib/gillijimproject (Alpha-first), not a subset.
- Preserve binary semantics, offsets, and bounds exactly as reference C++.
- Operate efficiently on multi-GB Alpha containers via streaming IO (no ReadAllBytes).
- Validate on real 0.5.3 data across many tiles; zero tolerance for silent fallback.

## Scope (Complete Coverage)
- Alpha WDT container handling with tolerant MAIN discovery.
- MAIN → M(P/M)HD → MHDR → MCIN → MCNK pipeline with Alpha header-driven offsets.
- Terrain subchunks: MCVT, MCNR, MCCV, MCLQ (Alpha fallback rules), MCSE, MCBB.
- World object chunks: MWMO/MODF, MMDX/MMID, MWID/MDDF, etc.
- WDL and map index structures where used by original.
- Diagnostics and CLI outputs mirroring original’s visibility.

## IO Design (Streaming, Large Files)
- Use FileStream + BinaryReader throughout. No byte[] file-wide buffers.
- All APIs accept (BinaryReader br, long baseOffset) where relevant.
- Seek to absolute offsets; compute relative offsets exactly like the reference.
- Implement tolerant MAIN locator: scan the container (4-byte aligned) for tag 'MAIN'; for each hit, read size and validate bounds; verify 16-byte stride table shape; select first valid.

## Path Conventions
- All paths are relative to `./` (refactor/gp-csharp/).
- Examples: `./Utilities/Utilities.cs`, `./WowFiles/Mcin.cs`, `./GillijimProject.csproj`.

## Repository Layout
- ./
  - GillijimProject.csproj (net9.0)
  - README.md
  - PLAN.md (this file)
  - Utilities/
    - Utilities.cs (FourCC, endian helpers, BinaryReader safe reads, assertions)
  - WowFiles/
    - WowChunkedFormat.cs (streaming chunk helpers; MAIN locator)
    - Chunk.cs (ChunkHeader reader from stream + offset)
    - ChunkHeaders.cs (tag constants)
    - Wdt.cs (orchestrator, streaming)
    - Main.cs (Alpha MAIN parser, 16-byte stride)
    - Mphd.cs (Alpha MPHD/MPhd variants as applicable)
    - Mhdr.cs (Alpha MHDR parser)
    - Mcin.cs (Alpha MCIN parser, 256 entries)
    - Mcnk.cs (Alpha MCNK header; subchunk offset derivation; bounds)
    - Terrain/
      - Mcvt.cs, Mcnr.cs, Mccv.cs, Mclq.cs, Mcse.cs, Mcbb.cs (decoders or validators)
    - Objects/
      - Mmdx.cs, Mmid.cs, Mwmo.cs, Mwid.cs, Mddf.cs, Modf.cs (index/object tables)
    - Wdl/
      - Wdl.cs (as per original usage)

## Exact Semantics To Preserve (Alpha)
- MAIN: 16-byte stride entries; first uint = MHDR offset (relative to MAIN payload base); Size may be 0; tile presence if Offset != 0.
- MHDR: first uint = MCIN offset relative to MHDR payload base.
- MCIN: 256 entries, 8 bytes each (uint32 ofsRel, uint32 sizeUnused).
  - Absolute MCNK offset = tileStart + ofsRel (tileStart = MAIN.PayloadBase + entry.MhdrOffset).
  - Validate MCNK tag before accepting entry.
- MCNK (Alpha header size = 0x80):
  - mcvtRel @ 0x18; mcnrRel @ 0x1C; chunksSize @ 0x64; mclqRel @ 0x68 (if > 0)
  - MCVT size = 145 * 4 = 580 bytes
  - MCNR size = 145 * 3 = 435 bytes
  - Strict bounds: subchunks must be within [mcnkPayload, mcnkPayload + chunksSize]
- MCLQ: keep Alpha-specific fallback/semantics when unknown.
- No scanning heuristics for subchunks beyond header-driven offsets.

## Milestones (1:1 Parity Track)
1) Streaming foundation
   - Replace all byte[] file loads with FileStream + BinaryReader.
   - Add safe read helpers with bounds checks.
2) MAIN locator
   - Implement tolerant 'MAIN' search; validate candidate by bounds and table shape.
3) MAIN parsing (Alpha)
   - Parse 16-byte stride; presence by offset!=0; expose entries with absolute offsets.
4) MHDR → MCIN resolution
   - Resolve MCIN header via MHDR payload-relative offset; return absolute positions.
5) MCIN → MCNK
   - Parse 256 entries; compute absolute MCNK offsets; validate MCNK tag.
   - Read MCNK header (0x80) and compute absolute subchunk offsets.
6) Terrain subchunks
   - Implement readers/validators for MCVT, MCNR; add MCCV/MCSE/MCBB where applicable; MCLQ fallback.
7) World object/index chunks
   - Implement MMDX/MMID, MWMO/MWID, MDDF/MODF parsing as per original usage.
8) WDL (if used by reference flows)
   - Implement required structures and parsers.
9) CLI parity & diagnostics
   - `dump-wdt <path> [--start N] [--tiles K] [--mcnk M] [--verbose]`.
   - Show MAIN entries, MHDR presence, MCIN stats, sample MCNK subchunk validations.
10) Real-data validation
   - Run on `test_data/0.5.3/alphawdt/Kalimdor.wdt` and other maps.
   - Confirm non-zero heights/normals, subchunk bounds, object tables sanity.

## Validation Strategy (Real Data Only)
- Use the provided `test_data/` Alpha files; no mocks.
- For each phase, validate offsets/tags/counts across multiple tiles.
- Terrain checks: MCVT=580, MCNR=435 within bounds, values non-zero in representative tiles.
- Objects: counts and offsets consistent with indices.
- Add verbose diagnostics to surface offset math and bounds quickly.

## CLI Behavior (Performance-aware)
- Streamed reads only; never load the entire file.
- Bounded sampling: by default process first K tiles and first M MCNKs unless --all is set.
- `--verbose` prints absolute offsets for MAIN/MHDR/MCIN/MCNK/subchunks and bounds evaluations.

## Performance & Safety
- Sequential scan for MAIN with buffered IO; 4-byte alignment by default; allow unaligned if needed via flag.
- All seeks validated; prevent overflow/underflow on offset arithmetic (checked math where necessary).
- Early-exit on invalid structures with precise error messages including offsets/tags.

## Notes
- All operations are little-endian.
- Preserve Alpha header-driven computations; avoid heuristic rescans for subchunks.
- Keep implementation modular to extend to additional chunks without regressions.