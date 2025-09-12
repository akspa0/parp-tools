# WDL Parsing Plan (Next)

- Document: `next/docs/wdl-parsing-plan.md`
- Owner: Next Core (IO + Domain)
- Date: 2025-09-05
- Status: Approved

## Overview

- Goal: Implement robust WDL parsing for low-resolution terrain meshes, aligned with Noggit’s reading model and wowdev WDL v18 documentation.
- Outcomes:
  - Access a 64x64 grid of WDL tiles, each with two height grids (17x17 and 16x16).
  - Optional hole mask (MAHO) parsing per tile.
  - Compatibility with forward and reversed FourCC chunk tags.
  - Foundation for CLI verbs (dump/build) and a WDL writer.

## References

- Code
  - `next/src/GillijimProject.Next.Core/IO/AlphaReader.cs`
  - `next/src/GillijimProject.Next.Core/Domain/AdtAlpha.cs` (temporary home for `Wdl`/`WdlTile`; will move to `Domain/Wdl.cs`)
  - `next/tests/GillijimProject.Next.Tests/SmokeTests.cs`
- Format
  - `reference_data/wowdev.wiki/WDL_v18.md`
- Porting source-of-truth
  - `src/gillijimproject-csharp` (ignore `refactor/gp-csharp`)

## Goals

- Parse WDL files and expose a domain model with:
  - 64x64 `WdlTile` grid.
  - For each present tile: `short[17,17]` + `short[16,16]` height grids.
  - Support `MAOF` (offset table), `MARE` (tile data), and optional `MAHO` (holes).
  - Recognize `MVER` (expect version 18 if present).
  - Handle reversed FourCCs (e.g., `FOAM`/`ERAM`/`REVM`/`OHAM`).
  - Implement robust size, bounds, and padding handling.

## Non-Goals (now)

- Writing WDL files (follow-up).
- Rendering images/CSV in this phase (CLI dump later).
- Detailed logging/telemetry beyond correctness checks.

## Domain Model

- File: `next/src/GillijimProject.Next.Core/Domain/Wdl.cs` (split from `AdtAlpha.cs`)
- Types
  - `Wdl`
    - `string Path`
    - `WdlTile?[,] Tiles` (64x64)
  - `WdlTile`
    - `const int OuterGrid = 17`
    - `const int InnerGrid = 16`
    - `short[,] Height17` (17x17)
    - `short[,] Height16` (16x16)
    - Holes:
      - `ushort[] HoleMask16` (length 16), each entry is a 16-bit row mask (bit set => hole).
      - Helper: `bool IsHole(int y, int x) => (HoleMask16[y] & (1 << x)) != 0;`

## Parser Design

- File: `next/src/GillijimProject.Next.Core/IO/AlphaReader.cs`
- Entry point: `AlphaReader.ParseWdl(string wdlPath)`
- Behavior
  - Scan top-level chunks, tolerate unknowns; stop after `MAOF` (Noggit behavior).
  - `MVER`
    - If present and size >= 4, read version; tolerate absence; TODO: log mismatch if not 18.
    - Skip trailing payload; apply padding if size is odd.
  - `MAOF`
    - Read up to 4096 offsets (64x64) as `uint32`, absolute file offsets; zero = empty.
    - For each non-zero offset:
      - Seek to offset; expect `MARE` (reversed allowed). Validate size/bounds with `ulong` arithmetic.
      - Read 17x17 then 16x16 `int16` heights. Tolerate extra bytes by skipping to end of `MARE` payload.
      - Apply `MARE` data padding if size is odd before reading the next chunk.
      - Immediately check for `MAHO` (reversed allowed):
        - Read size and up to 16 x `UInt16`; zero-fill if short; skip extras if long; apply padding.
        - If no `MAHO`, default to zero masks.
      - Construct `WdlTile` with heights + hole mask and assign `Tiles[y,x]`.
- Reversed FourCC support
  - `Matches("MAOF")` => `"MAOF"` or `"FOAM"`
  - `Matches("MARE")` => `"MARE"` or `"ERAM"`
  - `Matches("MVER")` => `"MVER"` or `"REVM"`
  - `Matches("MAHO")` => `"MAHO"` or `"OHAM"`

## Edge Cases & Safety

- Use `ulong` for boundary math to prevent overflow on `offset + size`.
- Check `fs.Position + needed <= len` before reads; if violated, skip tile safely.
- For unknown chunks, skip `size` bytes and pad when `size` is odd.
- If `MARE` size < expected 1090 bytes (17*17*2 + 16*16*2), skip tile.
- If holes chunk is missing, treat as all zeros (per wowdev).

## Testing Strategy

- Unit/Smoke tests: `next/tests/GillijimProject.Next.Tests/SmokeTests.cs`
  - Minimal WDL with single `MARE` tile; verify sample heights.
  - Reversed FourCC WDL (`REVM`, `ERAM`, `FOAM`); verify sample heights.
  - MAHO synthetic
    - Normal FourCC: `MARE` then `MAHO` with known bit pattern; validate `IsHole(y,x)`.
    - Reversed FourCC: `ERAM` then `OHAM` with known bit pattern; validate.
  - Fixture-based integration (skip-if-missing)
    - Search `next/test_data/` or `../test_data/` for any `.wdl`.
    - If none, log `[skip] …` and return.
    - If found, parse and assert at least one tile present; optionally spot-check.
- Style
  - Use temp dirs for synthetic tests; cleanup in `finally`.
  - Keep assertions defensive (bounds, non-null tiles, expected samples).

## CLI (Future)

- `wdl-dump`
  - Output: CSV or image heatmap of low-res terrain.
  - Options: region selection, normalization/scaling.
- `wdl-build`
  - Synthesize WDL from ADTs (low-res mesh generation).
  - Compare vs WDL provided by AlphaWDT; report differences.

## Writer (Future)

- Generate WDL
  - Compute 17x17 + 16x16 heights per tile from ADT data.
  - Build `MAOF` with absolute offsets.
  - Write `MARE` tiles and optional `MAHO` masks.
  - Respect reversed FourCC on disk if required.
- Verification
  - Compare against fixture WDLs; report tile coverage and statistical diffs.

## Milestones

- M1 — Organization
  - Split `Wdl`/`WdlTile` into `Domain/Wdl.cs`.
- M2 — Parser Hardening
  - Ensure consistent padding after chunk reads (esp. after `MARE`).
  - Keep `MVER` read/tolerance; TODO for logging version mismatch.
- M3 — MAHO Support
  - Implement holes parsing (`MAHO`/`OHAM`) and extend domain model.
- M4 — Tests
  - Add synthetic MAHO tests (normal + reversed).
  - Add skip-if-missing WDL fixture test.
- M5 — Docs & Memory Bank
  - Land this plan; update Next memory bank `activeContext.md` and `progress.md` after implementation.

## Acceptance Criteria

- Parsing
  - `MARE` heights read correctly for present tiles; bounds-safe behavior on malformed inputs.
  - `MAHO` parsed when present; zero masks when absent.
  - Reversed FourCCs recognized for `MVER`/`MAOF`/`MARE`/`MAHO`.
- Tests
  - All synthetic smoke tests pass.
  - Fixture-based test skips when missing; passes basic assertions when present.
- Organization
  - `Wdl`/`WdlTile` exist in `Domain/Wdl.cs`.

## Risks & Mitigations

- Real files with unusual chunk ordering
  - Mitigation: Scan unknown chunks safely; stop after `MAOF` (Noggit pattern).
- Incomplete or malformed chunks
  - Mitigation: Strict bounds checks; skip tiles on violation; default hole masks.
- Future format variants
  - Mitigation: Keep implementation modular; easy to extend chunk recognition.

## Open Questions

- Expose `bool[,] HoleMask16x16` in addition to `ushort[] HoleMask16`?
- For integration tests, prioritize any specific world/continent if multiple `.wdl` files are found?

## Work Items (mapped to TODOs)

- Inventory mapping (src/gillijimproject-csharp): in_progress
- WDL parser implementation (basic): completed
- Split Wdl/WdlTile into Domain/Wdl.cs: completed
- Parser hardening (MVER, padding, size checks): completed
- MAHO support (holes domain + parsing): completed
- WDL integration tests (skip-if-missing): completed
- CLI verbs (wdl-dump, wdl-build): pending
- WDL writer and verification: pending
