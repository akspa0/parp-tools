# System Patterns

- Architecture: Chunked binary parsing (FourCC + size + data) composed into higher-level ADT/WDT structures.
- Layers:
  - Primitives: `WowChunkedFormat`, `Chunk`, Utilities.
  - File Models: `WdtAlpha`, `Wdt`, `AdtAlpha`, `AdtLk`.
  - CLI / Library: CLI orchestrates; library exposes public APIs for parse/convert/write.
- Patterns:
  - Stream/Span-based IO; little-endian; immutable value exposure; `using` statements for streams.
  - FourCC: forward in memory; reversed on disk during serialization.
  - Offsets in MHDR-like chunks stored as 32-bit LE within `Chunk.Data`; use `Chunk.GetOffset()`; absolute = MHDR start-of-data + offset.
  - Index-chunk construction: `MMDX`/`MWMO` = NUL-separated names; `MMID`/`MWID` = 32-bit offsets at each NUL boundary, excluding the final terminator (no `i+1 == Data.Length`).
  - MCNK ordering: write `MCLQ` last; update header offsets accordingly. Omit `MH2O` when empty.
- Scope: LK-only (no Cataclysm) for now. Porting conventions: one C++ file → one C# file; `[PORT]` notes; XML docs for public APIs.
- Constructor pattern: Initialize all non-nullable fields to avoid CS8618.
- Inheritance pattern: Use `new` when hiding base members (CS0108).
- Type consistency: Keep method signatures aligned across call sites (e.g., `Mcal` vs `Chunk`).

## Library Architecture (Upcoming)
- Projects: `GillijimProject.Core` (Class Library), `GillijimProject.Cli` (Console wrapper).
- Public API: parse (Alpha), convert (to LK), write (ADT/WDT). Exceptions for errors; cancellation-friendly where appropriate.
- Integration: adapters/facades to `Warcraft.NET` writer components to improve safety and performance while preserving output compatibility.
