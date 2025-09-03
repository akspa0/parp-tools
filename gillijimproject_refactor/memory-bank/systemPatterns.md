# System Patterns

- Architecture: Chunked binary parsing (FourCC + size + data) composed into higher-level ADT/WDT structures.
- Layers:
  - Primitives: `WowChunkedFormat`, `Chunk`, Utilities.
  - File Models: `WdtAlpha`, `Wdt`, `AdtAlpha`, `AdtLk`.
  - CLI: `Program` orchestrates read/convert/write.
- Patterns: Stream/Span-based IO; little-endian; immutable value exposure; `IDisposable` via using statements on streams.
- FourCC policy: forward literals in code; on-disk reversed; reversal centralized in `WowFiles/Chunk.cs`; no reversed literals outside `Chunk.cs`.
- Review tokens to grep for reversed literals in code: KNCM, NICM, TVCM, RNCM, HSCM, LACM, QLCM, OBFM, FXTM, RDHM.
- Offsets in MHDR-like chunks are stored as 32-bit LE integers in `Chunk.Data`; use `Chunk.GetOffset()` and add to MHDR start-of-data for absolute positions.
- Index-chunk construction: `MMDX`/`MWMO` are NUL-separated name pools; `MMID`/`MWID` store 32-bit offsets starting at 0 at each NUL boundary.
- Scope: LK-only (no Cataclysm). Porting conventions: one C++ file → one C# file; `[PORT]` notes; XML docs for public APIs.
- Constructor pattern: All non-nullable fields must be initialized in all constructors (including base constructors) to avoid CS8618 warnings.
- Inheritance pattern: Use `new` keyword when hiding base class members with the same name to avoid CS0108 warnings.
- Type consistency pattern: Ensure parameter types match between method signatures and their implementations/callers (e.g., `Mcal` vs `Chunk`).
- Base wrapper pattern: Abstract chunk wrappers (e.g., `Mcnk`) expose a `(string letters, int givenSize, byte[] data)` constructor that forwards to `Chunk` to support derived wrappers assembling payloads in memory.
