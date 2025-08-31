# System Patterns

- Architecture: Chunked binary parsing (FourCC + size + data) composed into higher-level ADT/WDT structures.
- Layers:
  - Primitives: `WowChunkedFormat`, `Chunk`, Utilities.
  - File Models: `WdtAlpha`, `Wdt`, `AdtAlpha`, `AdtLk`.
  - CLI: `Program` orchestrates read/convert/write.
- Patterns: Stream/Span-based IO; little-endian; immutable value exposure; `IDisposable` via using statements on streams.
