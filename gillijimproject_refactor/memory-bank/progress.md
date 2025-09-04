# Progress

- Works:
  - 1:1 C++ → C# parity achieved for Alpha WDT → Wrath ADT conversion; outputs validated with 010 templates.
  - Core primitives: `Utilities.cs`, `WowChunkedFormat.cs`, `Chunk.cs` stable.
  - Alpha: `WdtAlpha` parsing; `AdtAlpha` to `AdtLk` conversion pipeline.
  - LK: `AdtLk` assembly and serialization; MHDR/MCIN offsets stable for current assets.
  - Fixed bugs and correctness items:
    - MMID/MWID off-by-one eliminated (no extra final offset).
    - MCVT forward FourCC in memory; on-disk reversed correctly.
    - MH2O omitted when empty; MCNK writes `MCLQ` last.
    - Index and offset handling consolidated in `Chunk` and helpers.

- Pending:
  - Refactor into `GillijimProject.Core` (Class Library, net9.0) + `GillijimProject.Cli` (thin wrapper).
  - Define public API for parse/convert/write; add XML docs.
  - Add test project and smoke/integration tests with fixtures.
  - Integrate Warcraft.NET writer APIs via adapters/facades where applicable.
  - Prepare NuGet packaging for the core library; basic CI later.

- Known Issues / Follow-ups:
  - Broader asset coverage and performance profiling TBD.
  - Document edge cases and invariants (e.g., optional chunks, alignment/padding) as we expand tests.
