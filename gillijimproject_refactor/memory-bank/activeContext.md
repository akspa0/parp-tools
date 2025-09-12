# Active Context

- Current Focus: Refactor the working parity implementation into a reusable library (`GillijimProject.Core`) and integrate Warcraft.NET for safer, more efficient ADT writing. Maintain LK output compatibility.
- Recent Changes:
  - Achieved 1:1 C++ → C# parity for Alpha WDT → Wrath ADT conversion.
  - Fixed MMID/MWID off-by-one (no spurious final offset); corrected MCVT forward FourCC usage; ensured MH2O is omitted when empty; enforced MCLQ written last in MCNK.
  - Stabilized ADT outputs validated with 010 Editor templates.
- Next Steps:
  - Create `GillijimProject.Core` (Class Library, net9.0) and move core parsing/conversion/writing logic behind a public API.
  - Keep `GillijimProject.Cli` as a thin wrapper over the library.
  - Add test project for smoke/integration tests over representative Alpha WDT/ADT samples.
  - Introduce Warcraft.NET as writer backend where applicable; add adapters/facades as needed.
  - Document the public API (XML docs) and prepare for NuGet packaging.
- Decisions:
  - Library-first architecture; CLI delegates to library.
  - Forward FourCC in memory; reversed on disk by serializer.
  - Exceptions for error handling; immutable domain where practical; builders/writers handle mutation.
