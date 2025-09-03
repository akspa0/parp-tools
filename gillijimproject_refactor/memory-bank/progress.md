# Progress

- Works:
  - Project scaffolded (.csproj, folders)
  - `Utilities.cs`, `WowChunkedFormat.cs`, `Chunk.cs`
  - Alpha: `MainAlpha`, `Mdnm`, `Monm` implemented; `WdtAlpha` implemented
  - FourCC on-disk handling confirmed; `Chunk.GetOffset()` fixed to read LE int from `Data`
  - ADT Alpha: parsed `MHDR`, `MCIN`, `MTEX`, `MDDF`, `MODF`; `MCIN` offsets implemented
  - `Mddf`/`Modf` helpers implemented (indices, LK remap, height helpers)
  - Index chunk builders implemented: `Mmdx`/`Mmid`, `Mwmo`/`Mwid` from indices + name lists
  - `AdtAlpha` builds richer LK ADTs (MVER, stub MHDR/MH2O, MTEX, MMDX/MMID, MWMO/MWID, MDDF/MODF w/ remapped indices)
  - CLI: `Program.cs` loads Alpha WDT and prints tile/model samples
  - Skeletons: Alpha (`McnkAlpha`, `McnrAlpha`, `McvtAlpha`), Core (`Main`, `Mhdr`, `Mcin`, `Mcal`, `Mcrf`, `Mddf`, `Mh2o`, `Mmdx`, `Mmid`, `Modf`, `Mwid`, `Mwmo`, `Wdl`, `Mcnk`, `ChunkHeaders`), LK (`McnkLk`, `McnrLk`)
  - Fixed method references: renamed `ByteArrayToStructure` to `ByteArrayToStruct` across codebase
  - Fixed McnkHeader field accessors: changed lowercase to PascalCase
  - Fixed constructor issues in `Mh2o`, `Mmid`, `Mmdx`, and `Mwid` classes
  - Implemented `ReadBytes` method in `WowChunkedFormat`
  - Implemented `Mcrf.UpdateIndicesForLk()` to parse from `Data` and remap via dictionaries
  - `MainAlpha.ToMain()` now passes `LichKing.MhdrOffset[]`
  - `AdtAlpha` builds ordered MMDX/MWMO lists and correct maps for `McnkAlpha.ToMcnkLk()`
  - Added 'new' keyword to hide warnings for shadowed constants in `McnkAlpha` and `McnkLk` classes
  - Fixed parameter type mismatch in `McnkAlpha` (changed from `Chunk` to `Mcal`)
  - Initialized non-nullable fields in constructors to fix CS8618 warnings
  - FourCC policy enforced: standardized forward FourCC literals across WowFiles; reversal centralized in `WowFiles/Chunk.cs`; rules updated in `.windsurf/rules/csharp-port.md`; audit confirms no reversed literals remain outside `Chunk.cs` (KNCM, NICM, TVCM, RNCM, HSCM, LACM, QLCM, OBFM, FXTM, RDHM)
  - Resolved most build errors by creating stub classes, fixing inheritance, and correcting method signatures.

- Pending:
  - Add `Mcnk(string letters, int givenSize, byte[] chunkData)` constructor overload in `WowFiles/Mcnk.cs` and rebuild.
  - Implement functionality for stubbed classes (`Mcal`, `Mclq`, `McnkLk`, etc.).
  - Build solution and smoke test with known assets (WDT/ADT) to output directory (`-o/--out`)
  - Populate LK `MHDR`/`MCIN` with correct offsets and finalize chunk order in `AdtLk`
  - Port `McnkAlpha` parsing and LK conversion
  - Validate against `reference_data/wowdev.wiki`

- Known Issues:
  - 2 build errors remain (CS1729) due to missing `Mcnk(string,int,byte[])` base constructor referenced by `McnkAlpha`.
  - Many chunk classes are skeletons and lack full implementation (`Mcal`, `Mclq`, `Mh2o`, etc.).
  - `MHDR` is currently stubbed; MCNK chunks are not yet fully written to the final ADT.
  - 25 warnings remain in the codebase.
