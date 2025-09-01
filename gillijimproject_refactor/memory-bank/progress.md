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
  - Fixed `Mcrf` constructor and added missing methods (`UpdateIndicesForLk`, `GetDoodadsIndices`, `GetWmosIndices`)
  - Added 'new' keyword to hide warnings for shadowed constants in `McnkAlpha` and `McnkLk` classes
  - Fixed parameter type mismatch in `McnkAlpha` (changed from `Chunk` to `Mcal`)
  - Initialized non-nullable fields in constructors to fix CS8618 warnings

- Pending:
  - Fix remaining 23 build errors and 25 warnings
  - Build solution and smoke test with known assets (WDT/ADT) to output directory (`-o/--out`)
  - Populate LK `MHDR`/`MCIN` with correct offsets and finalize chunk order in `AdtLk`
  - Port `McnkAlpha` parsing and LK conversion
  - Validate against `reference_data/wowdev.wiki`

- Known Issues:
  - 23 build errors remain in the codebase
  - `MHDR` currently stubbed; MCNK chunks not yet written
  - Parameter names and types in some methods need to be corrected
  - Some non-nullable references may still be uninitialized
