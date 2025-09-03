# Active Context

- Current Focus: Complete the MCNK pipeline. Immediate fix: add missing base constructor `Mcnk(string letters, int givenSize, byte[] chunkData)` and rebuild.
- Recent Changes:
  - `Mcrf.UpdateIndicesForLk()` now parses indices from `Data` (M2s then WMOs), remaps via dictionaries, and returns a new chunk.
  - `MainAlpha.ToMain()` now constructs and passes `LichKing.MhdrOffset[]` to `LichKing.Main`.
  - `AdtAlpha` builds ordered MMDX/MWMO lists from Alpha (`_mddf`, `_modf`) and passes correct maps to `McnkAlpha.ToMcnkLk()`.
  - Utilities provides `ByteArrayToStruct<T>`, `StructToByteArray<T>`, and `GetByteArrayFromFile(...)` helpers used across WowFiles.
- Next Steps:
  - Add `Mcnk(string,int,byte[])` constructor overload in `WowFiles/Mcnk.cs`, rebuild, and triage remaining issues.
  - Continue MCNK payload assembly (height/normal layers, RF/LY integration) and finalize MHDR/MCIN offsets in `AdtLk`.
- Decisions:
  - Abstract chunk wrappers (e.g., `Mcnk`) expose a `(string letters, int givenSize, byte[] data)` constructor that forwards to `Chunk` to support derived wrappers that assemble payloads in memory.
