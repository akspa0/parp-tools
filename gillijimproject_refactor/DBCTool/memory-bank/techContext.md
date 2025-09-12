# Tech Context

- Runtime: .NET 9.0; 64-bit only (x64)
- Project: `DBCTool/DBCTool.csproj` with `PlatformTarget` = `x64`
- Libraries:
  - DBCD via ProjectReference: `..\lib\wow.tools.local\DBCD\DBCD\DBCD.csproj` (no vendoring)
  - WoWDBDefs (DBD definitions) default path: `lib/WoWDBDefs/definitions`

## Filesystem-only Mode

- Input is a local DBC directory (typically `.../DBFilesClient/`).
- Build can be provided as an alias or inferred from the path:
  - Aliases: `0.5.3`, `0.5.5`, `3.3.5`
  - Canonical builds (for DBCD): `0.5.3.3368`, `0.5.5.3494`, `3.3.5.12340`
- Output is written to version folders without timestamps:
  - `out/0.5.3/<Table>.csv`, `out/0.5.5/<Table>.csv`, `out/3.3.5/<Table>.csv`

## CLI Usage

- Export single version (alias in input):
  ```powershell
  dotnet run --project .\DBCTool\DBCTool.csproj -- \
    --dbd-dir .\lib\WoWDBDefs\definitions \
    --out out \
    --locale enUS \
    --table AreaTable --table Map \
    --input 3.3.5=H:\extract\DBFilesClient
  ```

- Export single version (bare directory; alias inferred from path tokens 0.5.3|0.5.5|3.3.5):
  ```powershell
  dotnet run --project .\DBCTool\DBCTool.csproj -- \
    --dbd-dir .\lib\WoWDBDefs\definitions \
    --out out \
    --locale enUS \
    --table AreaTable \
    --input ..\test_data\0.5.3\tree\DBFilesClient
  ```

- Compare AreaTable across versions (map to 3.3.5 IDs, check ContinentID consistency):
  ```powershell
  dotnet run --project .\DBCTool\DBCTool.csproj -- \
    --dbd-dir .\lib\WoWDBDefs\definitions \
    --out out \
    --locale enUS \
    --compare-area \
    --input 0.5.3=..\test_data\0.5.3\tree\DBFilesClient \
    --input 0.5.5=..\test_data\0.5.5\tree\DBFilesClient \
    --input 3.3.5=..\test_data\3.3.5\tree\DBFilesClient
  ```

## Output Layout

- Exports write to `DBCTool/out/<alias>/` without timestamps.
- Comparison outputs:
  - `out/compare/AreaTable_mapping_0.5.3_0.5.5_3.3.5.csv`
  - `out/compare/AreaTable_continent_mismatches.csv`

## Principles

- Use upstream DBCD as a ProjectReference
- Keep changes local to the tool; do not modify shared core libraries
- Filesystem-only; MPQ support removed
