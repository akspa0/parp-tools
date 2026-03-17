# AI Guidelines

- Read source files before coding. Preserve existing functionality. Follow existing code style.
- MPQ discovery: call `_mpq.GetAllKnownFiles()` → `_fileSet`. Use `StringComparer.OrdinalIgnoreCase`.
- Nested WMO archives: `ScanWmoMpqArchives()`. Path normalization: `file.Replace('/', '\\')`.
- Build: `dotnet build --no-restore`. Always verify build succeeds after changes.
5. Check path normalization
