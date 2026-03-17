# AI Agent Guidelines for MdxViewer

- Read source files before coding. Preserve working features. Follow existing code style.
- MPQ discovery: call `_mpq.GetAllKnownFiles()` and add to `_fileSet`. Check `[MpqDataSource]` logs.
- Case sensitivity: Alpha uses uppercase (.MPQ). Use `StringComparer.OrdinalIgnoreCase`.
- WMO nested archives: use `ScanWmoMpqArchives()` for `.wmo.MPQ` files.
- Path normalization: `file.Replace('/', '\\')`.
- Extensions: `.ToLowerInvariant()` then match `.mdx`, `.wmo`, `.m2`, `.blp`.
