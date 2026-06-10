# Coding Standards — wow-viewer

## FourCC Handling
- Define constants readable: `const string SIG_MTEX = "MTEX";`
- Reverse on read, reverse on write. Never compare against reversed strings.
- Log readable signatures only.

## Code Style
- File-scoped namespaces (`namespace WowViewer.Core.IO;`)
- `var` when type is obvious, explicit types otherwise
- Auto-properties over fields
- XML comments (`/// <summary>`) for public APIs

## Project Structure
- Library-first: format readers live in `WowViewer.Core.*` libraries
- Tools are thin CLI wrappers in `tools/`
- Tests in `tests/` mirroring the library structure
- One project, one concern

## Memory Bank
- Update `activeContext.md` after every logical change
- Keep it lean (< 100 lines); use `progress.md` for chronological log
- Archive old spec dirs rather than deleting
