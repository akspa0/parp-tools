# Coding Standards

## FourCC: readable in memory, reversed only at I/O boundaries. Never `"XETM"` in constants/logic.

## Style: file-scoped namespaces, `var` when type obvious, auto-properties, XML comments for public APIs.

## Structure: each CLI tool gets own project; shared parsing in `WoWRollback.Core` or library projects.

## Memory Bank Rules
- `activeContext.md`: ~50 lines max, only current focus
- `progress.md`: status tables only, no session logs
- Archive detailed logs to `archive/`
