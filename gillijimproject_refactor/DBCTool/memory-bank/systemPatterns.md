# System Patterns

- Architecture:
  - CLI parses args → chooses provider (`FilesystemDBCProvider` or `MpqDBCProvider`) → uses DBCD to load → exports CSV.
  - Output is organized by build and timestamp under `out/`.

- MPQ Reading:
  - Locale-only filter: restrict to `Data/<locale>/` archives; additionally include core base/patch MPQs so patch overlays have a base.
  - Exclusions: ignore speech/backup MPQs.
  - Patch chain (WotLK):
    - Core patches: `patch.MPQ`, `patch-2.MPQ`, `patch-3.MPQ`
    - Locale bases: `base-<locale>.MPQ`, `locale-<locale>.MPQ`, `expansion-locale-<locale>.MPQ`, `lichking-locale-<locale>.MPQ`
    - Locale patches: `patch-<locale>.MPQ`, `patch-<locale>-2.MPQ`, `patch-<locale>-3.MPQ`
  - Locale prefix: pass `<locale>\` when calling `SFileOpenPatchArchive` for locale MPQs.
  - Validation: after open, verify `WDBC` header before returning a stream.

- Diagnostics:
  - `--mpq-verbose` prints archive opens, has-file checks, scope attempts, sizes, and header checks.
  - `--debug-mpq-file <TableName>`: probe which archives expose `DBFilesClient/<Table>.dbc`.
  - `--mpq-list <archive> [--mask <pattern>]`: list files (best-effort without listfile).
  - `--mpq-test-open <archive> <mpqPath>`: open and dump header bytes for a single file.

- Performance:
  - Direct-open pass targets core archives only (locale patch fragments require composite view).
  - Chunked reads with size guards; unknown-size fallback with a 64MB ceiling.

- Principles:
  - Do not vendor DBCD; reference it as a project dependency.
  - Keep tool logic local; do not alter unrelated core libraries.
