# CLI Contract: `inspect adt-ahdr`

Host: `tools/inspect/WowViewer.Tool.Inspect`. This is a thin wrapper over `WowViewer.Core.IO.Maps` (Library-First).
After implementation, `docs/CLI-TOOLS.md` must be diffed against the real argument parser.

## `adt-ahdr inventory`

```
inspect adt-ahdr inventory --root <dir> [--recursive] [--json <out.json>] [--limit <n>]
```

- Walks every file whose first chunk is `AHDR` (extension-agnostic; includes `.error`).
- **stdout**: one line per file (`kind version size chunks unaccounted=<bytes> status`), then aggregates: version histogram, chunk-occurrence table (id, parent, count, size min/median/max), unknown chunks, documented-size disagreements, failed files.
- **--json**: the full `AdtAhdrInventory` (files with SHA-256, chunk records, aggregates, distinct ATEX/ADOO names).
- **Exit code**: 0 when every file was walked (even with disagreements); 2 when any file failed to walk; 1 on usage error.

## `adt-ahdr dump`

```
inspect adt-ahdr dump --file <path> [--chunk <x>,<y>] [--json <out.json>]
```

- Header; name tables; height/normal stats (min/max/mean, normal magnitude histogram); per-chunk area id, layers (texture name, flags, alpha encoding and reason), shadow coverage %, placements (model name, target, position, rotation, scale, uniqueId); diagnostics.
- `--chunk` limits the per-chunk section to one chunk.
- **Exit code**: 0 on decode (diagnostics allowed); 2 when the file is not an AHDR file.

## `adt-ahdr layout-probe`

```
inspect adt-ahdr layout-probe --root <dir> [--json <out.json>]
```

- Runs the R4/R5/R6/R9 probes. For each probe: candidates, score per candidate, winner, and margin to the runner-up, plus the **detector power control** result (score of a deliberately corrupted correct candidate).
- Prints `INCONCLUSIVE` rather than a winner when the margin is below the probe's threshold or the power control fails.
- **Exit code**: 0 when every probe is conclusive; 3 when any probe is inconclusive.

## Existing surface change

`inspect map <file>` on an AHDR file prints `ADT/v22 semantics:` or `ADT/v23 semantics:` according to the detected version (currently always `v23`).
