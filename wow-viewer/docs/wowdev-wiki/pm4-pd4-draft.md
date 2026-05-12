# Draft: wowdev.wiki PM4 / PD4 Page Refresh

Status: concise handoff draft for the existing wowdev PM4 and PD4 pages.

## Proposed Changes

1. Keep the current split:

- `PD4` owns the common chunk family: `MVER`, `MCRC`, `MSHD`, `MSPV`, `MSPI`, `MSCN`, `MSLK`, `MSVT`, `MSVI`, `MSUR`
- `PM4` references those and documents the PM4-only chunks: `MPRL`, `MPRR`, `MDBH`, `MDBI`, `MDBF`, `MDOS`, `MDSF`

1. Add parser-stable record sizes:

- `MSLK`: 20 bytes
- `MSUR`: 32 bytes
- `MPRL`: 24 bytes
- `MPRR`: 4 bytes
- `MDBH`: 4 bytes
- `MDBI`: 4 bytes
- `MDOS`: 8 bytes
- `MDSF`: 8 bytes

1. Slightly improve `PD4#MSLK` without renaming the page-level field table.

Suggested note:

> Keep the raw field names on the wiki for now. If a short note is useful, mention that current shared tooling informally treats `_0x04` as an object-group key candidate, `MSPI_first_index` / `_0x0b` as a validated index window, and `_0x10` as an active relationship candidate. Do not turn those into global field renames yet.

1. Slightly improve `PD4#MSUR` the same way.

Suggested note:

> Keep the raw field names on the wiki for now. A short supporting note can say that current tooling uses `_0x01` as the validated index count, `MSVI_first_index` as the validated offset into `MSVI`, and treats `_0x18` / `_0x1c` as useful relationship-bearing fields without claiming final semantic names.

1. Add a short `MSCN` caution:

> Current corpus tooling treats `MSCN` as meaningful connector-family evidence when grouped through `MSUR` / `MDOS`; it should not be dismissed as “probably normals” without further proof.

1. Tighten `PM4#MPRL`:

> `MPRL` is currently best treated as a position-reference chunk. Shared tooling uses `position` directly for placement-space comparison and treats `_0x14` as a likely rotation-bearing field used only as supporting evidence for now.

1. Keep `PM4#MPRR` conservative:

> `MPRR` parses cleanly as `uint16` pairs, but its runtime relationship to `MPRL`, `MSVT`, or placement matching is still unresolved.

1. Make the PM4-only relationship chunks more explicit:

- `MDBH`: destructible-building count
- `MDBI`: destructible-building index
- `MDBF`: destructible-building filename
- `MDOS`: destructible-building index + destruction state
- `MDSF`: `MSUR` index + `MDOS` index

## Editorial Style

- keep raw `_0xNN` fields where semantics are not closed
- if a later wowdev page already has the right name for the same thing, reuse that existing wiki term
- do not promote tooling aliases into global wiki field names unless they are backed by a real semantic reason
- if a tooling alias is still useful, keep it in prose as a local note rather than renaming the structure table
- prefer “Verified” or “Working interpretation” over stronger claims

## Repo-Backed Sources

- `wow-viewer/src/core/WowViewer.Core.PM4/Services/Pm4ResearchReader.cs`
- `wow-viewer/src/core/WowViewer.Core.PM4/Models/Pm4ResearchChunkModels.cs`
- `wow-viewer/src/core/WowViewer.Core.PM4/Research/Pm4ResearchUnknownsAnalyzer.cs`
- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Pm4MatchSupport.cs`
