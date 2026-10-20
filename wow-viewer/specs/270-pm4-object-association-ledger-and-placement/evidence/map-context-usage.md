# Whole-map PM4 context and operator-authored object pools

## Whole-map audit — operator-owned run, not executed by the agent

From the repository root in PowerShell 7:

```powershell
$pm4Map = 'wow-viewer/test_data/original_development/World/Maps/development'
$museumMap = 'wow-viewer/test_data/WoWMuseum/335-dev/World/Maps/development'
$output = 'wow-viewer/output/pm4-map-audit-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
dotnet run --project 'wow-viewer/specs/270-pm4-object-association-ledger-and-placement/evidence/map-partition-audit/MapPartitionAudit.csproj' -c Debug -- `
    $pm4Map $museumMap $output
```

This reads **every PM4 in the map directory**, not five selected tiles. It ingests every authored
placement ADT in the Museum directory, including tiles without PM4s; texture-only companions are
excluded. Base/object companions and duplicate placement references retain their source provenance.
It can consume substantial memory/disk: raw decoded documents and binary chunk payloads are
retained alongside twelve full partition hypotheses. Do not run it against installed-client folders
without supplying the intended extracted map and authored placement roots.

Outputs:

- `manifest.json`: all input names, sizes, SHA-256 fingerprints, roots and failures. Dataset/build
  identity is not inferred; record the actual source revision separately.
- `decoded-documents.jsonl`: all decoded chunk sets **and original chunk payloads**, not just zero
  geometry. This retains context for later MPRL/MPRR/MSCN/MSPV/MSPI research.
- `authored-catalogs.jsonl`: model paths, IDs, positions, rotations and scales from authored ADTs.
- `wl-source-hints.json`: source-name hints; not established zone footprints or tile assignments.
- `map-partition-audit.json`: file-qualified global IDs; every MSUR/MSLK entry; window owners,
  invalid/unowned/anchor-only records; raw field incidence; rival window controls; cross-file
  geometric coincidences; partition hypotheses and normalized candidate families.
- `COMPLETE.json`: appears only after all inputs pass the conservative evidence gate and the
  entire map analysis is serialized. It proves completion of this instrument, **not** decoded
  semantics, correct object partitions, or accurate asset matching.

Any PM4 reader diagnostic or malformed/unresolved authored placement record prevents a completion
claim. Evidence/failures remain in the output for review. A diagnostic is not automatically a
proven format bug; do not change or relax shared readers merely to finish the audit.

The window graph treats `_0x18` and attrMask as a proposed start/length relation, not a settled
meaning. The report also retains attrMask values/bits and compares rival lengths. Same raw group
values are tested under local and map scope as separate hypotheses. None is selected as truth.
**Subsequent measured correction:** [whole-map structural evidence](structural-foundations-2026-10-20.md)
validates the start/count relation in this corpus and packed target-tile addressing. Two addressed
graph hypotheses are now included; local-reference graphs remain explicitly competing controls.
Normalized family equality is a lossy diagnostic (fan triangles, 0.02 dimensionless bins); it can
join unrelated simple shapes or split differently clipped copies. Absolute radii/centroids and
memberships remain available. Merged copies are not assumed separable by connected components.

## Persistent tile object pools

Build the utility once:

```powershell
dotnet build 'wow-viewer/specs/270-pm4-object-association-ledger-and-placement/evidence/object-pools/ObjectPools.csproj' -c Debug
$poolsTool = 'wow-viewer/specs/270-pm4-object-association-ledger-and-placement/evidence/object-pools/bin/Debug/net10.0/ObjectPools.dll'
$manifest = 'wow-viewer/output/pm4-context/development.json'
dotnet $poolsTool init $manifest 'development'
dotnet $poolsTool hints $manifest 'wow-viewer/test_data/original_development/World/Maps/development'
```

The manifest is ordinary editable JSON. Pool definitions contain operator labels (`DesignKit`,
`ObjectClasses`) and actual selectors (`AssetKinds`, explicit `AssetPaths`, `PathPrefixes`). Labels
do not magically resolve to asset paths. Prefixes select directories, not arbitrary substrings.
Assignments contain a pool ID, inclusive tile rectangle and operator notes. Multiple assigned
pools form a union. Unassigned tiles/maps use the original database unchanged; an assigned pool
with no eligible assets stays empty rather than guessing from unrelated kits.

Use `--help` for the positional commands:

```powershell
dotnet $poolsTool --help
```

- `pool <manifest> <pool-id> <design-kit> <object-class> <m2|wmo|both> <prefix> [more-prefixes]`
- `assign <manifest> <pool-id> <min-x> <min-y> <max-x> <max-y> <operator-note>`
- `filter <manifest> <input-surface-db> <map-name> <tile-x> <tile-y> <new-output-db>`

Choose real pool selectors and tile coordinates from evidence; **no Zul'Drak development tile
coordinates or full asset kit are invented here**. Explicit asset paths or additional selectors
can be edited directly in the manifest.

The filter emits a normal surface database plus `.context.json` with active pool IDs and
before/after counts. Pass that database to the existing `pm4 match-surfaces` workflow's
`--wmo-surface-db` argument; despite that legacy flag name, records retain their declared kinds.
The output is for the selected tile/map only, not a replacement global DB for a whole-map job.
No geometric fingerprint, score, type gate, or existing WMO matcher behavior is modified.

## WL-family hints and abbreviations

Default aliases are only the operator-provided HFjord/Howling Fjord → Howling Fjord/Northrend and
Grizzly → Grizzly Hills/Northrend. All 70 local WL-family source names were retained by the smoke
run; 11 matched these aliases. `Griz - Demo Pond.wlw` is retained as unrecognized by default, not
silently equated to a zone. Add explicit alias entries to `HintAliases` when confirming abbreviations:

```json
[{ "Token": "Griz", "Zone": "Grizzly Hills", "DesignKit": "Northrend" }]
```

Custom aliases replace the defaults for that manifest, so include any defaults you want to keep.
Hint import preserves confirmed records and sources absent from the new import. Multiple matches
remain multiple suggestions. A filename supplies design context, not exact tile ownership: spatial
association using WL geometry is **not implemented**, and imports do not create hard restrictions.

## Open work

- Operator large raw/partition export and analysis against Museum placements; lightweight whole-map
  structural summaries **have run** (see the subsequent receipt), but no real partition accuracy or
  improvement over the operator-reported ~70% WMO baseline is claimed.
- Interpret remaining chunk relationships and determine whether unknown fields split merged,
  scaled copies. Raw context is retained but not newly decoded by this instrument.
- Viewer annotation UI and automatic WL footprint-to-tile association remain unimplemented.
