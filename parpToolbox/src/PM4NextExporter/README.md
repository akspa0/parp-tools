# PM4NextExporter (pm4next-export)

Focused CLI for exporting PM4 data using multiple object assembly strategies. Built on the parpToolbox core library.

- parpToolbox = core library + analysis/diagnostics CLI (PM4/PD4/WMO)
- PM4NextExporter = specialized PM4 exporter (OBJ today; glTF/GLB planned)
- All outputs are written to timestamped directories under `project_output/`

## Build & Run

Requires .NET 9.0 or later.

```bash
# From repository root
dotnet build

# Run PM4NextExporter
# Single file
dotnet run --project src/PM4NextExporter -- "development_00_00.pm4" [options]

# Or a directory (batch mode)
dotnet run --project src/PM4NextExporter -- "./pm4_dir" --batch [options]
```

A timestamped output directory is created automatically via `ProjectOutput.CreateOutputDirectory(<baseName>)`.
Logs are also written to `run.log` in that directory.

## Usage

```
pm4next-export <pm4Input|directory> [--out <dir>] [--include-adjacent] [--format obj|gltf|glb]
  [--assembly parent-index|msur-indexcount|surface-key|surface-key-aa|composite-hierarchy|container-hierarchy-8bit|composite-bytepair|parent16|mslk-parent|mslk-instance|mslk-instance+ck24]
  [--group parent16|parent16-container|parent16-object|surface|flags|type|sortkey|tile]
          [--parent16-swap] [--csv-diagnostics] [--csv-out <dir>] [--correlate <keyA:keyB>]
  [--batch] [--legacy-obj-parity] [--audit-only] [--no-remap] [--ck-split-by-type]
  [--mslk-parent-min-tris <N>] [--mslk-parent-allow-fallback] [--export-mscn-obj] [--export-tiles] [--project-local] [--name-with-tile]
```

- --out <dir>: Base name for output session directory (default: `pm4next`)
- --include-adjacent: Load cross-tile vertices for complete geometry
- --format: `obj` (supported), `gltf|glb` (not implemented yet)
- --assembly: choose object assembly strategy (default: `composite-hierarchy`)
- --group: optional secondary grouping for summaries/diagnostics
- --csv-diagnostics: write CSV diagnostics to output directory (or `--csv-out`)
- --legacy-obj-parity: legacy OBJ winding/parity behavior for compatibility
- --audit-only: run cross-tile audit and exit (no export)
- --no-remap: disable MSCN-based vertex remapping
- --ck-split-by-type: split by composite-key type buckets
- --export-mscn-obj: export MSCN vertices as OBJ layers
- --export-tiles: per-tile OBJ export using global coordinates (preserves object boundaries)
- --project-local: local projection for OBJ (ignored for per-tile exports)
- --name-with-tile: include tile name in object names
- --mslk-parent-min-tris <N>: minimum triangles for MSLK parent groups (experimental)
- --mslk-parent-allow-fallback: enable fallback scanning for missing cross-tile parents (experimental)
- --parent16-swap: swap parent16 pair halves (when using `--assembly parent16`)
- --correlate <keyA:keyB>: print correlation of two keys in diagnostics
- --batch: treat input path as directory and process all PM4 files inside

## Recommended Examples

- Basic OBJ export (default strategy):
```bash
dotnet run --project src/PM4NextExporter -- "development_00_00.pm4" --format obj
```

- Include adjacent tiles, per-tile export, and CSV diagnostics:
```bash
dotnet run --project src/PM4NextExporter -- "development_00_00.pm4" \
  --include-adjacent --export-tiles --csv-diagnostics
```

- MSCN OBJ layers for validation:
```bash
dotnet run --project src/PM4NextExporter -- "development_00_00.pm4" --export-mscn-obj
```

- Legacy OBJ parity and local projection:
```bash
dotnet run --project src/PM4NextExporter -- "development_00_00.pm4" \
  --legacy-obj-parity --project-local
```

- Experimental MSLK parent assembly with min triangle threshold:
```bash
dotnet run --project src/PM4NextExporter -- "development_00_00.pm4" \
  --assembly mslk-parent --mslk-parent-min-tris 500
```

## Notes on Assembly Strategies

- composite-hierarchy (default): robust per-object grouping across tiles
- surface-key / surface-key-aa: useful for experiments; may merge unrelated geometry
- msur-indexcount: historical method; aligns with prior analyses
- mslk-parent / mslk-instance / mslk-instance+ck24: experimental strategies using MSLK relations
- parent-index / parent16 / container-hierarchy-8bit / composite-bytepair: research/diagnostics modes

## Output and Logging

- Output directory: `project_output/<timestamped>/` (auto-created)
- OBJ export: writes `.obj` and `.mtl` as applicable
- Per-tile export: preserves global coordinates; `--project-local` is ignored for per-tile
- Log file: `<outDir>/run.log`

## Canonical References

- PM4 Format: ../../docs/formats/PM4.md
- PM4 Chunk Reference: ../../docs/formats/PM4-Chunk-Reference.md
- PM4 Field Reference (Complete): ../../docs/formats/PM4-Field-Reference-Complete.md
- PM4 Assembly Relationships: ../../docs/formats/PM4_Assembly_Relationships.md

## See Also

- parpToolbox (core, analysis CLI): `src/parpToolbox/`
- Root README for high-level overview and build instructions
