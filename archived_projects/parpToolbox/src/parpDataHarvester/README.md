# parpDataHarvester (GLB-RAW)

Minimal CLI to export PM4 tiles to GLB while preserving MSLK hierarchy and grouping geometry by objects or surfaces.

Usage:

```
parpDataHarvester export-glb-raw --in <path> --out <dir> [--per-region] [--mode objects|surfaces]
```

Notes:
- Implementation is in-progress. Current build stubs out the export but validates CLI and project wiring.
- Next steps: integrate PM4Rebuilder loaders, build MSLK node graph, assemble geometry, adapt GLB writer.

## Canonical References

- PM4 Format: ../../docs/formats/PM4.md
- PM4 Chunk Reference: ../../docs/formats/PM4-Chunk-Reference.md
- PM4 Field Reference (Complete): ../../docs/formats/PM4-Field-Reference-Complete.md
- PM4 Assembly Relationships: ../../docs/formats/PM4_Assembly_Relationships.md
