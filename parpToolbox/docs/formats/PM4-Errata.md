# PM4 Errata and Historical Notes

Last updated: 2025-08-19

This unified errata consolidates deprecated claims and historical context across the PM4 documentation set. Refer back here when older sections mention global mesh systems, MPRR-based building boundaries, or fully decoded fields.

## Current Guidance (Authoritative)
- Per-tile processing only: assemble one PM4 tile at a time; do not merge tiles into a global scene.
- Hierarchical containers: identify containers via `MSLK.MspiFirstIndex = -1`; traverse to geometry-bearing links.
- Placement link: `MPRL.Unknown4` equals `MSLK.ParentIndex` (confirmed).
- MPRR: `Value1 = 65535` are property separators, not building/object boundaries.
- MSUR.IndexCount: diagnostic grouping only; not a true object identifier, although sometimes useful for visualization.

## Deprecated or Overconfident Claims
- Global mesh architecture requiring directory-wide multi-tile loading to resolve geometry. This is deprecated for exporter workflows; process per tile. Any cross-tile references should be treated as non-rendering metadata unless new evidence dictates otherwise.
- MPRR sentinel-based building boundaries as the definitive grouping method. Sentinels mark property separators; they do not define object boundaries.
- "Fully decoded" field sets across all chunks. Several fields remain partially understood; confidence varies by field and file.

## Where to Find Updated Details
- `docs/formats/PM4-Spec.md` – overview, conventions, and per-tile pipeline.
- `docs/formats/PM4-Chunk-Reference.md` – chunk relationships with container traversal focus.
- `docs/formats/PM4-Field-Reference-Complete.md` – corrected mappings and known/unknown field tables.
- `docs/formats/PM4-Object-Grouping.md` – container traversal guidance and pitfalls.
- `docs/formats/PM4_Assembly_Relationships.md` – assembly pipeline and analyzer-backed samples.

## Tracked Open Items
- Exact semantics and usage of `MSLK.ReferenceIndex` (open).
- `MSUR` attribute masks/flags: semantics under investigation; dataset-dependent.
- Role and possible gating behavior of `MPRR` sections (open).

## Rationale and Evidence
- Analyzer outputs (links, placements, signatures) confirm: `MPRL.Unknown4 ↔ MSLK.ParentIndex`, frequent container nodes with `MspiFirstIndex = -1`, and strong distributional evidence against MPRR-as-boundary.
- Empirical exports demonstrate that per-tile assembly with container traversal avoids fragmentation and aligns with observed data distributions.

If future analysis overturns any of the above, update this page first and add cross-references back to each specific document section.
