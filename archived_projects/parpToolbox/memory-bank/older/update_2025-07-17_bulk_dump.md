# Bulk Dump & Compound Field Insights (2025-07-17)

## New Capability
* Implemented `--bulk-dump` flag for the `pm4` CLI command.
* Adds `Pm4BulkDumper` service which:
  * Writes comprehensive `msur_dump.csv` (all raw MSUR fields).
  * Exports OBJs per grouping strategy:
    * `by_surfacekey/` – groups by full 32-bit `Unknown_0x1C`.
    * `by_ground_subbucket/` – for surfaces where that key is zero, second-level grouping by `(SurfaceGroupKey<<8)|Flags0x00`.
* Output structure: `project_output/<tile>/bulk_dump/<strategy>/GXXXXXXXX.obj`.

## Format Discoveries
* **Compound IDs:** Evidence suggests many 32-bit fields are actually two 16-bit words (high|low).  Confirmed candidates:
  * `MSUR.Unknown_0x1C` (render-object key)
  * `MSLK.LinkIdRaw` low word (sub-key) / high word (`0xFFFF` pad or container).
  * `MSLK.ReferenceIndex` splits into container (high byte) + child index (low byte).
* Next steps: update chunk models to expose these halves directly and add grouping folders `by_surfacekey_hi16/`, `by_surfacekey_low16/`.

## Build & CLI
* Fixed multi-line help string in `Program.cs` (newline inside constant caused CS1010).
* Made `MsurChunk` public so `Pm4Scene.Surfaces` is accessible (resolved CS0053).
* Added `bulkDump` flag detection; default export now bypassed when `--bulk-dump` passed.

## Outstanding Tasks
1. Verify that `--bulk-dump` produces per-group OBJ sets – current output indicates fallback to full-tile export; debug filters.
2. Split compound fields in models (two `ushort` instead of one `uint`) for accuracy & perf.
3. Propagate new groupings into bulk dump exporter.
4. PD4 parity once PM4 pipeline validated.

---
_Remember to mirror any confirmed spec changes into `mslk_linkage_reference.md` and `msur_surface_reference.md`._
