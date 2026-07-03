# SUPERSEDED — V22 Consolidated Dataset (Spec 086)

**This spec is superseded by [`088-v22-enrichment-from-v18`](../088-v22-enrichment-from-v18/spec.md).**

## Reason

Spec 086 designed V22 as a parallel C# harvester stream that emits per-tile model payloads and a Python writer that dedupes them. The C# side never produced the three-message-class stream (tile / model-library / tileset-library) the spec required. The Python writer existed but `add_model()` / `add_tileset()` were never reached by a real producer. Result: zero populated `models/` or `tilesets/` groups in any V22 store, no end-to-end real-data build ever succeeded.

## What Replaces It

Spec 088 uses V18 as the substrate (untouched) and adds a separate C# `WowViewer.Tool.V22Enrich` tool that reads a finished V18 store, decodes every unique M2 / WMO / BLP exactly once, and writes a stable-keyed binary enrichment stream. The Python `build_v22_dataset.py` reads V18 + the enrichment stream and writes the V22 Zarr store.

## What Stays True

The V22 contract doc at [`docs/architecture/v22-dataset-signals-2026-06-30.md`](../../docs/architecture/v22-dataset-signals-2026-06-30.md) remains the canonical reference for the V22 store layout, root arrays, placement arrays, model library, and tileset library. Spec 088 implements that contract; Spec 086 attempted and failed to implement it.

## Action

Do not work on this spec. The contents (spec.md, plan.md, tasks.md) are kept for historical reference only. The new canonical work lives in `088-v22-enrichment-from-v18/`.
