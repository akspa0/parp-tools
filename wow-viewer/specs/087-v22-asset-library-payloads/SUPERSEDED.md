# SUPERSEDED — V22 Asset Library Payloads (Spec 087)

**This spec is superseded by [`088-v22-enrichment-from-v18`](../088-v22-enrichment-from-v18/spec.md).**

## Reason

Spec 087 inherited Spec 086's per-tile stream design. It used `Path.GetHashCode()` as the model key for the enrichment stream — which is randomized per process in .NET 6+. The same model gets different keys in every harvest run, breaking cross-run dedup.

Even if the keys were deterministic, the per-tile design still duplicates the same model payload once per tile that references it. For Azeroth-class maps with 700+ unique M2 placements, that is hundreds of MB of duplicated geometry in the stream.

## What Replaces It

Spec 088 uses **stable canonical path keys** (via `WowViewer.Core.M2.M2ModelIdentity.NormalizePath`) and a **build-wide library** (one entry per unique path), not per-tile. The C# enrich tool decodes each unique M2 / WMO / BLP exactly once and writes them to a separate enrichment stream keyed by canonical path. The Python Zarr writer accumulates per-build from that stream.

## What Stays True

The FR-008 (M2 fields) and FR-009 (WMO fields) field lists in this spec are reused verbatim by Spec 088 FR-008 and FR-009. Those field lists are correct; the broken part was the per-tile stream design that wrapped them.

## Action

Do not work on this spec. The contents (spec.md, plan.md, tasks.md) are kept for historical reference only. The new canonical work lives in `088-v22-enrichment-from-v18/`.
