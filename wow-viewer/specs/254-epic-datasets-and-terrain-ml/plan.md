# Plan — Epic 254 Datasets, Client Datastore & Terrain ML

**Status**: Implementation approach **not yet selected** (operator directive 2026-09-23: triage first).

## Dependencies

```text
D-01 patch-chain resolver ──> D-02 multi-build store ──> D-03 incremental ──> D-04 encoding
                                                    └──> D-05 viewer load ──> D-06 residency
D-11 clean-signal fix ──> D-12 later-client adapters
D-10 height model (independent of D-11; consumes the v50 dataset)
```

134 and 139 are one live codebase (`data-harvester/src/harvester/v60/clean_signal_*`); 139 anchors it.
132 and 140 duplicated the archaeology scope; D-21 is the merged residue.

## Design documents adopted by reference

| Item | Adopted design |
|---|---|
| D-01–D-05 | [old datastore epic](../archived/epic-client-datastore/epic.md) and specs [179](../archived/179-patch-chain-resolver/spec.md)–[183](../archived/183-datastore-viewer-load/spec.md) |
| D-06 | [206 spec](../archived/206-zarr-first-residency/spec.md) |
| D-10 | [123 spec](../archived/123-real-wdl-detailer/spec.md) |
| D-11, D-12 | [139 plan](../archived/139-v7-clean-signal-reconstruction/plan.md) · [134 plan](../archived/134-v60-unified-dataset-model/plan.md) |
| D-13 | [114 plan](../archived/114-direct-terrain-reconstruction/plan.md) |
| D-14 | [141 plan](../archived/141-terrain-method-translation/plan.md) |
| D-20 | [127 spec](../archived/127-weak-tile-explorer/spec.md) |
| D-21 | [132 plan](../archived/132-terrain-brush-signature-classification/plan.md) · [140 plan](../archived/140-terrain-paste-motif-archaeology/plan.md) |
| D-30 | [124 spec](../archived/124-legacy-detangle-runpod/spec.md) |

## Standing constraints

- **Python + TensorStore owns datastore IO**; C# emits ARRY blobs (memory: Python owns the datastore).
- Python work runs under `data-harvester/` with `uv`; training, harvests and GPU/cloud runs are
  user-executed (hand over exact PowerShell commands). No RunPod pod launch without an explicit "deploy".
- Validate on Kalimdor/Azeroth, never PVPZone02/Kalidar. Curation partitions, never silently filters.
- A target must be visible in the minimap RGB.
