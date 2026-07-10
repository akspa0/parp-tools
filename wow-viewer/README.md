# WoWViewer

Active development target inside `parp-tools`.

World viewer, CLI toolchain, shared format libraries, and data-harvester for staged World of Warcraft client data.

## Current focus

- **Spec 097 `097-v18-to-wdl-adt` — full-map V18 Zarr → stitched mesh + WDL + ADT round-trip.** Slice 1 (per-map stitched OBJ + baked atlas with edge alignment) is live as of 2026-07-10. Northrend smoke: 1,131 tiles → 7,453×10,023 heightmap, 74.7M vertices, 6.3 min wall. Slices 2/3/4 (WDL writer, ADT writer, round-trip smoke) are next-session work.
- **Spec 096 `096-v24-minimap-deploy` — V24 minimap-to-prior deployment wiring.** Trained the minimap-only Stage A checkpoint; ships `infer_v24_stage_a_png.py` (PNG → WDL prior NPZ), `v24_prior_to_obj.py` (NPZ → textured OBJ), and `v24_run_on_png.py` (one-shot wrapper for any PNG, with `--batch-dir` for folder processing). 40/40 v24 tests pass. Honest caveat: the minimap-only regime is 158× worse than the cheat regime on the held-out V24 prior validation. Spec 095 (learned minimap cleaner) is the next step.
- Spec 089 `089-dav2-height-predictor` — active height-model lane.
- Spec 088 `088-v22-enrichment-from-v18` — active V22 dataset contract.
- Spec 080 `080-wow-ui-consolidation` — active viewer-shell doc and compatibility lane.

Background, not front-of-queue:

- Spec 047 — focused V18 operator lane.
- Spec 079 — shared RunPod bundle/runtime pattern.
- Spec 076 and Spec 077 — paused/background until reopened.

## Hard boundaries

- `wow-viewer/` owns new implementation work.
- `gillijimproject_refactor/` is read-only reference.
- Staged clients only: `output/tmp/wowarchive-clients/`.
- Any `H:\CLIENTS` reference is stale and must be removed.

## Build

```powershell
dotnet build wow-viewer/WowViewer.slnx -c Debug
dotnet test wow-viewer/WowViewer.slnx -c Debug
```

## Run viewer

```powershell
dotnet run --project wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug
```

Normal startup path:

1. Open staged game folder.
2. Pick explicit client build.
3. Load world from viewer UI.

Automation path exists for capture/debug flows through `--game-path`, `--build`, `--world`, and capture flags.

## Data-harvester setup

```powershell
cd wow-viewer/data-harvester
uv sync
```

Use `uv run ...` from that directory for dataset, training, and inference work.

## Main surfaces

| Surface | Purpose | Path |
|------|------|------|
| Viewer app | 3D world viewer | `src/viewer/WoWViewer/` |
| Shared libraries | format/domain/runtime code | `src/core/` |
| CLI tools | inspect, convert, harvest, validation, animfarm | `tools/` |
| Tests | C# xUnit | `tests/` |
| Data harvester | Python dataset/training/inference | `data-harvester/` |
| Specs | feature packs | `specs/` |
| Architecture docs | long-form design notes | `docs/architecture/` |

## Canonical docs

- [AGENTS.md](/I:/parp/parp-tools/wow-viewer/AGENTS.md)
- [docs/DOCUMENTATION-STATUS.md](/I:/parp/parp-tools/wow-viewer/docs/DOCUMENTATION-STATUS.md)
- [docs/CLI-TOOLS.md](/I:/parp/parp-tools/wow-viewer/docs/CLI-TOOLS.md)
- [data-harvester/README.md](/I:/parp/parp-tools/wow-viewer/data-harvester/README.md)
- [docs/WoWViewer/USERGUIDE.md](/I:/parp/parp-tools/wow-viewer/docs/WoWViewer/USERGUIDE.md)
- [memory-bank/activeContext.md](/I:/parp/parp-tools/wow-viewer/memory-bank/activeContext.md)
- [memory-bank/progress.md](/I:/parp/parp-tools/wow-viewer/memory-bank/progress.md)

## Historical surfaces

- `specs/archived/` — closed or superseded.
- `specs/086-*` and `specs/087-*` — superseded by Spec 088; keep only as evidence.
- `plans/` — old planning notes unless a live spec points there.
- `docs/MdxViewer-legacy-documentation.tar.gz` — archive only.
