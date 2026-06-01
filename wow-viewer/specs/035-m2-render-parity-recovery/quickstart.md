# Quickstart: M2 Render Parity Recovery

## 1. Build

```powershell
dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
```

## 2. Run Adapter Probe on Known Tree Model

```powershell
dotnet run --project i:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -- --probe-m2-adapter "I:/parp/parp-tools/output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft" "WORLD/AZEROTH/ELWYNN/PASSIVEDOODADS/TREES/ELWYNNTREECANOPY03.M2" --build 3.3.5.12340 --archives "I:/parp/parp-tools/output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft/Data"
```

Expected: selected skin, geometry summary, and material/pass diagnostics are printed.

## 3. Run Runtime Probe on Same Model

```powershell
dotnet run --project i:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -- --probe-m2-runtime "I:/parp/parp-tools/output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft" "WORLD/AZEROTH/ELWYNN/PASSIVEDOODADS/TREES/ELWYNNTREECANOPY03.M2" --build 3.3.5.12340 --archives "I:/parp/parp-tools/output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft/Data"
```

Expected: section/material route state matches adapter-probe expectations.

## 4. World Runtime Spot-Check

1. Launch WoWViewer on staged `3_3_5_12340` client.
2. Load tile with known tree density.
3. Enable doodads and bounds overlays.
4. Confirm sampled tree placements show both bounds and rendered geometry.

## 5. Record Parity Evidence

Store outputs/logs under:

- `wow-viewer/output/tmp/m2-parity/`
- include sample id, build, tile, and timestamp in file names.

### Naming Conventions

Evidence files MUST follow this naming scheme:

```
m2-parity-{phaseId}-{sampleId}-{buildId}-{tileX}_{tileY}-{timestamp}.txt
```

Where:
- `phaseId` = evidence phase group: `us1`, `us2`, `us3`, `probe-adapter`, `probe-runtime`
- `sampleId` = sample identifier from parity-samples.md (e.g. `elwynn-tree-01`)
- `buildId` = staged build label (e.g. `335-12340`)
- `tileX_tileY` = tile coordinates in `{x}_{y}` format (e.g. `27_57`)
- `timestamp` = capture UTC in `YYYYMMDDTHHmmssZ` format

Example:

```
m2-parity-us1-elwynn-tree-01-335-12340-27_57-20260601T143000Z.txt
```
