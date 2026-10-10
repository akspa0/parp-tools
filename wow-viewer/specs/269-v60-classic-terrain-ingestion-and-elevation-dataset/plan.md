# Plan: Spec 269 v60 Classic Terrain Zarr Ingestion & Felwood Clean Elevation Training

## 1. Technical Architecture & Approach

This spec provides the tooling and dataset bridge between the 3.3.5 client root and the clean elevation model under the strict V7 architecture.

### Architectural Flow:
```
                                 [ WoW 3.3.5 Client ]
                     (H:\CLIENTS\...\World of Warcraft)
                                         │
                                         ▼
                             v60_build_dataset.py
                     (--maps Kalimdor,Azeroth --tile-workers 4)
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
         3_3_5_12340-Kalimdor.zarr                   3_3_5_12340-Azeroth.zarr
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                            Unified v60 Zarr Datastore
                               (v60_unified.zarr)
                                         │
                                         ▼
                           ZarrElevationDataset (V7)
              (6-Channel Features + Residual ΔZ + Felwood Indentations)
                                         │
                                         ▼
                              v60_train_trestle_model.py
                                         │
                                         ▼
                               TrestleElevationUNet
                      (Clean Classic Indentations Restored)
```

---

## 2. Component Design & Changes

### A. Client Root & Map Selection in `v60_build_dataset.py`
1. **Client Root Discovery**:
   - Update `_discover_clients`:
     - Inspect `client_root` directly for `Data/` or `WTF/`. If present, recognize `client_root` immediately.
     - Extract normalized `build_id` (e.g. `3_3_5_12340` from `3.X_Retail_Windows_enUS_3.3.5.12340`).
     - Support `--build-id` CLI flag to override the identifier.
2. **Targeted Map Selection**:
   - Add `--maps` CLI argument (comma-separated list, e.g. `Kalimdor,Azeroth`).
   - Filter discovered maps against `--maps` so only desired continents are streamed.
3. **Stream Concurrency & Tile Coordinates**:
   - Add `--tile-workers` option (default 4) forwarded to `harvest-stream --tile-workers`.
   - Ensure tile decoding extracts `tile_x` and `tile_y` from `_metadata` when deserializing tile blobs, preserving exact grid coordinates in `index.parquet`.
   - Retain clean signals: `height_257`, `ground_intent_height_257`, `minimap_rgb`, `normal_xyz`, `terrain_shadow_256`.

### B. Zarr Dataset Loader (`zarr_elevation_dataset.py`)
1. Create `ZarrElevationDataset` and `build_or_load_zarr_elevation_corpus` in `data-harvester/src/harvester/v60/zarr_elevation_dataset.py`:
   - Query `index.parquet` for target builds (`3_3_5_12340`) and maps (`Kalimdor`, `Azeroth`).
   - Support selective geographic filtering / weighting for Felwood canyon tiles (Kalimdor $X \in [25, 29], Y \in [19, 26]$).
   - Compute coarse 17x17 WDL lattice from ground-truth height (`ground_intent_height_257` or `height_257`), upsample to 257x257 trestle prior, and calculate physical residual $\Delta Z = Z_{\text{gt}} - Z_{\text{trestle}}$.
   - Compute photometric normals from `minimap_rgb`.
   - Return 6-channel tensors: `[Minimap RGB (3), Normalized WDL Trestle (1), Photometric Normals (2)]`.

### C. Training Integration (`v60_train_trestle_model.py`)
1. Update `v60_train_trestle_model.py`:
   - Add `--zarr-store` option pointing to unified or per-build Zarr store.
   - When specified, loads training samples using `ZarrElevationDataset`.
   - Trains `TrestleElevationUNet` (4.7M parameters) with composite residual loss.
   - Saves checkpoint (e.g. `output/models/trestle_elevation_classic_v1.pt`).

---

## 3. Phased Implementation Roadmap

- **Phase 1**: Update `v60_build_dataset.py` with flexible client root discovery, `--maps` filtering, `--build-id` normalization, and coordinate tracking.
- **Phase 2**: Add unit tests in `tests/v60/test_v60_build_dataset.py` validating client discovery, map filtering, and build ID normalization.
- **Phase 3**: Implement `ZarrElevationDataset` in `harvester/v60/zarr_elevation_dataset.py` with Felwood indentation awareness and V7 tensor contract.
- **Phase 4**: Add unit tests in `tests/v60/test_zarr_elevation_dataset.py` validating tensor shapes, residual calculation, and Felwood filtering.
- **Phase 5**: Integrate `--zarr-store` support into `v60_train_trestle_model.py`.
- **Phase 6**: Prepare validated PowerShell 7 execution commands for the operator.
