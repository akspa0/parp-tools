# Spec 269: v60 Classic Terrain Zarr Ingestion & Felwood Clean Elevation Training

## 1. Overview & Problem Statement

In Spec 268, the multi-tile continuous quilt reconstruction and MCAL deciphering pipeline was successfully established on the pure V7 architectural foundation. However, testing on unauthored development tiles revealed a critical gap: **synthesized terrain lacked authentic classic terrain indentations (such as Felwood canyons, Ashenvale ravines, and classic continental mountain cuts)**.

### Root Cause
1. Prior neural elevation models (`TrestleElevationUNet`, `trestle_elevation_v1.pt`) were trained exclusively on 270 historical development tiles (`test_data/original_development`) and transferred Northrend 3.0.1.8303 prototype WDL lattices.
2. The model had zero exposure to classic Kalimdor (988 tiles) and classic Azeroth (687 tiles), preventing it from learning authentic classic terrain motifs like Felwood's deep corrupted forest canyons, steep cliff indentations, and river ravines.
3. Rather than relying on ad-hoc loose scripts, the project mandates using the unified **v60 Zarr datastore pipeline** (`v60_build_dataset.py`, `harvest-stream --stream-profile v22`) to ingest authentic 3.3.5 retail client data from `H:\CLIENTS\WoW335\3.X_Retail_Windows_enUS_3.3.5.12340\World of Warcraft`.

---

## 2. User Stories

### User Story 1 - Flexible Client Root & Map Selection in v60 Dataset Pipeline (Priority: P1)
As a pipeline operator, I want `v60_build_dataset.py` to seamlessly accept direct client installation paths (such as `H:\CLIENTS\WoW335\3.X_Retail_Windows_enUS_3.3.5.12340\World of Warcraft`) and selectively stream specified maps (`--maps Kalimdor,Azeroth`), assigning standard normalized build IDs (`3_3_5_12340`), so that Kalimdor and Azeroth terrain are ingested directly into the v60 Zarr datastore.

### User Story 2 - Complete Clean Terrain Signal Ingestion (Priority: P1)
As a machine learning researcher, I want `harvest-stream` via the v22 profile to stream raw height (`height_257`), clean object-masked terrain elevation (`ground_intent_height_257`), minimap RGB (`minimap_rgb`), surface normals (`normal_xyz`), and shadows (`terrain_shadow_256`) into per-build Zarr stores and merge them into the unified v60 Zarr store with tile-level coordinate lineage (`tile_x`, `tile_y`).

### User Story 3 - Zarr-Backed Clean Elevation Dataset with Felwood Indentation Supervision (Priority: P1)
As a model architect, I want a dedicated dataset builder and PyTorch dataset (`ZarrElevationDataset`) that loads directly from the v60 Zarr store, extracting 6-channel input tensors paired with clean ground-truth elevation and residual $\Delta Z = Z_{\text{gt}} - Z_{\text{trestle}}$, explicitly supervising on authentic Felwood canyons (Kalimdor tiles $X \in [25, 29], Y \in [19, 26]$) and classic indentation features.

### User Story 4 - Elevation Training Integration under Strict V7 Architecture (Priority: P1)
As a model trainer, I want `v60_train_trestle_model.py` to support training `TrestleElevationUNet` directly from the v60 Zarr store, enforcing the lean $< 10\text{M}$ parameter budget, strict V7 macro-prior + residual refinement, and zero post-V7 or deprecated tool dependencies.

---

## 3. Acceptance Criteria

| ID | Criterion | Requirement | Target Metric |
|---|---|---|---|
| **AC-001** | Direct Client Root Discovery | `v60_build_dataset.py` discovers clients from direct path, parent path, or root library | Valid `(build_id, client_path)` returned without empty list failure |
| **AC-002** | Targeted Map Filtering & Build Normalization | Support `--maps Kalimdor,Azeroth` and normalized build ID `3_3_5_12340` | Only specified maps streamed; per-build store named `3_3_5_12340-<map>.zarr` |
| **AC-003** | Tile Coordinate & Signal Preservation | Ensure `tile_x`, `tile_y` and clean terrain signals are preserved in `index.parquet` and Zarr arrays | 100% of streamed tiles have valid `tile_x`, `tile_y` $\in [0, 63]$; `height_257` and `ground_intent_height_257` stored |
| **AC-004** | Zarr Dataset Loader & Felwood Indentation Extraction | `ZarrElevationDataset` loads 6-channel tensors + residual $\Delta Z$ targets with Felwood feature tagging | Extracts valid $(6, 256, 256)$ features and $(257, 257)$ targets; detects Felwood coordinates |
| **AC-005** | Strict V7 Training Pipeline Integration | `v60_train_trestle_model.py` accepts `--zarr-store` and prepares training/validation splits | Successful dataset instantiation and PyTorch DataLoader batch emission under V7 architecture |

---

## 4. Architectural Constraints & Code Ownership

- **Code Ownership (AGENTS.md §4)**: Core format I/O remains in `src/core/WowViewer.Core.IO/`. Python dataset, ML, and harvest tooling lives strictly under `wow-viewer/data-harvester/` using `uv`.
- **God-Class Freeze (AGENTS.md §10)**: Zero member additions to `WorldScene.cs` or `ViewerApp.cs`.
- **Operator-Owned Operations (AGENTS.md §5)**: Large client data harvests and deep neural network training are operator-owned. Provide exact, validated PowerShell 7 commands and stop before execution.
- **Portability & Client Roots (AGENTS.md §5)**: `H:\CLIENTS` is runtime configuration passed via CLI options; never hardcode machine-local paths into source code or test fixtures.
- **Strict V7 Governance**: Maintain pure V7 macro-prior + residual refiner architecture. Zero post-V7 versions, zero Terra, no proprietary authoring tool names.
