# Plan 297 — Automated Known-Good Pairing Harvester & Streamlined Matching Workbench

## Architecture & Technical Design

### Track 1: Automated Donor / Placed Object Harvester
In `Pm4DoodadDuplicationService`:
1. `HarvestPairingsFromPlacedObjects(IViewerAppHost host, float maxCentroidDistance = 15.0f)`:
   - Enumerate all placed objects across `TileMdxInstances` and `TileWmoInstances` for all loaded tiles.
   - For each placed object:
     - Query `Pm4Overlay._pm4TileObjects` on the object's tile (and adjacent tiles if near boundary).
     - Filter candidate PM4 hulls by centroid distance $\le \text{maxCentroidDistance}$ and vertical bounding overlap.
     - If an unambiguous, best-fitting hull is identified (highest footprint/bounding agreement, distance $< \text{threshold}$):
       - Extract surface vertices and footprint prism.
       - Solve scale and yaw offset relative to the placed model's rotation.
       - Construct `TaughtPairing` record and add to `_taughtPairings`.
   - Deduplicate pairings by model path and hull shape signature.
   - Save to disk: `SaveTaughtPairingsToDisk()` (`<map>.pairings.json` and `<map>.manifest.json`).
   - Return detailed result summary: total scanned, newly harvested, already known, and ambiguous/unmatched.

### Track 2: Streamlined Unified Workbench UI
In `Pm4WorkbenchSidebarService` and `Pm4ShapeMatchWorkbenchService`:
1. Restructure the main PM4 Sidebar into 4 clean sections:
   - **Section 1: Cockpit Status & Action Bar**:
     - Summary pills: Loaded Tiles, Placed Models, PM4 Hulls, Learned Rules.
     - One-click buttons:
       - `[ Harvest Pairings from Placed Models ]`
       - `[ Apply Learned Rules to All Tiles ]`
       - `[ Save / Export Manifest ]`
   - **Section 2: Active Selection & Single-Pair Curation**:
     - When an object or hull is selected: clean side-by-side card showing Model Name, PM4 Hull `[Tile] Obj #<Part> (Z=<z> yd)`, distance, and dimensions.
     - Big action buttons: `[ Pair Selected Model with Hull ]` and `[ Snap & Align ]`.
   - **Section 3: Learned Rules Ledger**:
     - Compact table/list of learned pairings with model name, hull dimensions, and live count of matching unoccupied hulls across loaded tiles.
     - Per-rule `[ Apply ]` and `[ Delete ]` actions.
   - **Section 4: Advanced Forensics (Collapsed by Default)**:
     - Foldout header: `Advanced Raw Forensics & Chunk Explorer`.
     - Contains the deep binary debug inspector (`MSHD`, `MSLK`, raw bytes, regional peer lists) so it remains accessible for debugging but never clutters normal matching.

### Track 3: Verification & Governance
1. Unit tests for `HarvestPairingsFromPlacedObjects` with simulated placements and PM4 hulls.
2. Full solution build, unit tests, and `Verify-UiReference.ps1`.
