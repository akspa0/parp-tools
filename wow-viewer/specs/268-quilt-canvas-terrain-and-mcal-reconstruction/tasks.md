# Tasks: Spec 268 Multi-Tile Quilt Canvas Terrain Reconstruction & MCAL Deciphering

## Phase 1: Quilt Canvas Stitcher & Seam Boundary Solver (AC-001)

- [ ] **T268-01**: Implement `QuiltCanvasAssembler` in `data-harvester/src/harvester/v60/quilt_canvas_assembler.py`: maps arbitrary sets of tiles $(T_x, T_y)$ to continuous global quilt coordinates $(U_{\text{global}}, V_{\text{global}})$, loads stitched minimap RGB, and tracks tile boundaries.
- [ ] **T268-02**: Implement boundary seam solver in `QuiltCanvasAssembler`: enforces exact C0 elevation continuity and C1 normal smoothness across adjacent tile borders ($u=256$ to $u=0$ and $v=256$ to $v=0$) using Laplacian margin relaxation.
- [ ] **T268-03**: Add unit tests in `data-harvester/tests/v60/test_quilt_canvas_assembler.py` validating coordinate mapping, border vertex equality ($\|\Delta Z_{\text{seam}}\| \le 0.05$ yds), and border normal alignment ($\ge 0.98$ cosine similarity).

---

## Phase 2: Minimap Albedo De-Mixing & Dynamic MCAL Layer Decipherer (AC-002)

- [ ] **T268-04**: Implement `McalLayerDecipherer` in `data-harvester/src/harvester/v60/mcal_layer_decipherer.py`: extracts authentic tileset BLP color signatures from client/WDT catalogs and performs convex albedo de-mixing on minimap pixels.
- [ ] **T268-05**: Implement dynamic chunk layer stacker in `McalLayerDecipherer`: enforces the maximum 4-layer constraint per MCNK chunk while optimizing layer indices across chunk boundaries to maintain visual continuity across the quilt canvas.
- [ ] **T268-06**: Add unit tests in `data-harvester/tests/v60/test_mcal_layer_decipherer.py` validating $\le 4$ layers per chunk, correct BLP assignments, and seamless alpha blending across chunk borders.

---

## Phase 3: Bare Terrain Shadow Sieve & WDL Macro Trestle Quilt (AC-003, AC-004)

- [ ] **T268-07**: Implement `WdlQuiltSynthesizer` in `data-harvester/src/harvester/v60/wdl_quilt_synthesizer.py`: strips texture albedo and masked object footprints to isolate the bare photometric terrain shadow signal $S(u, v)$.
- [ ] **T268-08**: Integrate `TrestleElevationUNet` (Spec 266) across multi-tile quilt coordinates, synthesizing continuous $64 \times 64$ continent WDL elevation lattices spanning authentic $>250\text{--}425$ yards mountain relief with zero boundary shear.
- [ ] **T268-09**: Add unit tests in `data-harvester/tests/v60/test_wdl_quilt_synthesizer.py` verifying bare shadow extraction ($\ge 85\%$ correlation to ground-truth illumination) and seamless WDL lattice stitching.

---

## Phase 4: Inches-Scale Refiner & 3D Fractal Pastes/Scars Engine (AC-005, AC-006)

- [ ] **T268-10**: Implement `QuiltFractalRefiner` in `data-harvester/src/harvester/v60/quilt_fractal_refiner.py`: constructs high-density $36\times$ sub-cell coordinate space reflecting authentic DAT project resolution ($1\text{ yd} = 36\text{ inches}$).
- [ ] **T268-11**: Integrate v7-era 3D fractal brush fitting, multi-tile prefabricated pastes, and historical brush scars detection on height and alpha layers across the quilt canvas using matching pursuit.
- [ ] **T268-12**: Implement deterministic area-weighted downsampling from inches-resolution to standard 145-vertex MCVT client yards ($9 \times 9$ outer + $8 \times 8$ inner per chunk) with anti-aliasing.
- [ ] **T268-13**: Add unit tests in `data-harvester/tests/v60/test_quilt_fractal_refiner.py` asserting inches-scale sub-cell precision, fractal brush NCC $\ge 0.80$, and ringing-free yards reduction.

---

## Phase 5: Monolithic ADT Patching & Multi-Tile Quilt CLI Runner (AC-007)

- [ ] **T268-14**: Implement `QuiltAdtMaterializer` in `data-harvester/src/harvester/v60/quilt_adt_materializer.py`: patches or constructs monolithic 3.3.5 ADTs preserving 100% of authentic chunks (`MCVT`, `MCNR`, `MCLY`, `MCAL`, `MCCV`, `MMDX`, `MWMO`, `MDDF`, `MODF`, `MFBO`, `MCLQ`).
- [ ] **T268-15**: Author end-to-end CLI tool `data-harvester/scripts/v60_reconstruct_quilt.py` supporting `--tiles`, `--bbox`, `--map`, and continuous OBJ/GLB mesh exports.
- [ ] **T268-16**: Add unit tests in `data-harvester/tests/v60/test_quilt_adt_materializer.py` validating byte-level chunk completeness, valid MCCV preservation, and clean file generation.

---

## Phase 6: Benchmark Sweep & Operator Verification

- [ ] **T268-17**: Run quilt reconstruction on benchmark tile clusters (`development_16_32` + `development_16_33` alpine massif, and `development_0_0` + `development_0_1` flat basin).
- [ ] **T268-18**: Export multi-tile 3D OBJ/GLB meshes and verify watertight upward normals and zero boundary seams in 3D viewer.
- [ ] **T268-19**: Generate diagnostic visual comparison sheets and document receipts in `specs/268-quilt-canvas-terrain-and-mcal-reconstruction/evidence/receipt-spec268.md`.
