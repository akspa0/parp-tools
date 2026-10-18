# Tasks: Spec 274 Minimap Hillshade Correlation & Domain Translation

**Spec**: [Spec 274: Minimap Hillshade Correlation & Domain Translation](spec.md)  
**Plan**: [Technical Design Plan](plan.md)  
**Status**: Implemented & Verified (Gates A, B, C Passed)  
**Receipt**: [evidence/receipt-spec274.md](evidence/receipt-spec274.md)  

---

## Phase 1: Headless Batch Export Pipeline (Clean Pairs)

- [x] **T001**: Implement fast DXT1 / BC1 block quantization parity codec in `data-harvester/src/harvester/v60/dxt1_parity.py` matching `Dxt1TileCodec.cs` (4x4 block RGB565 endpoints and 2-bit indexing).
- [x] **T002**: Implement `batch_export_clean_pairs.py` in `data-harvester/src/harvester/v60/` to scan Zarr v3 map stores, extract aligned pairs (`minimap_rgb`, `synthetic_hillshade_dxt1`, `wdl_lattice`, `height_257`), and output a unified clean paired store.
- [x] **T003**: Create CLI entry point `data-harvester/scripts/v60_batch_export_clean_pairs.py` with `--dry-run`, `--limit`, `--maps`, and `--output` options.
- [x] **T004**: Author unit tests in `data-harvester/tests/v60/test_batch_export_clean_pairs.py` testing DXT1 parity, signal presence, and pairing integrity.
- [x] **Gate A**: Run `uv run pytest tests/v60/test_batch_export_clean_pairs.py` and verify all tests pass with zero warnings/errors. (Verified: 4 passed in 4.39s)

---

## Phase 2: PyTorch Clean Pair Dataset & DataLoader

- [x] **T005**: Implement `CleanPairDataset` in `data-harvester/src/harvester/v60/clean_pair_dataset.py` with PyTorch `Dataset` interface, loading paired tensors with data augmentation (random flips, brightness/contrast jitter).
- [x] **T006**: Author unit tests in `data-harvester/tests/v60/test_clean_pair_dataset.py` verifying tensor dimensions `(3, 256, 256)` and `(257, 257)`, value ranges, and train/val splits.
- [x] **Gate B**: Run `uv run pytest tests/v60/test_clean_pair_dataset.py` and verify 100% pass. (Verified: 2 passed in 5.72s)

---

## Phase 3: Lightweight Domain Translation Model & Training Script

- [x] **T007**: Implement `DomainTranslationUNet` and composite losses ($L_1$ + Sobel gradient + SSIM) in `data-harvester/src/harvester/v60/domain_translation_model.py` ($\le 10\text{M}$ parameter budget, actual 5.54M).
- [x] **T008**: Implement CLI training script `data-harvester/scripts/v60_train_domain_translation.py` with dry-run support, batch loop, checkpointing, and evaluation metrics.
- [x] **T009**: Author unit tests in `data-harvester/tests/v60/test_domain_translation_model.py` verifying forward pass, loss calculation, backward gradient flow, and parameter count.
- [x] **Gate C**: Run `uv run pytest tests/v60/test_domain_translation_model.py` and verify all tests pass. (Verified: 4 passed in 3.56s)

---

## Phase 4: Terrain Implication Correlation & Operator Verification

- [x] **T010**: Implement terrain elevation correlation bridge connecting translated clean hillshades to macro WDL + micro ADT elevation inference (`minimap_hillshade_correlator.py`).
- [x] **T011**: Prepare operator PowerShell 7 execution commands for full corpus batch export and GPU domain translation training.
- [x] **T012**: Write verification receipt in `evidence/receipt-spec274.md`.
