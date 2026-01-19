# Active Context

## Current Focus: V8 Implementation (Jan 18, 2026)

### Status Summary

**V8 RECONSTRUCTION BRANCH**: Core infrastructure implemented.
- **Binary Format Integrated**: `.bin` files (Heights/Normals/Shadows/Alpha) now used for training.
- **Split ADT Support**: `ExtractFromLkAdt` updated to handle WotLK/Cata split files.
- **Shadow Map Fix**: Corrected truncation bug (64 -> 512 bytes).
- **Python Pipeline**: `train_v8.py` and `v8_utils.py` updated to consume `.bin` files directly.

**Next Steps**: Validation run of `vlm-batch` and initial V8 training loop.

### V8 Key Design Decisions

| Decision | Value | Rationale |
|----------|-------|-----------|
| Input Channels | **15** | RGB MCCV (3ch) instead of luminance (1ch) |
| Texture Embeddings | **16-dim** | Better texture differentiation |
| Object Embeddings | **128-dim** | Sufficient for instance segmentation |
| Texture Output | **20 ch** | 4 alpha + 16 embedding (4 per layer) |
| **Heightmap Resolution** | **145×145** | Native ADT, avoids upsampling artifacts from V7 |
| **Segmentation Models** | **BiRefNet/LayerD** | Efficient multi-layer matting & brush detection (~200M params) |
| PM4 Integration | **Deferred to V10** | Focus on minimap-based detection first |

### Technical Strategy: "Neural Cartography"
- **Layer-Aware Matting**: Using `layerd-birefnet` iterative decomposition to separate 4 texture layers.
- **Edge Refinement**: Using `BEN2` CGM concepts to sharpen texture blending boundaries.
- **Brush Archeology**: Using `BiRefNet` to extract and categorize designer brush patterns (V9 precursor).
- **Efficiency**: All models targeted for consumer hardware (8-12GB VRAM), avoiding expensive "GIANT" foundation models where possible.
- **Funding Framing**: Reconstructing lost Worlds as a demonstration of "Neural Cartography" capabilities.

### V8 Training Data Priority

| Version | Maps | Reason |
|---------|------|--------|
| 0.5.3 Alpha | Azeroth, Kalimdor | Base training, original terrain |
| 3.3.5 WotLK | Northrend + updated EK/Kali | MCCV vertex colors |
| 4.3.4 Cata | LostIsles, MaelstromZone, Deepholm | Dev map tile sources |

---

## Key Documents

| Document | Purpose |
|----------|---------|
| `docs/V8_UNIVERSAL_MINIMAP_TRANSLATOR.md` | Full V8 specification |
| `docs/V7_HEIGHT_REGRESSOR.md` | V7 training documentation |
| `src/WoWMapConverter/scripts/train_v7.py` | Current training script |

---

## Alpha Mask Insight (Critical for V8)

> Alpha masks are composed of **preset brush patterns** - collections of low-resolution data
> that artists used to create complex non-repeating effects. This is WoW's "cheat code" for
> encoding texture transformations into terrain painting.

The V8 texture prediction head needs to implicitly learn these brush patterns.

---

1. **C# Infrastructure** - [x] `vlm-batch` Exporter, [x] Binary `.bin` Writer, [x] Split ADT Support.
2. **Python Infrastructure** - [x] `load_adt_bin` loader, [x] `train_v8.py` binary integration.
3. **Multi-Client Batch Export** - [ ] Validation run.


---

## Scripts Usage

```bash
# Current V7 training (running)
python src/WoWMapConverter/scripts/train_v7.py

# Future V8 (not yet implemented)
# V8 Training
dotnet run -- vlm-batch --config v8_config.json
python scripts/train_v8.py --dataset vlm_output/v8_dataset
```


---

## Scripts Usage

```bash
# Validate dataset
python scripts/prepare_v6_datasets.py --dataset test_data/vlm-datasets/053_azeroth_v11 --validate

# Render normalmaps
python scripts/prepare_v6_datasets.py --dataset test_data/vlm-datasets/053_azeroth_v11 --render-normalmaps

# Start training
python src/WoWMapConverter/scripts/train_height_regressor_v6_absolute.py
```

---

## Dataset Exports

| Map | Version | Tiles | Status |
|-----|---------|-------|--------|
| Azeroth | v11 | 685 | ✅ Ready for V6.1 |
| Kalimdor | v6 (pending) | ~951 | 🔧 Needs export |
| Shadowfang | v1 | ~20 | ✅ Complete |
| Deadmines | v2 | ~15 | ✅ Complete |

**Client Path**: `H:\053-client\`

---

## Key Files

| File | Purpose |
|------|---------|
| `train_height_regressor_v6_absolute.py` | V6.1 training script |
| `prepare_v6_datasets.py` | Dataset validation/fixing |
| `render_normalmaps.py` | Normalmap rendering from MCNR |
| `VlmDatasetExporter.cs` | C# dataset export |
| `VlmTrainingSample.cs` | JSON model definitions |

---

## Technical Notes

- **Alpha MCVT format**: 81 outer (9×9) FIRST, then 64 inner (8×8)
- **MCNR padding**: Alpha=448 bytes, LK=435 bytes (truncate to 435)
- **WDL upsampling**: 17×17 → 256×256 bilinear interpolation
- **Height normalization**: `(h - global_min) / (global_max - global_min)`

