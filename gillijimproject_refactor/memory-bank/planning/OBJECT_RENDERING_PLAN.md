# Object Rendering Plan for V6 Training

> **Last Updated**: January 16, 2026
> **Status**: Planned - Not Yet Implemented
> **Priority**: High

---

## Problem Statement

Minimaps show **objects** (buildings, trees, rocks, props) from a top-down view, but heightmaps only contain **terrain data**. This mismatch causes training loss noise because the model is penalized for object shapes it can see in the minimap but can't predict from terrain-only heightmaps.

### Impact

- Loss spikes on tiles with many objects (cities, forests)
- Model learns to hallucinate flat areas where objects exist
- Reduced prediction quality in urban/forested areas

### Training Goal

**The model should learn to predict flat/smoothed terrain where buildings exist.** This is exactly what we want for terrain reconstruction - the minimap shows objects, but we want the model to output terrain-only heightmaps that can be used for mesh generation (objects are placed separately).

---

## Existing Infrastructure

### Object Data in Dataset JSON

Each tile JSON already contains object placements:

```json
{
  "objects": [
    {
      "name": "arathitree03",
      "name_id": 1120,
      "unique_id": 209539,
      "x": 20828.436,
      "y": 95.82229,
      "z": 18159.25,
      "rot_x": -12,
      "rot_y": 120.27,
      "rot_z": -9,
      "scale": 0.74121094,
      "category": "m2"
    }
  ]
}
```

### Hole Data in Dataset JSON

Each tile also contains **hole bitmasks** (256 values, one per MCNK chunk):

```json
{
  "holes": [0, 0, 0, 4096, 0, ...]  // 256 values
}
```

Holes mark areas where:
- Buildings have basements or go through terrain
- The mesh data exists but collision is disabled
- Heightmaps contain terrain data BUT the game doesn't render it

### Mesh Parsers Available

| Parser | Location | Capabilities |
|--------|----------|--------------|
| `WMOReader.cs` | `WoWFormatLib` | MOVT (vertices), MOVI (indices), full WMO groups |
| `MDXReader.cs` | `WoWFormatLib` | VRTX (vertices), PVTX, NRMS (normals), GEOS (geosets) |
| `M2Reader.cs` | `WoWFormatLib` | Modern M2 format parsing |

> **Note**: `M2File.cs` in `WoWRollback.PM4Module` is for PM4 navigation mesh work, NOT for general MDX parsing.

> **WebWowViewerCpp Assessment**: This library targets modern WoW (Legion+, MD21, CASC). Adding 0.5.3 Alpha support would require implementing WMOv14 and MDLX parsers - significant work. For training pipeline, existing WoWFormatLib parsers are sufficient for bounding box extraction.

---

## Recommended Implementation

### Understanding the Data Relationship

1. **Heightmaps** = Terrain mesh data (MCVT heights from ADT chunks)
2. **Holes** = Collision flags - terrain data EXISTS but rendering is disabled
3. **Objects** = WMO/M2 placements with world position, rotation, scale

**Key Insight**: Heightmaps already contain terrain heights in hole areas. The "holes" just tell the game not to render/collide there. So heightmap targets are already "correct" - we need to help the model understand WHERE objects occlude the minimap.

### Phase 1: Hole Mask Rendering (Python - Using Existing Data)

The `holes` array already exists in JSON. Render hole masks to help the model understand where terrain doesn't render:

```python
def render_hole_mask(tile_json, output_size=256):
    """Render 256x256 hole mask from existing JSON data."""
    holes = tile_json.get("terrain_data", {}).get("holes", [])
    if len(holes) != 256:
        return np.zeros((output_size, output_size), dtype=np.float32)
    
    # holes is 256 values (16x16 chunks)
    # Each chunk maps to 16x16 pixels in 256x256 image
    mask = np.zeros((output_size, output_size), dtype=np.float32)
    
    for chunk_idx, hole_flags in enumerate(holes):
        if hole_flags == 0:
            continue
        
        chunk_y = chunk_idx // 16
        chunk_x = chunk_idx % 16
        
        # Each chunk has 4x4 sub-holes (16 bits)
        for sub_y in range(4):
            for sub_x in range(4):
                bit_idx = sub_y * 4 + sub_x
                if hole_flags & (1 << bit_idx):
                    # Mark 4x4 pixel region as hole
                    py_start = chunk_y * 16 + sub_y * 4
                    px_start = chunk_x * 16 + sub_x * 4
                    mask[py_start:py_start+4, px_start:px_start+4] = 1.0
    
    return mask
```

### Phase 2: Object Footprint Estimation (Python - No Mesh Loading)

For v1, estimate object footprint from position, scale, and category:

```python
# Approximate base sizes by category (in world units)
OBJECT_SIZES = {
    "m2": {"tree": 5.0, "shrub": 2.0, "rock": 3.0, "default": 4.0},
    "wmo": {"default": 20.0}  # WMOs are usually larger
}

def estimate_object_footprint(obj):
    """Estimate object footprint circle radius without loading mesh."""
    category = obj.get("category", "m2")
    name = obj.get("name", "").lower()
    scale = obj.get("scale", 1.0)
    
    # Heuristic based on name
    if "tree" in name:
        base_size = OBJECT_SIZES["m2"]["tree"]
    elif "shrub" in name or "bush" in name:
        base_size = OBJECT_SIZES["m2"]["shrub"]
    elif "rock" in name or "boulder" in name:
        base_size = OBJECT_SIZES["m2"]["rock"]
    elif category == "wmo":
        base_size = OBJECT_SIZES["wmo"]["default"]
    else:
        base_size = OBJECT_SIZES["m2"]["default"]
    
    return base_size * scale
```

### Phase 3: Training Integration

Two approaches to use this data:

**Option A: Hole + Object Mask as Loss Modifier**

```python
def combined_loss_with_masks(pred, target, hole_mask, object_mask):
    base_loss = F.l1_loss(pred, target, reduction='none')
    
    # In hole areas: terrain data exists but isn't rendered
    # Model should still learn the terrain, but we reduce weight slightly
    hole_weight = 0.5
    
    # In object areas: minimap shows objects, heightmap is terrain
    # Model should learn to predict underlying terrain (what we want!)
    object_weight = 0.3
    
    weights = torch.ones_like(base_loss)
    weights = weights * (1.0 - (1.0 - hole_weight) * hole_mask)
    weights = weights * (1.0 - (1.0 - object_weight) * object_mask)
    
    return (base_loss * weights).mean()
```

**Option B: Masks as Additional Input Channels**

Add hole_mask and object_mask as channels 9 and 10, letting the model learn to interpret object locations.

---

## Phase 4 (Future): Proper Mesh-Based Object Footprints

When ready for more accurate object silhouettes:

1. Use `WMOReader.cs` and `MDXReader.cs` from WoWFormatLib
2. Load mesh vertices for each unique model
3. Project top-down bounding box or convex hull
4. Cache results by model name

This is deferred because:
- Heuristic approach may be sufficient
- Mesh loading adds complexity
- Need to handle MPQ/CASC extraction for model files

---

## Implementation Order

1. **Phase 1** (Python): Render hole masks from existing JSON
2. **Phase 2** (Python): Estimate object footprints heuristically  
3. **Phase 3** (Python): Add masks to training as loss modifiers
4. **Phase 4** (Optional): Implement proper mesh projection later

---

## Files to Modify

| File | Change |
|------|--------|
| `prepare_v6_datasets.py` | Add `--render-hole-masks` and `--render-object-masks` |
| `train_height_regressor_v6_absolute.py` | Add mask-based loss weighting |
| `VlmDatasetExporter.cs` | (Future) Add bbox from mesh parsing |

