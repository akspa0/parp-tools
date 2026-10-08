# Data Model: Spec 262 — Minimap Residual Model & 3D Fractal Editor Brushes

**Spec**: [Spec 262](spec.md)  
**Date**: 2026-10-06  

---

## 1. Unified Zarr v3 Datastore Contract

All arrays and tensors are stored under a versioned, immutable Zarr v3 store at:
`output/datasets/v60/residual-datastore.zarr/` accompanied by an Apache Parquet index `catalog.parquet`.

### 1.1 Tensor Arrays (`residual-datastore.zarr/`)

| Array Name | Shape | Dtype | Chunk Size | Codec | Description |
|---|---|---|---|---|---|
| `real_minimap_rgb_256` | `(N, 256, 256, 3)` | `uint8` | `(1, 256, 256, 3)` | zstd (lvl 3) | Authentic 0.5.3 minimap RGB BLPs |
| `synth_control_shadow_256` | `(N, 256, 256)` | `float32` | `(1, 256, 256)` | zstd (lvl 3) | Synthetic terrain shadow with DXT1 block simulation |
| `object_contamination_mask_256` | `(N, 256, 256)` | `float32` | `(1, 256, 256)` | zstd (lvl 3) | Binary/soft mask from SAM 2.1 & PaliGemma 2 |
| `stripped_residual_shadow_256` | `(N, 256, 256)` | `float32` | `(1, 256, 256)` | zstd (lvl 3) | Albedo-normalized infilled bare terrain shadow |
| `shadow_difference_delta_256` | `(N, 256, 256)` | `float32` | `(1, 256, 256)` | zstd (lvl 3) | Pixel difference $\Delta S = S_{\text{real}} - S_{\text{synth}}$ |
| `ground_truth_height_257` | `(N, 257, 257)` | `float32` | `(1, 257, 257)` | zstd (lvl 5) | Absolute ground truth 0.5.3 terrain elevation (MCVT) |
| `ground_truth_normal_256` | `(N, 256, 256, 3)` | `float32` | `(1, 256, 256, 3)` | zstd (lvl 3) | Surface normal vectors derived from MCVT/MCNR |

### 1.2 3D Fractal Editor Brush Subgroup (`residual-datastore.zarr/fractal_brushes_3d/`)

| Array / Field | Shape | Dtype | Chunk Size | Description |
|---|---|---|---|---|
| `brush_displacement_kernels` | `(B, 65, 65)` | `float32` | `(1, 65, 65)` | Spatial vertical height displacement $\Delta Z(u, v)$ |
| `brush_alpha_kernels` | `(B, 65, 65)` | `float32` | `(1, 65, 65)` | Associated texture alpha splatting weight $\alpha(u, v)$ |
| `brush_metadata` | Parquet table | — | — | Footprint size (m), curvature spectrum, source tile ID |

---

## 2. Parquet Index Catalog Schema (`catalog.parquet`)

| Column Name | Arrow Type | Nullable | Description |
|---|---|---|---|
| `row_id` | `int64` | No | Contiguous integer index corresponding to Zarr row index |
| `build` | `string` | No | Client build identifier (e.g. `"0.5.3.3368"`) |
| `map_name` | `string` | No | Map identifier (e.g. `"Azeroth"`, `"Kalimdor"`) |
| `tile_x` | `int32` | No | ADT tile column coordinate ($0 \dots 63$) |
| `tile_y` | `int32` | No | ADT tile row coordinate ($0 \dots 63$) |
| `is_whiteplate` | `bool` | No | True if tile lacks complex diffuse textures (pure shadow signal) |
| `split` | `string` | No | Partition identifier (`"train"`, `"val"`, `"test_heldout"`) |
| `terrain_amplitude` | `float32` | No | Height range $\max(H) - \min(H)$ in world meters |
| `lighting_calib_mae` | `float32` | Yes | Photometric error achieved during calibration |
| `object_coverage_pct` | `float32` | No | Percentage of tile area occupied by doodads/structures |
| `fractal_brush_count` | `int32` | No | Number of identified 3D editor brush instances in tile |

---

## 3. C# Data Contracts (`WowViewer.Core/Maps/`)

### 3.1 `MinimapLightingParameters.cs`
```csharp
namespace WowViewer.Core.Maps;

/// <summary>
/// Parameter set defining the physical lighting and specular response of terrain minimap generation.
/// </summary>
public readonly record struct MinimapLightingParameters(
    float SolarAzimuth,        // θ in radians, [0, 2π)
    float SolarElevation,      // φ in radians, (0, π/2]
    float AmbientIntensity,    // A in [0.0, 1.0]
    float DiffuseIntensity,    // D in [0.0, 1.0]
    float CastShadowStrength,  // [0.0, 1.0]
    float SpecularIntensity,   // ks in [0.0, 1.0]
    float SpecularPower,       // p >= 1.0
    bool ApplyDxt1Quantization
);
```

### 3.2 `FractalEditorBrush3D.cs`
```csharp
namespace WowViewer.Core.Maps;

/// <summary>
/// Discrete 3D editor brush reproducing authentic WoWEdit procedural/fractal stamping operations.
/// </summary>
public sealed class FractalEditorBrush3D
{
    public required string BrushId { get; init; }
    public required float FootprintMeters { get; init; } // e.g. 16.0f, 33.33f, 64.0f
    public required int Resolution { get; init; }       // e.g. 65 or 129
    public required float[,] DisplacementKernel { get; init; } // Normalized height delta ΔZ(u, v)
    public required float[,] AlphaKernel { get; init; }        // Normalized texture alpha weight α(u, v)
    public string DominantBiome { get; init; } = string.Empty;
    public float CurvatureSharpness { get; init; }
}
```

---

## 4. Python Data Contracts (`data-harvester/src/harvester/v60/`)

### 4.1 `LightingCalibrationProfile`
```python
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True, slots=True)
class LightingCalibrationProfile:
    build: str
    map_name: str
    solar_azimuth_rad: float
    solar_elevation_rad: float
    ambient_intensity: float
    diffuse_intensity: float
    specular_intensity: float
    specular_power: float
    photometric_mae: float
    converged: bool
    iterations: int
    provenance_hash: str
```

### 4.2 `FractalBrushSample3D`
```python
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class FractalBrushSample3D:
    brush_id: str
    footprint_meters: float
    displacement_kernel: np.ndarray  # Shape (65, 65), float32 in [-1.0, 1.0]
    alpha_kernel: np.ndarray         # Shape (65, 65), float32 in [0.0, 1.0]
    dominant_layer_slot: int
    radial_falloff_exponent: float
    motif_signature_hash: str
```
