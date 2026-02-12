# V8 Universal Minimap Translator - Tooling Specification

> **Purpose:** Comprehensive technical specification for implementing V8 tooling infrastructure that translates minimaps → terrain + textures + objects for WoW 0.5.3, 3.x, and 4.x clients.

---

## 1. Executive Summary

The V8 system extends the V7 height regressor with three major enhancements:

1. **Texture Prediction**: Alpha masks + texture path embeddings for exact tileset matching
2. **Object Segmentation**: Instance-level object detection from minimap pixels
3. **Multi-Version Support**: Unified export pipeline for Alpha 0.5.3, WotLK 3.x, and Cata 4.x

This specification defines the data formats, interfaces, and implementation details needed to build this system.

---

## 1.1 V7 Lessons Learned

> [!WARNING]
> V7 training plateau'd at val loss 0.106 (epoch 79/500) and produced exaggerated heights. These issues inform V8 design.

### Observed Problems

| Issue | Symptom | Root Cause Hypothesis |
|-------|---------|----------------------|
| **Height Exaggeration** | Mesh too spiky/extreme | 512×512 upsampled targets may have lost normalization anchor |
| **Training Plateau** | No improvement after epoch 79 | Insufficient discriminative input signals (only 11 channels) |
| **Blocky Artifacts** | L-shaped blocks in predictions | Liquid masking interference or chunk boundary issues |
| **Early Stopping** | Patience exhausted at 25 epochs | Model capacity vs data complexity mismatch |

### V8 Mitigations

1. **Native Resolution Training**: Train heightmaps at 145×145 (ADT native) instead of 512×512
   - Avoids upsampling artifacts in ground truth
   - Faster training, less memory
   - Direct match to ADT output format

2. **More Input Context**: 15 channels (RGB MCCV + designkit) gives model more signals to disambiguate

3. **Brush Pattern Awareness**: Future brush segmentation tool will provide explicit texture context

4. **Multi-Task Learning**: Forcing model to also predict textures/objects may regularize height predictions

5. **Per-Tile Height Bounds**: Use actual `height_min`/`height_max` from JSON for denormalization, not global constants

## 2. Data Formats & Schemas

### 2.1 Enhanced VlmTrainingSample JSON Schema

```json
{
  "$schema": "v8_training_sample",
  "version": "8.0",
  "image": "images/Azeroth_32_48.png",
  "terrain_data": {
    "tile_name": "Azeroth_32_48",
    "tile_coord": { "x": 32, "y": 48 },
    "wow_version": "0.5.3",
    
    "heights": [...],
    "heightmap": "heightmaps/Azeroth_32_48_heightmap.png",
    "heightmap_global": "heightmaps/Azeroth_32_48_heightmap_global.png",
    "heightmap_local": "heightmaps/Azeroth_32_48_heightmap_local.png",
    "normalmap": "normalmaps/Azeroth_32_48_normalmap.png",
    
    "height_min": -50.0,
    "height_max": 150.0,
    "height_global_min": -1000.0,
    "height_global_max": 3000.0,
    
    "wdl_heights": {
      "outer_17": [/* 289 floats */],
      "inner_16": [/* 256 floats */]
    },
    
    "textures": [
      "Tileset\\Azeroth\\Elwynn\\ElwynnGrass.blp",
      "Tileset\\Azeroth\\Elwynn\\ElwynnDirt.blp"
    ],
    "texture_paths_full": {
      "Tileset\\Azeroth\\Elwynn\\ElwynnGrass.blp": {
        "designkit": "Azeroth/Elwynn",
        "category": "Grass",
        "embedding_id": 142
      }
    },
    
    "layers": [
      {
        "texture_id": 0,
        "texture_path": "Tileset\\Azeroth\\Elwynn\\ElwynnGrass.blp",
        "alpha_mask": "alpha/Azeroth_32_48_layer0.png",
        "flags": 0
      }
    ],
    
    "alpha_masks": [
      "alpha/Azeroth_32_48_layer0.png",
      "alpha/Azeroth_32_48_layer1.png",
      "alpha/Azeroth_32_48_layer2.png",
      "alpha/Azeroth_32_48_layer3.png"
    ],
    
    "mccv": {
      "path": "mccv/Azeroth_32_48_mccv.png",
      "has_vertex_colors": true
    },
    
    "liquid_mask": "liquids/Azeroth_32_48_liq_mask.png",
    "liquid_height": "liquids/Azeroth_32_48_liq_height.png",
    
    "objects": [
      {
        "object_type": "wmo",
        "asset_path": "World\\wmo\\Azeroth\\Buildings\\GoldshireInn.wmo",
        "name": "GoldshireInn",
        "pos_x": 100.5,
        "pos_y": 200.3,
        "pos_z": 50.0,
        "rot_x": 0.0,
        "rot_y": 0.0,
        "rot_z": 45.0,
        "scale": 1.0,
        "bounds_min": [-15.0, -15.0, 0.0],
        "bounds_max": [15.0, 15.0, 20.0],
        "footprint_pixels": [[100, 120], [130, 120], [130, 150], [100, 150]],
        "instance_id": 12345,
        "designkit": "Azeroth/Buildings"
      }
    ],
    
    "object_footprint_mask": "objects/Azeroth_32_48_footprint.png",
    "object_instance_mask": "objects/Azeroth_32_48_instances.png",
    
    "zone_id": 12,
    "zone_name": "Elwynn Forest",
    "designkit_class": 3
  }
}
```

### 2.2 Object Library Entry Schema

```json
{
  "name": "GoldshireInn",
  "asset_path": "World\\wmo\\Azeroth\\Buildings\\GoldshireInn.wmo",
  "designkit": "Azeroth/Buildings",
  "object_type": "wmo",
  
  "instances": [
    {
      "tile_name": "Azeroth_32_48",
      "instance_id": 12345,
      "scale": 1.0,
      "rotation": 45.0,
      "crop_path": "crops/GoldshireInn_12345.png",
      "mask_path": "masks/GoldshireInn_12345.png",
      "context": {
        "terrain_type": "grass",
        "nearby_objects": ["Tree_Oak", "Fence_Wood"]
      }
    }
  ],
  
  "geometry": {
    "bounds_local": {
      "min": [-15.0, -15.0, 0.0],
      "max": [15.0, 15.0, 20.0]
    },
    "footprint_area_m2": 900.0,
    "height_m": 20.0
  },
  
  "embedding": [/* 128-dim float vector from ResNet */],
  "variants_count": 1
}
```

### 2.3 Texture Library Entry Schema

```json
{
  "texture_id": 142,
  "full_path": "Tileset\\Azeroth\\Elwynn\\ElwynnGrass.blp",
  "relative_path": "Azeroth/Elwynn/ElwynnGrass.png",
  "designkit": "Azeroth/Elwynn",
  "category": "Grass",
  
  "file_hash": "a1b2c3d4e5f6...",
  "dimensions": { "width": 256, "height": 256 },
  
  "embedding": [/* 16-dim float vector */],
  
  "variants": [
    {
      "path": "Tileset\\Expansion01\\Elwynn\\ElwynnGrass.blp",
      "designkit": "Expansion01/Elwynn"
    }
  ]
}
```

---

## 3. C# Infrastructure Components

### 3.1 TilesetExporter (New: Preserves Directory Structure)

**File:** `WoWMapConverter.Core/VLM/TilesetExporter.cs`

```csharp
public class TilesetExporter
{
    /// <summary>
    /// Export tileset texture with full directory structure preserved.
    /// Input:  "Tileset\Azeroth\Elwynn\ElwynnGrass.blp"
    /// Output: "{outputDir}/tilesets/Azeroth/Elwynn/ElwynnGrass.png"
    /// </summary>
    public string ExportTileset(string blpPath, string outputDir, MpqArchiveService mpqService);
    
    /// <summary>
    /// Extract designkit from path.
    /// "Tileset\Azeroth\Elwynn\ElwynnGrass.blp" -> "Azeroth/Elwynn"
    /// </summary>
    public static string ExtractDesignkit(string blpPath);
    
    /// <summary>
    /// Build texture library with deduplicated embeddings.
    /// </summary>
    public TextureLibrary BuildTextureLibrary(string tilesetsDir);
}

public record TextureLibrary(
    Dictionary<string, TextureLibraryEntry> Entries,
    int TotalCount,
    string Version
);

public record TextureLibraryEntry(
    int TextureId,
    string FullPath,
    string RelativePath,
    string Designkit,
    string Category,
    string FileHash,
    float[] Embedding
);
```

**Key Implementation Details:**
1. Preserve full path hierarchy when exporting (no hash flattening)
2. Handle duplicates by comparing file hashes
3. Generate `texture_library.json` with all entries

### 3.2 ObjectLibraryBuilder (New: Geometry-Masked Crops)

**File:** `WoWMapConverter.Core/VLM/ObjectLibraryBuilder.cs`

```csharp
public class ObjectLibraryBuilder
{
    /// <summary>
    /// Build object library from all VLM-exported tiles.
    /// </summary>
    public async Task<ObjectLibrary> BuildLibraryAsync(
        string datasetDir,
        string clientPath,
        MpqArchiveService mpqService,
        IProgress<string>? progress = null
    );
    
    /// <summary>
    /// Rasterize 3D mesh geometry to 2D top-down mask.
    /// </summary>
    public Image<L8> RasterizeTopDown(Mesh mesh, BoundingBox worldBbox, int targetSize = 512);
    
    /// <summary>
    /// Transform world coordinates to minimap pixel coordinates.
    /// Formula: pixels = (world_units / 533.33) * 512
    /// </summary>
    public Rectangle WorldToMinimapBounds(BoundingBox worldBbox, string tileName, int minimapSize = 512);
    
    /// <summary>
    /// Extract minimap crop with geometry mask applied.
    /// </summary>
    public (byte[] crop, byte[] mask) ExtractMaskedCrop(
        Image<Rgba32> minimap,
        VlmObjectPlacement obj,
        Mesh mesh
    );
}

public record ObjectLibrary(
    Dictionary<string, ObjectLibraryEntry> Entries,
    int TotalInstances,
    string Version
);

public record ObjectLibraryEntry(
    string Name,
    string AssetPath,
    string Designkit,
    string ObjectType,
    List<ObjectInstance> Instances,
    ObjectGeometry Geometry,
    float[] Embedding
);

public record ObjectInstance(
    string TileName,
    uint InstanceId,
    float Scale,
    float Rotation,
    string CropPath,
    string MaskPath,
    ObjectContext Context
);

public record ObjectContext(
    string TerrainType,
    string[] NearbyObjects
);

public record ObjectGeometry(
    float[] BoundsMin,
    float[] BoundsMax,
    float FootprintAreaM2,
    float HeightM
);
```

**Key Implementation Details:**
1. Load WMO/M2 mesh via existing `AlphaMpqReader` and mesh parsers
2. Rasterize to 2D using orthographic projection (top-down)
3. Apply masks to minimap crops to isolate object pixels only
4. Store metadata for later matching during inference

### 3.3 MultiVersionAdtParser (Enhanced: Version Detection)

**File:** `WoWMapConverter.Core/VLM/MultiVersionAdtParser.cs`

```csharp
public enum WowVersion
{
    Alpha_053,    // Embedded ADT in WDT, no MCCV
    WotLK_303,    // Separate ADT files, MCCV present
    Cata_400,     // Split ADT (_obj0, _tex0, terrain)
    Unknown
}

public class MultiVersionAdtParser
{
    /// <summary>
    /// Detect WoW version from ADT/WDT file structure.
    /// </summary>
    public WowVersion DetectVersion(string adtPath, string wdtPath, MpqArchiveService mpqService);
    
    /// <summary>
    /// Parse ADT data into unified VlmTerrainData format.
    /// </summary>
    public VlmTerrainData ParseAdtUnified(
        string path,
        WowVersion version,
        int tileIndex,
        MpqArchiveService mpqService
    );
    
    /// <summary>
    /// Extract MCCV vertex colors (WotLK+).
    /// </summary>
    public Image<Rgba32>? ExtractMccv(byte[] adtBytes, string tileName, string outputDir);
}
```

**Version Detection Logic:**
```
1. Check if ADT is embedded in WDT (file size > 100KB) -> Alpha 0.5.3
2. Check for _obj0.adt / _tex0.adt files -> Cata 4.0+
3. Check for MCCV chunk in MCNK -> WotLK 3.x
4. Fallback to Alpha format
```

### 3.4 VlmDatasetExporterV8 (Enhanced Main Exporter)

**Modifications to existing `VlmDatasetExporter.cs`:**

```csharp
// New export options
public class VlmExportOptionsV8
{
    public bool PreserveTilesetPaths { get; set; } = true;
    public bool BuildObjectLibrary { get; set; } = true;
    public bool ExportMccv { get; set; } = true;
    public bool GenerateObjectMasks { get; set; } = true;
    public bool ComputeTextureEmbeddings { get; set; } = false; // Python handles this
    public WowVersion? ForceVersion { get; set; } = null;
}

// New methods in VlmDatasetExporter
public async Task<VlmExportResult> ExportMapV8Async(
    string clientPath,
    string mapName,
    string outputDir,
    VlmExportOptionsV8 options,
    IProgress<string>? progress = null
);

// Export per-tile object footprint mask (512x512, grayscale)
private async Task ExportObjectFootprintMask(
    List<VlmObjectPlacement> objects,
    string tileName,
    string outputDir
);

// Export per-tile object instance mask (512x512, indexed colors = instance IDs)
private async Task ExportObjectInstanceMask(
    List<VlmObjectPlacement> objects,
    string tileName,
    string outputDir
);
```

---

## 4. Python Infrastructure Components

### 4.1 TextureLibrary (FAISS-based Embedding Search)

**File:** `scripts/texture_library.py`

```python
from dataclasses import dataclass
from pathlib import Path
import faiss
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image

@dataclass
class TextureEntry:
    texture_id: int
    full_path: str
    relative_path: str
    designkit: str
    embedding: np.ndarray

class TextureLibrary:
    """
    Texture embedding library with FAISS nearest-neighbor search.
    Uses 16-dim embeddings for improved texture differentiation.
    """
    
    def __init__(self, tileset_dir: Path, embedding_dim: int = EMBEDDING_DIM):
        self.tileset_dir = tileset_dir
        self.embedding_dim = embedding_dim  # 16
        self.entries: dict[str, TextureEntry] = {}
        self.index: faiss.IndexFlatIP = None  # Cosine similarity via normalized L2
        self._embedder = self._build_embedder()
    
    def _build_embedder(self) -> torch.nn.Module:
        """Build lightweight texture embedder (ResNet18 truncated)."""
        resnet = models.resnet18(pretrained=True)
        # Remove final FC, keep up to avgpool -> 512-dim
        embedder = torch.nn.Sequential(*list(resnet.children())[:-1])
        # Project to embedding_dim
        embedder.add_module('flatten', torch.nn.Flatten())
        embedder.add_module('proj', torch.nn.Linear(512, self.embedding_dim))
        embedder.eval()
        return embedder
    
    def compute_embedding(self, img: Image.Image) -> np.ndarray:
        """Compute texture embedding from PIL image."""
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        with torch.no_grad():
            tensor = transform(img).unsqueeze(0)
            emb = self._embedder(tensor).numpy().flatten()
            return emb / (np.linalg.norm(emb) + 1e-8)  # L2 normalize
    
    def build_from_directory(self):
        """Scan tileset directory and build embedding index."""
        embeddings = []
        texture_id = 0
        
        for png_path in self.tileset_dir.rglob("*.png"):
            rel_path = png_path.relative_to(self.tileset_dir)
            designkit = "/".join(rel_path.parts[:-1])
            
            img = Image.open(png_path).convert("RGB")
            emb = self.compute_embedding(img)
            
            entry = TextureEntry(
                texture_id=texture_id,
                full_path=str(png_path),
                relative_path=str(rel_path),
                designkit=designkit,
                embedding=emb
            )
            self.entries[str(rel_path)] = entry
            embeddings.append(emb)
            texture_id += 1
        
        # Build FAISS index
        embeddings_np = np.vstack(embeddings).astype('float32')
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings_np)
    
    def find_nearest(self, pred_embedding: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """Find k nearest textures by embedding similarity."""
        pred_embedding = pred_embedding.reshape(1, -1).astype('float32')
        D, I = self.index.search(pred_embedding, k)
        
        results = []
        paths = list(self.entries.keys())
        for i, (idx, score) in enumerate(zip(I[0], D[0])):
            if idx < len(paths):
                results.append((paths[idx], float(score)))
        return results
    
    def save(self, path: Path):
        """Save library to disk."""
        import json
        data = {
            "entries": {k: {
                "texture_id": v.texture_id,
                "full_path": v.full_path,
                "relative_path": v.relative_path,
                "designkit": v.designkit,
                "embedding": v.embedding.tolist()
            } for k, v in self.entries.items()},
            "embedding_dim": self.embedding_dim
        }
        path.write_text(json.dumps(data, indent=2))
        faiss.write_index(self.index, str(path.with_suffix('.faiss')))
    
    def load(self, path: Path):
        """Load library from disk."""
        import json
        data = json.loads(path.read_text())
        self.embedding_dim = data["embedding_dim"]
        for k, v in data["entries"].items():
            self.entries[k] = TextureEntry(
                texture_id=v["texture_id"],
                full_path=v["full_path"],
                relative_path=v["relative_path"],
                designkit=v["designkit"],
                embedding=np.array(v["embedding"])
            )
        self.index = faiss.read_index(str(path.with_suffix('.faiss')))
```

### 4.2 ObjectLibrary (Feature Matching)

**File:** `scripts/object_library.py`

```python
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import faiss

@dataclass
class ObjectEntry:
    name: str
    asset_path: str
    designkit: str
    object_type: str
    embedding: np.ndarray
    instances: list[dict]

class ObjectLibrary:
    """
    Object embedding library for matching minimap crops to known objects.
    """
    
    def __init__(self, library_dir: Path, embedding_dim: int = 128):
        self.library_dir = library_dir
        self.embedding_dim = embedding_dim
        self.entries: dict[str, ObjectEntry] = {}
        self.index: faiss.IndexFlatIP = None
        self._embedder = self._build_embedder()
    
    def _build_embedder(self) -> torch.nn.Module:
        """Build object embedder (ResNet50 for more detail)."""
        resnet = models.resnet50(pretrained=True)
        embedder = torch.nn.Sequential(*list(resnet.children())[:-1])
        embedder.add_module('flatten', torch.nn.Flatten())
        embedder.add_module('proj', torch.nn.Linear(2048, self.embedding_dim))
        embedder.eval()
        return embedder
    
    def compute_embedding(self, crop: Image.Image, mask: Image.Image = None) -> np.ndarray:
        """Compute object embedding from masked crop."""
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        if mask:
            # Apply mask (black out non-object pixels)
            crop = crop.copy()
            mask_arr = np.array(mask.convert('L'))
            crop_arr = np.array(crop)
            crop_arr[mask_arr == 0] = 0
            crop = Image.fromarray(crop_arr)
        
        with torch.no_grad():
            tensor = transform(crop).unsqueeze(0)
            emb = self._embedder(tensor).numpy().flatten()
            return emb / (np.linalg.norm(emb) + 1e-8)
    
    def match_crop(self, crop: Image.Image, mask: Image.Image = None, k: int = 5) -> list[tuple[str, float]]:
        """Match a crop against the library."""
        emb = self.compute_embedding(crop, mask)
        emb = emb.reshape(1, -1).astype('float32')
        D, I = self.index.search(emb, k)
        
        results = []
        names = list(self.entries.keys())
        for idx, score in zip(I[0], D[0]):
            if idx < len(names):
                results.append((names[idx], float(score)))
        return results
```

### 4.3 WoWTileDatasetV8 (15 Input Channels)

**File:** `scripts/train_v8.py` (Dataset portion)

```python
class WoWTileDatasetV8(Dataset):
    """
    V8 Dataset with 15 input channels:
    - 0-2:   Minimap RGB (blurred)
    - 3-5:   Normal Map RGB
    - 6:     WDL Height
    - 7-8:   H_Min, H_Max (tile height bounds)
    - 9:     Water Mask
    - 10:    Object Footprint
    - 11-13: MCCV RGB (vertex colors - full color info)
    - 14:    Designkit Class (zone category)
    
    Targets:
    - Terrain: [2, 145, 145] - Global + Local heightmaps (ADT native)
    - Textures: [20, 512, 512] - 4 alpha + 16 embedding dims (4 per layer)
    - Objects: [64, 512, 512] - Instance segmentation embeddings
    """
    
    INPUT_CHANNELS = 15
    TERRAIN_CHANNELS = 2
    TEXTURE_CHANNELS = 20  # 4 alpha + 16 embedding (4 per layer)
    OBJECT_CHANNELS = 64   # Instance embedding dimension
    
    def __init__(self, dataset_roots: list[Path], texture_library: TextureLibrary):
        self.samples = []
        self.texture_library = texture_library
        self._load_samples(dataset_roots)
    
    def _load_mccv(self, mccv_path: Path) -> torch.Tensor:
        """Load MCCV as single luminance channel."""
        if mccv_path.exists():
            img = Image.open(mccv_path).convert("L")
            img = img.resize((512, 512), Image.BILINEAR)
            return transforms.ToTensor()(img)
        return torch.full((1, 512, 512), 0.5)
    
    def _encode_designkit(self, zone_name: str) -> torch.Tensor:
        """Encode zone/designkit as class index (0-15 categories)."""
        # Category mapping (example)
        categories = {
            "forest": 0, "grassland": 1, "desert": 2, "snow": 3,
            "swamp": 4, "mountain": 5, "beach": 6, "volcanic": 7,
            "urban": 8, "dungeon": 9, "underwater": 10, "void": 11,
            "fel": 12, "plague": 13, "jungle": 14, "other": 15
        }
        # Normalize zone name to category
        zone_lower = zone_name.lower()
        for cat, idx in categories.items():
            if cat in zone_lower:
                return torch.full((1, 512, 512), idx / 15.0)
        return torch.zeros(1, 512, 512)
    
    def _prepare_texture_target(self, sample: dict) -> torch.Tensor:
        """
        Prepare texture target: 4 alpha channels + 8 embedding dimensions.
        """
        texture_target = torch.zeros(self.TEXTURE_CHANNELS, 512, 512)
        terrain = sample.get("terrain_data", {})
        
        # Load alpha masks (channels 0-3)
        alpha_paths = terrain.get("alpha_masks", [])
        for i, path in enumerate(alpha_paths[:4]):
            full_path = sample["root"] / path
            if full_path.exists():
                alpha = Image.open(full_path).convert("L").resize((512, 512))
                texture_target[i] = transforms.ToTensor()(alpha).squeeze(0)
        
        # Compute per-layer texture embeddings (channels 4-11)
        # Average embedding across the tile for each layer
        layers = terrain.get("layers", [])
        for i, layer in enumerate(layers[:4]):
            tex_path = layer.get("texture_path", "")
            if tex_path and tex_path in self.texture_library.entries:
                emb = self.texture_library.entries[tex_path].embedding
                # Broadcast 8-dim embedding to 2 channels each per layer
                start_ch = 4 + i * 2
                texture_target[start_ch:start_ch+2] = torch.tensor(emb[:2]).view(2, 1, 1).expand(-1, 512, 512)
        
        return texture_target
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        terrain = sample["terrain_data"]
        root = sample["root"]
        
        # === INPUT CHANNELS ===
        # 0-2: Minimap RGB
        minimap = Image.open(root / sample["minimap"]).convert("RGB")
        minimap = self.blur(minimap.resize((512, 512)))
        minimap_t = self.normalize(self.to_tensor(minimap))
        
        # 3-5: Normal Map RGB
        normal_path = root / terrain.get("normalmap", "")
        normalmap = Image.open(normal_path).convert("RGB").resize((512, 512))
        normalmap_t = self.normalize(self.to_tensor(normalmap))
        
        # 6: WDL Height
        wdl_t = self._render_wdl(terrain.get("wdl_heights"))
        
        # 7-8: H_Min, H_Max
        h_min_n = (terrain.get("height_min", 0) + 1000) / 4000
        h_max_n = (terrain.get("height_max", 100) + 1000) / 4000
        h_min_mask = torch.full((1, 512, 512), h_min_n)
        h_max_mask = torch.full((1, 512, 512), h_max_n)
        
        # 9: Water Mask
        water_mask = self._load_mask(root / terrain.get("liquid_mask", ""))
        
        # 10: Object Footprint
        object_mask = self._load_mask(root / terrain.get("object_footprint_mask", ""))
        
        # 11: MCCV Luminance
        mccv = self._load_mccv(root / terrain.get("mccv", {}).get("path", ""))
        
        # 12: Designkit Class
        designkit = self._encode_designkit(terrain.get("zone_name", ""))
        
        # Concatenate all 13 channels
        input_tensor = torch.cat([
            minimap_t,      # 0-2
            normalmap_t,    # 3-5
            wdl_t,          # 6
            h_min_mask,     # 7
            h_max_mask,     # 8
            water_mask,     # 9
            object_mask,    # 10
            mccv,           # 11
            designkit       # 12
        ], dim=0)
        
        # === TARGETS ===
        terrain_target = self._load_terrain_target(sample)
        texture_target = self._prepare_texture_target(sample)
        object_target = self._load_object_target(sample)
        
        return {
            "input": input_tensor,           # [13, 512, 512]
            "terrain": terrain_target,       # [2, 512, 512]
            "textures": texture_target,      # [12, 512, 512]
            "objects": object_target         # [64, 512, 512]
        }
```

### 4.4 MultiChannelUNetV8 (Multi-Head Architecture)

**File:** `scripts/train_v8.py` (Model portion)

```python
class MultiChannelUNetV8(nn.Module):
    """
    V8 Multi-Head U-Net:
    - Shared encoder (512 -> 16)
    - Terrain head (2 channels: global + local height)
    - Texture head (12 channels: 4 alpha + 8 embedding)
    - Object head (64 channels: instance embeddings)
    """
    
    def __init__(self, in_channels=13):
        super().__init__()
        
        # === SHARED ENCODER ===
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        self.enc5 = self._conv_block(512, 1024)
        
        self.bottleneck = self._conv_block(1024, 2048)
        
        # === TERRAIN HEAD ===
        self.terrain_up5 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.terrain_dec5 = self._conv_block(2048, 1024)
        self.terrain_up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.terrain_dec4 = self._conv_block(1024, 512)
        self.terrain_up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.terrain_dec3 = self._conv_block(512, 256)
        self.terrain_up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.terrain_dec2 = self._conv_block(256, 128)
        self.terrain_up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.terrain_dec1 = self._conv_block(128, 64)
        self.terrain_out = nn.Conv2d(64, 2, 1)
        
        # === TEXTURE HEAD ===
        self.texture_up5 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.texture_dec5 = self._conv_block(2048, 512)  # Smaller
        self.texture_up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.texture_dec4 = self._conv_block(768, 256)
        self.texture_up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.texture_dec3 = self._conv_block(384, 128)
        self.texture_up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.texture_dec2 = self._conv_block(192, 64)
        self.texture_up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.texture_dec1 = self._conv_block(96, 32)
        self.texture_out = nn.Conv2d(32, 12, 1)  # 4 alpha + 8 embedding
        
        # === OBJECT HEAD ===
        self.object_up5 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.object_dec5 = self._conv_block(2048, 512)
        self.object_up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.object_dec4 = self._conv_block(768, 256)
        self.object_up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.object_dec3 = self._conv_block(384, 128)
        self.object_up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.object_dec2 = self._conv_block(192, 64)
        self.object_up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.object_dec1 = self._conv_block(96, 32)
        self.object_out = nn.Conv2d(32, 64, 1)  # Instance embeddings
        
        # === BOUNDS HEAD (Global) ===
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.bounds_fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4)  # tile_min, tile_max, global_min, global_max
        )
        
        self.pool = nn.MaxPool2d(2)
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # Shared encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        b = self.bottleneck(self.pool(e5))
        
        # Bounds prediction
        g = self.global_pool(b).view(b.size(0), -1)
        bounds = self.bounds_fc(g)
        
        # Terrain decoder
        td5 = torch.cat([self.terrain_up5(b), e5], dim=1)
        td5 = self.terrain_dec5(td5)
        td4 = torch.cat([self.terrain_up4(td5), e4], dim=1)
        td4 = self.terrain_dec4(td4)
        td3 = torch.cat([self.terrain_up3(td4), e3], dim=1)
        td3 = self.terrain_dec3(td3)
        td2 = torch.cat([self.terrain_up2(td3), e2], dim=1)
        td2 = self.terrain_dec2(td2)
        td1 = torch.cat([self.terrain_up1(td2), e1], dim=1)
        td1 = self.terrain_dec1(td1)
        terrain = torch.sigmoid(self.terrain_out(td1))
        
        # Texture decoder (similar pattern, omitted for brevity)
        # ... 
        textures = torch.sigmoid(self.texture_out(txd1))
        
        # Object decoder (similar pattern, omitted for brevity)
        # ...
        objects = self.object_out(od1)  # No sigmoid - raw embeddings
        
        return {
            "terrain": terrain,      # [B, 2, 512, 512]
            "textures": textures,    # [B, 12, 512, 512]
            "objects": objects,      # [B, 64, 512, 512]
            "bounds": bounds         # [B, 4]
        }
```

### 4.5 Multi-Task Loss Function

```python
def v8_loss(pred: dict, target: dict, texture_library: TextureLibrary) -> tuple[torch.Tensor, dict]:
    """
    V8 multi-task loss function.
    
    Returns:
        total_loss: Weighted sum of all losses
        components: Dict of individual loss values for logging
    """
    # === TERRAIN LOSS ===
    terrain_pred = pred["terrain"]
    terrain_gt = target["terrain"]
    
    l_global = F.l1_loss(terrain_pred[:, 0], terrain_gt[:, 0])
    l_local = F.l1_loss(terrain_pred[:, 1], terrain_gt[:, 1])
    l_edge = edge_loss(terrain_pred, terrain_gt)
    
    l_terrain = 0.4 * l_global + 0.4 * l_local + 0.2 * l_edge
    
    # === TEXTURE LOSS ===
    texture_pred = pred["textures"]
    texture_gt = target["textures"]
    
    # Alpha loss (channels 0-3)
    l_alpha = F.mse_loss(texture_pred[:, :4], texture_gt[:, :4])
    
    # Embedding loss (channels 4-11): cosine similarity
    emb_pred = texture_pred[:, 4:]
    emb_gt = texture_gt[:, 4:]
    l_embed = 1 - F.cosine_similarity(emb_pred, emb_gt, dim=1).mean()
    
    l_texture = 0.6 * l_alpha + 0.4 * l_embed
    
    # === OBJECT LOSS ===
    object_pred = pred["objects"]
    object_gt = target["objects"]
    
    # Instance-aware contrastive loss
    l_object = instance_contrastive_loss(object_pred, object_gt)
    
    # === BOUNDS LOSS ===
    l_bounds = F.mse_loss(pred["bounds"], target["bounds"])
    
    # === TOTAL ===
    total = (
        0.30 * l_terrain +
        0.25 * l_texture +
        0.35 * l_object +
        0.10 * l_bounds
    )
    
    return total, {
        "terrain": l_terrain.item(),
        "terrain_global": l_global.item(),
        "terrain_local": l_local.item(),
        "texture": l_texture.item(),
        "alpha": l_alpha.item(),
        "embed": l_embed.item(),
        "object": l_object.item(),
        "bounds": l_bounds.item()
    }


def instance_contrastive_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Instance-aware contrastive loss for object segmentation.
    Pixels belonging to same instance should have similar embeddings.
    """
    # Flatten spatial dimensions
    B, C, H, W = pred.shape
    pred_flat = pred.permute(0, 2, 3, 1).reshape(B * H * W, C)
    gt_flat = gt.permute(0, 2, 3, 1).reshape(B * H * W, C)
    
    # Cosine similarity between pred and gt embeddings
    similarity = F.cosine_similarity(pred_flat, gt_flat, dim=1)
    return 1 - similarity.mean()
```

---

## 5. Directory Structure

```
vlm-datasets/
├── {version}_{mapName}_v30/
│   ├── images/                    # Minimap tiles (256x256 or 512x512)
│   ├── heightmaps/                # Local heightmaps (512x512, 16-bit)
│   ├── heightmaps_global/         # Global heightmaps (relative to map min/max)
│   ├── normalmaps/                # Normal maps (512x512, RGB)
│   ├── shadows/                   # Per-chunk shadow maps (64x64)
│   ├── alpha/                     # Tile-level stitched alpha masks (512x512)
│   ├── masks/                     # Per-chunk alpha masks
│   ├── liquids/                   # Liquid height & mask maps
│   ├── mccv/                      # Vertex color maps (for WotLK+)
│   ├── objects/                   # Object footprint & instance masks
│   ├── tilesets/                  # Tileset textures (FULL path preserved)
│   │   ├── Azeroth/
│   │   │   └── Elwynn/
│   │   │       ├── ElwynnGrass.png
│   │   │       └── ElwynnDirt.png
│   │   └── Expansion01/
│   ├── object_library/            # Object crops & masks
│   │   ├── Inn.wmo/
│   │   │   ├── metadata.json
│   │   │   └── crops/
│   │   └── Tree_Oak.m2/
│   ├── dataset/                   # JSON per tile
│   ├── stitched/                  # Full map stitched images
│   ├── texture_library.json       # Texture embeddings index
│   ├── texture_library.faiss      # FAISS index
│   ├── object_library.json        # Object embeddings index
│   └── object_library.faiss
```

---

## 6. Verification Plan

### 6.1 C# Unit Tests

**Location:** `tests/GillijimProject.Next.Tests/`

```bash
# Run all tests
dotnet test tests/GillijimProject.Next.Tests/

# Run specific test class
dotnet test --filter "FullyQualifiedName~TilesetExporterTests"
```

**New Test Files to Create:**
- `TilesetExporterTests.cs` - Verify path preservation, designkit extraction
- `ObjectLibraryBuilderTests.cs` - Verify mesh rasterization, crop extraction
- `MultiVersionAdtParserTests.cs` - Verify version detection for all formats

### 6.2 Python Smoke Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run dataset loading test
python -c "
from train_v8 import WoWTileDatasetV8
from texture_library import TextureLibrary
from pathlib import Path

# Test texture library
tlib = TextureLibrary(Path('test_data/tilesets'), embedding_dim=8)
tlib.build_from_directory()
assert len(tlib.entries) > 0, 'No textures loaded'

# Test dataset
ds = WoWTileDatasetV8([Path('test_data/vlm-datasets/053_Azeroth_v30')], tlib)
sample = ds[0]
assert sample['input'].shape == (13, 512, 512), f'Bad input shape: {sample[\"input\"].shape}'
assert sample['terrain'].shape == (2, 512, 512), f'Bad terrain shape'
assert sample['textures'].shape == (12, 512, 512), f'Bad texture shape'
print('All tests passed!')
"
```

### 6.3 End-to-End Pipeline Test

```bash
# 1. Export dataset with V8 options
dotnet run --project src/WoWMapConverter -- vlm-export \
  --client "J:\Alpha\0.5.3" \
  --map "Azeroth" \
  --output "test_output/053_Azeroth_v8" \
  --v8-options

# 2. Verify output structure
python -c "
from pathlib import Path
import json

output = Path('test_output/053_Azeroth_v8')

# Check directories exist
assert (output / 'tilesets').exists(), 'Missing tilesets'
assert (output / 'objects').exists(), 'Missing objects'
assert (output / 'mccv').exists(), 'Missing mccv'

# Check JSON schema
sample_json = next((output / 'dataset').glob('*.json'))
data = json.loads(sample_json.read_text())
assert 'texture_paths_full' in data.get('terrain_data', {}), 'Missing texture_paths_full'
assert 'object_footprint_mask' in data.get('terrain_data', {}), 'Missing object_footprint_mask'

print('V8 export validated successfully!')
"

# 3. Build texture library
python scripts/build_texture_library.py \
  --input test_output/053_Azeroth_v8/tilesets \
  --output test_output/053_Azeroth_v8/texture_library.json

# 4. Build object library
python scripts/build_object_library.py \
  --input test_output/053_Azeroth_v8 \
  --output test_output/053_Azeroth_v8/object_library.json

# 5. Train V8 model (smoke test - 1 epoch)
python scripts/train_v8.py \
  --dataset test_output/053_Azeroth_v8 \
  --epochs 1 \
  --batch-size 2
```

---

## 7. Success Criteria

| Component | Metric | Target |
|-----------|--------|--------|
| Tileset Export | Path preservation accuracy | 100% (no flattening) |
| Object Library | Crops extracted per map | > 5,000 |
| Version Detection | Correct identification | 100% (all 3 versions) |
| MCCV Export | Files generated for WotLK | > 0 per tile |
| Texture Embedding | Top-1 match accuracy | > 85% |
| Object Matching | Top-5 match recall | > 90% |
| V8 Val Loss | Combined multi-task | < 0.10 |
| Edge Loss | Terrain boundary accuracy | < 0.05 |

---

## 8. Implementation Priority

1. **Week 1 (C# Core)**
   - TilesetExporter with path preservation
   - MultiVersionAdtParser with detection logic
   - Extended VlmTrainingSample schema

2. **Week 2 (C# Objects)**
   - ObjectLibraryBuilder with mesh rasterization
   - MCCV export for WotLK+
   - Object footprint/instance masks

3. **Week 3 (Python)**
   - TextureLibrary with FAISS (16-dim embeddings)
   - ObjectLibrary with feature matching (128-dim embeddings)
   - WoWTileDatasetV8 (15 channels)

4. **Week 4 (Training)**
   - MultiChannelUNetV8 architecture
   - Multi-task loss function
   - Training pipeline with V7 checkpoint init

---

## 9. Multi-Client Batch Processing

> [!NOTE]
> PM4 integration is deferred to **V10**. This section covers processing multiple client versions in a unified pipeline.

### 9.1 Client Configuration Schema

```json
{
  "clients": [
    {
      "path": "J:\\WoW\\Alpha\\0.5.3",
      "version": "0.5.3",
      "build": 3368,
      "maps": ["Azeroth", "Kalimdor", "Kalidar"]
    },
    {
      "path": "J:\\WoW\\WotLK\\3.3.5a",
      "version": "3.3.5",
      "build": 12340,
      "maps": ["Azeroth", "Kalimdor", "Northrend", "EasternKingdoms"]
    },
    {
      "path": "J:\\WoW\\Cata\\4.3.4",
      "version": "4.3.4",
      "build": 15595,
      "maps": ["Azeroth", "Kalimdor", "Northrend", "LostIsles", "MaelstromZone", "Deepholm"]
    }
  ]
}
```

### 9.2 Version Detection from Folder Path

```csharp
public class ClientVersionDetector
{
    /// <summary>
    /// Detect version from folder name patterns like "0.5.3", "3.3.5a", "4.3.4"
    /// </summary>
    public static (string Version, int Build)? DetectFromPath(string clientPath)
    {
        var folderName = Path.GetFileName(clientPath);
        
        // Pattern: "0.5.3", "3.3.5a", "4.3.4", etc.
        var versionMatch = Regex.Match(folderName, @"(\d+\.\d+\.\d+[a-z]?)");
        if (versionMatch.Success)
        {
            var version = versionMatch.Groups[1].Value;
            var build = GuessBuildFromVersion(version);
            return (version, build);
        }
        
        // Fallback: read .build.info or WoW.exe version
        return DetectFromExecutable(clientPath);
    }
    
    private static int GuessBuildFromVersion(string version) => version switch
    {
        "0.5.3" => 3368,
        "3.3.5" or "3.3.5a" => 12340,
        "4.3.4" => 15595,
        _ => 0
    };
}
```

### 9.3 Priority Maps by Version

| Version | Priority Maps | Reason |
|---------|--------------|--------|
| **0.5.3 Alpha** | Azeroth, Kalimdor | Base training data, original terrain |
| **3.3.5 WotLK** | Northrend, (updated Azeroth/Kali) | MCCV vertex colors, refined textures |
| **4.3.4 Cata** | LostIsles, MaelstromZone, Deepholm | Dev map tile sources, Cata-specific zones |

### 9.4 Batch Export Command

```bash
# Export all clients in sequence
dotnet run --project src/WoWMapConverter -- vlm-batch-export \
  --config clients.json \
  --output "J:\vlm-datasets" \
  --v8-options
```

---

## 10. Resolved Design Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| Texture Embedding Dimension | **16-dim** | Better differentiation for similar textures |
| Object Embedding Dimension | **128-dim** | Sufficient for instance segmentation |
| MCCV Handling | **RGB (3 channels)** | Preserves color info for lighting reconstruction |
| PM4 Integration | **Deferred to V10** | Focus on minimap-based detection first |
| Training Data | **All versions** | 0.5.3, 3.x, 4.x with priority maps above |
| Input Channels | **15 total** | RGB MCCV adds 2 channels vs original plan |
| Texture Output | **20 channels** | 4 alpha + 16 embedding (4 per layer) |
| **Heightmap Resolution** | **145×145** | Native ADT resolution, avoids upsampling artifacts |

---

## 11. Future Work: Brush Segmentation Tool

> [!TIP]
> This is a **V9+** feature that will provide explicit texture context for minimap de-baking.

### Background

Alpha masks contain **preset brush patterns** - collections of low-resolution patterns artists used to create complex non-repeating textures. These are effectively "prefabs" that originated from Warcraft 3's texture system.

### Proposed Tool: `brush_segmenter.py`

```python
# Extract all 64×64 alpha chunks from dataset
# Cluster into canonical brush library
# Output: brush_library.json with brush IDs + canonical patches

def extract_brush_patches(dataset_dir: Path) -> list[np.ndarray]:
    """Extract all 64×64 alpha chunks."""
    patches = []
    for alpha_png in (dataset_dir / "alpha").glob("*.png"):
        img = np.array(Image.open(alpha_png).convert("L"))
        # Alpha masks are 512×512, divided into 64×64 chunks
        for cy in range(8):
            for cx in range(8):
                patch = img[cy*64:(cy+1)*64, cx*64:(cx+1)*64]
                patches.append(patch)
    return patches

def cluster_brushes(patches: list[np.ndarray], k: int = 256) -> BrushLibrary:
    """Cluster patches into canonical brushes using k-means."""
    # Use VAE or ResNet embeddings
    # Cluster with k-means or DBSCAN
    # Return canonical patches + assignments
    pass
```

### Use Cases

1. **Minimap De-baking**: Predict brush IDs instead of raw alpha pixels
2. **Noggit Integration**: Export brush library as PNG brush set
3. **V8 Regularization**: Add brush classification as auxiliary task

### Output Format

```json
{
  "brushes": [
    {
      "brush_id": 0,
      "canonical_patch": "brushes/brush_000.png",
      "frequency": 15234,
      "categories": ["splat", "organic"]
    }
  ],
  "assignments": {
    "Azeroth_32_48_layer1_chunk_3_5": 42
  }
}
```

---

## 12. Synthetic Flat Minimap Generation

> [!IMPORTANT]
> This is a key insight for training texture prediction: we can generate "flat" minimaps from our dataset without 3D terrain deformation, providing clean ground truth for un-baking.

### The Deformation Problem

Real minimaps (256×256 native) include **3D perspective distortion** from terrain mesh:
- Hills/valleys stretch or compress textures
- Camera angle affects pixel coverage
- This makes minimap → alpha mask prediction harder

### Solution: Synthetic Ground Truth

We have all the data needed to synthesize **flat minimaps** (no 3D deformation):

```python
def synthesize_flat_minimap(tile_json: dict, tileset_dir: Path) -> np.ndarray:
    """
    Generate a flat minimap by compositing alpha layers + textures.
    No terrain mesh deformation - pure 2D blending.
    """
    canvas = np.zeros((512, 512, 3), dtype=np.float32)
    
    for layer in tile_json["layers"]:
        # Load texture
        tex_path = tileset_dir / layer["texture_path"].replace(".blp", ".png")
        texture = np.array(Image.open(tex_path).convert("RGB"))
        texture = np.tile(texture, (2, 2, 1))[:512, :512]  # Tile to 512×512
        
        # Load alpha mask
        alpha = np.array(Image.open(layer["alpha_mask"]).convert("L")) / 255.0
        alpha = alpha[:, :, np.newaxis]
        
        # Blend
        canvas = canvas * (1 - alpha) + texture * alpha
    
    return canvas.astype(np.uint8)
```

### Deformation Map Extraction

```python
def extract_deformation(real_minimap: np.ndarray, flat_minimap: np.ndarray) -> np.ndarray:
    """
    Extract the 3D deformation effect by comparing real vs flat.
    
    Steps:
    1. Downscale flat to 256×256 (native minimap res)
    2. Subtract from real minimap
    3. Result = deformation-induced color/brightness shift
    """
    # Downscale - try both to determine which WoW uses
    flat_256_bilinear = cv2.resize(flat_minimap, (256, 256), interpolation=cv2.INTER_LINEAR)
    flat_256_nearest = cv2.resize(flat_minimap, (256, 256), interpolation=cv2.INTER_NEAREST)
    
    # Compare to real
    diff_bilinear = real_minimap.astype(float) - flat_256_bilinear.astype(float)
    diff_nearest = real_minimap.astype(float) - flat_256_nearest.astype(float)
    
    return diff_bilinear, diff_nearest
```

### Training Strategy

| Data Type | Resolution | Purpose |
|-----------|------------|---------|
| Real Minimap | 256×256 (native) | Input |
| Flat Minimap (synthesized) | 512×512 → 256×256 | Ground truth for texture |
| Deformation Map | 256×256 | Auxiliary signal (optional) |
| Alpha Masks | 64×64 per chunk | Ground truth for layer weights |

### Key Insight: Brush Workflow

> The level designers likely used a **brush pattern interface** rather than painting directly.
> Alpha masks show repeated brush patterns → artists selected from a brush library, not freehand painting.

This means:
1. Alpha masks are deterministic (brush ID + position + parameters)
2. Predicting brush IDs may be easier than predicting raw alpha pixels
3. Downscaling uses **nearest neighbor** for alpha (discrete values), **bilinear** for color

### Resolution Notes

- **Minimaps**: 256×256 native (game files)
- **Alpha masks**: 64×64 per chunk (native), stored as 8-bit discrete
- **Heightmaps**: 145×145 per tile (9×9 outer + 8×8 inner per chunk × 256 chunks)
- **Synthesized data**: Can be any resolution we want - use game-res for training

---

## 13. Current HuggingFace Models (January 2026)

> [!WARNING]
> We already tried DPT and Qwen3-VL-8B - they didn't work for our use case. 
> The U-Net architecture was chosen after those experiments failed.
> Hardware constraints: These models should be used as *guidance* or *pre-processors*, not necessarily as replacements for our efficient U-Net.

### 13.1 Relevant Trending Models

| Model | Category | Size | WoW Application |
|-------|----------|------|-----------------|
| **cyberagent/layerd-birefnet** | Matting/Seg | ~200M | **Multi-layer decomposition** - iterative alpha layering! |
| **PramaLLC/BEN2** | Seg/Matting | - | **Edge refinement** - clean texture blending boundaries |
| **ZhengPeng7/BiRefNet** | DIS | 200M | **Brush detection** - discrete pattern recognition |
| **depth-anything/DA3-GIANT** | Depth | 1B | Foundation - validates plain transformer sufficiency |

### 13.2 Layer-Aware Iterative Decomposition

The `layerd-birefnet` (matting module of LayerD) uses an **iterative process** of "top-layer matting and background completion."

**WoW Application:**
WoW ground textures are stacked (Layer 1 → Layer 2 → Layer 3).
1. **Iteration 1**: Segment "topmost" texture (Layer 4 alpha).
2. **Background Completion**: "Inpaint" what was behind Layer 4.
3. **Iteration 2**: Segment next layer (Layer 3 alpha).
4. **Result**: A clean stack of 4 alpha masks + 4 texture IDs.

### 13.3 BEN2: Confidence Guided Matting (CGM)

BEN2 uses a **Refiner Network** specifically for low-confidence pixels.

**WoW Application:**
Our U-Net might be "unsure" about the exact edge of a grass-to-dirt transition. 
- A specialized CGM refiner (like BEN2) can take the U-Net's "unsure" mask and sharpen the edges to match ADT high-resolution blending.

### 13.4 BiRefNet: Bilateral Reference for Brushes

BiRefNet captures both local (detailed) and global (contextual) info.

**WoW Application (V9+):**
- **Local**: Captures the exact shape of a "Brush 04" dab.
- **Global**: Understands that this dab is part of a larger path/road.
- **Workflow**: BiRefNet segments the "brush stroke" -> we match it against the Noggit brush library (`Noggit/brushes/*.png`).

### 13.5 Hardware & Funding Strategy

> [!IMPORTANT]
> **Autistic Advocacy & Project Funding**
> Framing these technical achievements is key to attracting interest/funding:
> 1. **"Any-to-Any" Reconstruction**: We aren't just making a map; we are reconstructing a lost 3D world from 2D pixels.
> 2. **Efficient Design**: Using 200M parameter models (BiRefNet) instead of 8B param models shows engineering discipline.
> 3. **Open Source Contribution**: Releasing the "Synthetic Flat Minimap" dataset identifies you as a domain expert in "Neural Cartography."

**Hardware Feasibility:**
- **Inference**: All models mentioned (BiRefNet, U-Net) run comfortably on 8GB-12GB VRAM.
- **Training**: Small models (200M) can be fine-tuned on single consumer GPUs (3080/4090) or free Kaggle/Colab tiers.

### 13.6 Recommended Path Forward (Revised)

1. **V8.0 (U-Net Baseline)**: Implement the multi-head output as specified.
2. **V8.1 (Layer-Aware Matting)**: Incorporate the iterative background-completion logic from `layerd-birefnet` into the training loop/inference.
3. **V8.2 (Refiner Head)**: Add a CGM-style refiner (BEN2 concept) to sharpen the final ADT heightmap and alpha masks.
4. **V9.0 (Brush Archeology)**: Use BiRefNet to extract and categorize the 100+ brush patterns used by Blizzard designers.

