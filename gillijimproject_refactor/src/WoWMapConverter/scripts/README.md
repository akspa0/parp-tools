# WoW AI Terrain Tools

This directory contains a suite of tools for analyzing WoW ADT terrain data and training AI models to reconstruct 3D geometry from 2D images.

## 1. Topography & Analysis

### `terrain_librarian.py`
**Purpose**: Analyzes the dataset to find and catalog recurring "Prefab" patterns.
**Usage**:
```bash
python terrain_librarian.py
```
**Output**: 
- `prefab_library.json`: Database of unique chunks (hashed by geometry).
- `prefab_instances.json`: Map of all chunk locations.
- `Prefab_Zoo_X_Y.json`: Reconstructed ADT tiles visualizing the unique prefabs.

---

## 2. AI Training Pipeline (Image-to-Mesh)

### A. Data Generation: `generate_height_regression_dataset.py`
**Purpose**: Prepares the training dataset by pairing 64x64 Minimap crops with their corresponding 3D Height Arrays (145 floats).
**Usage**:
```bash
python generate_height_regression_dataset.py
```
**Output**: `height_regression.jsonl` (Training data).

### B. Training: `train_tiny_regressor.py`
**Purpose**: Fine-tunes a **Tiny Vision Transformer (ViT)** to predict height geometry from images.
**Features**:
- Uses `google/vit-base-patch16-224` (or Tiny variant).
- Custom Regression Head (145 outputs).
- RAM Caching & Label Normalization for speed/stability.
**Usage**:
```bash
python train_tiny_regressor.py
```
**Output**: Model saved to `j:\vlm_output\wow_tiny_vit_regressor`.

---

## 3. Inference

### `img2mesh.py`
**Purpose**: The end-user tool. Takes an image, runs it through the AI, and generates a 3D OBJ mesh with the image applied as a texture.
**Usage**:
```bash
python img2mesh.py path/to/minimap_chunk.png
```
**Output**: 
- `minimap_chunk.obj`: 3D Geometry.
- `minimap_chunk.mtl`: Texture material file.
