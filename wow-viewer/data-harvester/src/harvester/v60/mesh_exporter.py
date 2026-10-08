"""3D Terrain Mesh Exporters (Wavefront OBJ and Binary glTF 2.0 GLB).

Provides textured 3D mesh serialization for terrain surfaces reconstructed
from minimap residual shadows and 1.60 MCCV ground-truth vertex color fields.
"""

from __future__ import annotations

import io
import json
import logging
import struct
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def export_obj_mesh(
    height: np.ndarray,
    texture_path: Path,
    obj_path: Path,
    tile_size_yards: float = 533.333,
    height_scale: float = 40.0,
) -> None:
    """Export heightmap grid (H, W) to textured Wavefront OBJ + MTL.

    Parameters:
        height: 2D array of elevation values (e.g. 256x256 or 257x257).
        texture_path: Path to texture image file referenced by the MTL.
        obj_path: Output path for the .obj file.
        tile_size_yards: Lateral footprint size in yards (default 533.333 yards = 1 ADT tile).
        height_scale: Scaling factor applied to normalized heights in meters/yards.
    """
    h, w = height.shape
    rows = h - 1
    cols = w - 1
    mtl_name = obj_path.stem + ".mtl"
    mtl_path = obj_path.with_suffix(".mtl")

    h_min = float(np.min(height))
    h_max = float(np.max(height))
    h_span = max(1e-5, h_max - h_min)
    h_norm = ((height - h_min) / h_span) * height_scale

    # Write MTL
    with mtl_path.open("w", encoding="utf-8") as f:
        f.write("newmtl terrain\n")
        f.write("Ka 1.0 1.0 1.0\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write("Ks 0.05 0.05 0.05\n")
        f.write(f"map_Kd {texture_path.name}\n")

    # Write OBJ
    with obj_path.open("w", encoding="utf-8") as f:
        f.write(f"mtllib {mtl_name}\n")
        f.write("usemtl terrain\n")

        # Vertices (WoW coordinates: X East, Y North, Z Up)
        for y in range(h):
            for x in range(w):
                wx = (x / cols) * tile_size_yards
                wy = (y / rows) * tile_size_yards
                wz = float(h_norm[y, x])
                f.write(f"v {wx:.3f} {wy:.3f} {wz:.3f}\n")

        # Texture coordinates (glTF/OBJ UV: U in [0, 1], V in [0, 1])
        for y in range(h):
            for x in range(w):
                u = x / cols
                v = 1.0 - (y / rows)
                f.write(f"vt {u:.4f} {v:.4f}\n")

        # Normals (Upward baseline)
        for y in range(h):
            for x in range(w):
                f.write("vn 0.000 0.000 1.000\n")

        # Faces (quads divided into two triangles)
        for y in range(rows):
            for x in range(cols):
                v1 = y * w + x + 1
                v2 = y * w + (x + 1) + 1
                v3 = (y + 1) * w + (x + 1) + 1
                v4 = (y + 1) * w + x + 1
                f.write(f"f {v1}/{v1}/{v1} {v2}/{v2}/{v2} {v3}/{v3}/{v3}\n")
                f.write(f"f {v1}/{v1}/{v1} {v3}/{v3}/{v3} {v4}/{v4}/{v4}\n")

    logger.info("Exported OBJ mesh to %s (vertices: %d, faces: %d)", obj_path, h * w, rows * cols * 2)


def export_glb_mesh(
    height: np.ndarray,
    texture: Union[Path, Image.Image, np.ndarray],
    glb_path: Path,
    tile_size_yards: float = 533.333,
    height_scale: float = 40.0,
) -> None:
    """Export heightmap grid (H, W) to standalone binary glTF 2.0 (.glb) with embedded texture.

    Self-contained single file viewable directly in Windows 3D Viewer, Blender, and web viewers.

    Parameters:
        height: 2D array of elevation values (e.g. 256x256 or 257x257).
        texture: PIL Image, NumPy array (H, W, 3), or Path to image.
        glb_path: Output path for the .glb file.
        tile_size_yards: Lateral footprint size in yards (default 533.333).
        height_scale: Elevation amplitude scaling factor.
    """
    h, w = height.shape
    rows = h - 1
    cols = w - 1

    h_min = float(np.min(height))
    h_max = float(np.max(height))
    h_span = max(1e-5, h_max - h_min)
    h_norm = ((height - h_min) / h_span) * height_scale

    # Coordinates: glTF convention is X Right, Y Up (elevation), Z Forward (South)
    xs = np.linspace(0, tile_size_yards, w, dtype=np.float32)
    zs = np.linspace(0, tile_size_yards, h, dtype=np.float32)
    xx, zz = np.meshgrid(xs, zs)
    yy = h_norm.astype(np.float32)

    positions = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)

    # Compute vertex normals from height gradient
    spacing_x = float(tile_size_yards / cols)
    spacing_z = float(tile_size_yards / rows)
    dy_dz, dy_dx = np.gradient(yy, spacing_z, spacing_x)
    nx = -dy_dx
    ny = np.ones_like(dy_dx)
    nz = -dy_dz
    norm_len = np.sqrt(nx * nx + ny * ny + nz * nz)
    normals = np.stack([nx / norm_len, ny / norm_len, nz / norm_len], axis=-1).reshape(-1, 3).astype(np.float32)

    # Texture coordinates (U in [0, 1], V in [0, 1])
    us = np.linspace(0.0, 1.0, w, dtype=np.float32)
    vs = np.linspace(0.0, 1.0, h, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)
    uvs = np.stack([uu, vv], axis=-1).reshape(-1, 2).astype(np.float32)

    # Quad to triangle indexing
    r_idx = np.arange(rows, dtype=np.uint32)[:, None]
    c_idx = np.arange(cols, dtype=np.uint32)[None, :]
    v0 = (r_idx * w + c_idx).ravel()
    v1 = (r_idx * w + (c_idx + 1)).ravel()
    v2 = ((r_idx + 1) * w + (c_idx + 1)).ravel()
    v3 = ((r_idx + 1) * w + c_idx).ravel()

    t1 = np.stack([v0, v1, v2], axis=-1).ravel()
    t2 = np.stack([v0, v2, v3], axis=-1).ravel()
    indices = np.concatenate([t1, t2]).astype(np.uint32)

    # Convert texture to PNG bytes
    if isinstance(texture, (str, Path)):
        texture_img = Image.open(texture).convert("RGB")
    elif isinstance(texture, np.ndarray):
        arr = texture
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        texture_img = Image.fromarray(arr)
    elif isinstance(texture, Image.Image):
        texture_img = texture.convert("RGB")
    else:
        raise TypeError(f"Unsupported texture type: {type(texture)}")

    buf_img = io.BytesIO()
    texture_img.save(buf_img, format="PNG")
    img_bytes = buf_img.getvalue()

    def _pad4(b: bytes, pad_byte: bytes = b"\x00") -> bytes:
        rem = len(b) % 4
        return b if rem == 0 else b + (pad_byte * (4 - rem))

    pos_bytes = _pad4(positions.tobytes())
    norm_bytes = _pad4(normals.tobytes())
    uv_bytes = _pad4(uvs.tobytes())
    idx_bytes = _pad4(indices.tobytes())
    img_bytes_padded = _pad4(img_bytes)

    # Calculate buffer offsets
    offset_idx = 0
    len_idx = len(idx_bytes)
    offset_pos = offset_idx + len_idx
    len_pos = len(pos_bytes)
    offset_norm = offset_pos + len_pos
    len_norm = len(norm_bytes)
    offset_uv = offset_norm + len_norm
    len_uv = len(uv_bytes)
    offset_img = offset_uv + len_uv
    len_img = len(img_bytes_padded)

    total_bin = idx_bytes + pos_bytes + norm_bytes + uv_bytes + img_bytes_padded

    pos_min = positions.min(axis=0).tolist()
    pos_max = positions.max(axis=0).tolist()

    gltf = {
        "asset": {"version": "2.0", "generator": "WowViewer Spec 263 Exporter"},
        "scenes": [{"nodes": [0]}],
        "scene": 0,
        "nodes": [{"mesh": 0, "name": "TerrainTile"}],
        "meshes": [
            {
                "name": "TerrainMesh",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 1,
                            "NORMAL": 2,
                            "TEXCOORD_0": 3,
                        },
                        "indices": 0,
                        "material": 0,
                        "mode": 4,  # TRIANGLES
                    }
                ],
            }
        ],
        "materials": [
            {
                "name": "TerrainMaterial",
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.95,
                },
                "doubleSided": True,
            }
        ],
        "textures": [{"source": 0}],
        "images": [{"bufferView": 4, "mimeType": "image/png"}],
        "buffers": [{"byteLength": len(total_bin)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": offset_idx, "byteLength": len_idx, "target": 34963},  # ELEMENT_ARRAY_BUFFER
            {"buffer": 0, "byteOffset": offset_pos, "byteLength": len_pos, "target": 34962},  # ARRAY_BUFFER
            {"buffer": 0, "byteOffset": offset_norm, "byteLength": len_norm, "target": 34962},
            {"buffer": 0, "byteOffset": offset_uv, "byteLength": len_uv, "target": 34962},
            {"buffer": 0, "byteOffset": offset_img, "byteLength": len(img_bytes)},
        ],
        "accessors": [
            {"bufferView": 0, "byteOffset": 0, "componentType": 5125, "count": len(indices), "type": "SCALAR"},  # UNSIGNED_INT
            {"bufferView": 1, "byteOffset": 0, "componentType": 5126, "count": len(positions), "type": "VEC3", "min": pos_min, "max": pos_max},  # FLOAT
            {"bufferView": 2, "byteOffset": 0, "componentType": 5126, "count": len(normals), "type": "VEC3"},
            {"bufferView": 3, "byteOffset": 0, "componentType": 5126, "count": len(uvs), "type": "VEC2"},
        ],
    }

    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_padded = _pad4(json_bytes, b" ")

    total_len = 12 + 8 + len(json_padded) + 8 + len(total_bin)
    glb_header = struct.pack("<4sII", b"glTF", 2, total_len)
    json_chunk = struct.pack("<II", len(json_padded), 0x4E4F534A) + json_padded
    bin_chunk = struct.pack("<II", len(total_bin), 0x004E4942) + total_bin

    glb_path = Path(glb_path)
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    with glb_path.open("wb") as f:
        f.write(glb_header + json_chunk + bin_chunk)

    logger.info("Exported GLB mesh to %s (%d bytes)", glb_path, total_len)
