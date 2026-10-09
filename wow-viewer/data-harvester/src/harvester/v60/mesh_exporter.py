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
    width_yards: Optional[float] = None,
    height_yards: Optional[float] = None,
    height_scale: float = 40.0,
    is_world_yards: bool = False,
    y_up: bool = False,
    placements: Optional[List[Any]] = None,
    export_building_boxes: bool = False,
) -> None:
    """Export heightmap grid (H, W) to textured Wavefront OBJ + MTL with optional 3D building bounding boxes.

    Parameters:
        height: 2D array of elevation values (e.g. 256x256 or 257x257).
        texture_path: Path to texture image file referenced by the MTL.
        obj_path: Output path for the .obj file.
        tile_size_yards: Lateral footprint size in yards (fallback if width/height not set).
        width_yards: Footprint width along X (East) in yards.
        height_yards: Footprint height along Z/Y (South/North) in yards.
        height_scale: Scaling factor applied to normalized heights when is_world_yards is False.
        is_world_yards: If True, uses height array directly as world elevation in yards.
        y_up: If True (default for Blender/3D Viewer/Three.js), Y is elevation, Z is depth.
        placements: Optional list of WmoPlacement / M2Placement objects.
        export_building_boxes: If True, appends 3D collision bounding boxes for placed buildings.
    """
    h, w = height.shape
    rows = h - 1
    cols = w - 1
    mtl_name = obj_path.stem + ".mtl"
    mtl_path = obj_path.with_suffix(".mtl")

    w_span = width_yards if width_yards is not None else tile_size_yards
    h_span_yards = height_yards if height_yards is not None else tile_size_yards

    if is_world_yards:
        h_world = height.astype(np.float32)
    else:
        h_min = float(np.min(height))
        h_max = float(np.max(height))
        h_span = max(1e-5, h_max - h_min)
        h_world = ((height - h_min) / h_span) * height_scale

    # Compute surface normals from elevation gradients
    spacing_x = float(w_span / cols)
    spacing_z = float(h_span_yards / rows)
    dy_dz, dy_dx = np.gradient(h_world, spacing_z, spacing_x)
    if y_up:
        # Surface Y = f(X, Z): normal = (-dY/dX, 1.0, -dY/dZ)
        nx = -dy_dx
        ny = np.ones_like(dy_dx)
        nz = -dy_dz
        norm_len = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-7
        norm_x = nx / norm_len
        norm_y = ny / norm_len
        norm_z = nz / norm_len
    else:
        # Surface Z = f(X, Y): normal = (-dZ/dX, -dZ/dY, 1.0)
        nx = -dy_dx
        ny = -dy_dz
        nz = np.ones_like(dy_dx)
        norm_len = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-7
        norm_x = nx / norm_len
        norm_y = ny / norm_len
        norm_z = nz / norm_len

    # Write MTL
    with mtl_path.open("w", encoding="utf-8") as f:
        f.write("newmtl terrain\n")
        f.write("Ka 1.0 1.0 1.0\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write("Ks 0.05 0.05 0.05\n")
        f.write(f"map_Kd {texture_path.name}\n")
        if export_building_boxes and placements:
            f.write("\nnewmtl building_bounds\n")
            f.write("Ka 1.0 0.2 0.2\n")
            f.write("Kd 1.0 0.2 0.2\n")
            f.write("d 0.7\n")

    # Write OBJ
    with obj_path.open("w", encoding="utf-8") as f:
        f.write(f"mtllib {mtl_name}\n")
        f.write("o Terrain\n")
        f.write("usemtl terrain\n")

        # Vertices
        if y_up:
            # Universal standard: X East, Y Elevation (Up), Z South (Depth)
            for y in range(h):
                for x in range(w):
                    wx = (x / cols) * w_span
                    wy = float(h_world[y, x])
                    wz = (y / rows) * h_span_yards
                    f.write(f"v {wx:.3f} {wy:.3f} {wz:.3f}\n")
        else:
            # Spec 263 standard: X East, Y South, Z Elevation (Up)
            for y in range(h):
                for x in range(w):
                    wx = (x / cols) * w_span
                    wy = (y / rows) * h_span_yards
                    wz = float(h_world[y, x])
                    f.write(f"v {wx:.3f} {wy:.3f} {wz:.3f}\n")

        # Texture coordinates (glTF/OBJ UV: U in [0, 1], V in [0, 1])
        for y in range(h):
            for x in range(w):
                u = x / cols
                v = 1.0 - (y / rows)
                f.write(f"vt {u:.4f} {v:.4f}\n")

        # Vertex normals
        for y in range(h):
            for x in range(w):
                f.write(f"vn {norm_x[y, x]:.4f} {norm_y[y, x]:.4f} {norm_z[y, x]:.4f}\n")

        # Faces (counter-clockwise upward facing triangles)
        for y in range(rows):
            for x in range(cols):
                v1 = y * w + x + 1
                v2 = y * w + (x + 1) + 1
                v3 = (y + 1) * w + (x + 1) + 1
                v4 = (y + 1) * w + x + 1
                if y_up:
                    f.write(f"f {v1}/{v1}/{v1} {v4}/{v4}/{v4} {v3}/{v3}/{v3}\n")
                    f.write(f"f {v1}/{v1}/{v1} {v3}/{v3}/{v3} {v2}/{v2}/{v2}\n")
                else:
                    # Looking from +Z down: (v1, v2, v3) and (v1, v3, v4) have positive +Z cross product
                    f.write(f"f {v1}/{v1}/{v1} {v2}/{v2}/{v2} {v3}/{v3}/{v3}\n")
                    f.write(f"f {v1}/{v1}/{v1} {v3}/{v3}/{v3} {v4}/{v4}/{v4}\n")

        # Placed building 3D bounding boxes (opt-in only)
        if export_building_boxes and placements:
            v_offset = h * w
            f.write("\no Buildings\n")
            f.write("usemtl building_bounds\n")
            for idx, p in enumerate(placements):
                px_min, py_min, px_max, py_max = getattr(p, "pixel_box", (0, 0, 0, 0))
                if px_max <= px_min or py_max <= py_min:
                    continue
                bx0 = (px_min / 256.0) * w_span
                bx1 = (px_max / 256.0) * w_span
                by0 = (py_min / 256.0) * h_span_yards
                by1 = (py_max / 256.0) * h_span_yards

                b_name = getattr(p, "name", f"Building_{idx}")
                b_z = getattr(p, "pos", (0, 0, 0))[2]
                bz0 = float(b_z) if b_z != 0 else float(np.mean(h_world))
                bz1 = bz0 + 15.0  # Height of structure box

                # 8 vertices of box
                corners = [
                    (bx0, by0, bz0), (bx1, by0, bz0), (bx1, by1, bz0), (bx0, by1, bz0),
                    (bx0, by0, bz1), (bx1, by0, bz1), (bx1, by1, bz1), (bx0, by1, bz1),
                ]
                for cx, cy, cz in corners:
                    if y_up:
                        f.write(f"v {cx:.3f} {cz:.3f} {cy:.3f}\n")
                    else:
                        f.write(f"v {cx:.3f} {cy:.3f} {cz:.3f}\n")

                # 6 quad faces (12 triangles)
                b_base = v_offset + 1
                box_faces = [
                    (1, 2, 3, 4), (5, 8, 7, 6), (1, 5, 6, 2),
                    (2, 6, 7, 3), (3, 7, 8, 4), (4, 8, 5, 1)
                ]
                for f1, f2, f3, f4 in box_faces:
                    i1, i2, i3, i4 = b_base + f1 - 1, b_base + f2 - 1, b_base + f3 - 1, b_base + f4 - 1
                    f.write(f"f {i1} {i2} {i3}\n")
                    f.write(f"f {i1} {i3} {i4}\n")
                v_offset += 8

    logger.info("Exported OBJ mesh to %s (vertices: %d, faces: %d)", obj_path, h * w, rows * cols * 2)


def export_glb_mesh(
    height: np.ndarray,
    texture: Union[Path, Image.Image, np.ndarray],
    glb_path: Path,
    tile_size_yards: float = 533.333,
    width_yards: Optional[float] = None,
    height_yards: Optional[float] = None,
    height_scale: float = 40.0,
    is_world_yards: bool = False,
) -> None:
    """Export heightmap grid (H, W) to standalone binary glTF 2.0 (.glb) with embedded texture.

    Self-contained single file viewable directly in Windows 3D Viewer, Blender, and web viewers.

    Parameters:
        height: 2D array of elevation values (e.g. 256x256 or 257x257).
        texture: PIL Image, NumPy array (H, W, 3), or Path to image.
        glb_path: Output path for the .glb file.
        tile_size_yards: Lateral footprint size in yards (fallback if width/height not set).
        width_yards: Footprint width along X in yards.
        height_yards: Footprint height along Z in yards.
        height_scale: Elevation amplitude scaling factor when is_world_yards is False.
        is_world_yards: If True, uses height array directly as world elevation in yards.
    """
    h, w = height.shape
    rows = h - 1
    cols = w - 1

    w_span = width_yards if width_yards is not None else tile_size_yards
    h_span_yards = height_yards if height_yards is not None else tile_size_yards

    if is_world_yards:
        h_world = height.astype(np.float32)
    else:
        h_min = float(np.min(height))
        h_max = float(np.max(height))
        h_span = max(1e-5, h_max - h_min)
        h_world = ((height - h_min) / h_span) * height_scale

    # Coordinates: glTF convention is X Right, Y Up (elevation), Z Forward (South)
    xs = np.linspace(0, w_span, w, dtype=np.float32)
    zs = np.linspace(0, h_span_yards, h, dtype=np.float32)
    xx, zz = np.meshgrid(xs, zs)
    yy = h_world.astype(np.float32)

    positions = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)

    # Compute vertex normals from height gradient
    spacing_x = float(w_span / cols)
    spacing_z = float(h_span_yards / rows)
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

    # Counter-clockwise winding looking from +Y down: (v0, v2, v1) and (v0, v3, v2)
    t1 = np.stack([v0, v2, v1], axis=-1).ravel()
    t2 = np.stack([v0, v3, v2], axis=-1).ravel()
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
