from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from transformers import DepthAnythingConfig
import zarr
import zarr.codecs
import zarr.storage


CODEC = zarr.codecs.BloscCodec(cname="zstd", clevel=1, shuffle="bitshuffle")


def tiny_depthanything_config() -> DepthAnythingConfig:
    return DepthAnythingConfig(
        backbone_config={
            "model_type": "dinov2",
            "image_size": 56,
            "patch_size": 14,
            "num_channels": 3,
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "mlp_ratio": 4,
            "qkv_bias": True,
            "apply_layernorm": True,
            "reshape_hidden_states": False,
            "use_mask_token": True,
            "out_features": ["stage1", "stage2", "stage3", "stage4"],
            "out_indices": [1, 2, 3, 4],
            "stage_names": ["stem", "stage1", "stage2", "stage3", "stage4"],
        },
        reassemble_hidden_size=64,
        neck_hidden_sizes=[16, 32, 48, 64],
        fusion_hidden_size=32,
        head_hidden_size=16,
        patch_size=14,
        reassemble_factors=[4, 2, 1, 0.5],
    )


def write_model_config_json(path: Path) -> Path:
    path.write_text(json.dumps(tiny_depthanything_config().to_dict()), encoding="utf-8")
    return path


def make_synthetic_v22_store(path: Path, *, build: str = "3_3_5_12340", tile_count: int = 6) -> Path:
    if path.exists():
        import shutil

        shutil.rmtree(path)

    store = zarr.storage.LocalStore(str(path), read_only=False)
    root = zarr.group(store=store)

    rng = np.random.default_rng(23)
    minimap = np.zeros((tile_count, 256, 256, 3), dtype=np.uint8)
    alpha = np.zeros((tile_count, 256, 256, 4), dtype=np.float32)
    normals = np.zeros((tile_count, 257, 257, 3), dtype=np.float32)
    normals[..., 2] = 1.0
    mcnr_mask = np.ones((tile_count, 257, 257), dtype=bool)
    liquid_mask = np.zeros((tile_count, 256, 256), dtype=np.float32)
    liquid_height = np.zeros((tile_count, 256, 256), dtype=np.float32)
    object_precise = np.zeros((tile_count, 257, 257), dtype=np.float32)
    object_filtered = np.zeros((tile_count, 257, 257), dtype=np.float32)
    heights = np.zeros((tile_count, 257, 257), dtype=np.float32)
    mcly_tileset_ids = np.zeros((tile_count, 16, 16, 4), dtype=np.int32)

    tile_index: list[dict[str, object]] = []
    for idx in range(tile_count):
        yy, xx = np.meshgrid(np.linspace(0.0, 1.0, 257), np.linspace(0.0, 1.0, 257), indexing="ij")
        heights[idx] = (xx * 10.0) + (yy * 5.0) + float(idx)
        minimap[idx, ..., 0] = np.clip((xx[:256, :256] * 255.0) + idx * 5.0, 0, 255).astype(np.uint8)
        minimap[idx, ..., 1] = np.clip((yy[:256, :256] * 255.0), 0, 255).astype(np.uint8)
        minimap[idx, ..., 2] = np.uint8(64 + idx * 3)
        alpha[idx, ..., 0] = 1.0
        alpha[idx, 64:128, 64:128, 1] = 0.5
        mcly_tileset_ids[idx, ..., 0] = 7 + (idx % 2)
        mcly_tileset_ids[idx, ..., 1] = 8 + (idx % 3)
        if idx % 2 == 0:
            liquid_mask[idx, :8, :8] = 1.0
            liquid_height[idx, :8, :8] = 25.0 + idx
        if idx % 3 == 0:
            object_precise[idx, 96:128, 96:128] = 1.0
            object_filtered[idx, 96:128, 96:128] = 1.0
        heights[idx] += rng.normal(0.0, 0.05, size=(257, 257)).astype(np.float32)
        tile_index.append(
            {
                "tile_id": idx,
                "build": build,
                "map": "SyntheticMap",
                "tile_x": idx,
                "tile_y": idx + 1,
                "mtex_texture_paths": [],
                "placement_mddf_asset_paths": [],
                "placement_modf_asset_paths": [],
            }
        )

    root.create_array("minimap_rgb", data=minimap, chunks=(1, 256, 256, 3), compressors=CODEC)
    root.create_array("alpha_256", data=alpha, chunks=(1, 256, 256, 4), compressors=CODEC)
    root.create_array("normal_xyz", data=normals, chunks=(1, 257, 257, 3), compressors=CODEC)
    root.create_array("mcnr_mask_257", data=mcnr_mask, chunks=(1, 257, 257), compressors=CODEC)
    root.create_array("liquid_mask", data=liquid_mask, chunks=(1, 256, 256), compressors=CODEC)
    root.create_array("liquid_height", data=liquid_height, chunks=(1, 256, 256), compressors=CODEC)
    root.create_array("height_257", data=heights, chunks=(1, 257, 257), compressors=CODEC)
    root.create_array("object_precise_mask", data=object_precise, chunks=(1, 257, 257), compressors=CODEC)
    root.create_array("object_filtered_mask", data=object_filtered, chunks=(1, 257, 257), compressors=CODEC)
    root.create_array("mcly_tileset_ids", data=mcly_tileset_ids, chunks=(1, 16, 16, 4), compressors=CODEC)
    root.attrs["tile_index"] = tile_index
    return path
