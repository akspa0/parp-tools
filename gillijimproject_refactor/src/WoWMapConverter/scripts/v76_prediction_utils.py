from __future__ import annotations

import json
import os
import re
import shutil
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

DEFAULT_MODEL_PATH = Path("output_v7_6/checkpoints/latest.pth")
DEFAULT_INPUT_SIZE = 512
DEFAULT_TILE_SIZE = 533.3333
DEFAULT_MAX_HEIGHT = 1200.0
HEIGHT_SIGMA = 1.0
HEIGHT_KERNEL = 5

_SAMPLE_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sanitize_sample_id(value: str) -> str:
    sanitized = _SAMPLE_ID_RE.sub("_", value.strip()).strip("._")
    return sanitized or "sample"


def ensure_unique_sample_id(base_value: str, used_ids: set[str]) -> str:
    candidate = sanitize_sample_id(base_value)
    if candidate not in used_ids:
        used_ids.add(candidate)
        return candidate

    suffix = 2
    while True:
        alternate = f"{candidate}_{suffix}"
        if alternate not in used_ids:
            used_ids.add(alternate)
            return alternate
        suffix += 1


def autocast_context(device: str):
    if str(device).startswith("cuda"):
        return torch.amp.autocast("cuda")
    return nullcontext()


def load_v76_model(model_path: Path, device: str):
    from train_v7_6 import MultiHeadUNet

    model = MultiHeadUNet().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_source_image(source_path: Path, input_size: int = DEFAULT_INPUT_SIZE) -> Tuple[Image.Image, torch.Tensor, Dict[str, Any]]:
    source_image = Image.open(source_path).convert("RGB")
    original_width, original_height = source_image.size
    model_image = source_image
    resized = False
    if source_image.size != (input_size, input_size):
        model_image = source_image.resize((input_size, input_size), Image.Resampling.LANCZOS)
        resized = True

    tensor = TF.to_tensor(model_image).unsqueeze(0)
    source_meta = {
        "original_width": original_width,
        "original_height": original_height,
        "model_input_width": input_size,
        "model_input_height": input_size,
        "resized_for_model": resized,
    }
    return source_image, tensor, source_meta


def predict_height_and_albedo(model, input_tensor: torch.Tensor, device: str) -> Tuple[Image.Image, Image.Image, np.ndarray]:
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        with autocast_context(device):
            pred_h, pred_a = model(input_tensor)

    pred_h = TF.gaussian_blur(pred_h, kernel_size=[HEIGHT_KERNEL, HEIGHT_KERNEL], sigma=[HEIGHT_SIGMA, HEIGHT_SIGMA])
    height_np = pred_h.squeeze().float().cpu().numpy()
    height_u16 = np.clip(height_np * 65535.0, 0.0, 65535.0).astype(np.uint16)
    height_image = Image.fromarray(height_u16, mode="I;16")
    albedo_image = TF.to_pil_image(pred_a.squeeze().cpu())
    return height_image, albedo_image, height_u16


def prepare_prediction_layout(output_root: Path) -> Dict[str, Path]:
    layout = {
        "root": output_root,
        "predictions": output_root / "predictions",
        "sources": output_root / "sources",
        "heights": output_root / "heights",
        "albedo": output_root / "albedo",
        "meshes": output_root / "meshes",
        "stitched": output_root / "stitched",
    }
    for directory in layout.values():
        directory.mkdir(parents=True, exist_ok=True)
    return layout


def build_prediction_record(
    *,
    sample_id: str,
    source_kind: str,
    source_input_rel: str,
    source_meta: Dict[str, Any],
    model_path: Path,
    height_rel: str,
    albedo_rel: str,
    obj_rel: Optional[str],
    mtl_rel: Optional[str],
    source_dataset_root: Optional[str] = None,
    source_tile_json_path: Optional[str] = None,
    source_tile_name: Optional[str] = None,
    source_map_name: Optional[str] = None,
    source_client_label: Optional[str] = None,
) -> Dict[str, Any]:
    source_block: Dict[str, Any] = {
        "source_kind": source_kind,
        "input_image_path": source_input_rel,
        **source_meta,
    }
    if source_dataset_root:
        source_block["source_dataset_root"] = source_dataset_root
    if source_tile_json_path:
        source_block["source_tile_json_path"] = source_tile_json_path
    if source_tile_name:
        source_block["source_tile_name"] = source_tile_name
    if source_map_name:
        source_block["source_map_name"] = source_map_name
    if source_client_label:
        source_block["source_client_label"] = source_client_label

    prediction_block: Dict[str, Any] = {
        "height_prediction_path": height_rel,
        "albedo_prediction_path": albedo_rel,
    }
    if obj_rel:
        prediction_block["obj_path"] = obj_rel
    if mtl_rel:
        prediction_block["mtl_path"] = mtl_rel

    return {
        "schema_version": "wowterrain-v76-prediction-tile.v1",
        "sample_id": sample_id,
        "source": source_block,
        "model": {
            "model_family": "v7.6",
            "checkpoint_path": str(model_path),
            "input_channels": 3,
            "output_height_channels": 1,
            "output_albedo_channels": 3,
        },
        "predictions": prediction_block,
        "geometry_assumptions": {
            "height_encoding": "uint16-normalized",
            "max_height_assumed": DEFAULT_MAX_HEIGHT,
            "tile_size_assumed": DEFAULT_TILE_SIZE,
        },
    }


def _relative_path(root: Path, path: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def _copy_source_image(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() == destination_path.resolve():
        return
    shutil.copy2(source_path, destination_path)


def generate_obj(
    *,
    height_map: np.ndarray,
    albedo_path: Path,
    output_path: Path,
    mat_path: Path,
    tile_size: float = DEFAULT_TILE_SIZE,
    max_height: float = DEFAULT_MAX_HEIGHT,
) -> None:
    h, w = height_map.shape
    scale_x = tile_size / float(w)
    scale_z = tile_size / float(h)
    scale_y = max_height / 65535.0
    texture_reference = Path(os.path.relpath(albedo_path, start=mat_path.parent)).as_posix()

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(f"mtllib {mat_path.name}\n")
        for y in range(h):
            for x in range(w):
                px = x * scale_x
                py = float(height_map[y, x]) * scale_y
                pz = -y * scale_z
                handle.write(f"v {px:.4f} {pz:.4f} {py:.4f}\n")
                handle.write(f"vt {x / (w - 1):.4f} {(1 - y / (h - 1)):.4f}\n")

        handle.write("usemtl TerrainMat\n")
        for y in range(h - 1):
            for x in range(w - 1):
                i0 = y * w + x + 1
                i1 = i0 + 1
                i2 = (y + 1) * w + x + 2
                i3 = (y + 1) * w + x + 1
                handle.write(f"f {i0}/{i0} {i1}/{i1} {i2}/{i2} {i3}/{i3}\n")

    with open(mat_path, "w", encoding="utf-8") as handle:
        handle.write("newmtl TerrainMat\n")
        handle.write("Ka 1.0 1.0 1.0\n")
        handle.write("Kd 1.0 1.0 1.0\n")
        handle.write("Ks 0.0 0.0 0.0\n")
        handle.write(f"map_Kd {texture_reference}\n")


def write_prediction_sample(
    *,
    layout: Dict[str, Path],
    sample_id: str,
    source_path: Path,
    source_kind: str,
    source_meta: Dict[str, Any],
    model_path: Path,
    height_image: Image.Image,
    albedo_image: Image.Image,
    height_map: np.ndarray,
    write_mesh: bool,
    source_dataset_root: Optional[str] = None,
    source_tile_json_path: Optional[str] = None,
    source_tile_name: Optional[str] = None,
    source_map_name: Optional[str] = None,
    source_client_label: Optional[str] = None,
) -> Dict[str, Any]:
    source_dest = layout["sources"] / f"{sample_id}_input{source_path.suffix.lower()}"
    height_dest = layout["heights"] / f"{sample_id}_height_pred.png"
    albedo_dest = layout["albedo"] / f"{sample_id}_albedo_pred.png"
    prediction_json = layout["predictions"] / f"{sample_id}.json"

    _copy_source_image(source_path, source_dest)
    height_image.save(height_dest)
    albedo_image.save(albedo_dest)

    obj_rel: Optional[str] = None
    mtl_rel: Optional[str] = None
    if write_mesh:
        obj_dest = layout["meshes"] / f"{sample_id}.obj"
        mtl_dest = layout["meshes"] / f"{sample_id}.mtl"
        generate_obj(height_map=height_map, albedo_path=albedo_dest, output_path=obj_dest, mat_path=mtl_dest)
        obj_rel = _relative_path(layout["root"], obj_dest)
        mtl_rel = _relative_path(layout["root"], mtl_dest)

    record = build_prediction_record(
        sample_id=sample_id,
        source_kind=source_kind,
        source_input_rel=_relative_path(layout["root"], source_dest),
        source_meta=source_meta,
        model_path=model_path,
        height_rel=_relative_path(layout["root"], height_dest),
        albedo_rel=_relative_path(layout["root"], albedo_dest),
        obj_rel=obj_rel,
        mtl_rel=mtl_rel,
        source_dataset_root=source_dataset_root,
        source_tile_json_path=source_tile_json_path,
        source_tile_name=source_tile_name,
        source_map_name=source_map_name,
        source_client_label=source_client_label,
    )

    with open(prediction_json, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    return record


def finalize_prediction_dataset(
    *,
    layout: Dict[str, Path],
    model_path: Path,
    records: List[Dict[str, Any]],
    stitched_outputs: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    stitched_outputs = stitched_outputs or []
    source_kind_counter = Counter(record["source"]["source_kind"] for record in records)
    mesh_count = sum(1 for record in records if "obj_path" in record["predictions"])
    summary_rows = []

    for record in records:
        predictions = record["predictions"]
        summary_rows.append(
            {
                "sample_id": record["sample_id"],
                "prediction_json_path": f"predictions/{record['sample_id']}.json",
                "source_input_path": record["source"]["input_image_path"],
                "height_prediction_path": predictions["height_prediction_path"],
                "albedo_prediction_path": predictions["albedo_prediction_path"],
                "obj_path": predictions.get("obj_path"),
                "mtl_path": predictions.get("mtl_path"),
            }
        )

    manifest = {
        "schema_version": "wowterrain-v76-prediction-manifest.v1",
        "generated_at_utc": utc_now_iso(),
        "prediction_root": str(layout["root"].resolve()),
        "model_family": "v7.6",
        "checkpoint_path": str(model_path),
        "source_kind": summary_rows and (summary_rows[0] and next(iter(source_kind_counter)) if len(source_kind_counter) == 1 else "mixed") or "unknown",
        "sample_count": len(records),
        "samples": summary_rows,
    }
    if stitched_outputs:
        manifest["stitched_outputs"] = stitched_outputs

    manifest_path = layout["root"] / "v76_prediction_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    metadata_path = layout["root"] / "metadata.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        for record in records:
            line = {
                "sample_id": record["sample_id"],
                "source_kind": record["source"]["source_kind"],
                "input_image_path": record["source"]["input_image_path"],
                "height_prediction_path": record["predictions"]["height_prediction_path"],
                "albedo_prediction_path": record["predictions"]["albedo_prediction_path"],
                "obj_path": record["predictions"].get("obj_path"),
                "checkpoint_path": str(model_path),
            }
            handle.write(json.dumps(line) + "\n")

    dataset_info = {
        "dataset_type": "v76-predicted-output",
        "model_family": "v7.6",
        "sample_count": len(records),
        "has_meshes": mesh_count > 0,
        "has_stitched_outputs": bool(stitched_outputs),
        "source_kind_breakdown": dict(source_kind_counter),
        "checkpoint_path": str(model_path),
    }
    dataset_info_path = layout["root"] / "dataset_info.json"
    with open(dataset_info_path, "w", encoding="utf-8") as handle:
        json.dump(dataset_info, handle, indent=2)

    return manifest_path