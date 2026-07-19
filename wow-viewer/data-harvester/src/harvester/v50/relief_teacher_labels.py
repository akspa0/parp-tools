"""Pinned broad-image relief teacher labels for Spec 114.

The teacher is an offline corpus builder, never a deployment dependency. Exact v50 terrain heights
remain the authority for top-down rows; this module creates explicitly labeled pseudo-relief for
licensed or BYOD images that do not carry exact geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from PIL import Image, UnidentifiedImageError

from harvester.v50.universal_relief_contract import raster_to_rgb

TEACHER_SCHEMA = "v50-relief-teacher-labels-v1"
DEFAULT_TEACHER_ID = "Intel/dpt-hybrid-midas"
DEFAULT_TEACHER_REVISION = "17fb43d4437eb62c260a593400db13c22b04511a"
DEFAULT_TEACHER_WEIGHTS_SHA256 = "9599793d3ce64d7ebc85657360831596c1df9abc61f6820fe623fe7efb2e29c5"
DEFAULT_TEACHER_LICENSE = "apache-2.0"
DEFAULT_TEACHER_OUTPUT_SEMANTICS = "larger_is_closer_and_higher"
DEFAULT_TEACHER_WEIGHT_FILE = "model.safetensors"
IMAGE_SUFFIXES = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


@dataclass(frozen=True)
class TeacherIdentity:
    hub_id: str
    revision: str
    weight_file: str
    weights_sha256: str
    license: str
    output_semantics: str


@dataclass(frozen=True)
class TeacherSource:
    source_id: str
    relative_path: str
    source_sha256: str
    width: int
    height: int
    mode: str


@dataclass(frozen=True)
class TeacherLabelPlan:
    schema: str
    input_root: str
    output_store: str
    visual_family: str
    data_authority: str
    teacher: TeacherIdentity
    sources: tuple[TeacherSource, ...]


def default_teacher_identity() -> TeacherIdentity:
    return TeacherIdentity(
        hub_id=DEFAULT_TEACHER_ID,
        revision=DEFAULT_TEACHER_REVISION,
        weight_file=DEFAULT_TEACHER_WEIGHT_FILE,
        weights_sha256=DEFAULT_TEACHER_WEIGHTS_SHA256,
        license=DEFAULT_TEACHER_LICENSE,
        output_semantics=DEFAULT_TEACHER_OUTPUT_SEMANTICS,
    )


def validate_teacher_identity(identity: TeacherIdentity) -> None:
    lowered = f"{identity.hub_id} {identity.revision}".lower().replace("_", "-")
    if "depth-anything" in lowered or "depthanything" in lowered:
        raise ValueError("DepthAnything-family teachers are forbidden for this lane")
    if not identity.hub_id or len(identity.revision) != 40:
        raise ValueError("teacher hub_id and full 40-character revision are required")
    if identity.weight_file != "model.safetensors":
        raise ValueError("teacher must use the pinned safetensors weight file")
    if len(identity.weights_sha256) != 64:
        raise ValueError("teacher weights_sha256 must contain 64 hexadecimal characters")
    try:
        int(identity.weights_sha256, 16)
    except ValueError as exc:
        raise ValueError("teacher weights_sha256 must be hexadecimal") from exc
    if not identity.license:
        raise ValueError("teacher license must be recorded")
    if identity.output_semantics != DEFAULT_TEACHER_OUTPUT_SEMANTICS:
        raise ValueError(
            f"teacher output semantics must be {DEFAULT_TEACHER_OUTPUT_SEMANTICS!r}"
        )


def normalize_teacher_relief(
    predicted_depth: np.ndarray,
    *,
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
) -> np.ndarray:
    """Robustly normalize larger-is-closer teacher output to float32 ``[0,1]`` relief."""
    predicted = np.asarray(predicted_depth, dtype=np.float32)
    if predicted.ndim != 2 or predicted.size == 0:
        raise ValueError("teacher prediction must be a non-empty 2D array")
    if not np.isfinite(predicted).all():
        raise ValueError("teacher prediction must contain only finite values")
    if not 0.0 <= low_percentile < high_percentile <= 100.0:
        raise ValueError("teacher normalization percentiles are invalid")
    low, high = np.percentile(predicted, (low_percentile, high_percentile))
    if float(high - low) <= 1e-12:
        return np.zeros_like(predicted, dtype=np.float32)
    return np.clip((predicted - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _source_id(relative_path: str, source_sha256: str) -> str:
    payload = f"{relative_path}\0{source_sha256}".encode()
    return hashlib.sha256(payload).hexdigest()


def discover_teacher_sources(input_root: str | Path) -> tuple[TeacherSource, ...]:
    root = Path(input_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"teacher input directory does not exist: {root}")
    candidates = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    sources = []
    for path in candidates:
        try:
            with Image.open(path) as image:
                image.load()
                width, height, mode = image.width, image.height, image.mode
        except (UnidentifiedImageError, OSError):
            continue
        source_sha256 = _sha256_file(path)
        relative_path = path.relative_to(root).as_posix()
        sources.append(
            TeacherSource(
                source_id=_source_id(relative_path, source_sha256),
                relative_path=relative_path,
                source_sha256=source_sha256,
                width=width,
                height=height,
                mode=mode,
            )
        )
    if not sources:
        raise ValueError(f"no decodable raster images found under {root}")
    return tuple(sources)


def build_teacher_label_plan(
    *,
    input_root: str | Path,
    output_store: str | Path,
    visual_family: str,
    license_id: str | None = None,
    byod: bool = False,
    teacher: TeacherIdentity | None = None,
) -> TeacherLabelPlan:
    if bool(license_id) == bool(byod):
        raise ValueError("provide exactly one of license_id or byod=True")
    if not visual_family.strip():
        raise ValueError("visual_family must be non-empty")
    identity = teacher or default_teacher_identity()
    validate_teacher_identity(identity)
    output = Path(output_store).resolve()
    if output.suffix.lower() != ".zarr":
        raise ValueError("teacher output store must end with .zarr")
    if output.exists():
        raise FileExistsError(f"teacher output already exists: {output}")
    root = Path(input_root).resolve()
    return TeacherLabelPlan(
        schema=TEACHER_SCHEMA,
        input_root=str(root),
        output_store=str(output),
        visual_family=visual_family.strip(),
        data_authority=f"license:{license_id}" if license_id else "private-byod",
        teacher=identity,
        sources=discover_teacher_sources(root),
    )


def plan_to_dict(plan: TeacherLabelPlan) -> dict[str, Any]:
    result = asdict(plan)
    result["source_count"] = len(plan.sources)
    return result


def _row_identity(source: TeacherSource, teacher: TeacherIdentity) -> str:
    payload = {
        "source_sha256": source.source_sha256,
        "teacher_revision": teacher.revision,
        "teacher_weights_sha256": teacher.weights_sha256,
        "output_semantics": teacher.output_semantics,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def write_teacher_label_store(
    plan: TeacherLabelPlan,
    predictor: Callable[[Image.Image], np.ndarray],
) -> dict[str, Any]:
    """Write variable-shape pseudo-relief rows to one Zarr store using a supplied predictor."""
    output = Path(plan.output_store)
    if output.exists():
        raise FileExistsError(f"teacher output already exists: {output}")
    root = Path(plan.input_root)
    store = zarr.open_group(str(output), mode="w")
    rows = store.create_group("rows")
    created_utc = datetime.now(UTC).isoformat()
    row_records = []
    for source in plan.sources:
        with Image.open(root / source.relative_path) as image:
            image.load()
            rgb = Image.fromarray(raster_to_rgb(image), mode="RGB")
            raw_prediction = predictor(rgb)
        relief = normalize_teacher_relief(raw_prediction)
        if relief.shape != (source.height, source.width):
            raise ValueError(
                f"teacher prediction for {source.relative_path} must preserve source shape "
                f"{(source.height, source.width)}, got {relief.shape}"
            )
        row_id = _row_identity(source, plan.teacher)
        row = rows.create_group(row_id)
        row.create_array(
            "relative_relief",
            data=relief,
            chunks=(min(256, source.height), min(256, source.width)),
        )
        record = {
            **asdict(source),
            "row_id": row_id,
            "target_authority": "teacher_pseudo",
            "target_signal": "relative_relief",
            "relief_min": float(relief.min()),
            "relief_max": float(relief.max()),
        }
        row.attrs.update(record)
        row_records.append(record)

    summary = {
        "schema": plan.schema,
        "created_utc": created_utc,
        "input_root": plan.input_root,
        "output_store": plan.output_store,
        "visual_family": plan.visual_family,
        "data_authority": plan.data_authority,
        "target_authority": "teacher_pseudo",
        "target_signal": "relative_relief",
        "teacher": asdict(plan.teacher),
        "source_count": len(plan.sources),
        "rows": row_records,
    }
    store.attrs.update({key: value for key, value in summary.items() if key != "rows"})
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _download_and_load_teacher(identity: TeacherIdentity, device: str) -> Callable[[Image.Image], np.ndarray]:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    snapshot = Path(
        snapshot_download(
            repo_id=identity.hub_id,
            revision=identity.revision,
            allow_patterns=["*.json", "*.txt", identity.weight_file],
        )
    )
    weight_path = snapshot / identity.weight_file
    observed_sha256 = _sha256_file(weight_path)
    if observed_sha256 != identity.weights_sha256:
        raise ValueError(
            f"teacher weight hash mismatch: expected {identity.weights_sha256}, "
            f"observed {observed_sha256}"
        )
    processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForDepthEstimation.from_pretrained(
        snapshot,
        local_files_only=True,
        use_safetensors=True,
    ).to(device)
    model.eval()

    def predict(image: Image.Image) -> np.ndarray:
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.inference_mode():
            output = model(pixel_values=pixel_values).predicted_depth
            resized = torch.nn.functional.interpolate(
                output.unsqueeze(1),
                size=(image.height, image.width),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0).squeeze(0)
        return resized.float().cpu().numpy()

    return predict


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build pinned broad-image pseudo-relief labels; dry-run unless --confirm-run."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--family", required=True)
    authority = parser.add_mutually_exclusive_group(required=True)
    authority.add_argument("--license-id")
    authority.add_argument("--byod", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confirm-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    plan = build_teacher_label_plan(
        input_root=args.input_dir,
        output_store=args.output,
        visual_family=args.family,
        license_id=args.license_id,
        byod=args.byod,
    )
    print(json.dumps(plan_to_dict(plan), indent=2))
    if not args.confirm_run:
        print("DRY RUN: add --confirm-run to download the pinned teacher and write pseudo-relief.")
        return 0
    predictor = _download_and_load_teacher(plan.teacher, args.device)
    summary = write_teacher_label_store(plan, predictor)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    return 0
