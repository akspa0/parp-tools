"""Footprint-guided known-object identification and marker-map contracts.

The marker lane is intentionally separate from the object sieve.  It consumes an image and one
candidate footprint, predicts knownness, and produces an embedding for retrieval against the
read-only v50 object library.  Exact library IDs stay in metadata; they are not encoded as pixels
or used as an input channel.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional

SCHEMA = "v60-object-marker-v1"
VALIDATION_SCHEMA = "v60-object-marker-validation-v1"
IMAGE_SIGNAL = "minimap_rgb_256"
FOOTPRINT_SIGNAL = "object_candidate_mask_256"
KNOWN_SIGNAL = "known_object"
MARKER_SIGNAL = "known_object_marker_256"
PIXELS = 256
EMBEDDING_DIM = 64


class ObjectMarkerError(ValueError):
    """Raised when marker inputs or outputs violate the v60 contract."""


def _hash_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & 0xFFFFFFFF


def _asset_family(path: str) -> str:
    parts = [part for part in path.replace("\\", "/").lower().split("/") if part]
    return "/".join(parts[:2]) if len(parts) >= 2 else (parts[0] if parts else "unknown")


def _translate_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate a mask without wraparound so a negative candidate is deterministic."""
    source = np.asarray(mask, dtype=bool)
    output = np.zeros_like(source)
    height, width = source.shape
    source_x0 = max(0, -dx)
    source_x1 = min(width, width - dx) if dx >= 0 else width
    source_y0 = max(0, -dy)
    source_y1 = min(height, height - dy) if dy >= 0 else height
    target_x0 = max(0, dx)
    target_x1 = target_x0 + max(0, source_x1 - source_x0)
    target_y0 = max(0, dy)
    target_y1 = target_y0 + max(0, source_y1 - source_y0)
    if source_x1 > source_x0 and source_y1 > source_y0:
        output[target_y0:target_y1, target_x0:target_x1] = source[source_y0:source_y1, source_x0:source_x1]
    return output


def _as_hwc_image(array: np.ndarray) -> np.ndarray:
    image = np.asarray(array, dtype=np.float32)
    if image.shape == (3, PIXELS, PIXELS):
        image = np.moveaxis(image, 0, -1)
    if image.shape != (PIXELS, PIXELS, 3):
        raise ObjectMarkerError(f"image must have shape (256, 256, 3), got {image.shape}")
    if not np.isfinite(image).all() or float(image.min()) < -1e-5 or float(image.max()) > 1.00001:
        raise ObjectMarkerError("image must be finite and in [0, 1]")
    return image


def _as_mask(array: np.ndarray) -> np.ndarray:
    mask = np.asarray(array, dtype=np.float32)
    if mask.shape != (PIXELS, PIXELS):
        raise ObjectMarkerError(f"candidate footprint must have shape (256, 256), got {mask.shape}")
    if not np.isfinite(mask).all() or float(mask.min()) < -1e-5 or float(mask.max()) > 1.00001:
        raise ObjectMarkerError("candidate footprint must be finite and in [0, 1]")
    return mask


def _hash_array(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(array, dtype="<f4"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ObjectMarkerNet(nn.Module):
    """Small image-plus-footprint marker specialist.

    ``known_logit`` answers whether the supplied candidate is a known library object.  The
    normalized ``embedding`` is used for gallery retrieval and is independently checkpointed with
    the knownness head.
    """

    def __init__(self, base: int = 16, embedding_dim: int = EMBEDDING_DIM) -> None:
        super().__init__()
        if base < 1 or embedding_dim < 2:
            raise ValueError("base and embedding_dim must be positive (embedding_dim >= 2)")
        self.base = int(base)
        self.embedding_dim = int(embedding_dim)
        self.encoder = nn.Sequential(
            _block(4, base),
            _block(base, base * 2),
            _block(base * 2, base * 4),
            _block(base * 4, base * 8),
            _block(base * 8, base * 8),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding_head = nn.Linear(base * 8, embedding_dim)
        self.known_head = nn.Linear(embedding_dim, 1)

    def embedding(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.pool(self.encoder(inputs)).flatten(1)
        return functional.normalize(self.embedding_head(features), dim=1)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.embedding(inputs)
        return {
            "known_logit": self.known_head(embedding).squeeze(1),
            "embedding": embedding,
        }


def marker_loss(
    outputs: dict[str, torch.Tensor],
    known_targets: torch.Tensor,
    identity_indices: torch.Tensor | None = None,
    *,
    metric_weight: float = 1.0,
    margin: float = 0.35,
) -> dict[str, torch.Tensor]:
    """Knownness BCE plus a pairwise metric loss over positive library identities."""
    known = known_targets.float().reshape(-1)
    known_loss = functional.binary_cross_entropy_with_logits(outputs["known_logit"], known)
    metric = outputs["embedding"].new_zeros(())
    if identity_indices is not None and len(identity_indices) > 1:
        ids = identity_indices.reshape(-1)
        embeddings = outputs["embedding"]
        similarities = embeddings @ embeddings.T
        upper = torch.triu(torch.ones_like(similarities, dtype=torch.bool), diagonal=1)
        valid = known > 0.5
        positive = upper & valid[:, None] & valid[None, :] & (ids[:, None] == ids[None, :])
        negative = upper & valid[:, None] & valid[None, :] & (ids[:, None] != ids[None, :])
        terms: list[torch.Tensor] = []
        if positive.any():
            terms.append(((1.0 - similarities[positive]) ** 2).mean())
        if negative.any():
            terms.append(functional.relu(similarities[negative] - margin).pow(2).mean())
        if terms:
            metric = torch.stack(terms).mean()
    total = known_loss + float(metric_weight) * metric
    return {"total_loss": total, "known_loss": known_loss, "metric_loss": metric}


def retrieve_library_identity(
    query_embedding: np.ndarray,
    gallery_embeddings: np.ndarray,
    gallery_ids: Iterable[str],
    *,
    known_confidence: float,
    known_threshold: float = 0.55,
    top_k: int = 5,
) -> dict[str, Any]:
    """Return ranked gallery matches; reject a candidate below the similarity threshold."""
    query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    gallery = np.asarray(gallery_embeddings, dtype=np.float32)
    ids = [str(value) for value in gallery_ids]
    if query.ndim != 1 or gallery.ndim != 2 or gallery.shape[1] != query.shape[0]:
        raise ObjectMarkerError("query/gallery embedding shapes are incompatible")
    if len(ids) != len(gallery) or not ids:
        raise ObjectMarkerError("gallery IDs must be non-empty and align with gallery embeddings")
    query_norm = query / max(float(np.linalg.norm(query)), 1e-8)
    gallery_norm = gallery / np.maximum(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-8)
    scores = gallery_norm @ query_norm
    order = np.argsort(-scores, kind="stable")[: max(1, int(top_k))]
    ranked = [{"library_id": ids[int(index)], "similarity": float(scores[int(index)])} for index in order]
    best = ranked[0]
    accepted = float(known_confidence) >= float(known_threshold) and float(best["similarity"]) >= float(known_threshold)
    return {
        "known": bool(accepted),
        "known_confidence": float(known_confidence),
        "best_library_id": best["library_id"],
        "best_similarity": best["similarity"],
        "top_matches": ranked,
        "rejection_reason": None if accepted else "knownness_or_similarity_below_threshold",
    }


def build_marker_map(
    image_shape: tuple[int, int], candidates: list[dict[str, Any]], decisions: list[dict[str, Any]]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Rasterize accepted candidate footprints and produce the identity sidecar rows."""
    if image_shape != (PIXELS, PIXELS) or len(candidates) != len(decisions):
        raise ObjectMarkerError("marker map inputs have incompatible shape/count")
    marker = np.zeros(image_shape, dtype=np.uint16)
    identity_rows: list[dict[str, Any]] = []
    for candidate, decision in zip(candidates, decisions, strict=True):
        mask = _as_mask(np.asarray(candidate["mask"], dtype=np.float32)) >= 0.5
        if not bool(decision.get("known", False)) or not mask.any():
            continue
        marker_id = len(identity_rows) + 1
        if marker_id > np.iinfo(np.uint16).max:
            raise ObjectMarkerError("too many accepted candidates for uint16 marker map")
        free = mask & (marker == 0)
        marker[free] = marker_id
        identity_rows.append({
            "marker_instance_id": marker_id,
            "candidate_id": str(candidate.get("candidate_id", marker_id)),
            "library_id": str(decision["best_library_id"]),
            "asset_path": str(candidate.get("asset_path", "")),
            "footprint_coverage": float(mask.mean()),
            "written_pixel_count": int(free.sum()),
            "overlap_pixel_count": int((mask & ~free).sum()),
            "known_confidence": float(decision["known_confidence"]),
            "retrieval_similarity": float(decision["best_similarity"]),
        })
    return marker, identity_rows


def _library_provenance(library: Path) -> dict[str, Any]:
    assets = library / "assets.parquet"
    index = library / "index.parquet"
    if not library.is_dir() or not assets.is_file() or not index.is_file():
        raise ObjectMarkerError(f"object library is incomplete: {library}")
    return {
        "source_store": str(library.resolve()),
        "assets_sha256": _hash_bytes(assets),
        "index_sha256": _hash_bytes(index),
    }


def _eligible_library_ids(library: Path) -> set[str]:
    import pyarrow.parquet as pq
    import zarr

    group = zarr.open_group(str(library), mode="r")
    if "capture_rgb" not in group or "capture_mask" not in group:
        raise ObjectMarkerError(f"object library is missing capture arrays: {library}")
    rows = pq.read_table(library / "assets.parquet").to_pylist()
    ids: set[str] = set()
    for index, row in enumerate(rows):
        if row.get("capture_status") != "captured":
            continue
        mask = np.asarray(group["capture_mask"][index])
        if (mask > 0).any() and row.get("library_id"):
            ids.add(str(row["library_id"]))
    if not ids:
        raise ObjectMarkerError(f"object library has no eligible captured IDs: {library}")
    return ids


def build_library_gallery_inputs(library: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build deterministic neutral-background gallery inputs from the read-only library."""
    import pyarrow.parquet as pq
    import zarr
    from PIL import Image

    group = zarr.open_group(str(library), mode="r")
    if "capture_rgb" not in group or "capture_mask" not in group:
        raise ObjectMarkerError(f"object library is missing capture arrays: {library}")
    rows = pq.read_table(library / "assets.parquet").to_pylist()
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    ids: list[str] = []
    for index, row in enumerate(rows):
        if row.get("capture_status") != "captured":
            continue
        rgb = np.asarray(group["capture_rgb"][index], dtype=np.uint8)
        mask = np.asarray(group["capture_mask"][index], dtype=np.uint8) > 0
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        crop_rgb = rgb[y0:y1, x0:x1]
        crop_mask = mask[y0:y1, x0:x1]
        extent = 96
        scale = extent / max(crop_rgb.shape[:2])
        size = (max(1, round(crop_rgb.shape[1] * scale)), max(1, round(crop_rgb.shape[0] * scale)))
        resized_rgb = np.asarray(Image.fromarray(crop_rgb).resize(size, Image.Resampling.BILINEAR), dtype=np.uint8)
        resized_mask = np.asarray(
            Image.fromarray(crop_mask.astype(np.uint8) * 255).resize(size, Image.Resampling.BOX),
            dtype=np.uint8,
        ) > 0
        canvas = np.full((PIXELS, PIXELS, 3), 0.5, dtype=np.float32)
        footprint = np.zeros((PIXELS, PIXELS), dtype=np.float32)
        top = (PIXELS - resized_rgb.shape[0]) // 2
        left = (PIXELS - resized_rgb.shape[1]) // 2
        rgb_float = resized_rgb.astype(np.float32) / 255.0
        luma = np.clip(
            (rgb_float * np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)).sum(axis=2), 0.0, 1.0
        )
        canvas[top : top + resized_rgb.shape[0], left : left + resized_rgb.shape[1], :] = luma[..., None]
        footprint[top : top + resized_mask.shape[0], left : left + resized_mask.shape[1]] = resized_mask
        images.append(canvas)
        masks.append(footprint)
        ids.append(str(row.get("library_id", "")))
    if not ids:
        raise ObjectMarkerError(f"object library has no non-blank gallery captures: {library}")
    return np.stack(images), np.stack(masks), ids


def build_object_marker_corpus(
    *, sieve_corpus: Path, object_library: Path, output: Path, seed: int = 6001
) -> dict[str, Any]:
    """Derive candidate rows from the corrected real-library sieve corpus."""
    if output.exists():
        raise ObjectMarkerError(f"refusing to overwrite existing output: {output}")
    partial_output = output.with_name(output.name + ".partial")
    if partial_output.exists():
        raise ObjectMarkerError(f"refusing to reuse incomplete output: {partial_output}")
    sieve_manifest_path = sieve_corpus / "object_library_sieve_manifest.json"
    if not sieve_manifest_path.is_file():
        raise ObjectMarkerError(f"missing sieve manifest: {sieve_manifest_path}")
    sieve_manifest = json.loads(sieve_manifest_path.read_text(encoding="utf-8"))
    if sieve_manifest.get("schema") != "v60-object-library-sieve-v1":
        raise ObjectMarkerError("marker builder requires v60-object-library-sieve-v1")
    provenance = _library_provenance(object_library)
    eligible_library_ids = _eligible_library_ids(object_library)
    partial_output.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    skipped_instances: list[dict[str, Any]] = []
    family_splits: dict[str, str] = {}
    for sieve_row in sieve_manifest.get("rows", []):
        npz_path = sieve_corpus / str(sieve_row["npz"])
        with np.load(npz_path, allow_pickle=False) as payload:
            image_gray = np.asarray(payload["objectified_terrain_shadow_256"], dtype=np.float32)
            instance_ids = np.asarray(payload["object_instance_id_256"], dtype=np.uint16)
        if image_gray.shape != (PIXELS, PIXELS) or instance_ids.shape != (PIXELS, PIXELS):
            raise ObjectMarkerError(f"sieve row has incompatible arrays: {npz_path}")
        image = np.repeat(image_gray[..., None], 3, axis=2)
        split = str(sieve_row.get("split", "train"))
        for instance in sieve_row.get("object_instances", []):
            instance_id = int(instance["instance_id"])
            footprint = (instance_ids == instance_id).astype(np.float32)
            if not footprint.any():
                skipped_instances.append({
                    "source_sieve_row_id": str(sieve_row["row_id"]),
                    "instance_id": instance_id,
                    "library_id": str(instance.get("library_id", "")),
                    "reason": "occluded_or_overwritten_in_instance_id_map",
                })
                continue
            library_id = str(instance["library_id"])
            if library_id not in eligible_library_ids:
                raise ObjectMarkerError(f"sieve instance references unavailable library ID: {library_id}")
            library_family = str(instance.get("library_family") or _asset_family(str(instance.get("asset_path", ""))))
            prior = family_splits.setdefault(library_family, split)
            if prior != split:
                raise ObjectMarkerError(f"library family crosses split: {library_family}")
            positive_id = f"{sieve_row['row_id']}-instance-{instance_id}"
            positive_npz = f"{positive_id}.npz"
            np.savez_compressed(
                partial_output / positive_npz,
                **{IMAGE_SIGNAL: image, FOOTPRINT_SIGNAL: footprint, KNOWN_SIGNAL: np.uint8(1)},
            )
            rows.append({
                "row_id": positive_id,
                "source_sieve_row_id": str(sieve_row["row_id"]),
                "candidate_kind": "known_library_object",
                "split": split,
                "library_id": library_id,
                "library_family": library_family,
                "asset_path": str(instance.get("asset_path", "")),
                "known_object": 1,
                "footprint_coverage": float(footprint.mean()),
                "placement_metadata": instance,
                "npz": positive_npz,
            })

            dx = 31 + int(_stable_seed(seed, positive_id) % 17)
            dy = -23 - int(_stable_seed(positive_id, seed) % 13)
            shifted = _translate_mask(footprint, dx, dy).astype(np.float32)
            negative_id = f"{positive_id}-shifted"
            negative_npz = f"{negative_id}.npz"
            np.savez_compressed(
                partial_output / negative_npz,
                **{IMAGE_SIGNAL: image, FOOTPRINT_SIGNAL: shifted, KNOWN_SIGNAL: np.uint8(0)},
            )
            rows.append({
                "row_id": negative_id,
                "source_sieve_row_id": str(sieve_row["row_id"]),
                "candidate_kind": "shifted_or_unknown",
                "split": split,
                "library_id": None,
                "library_family": None,
                "asset_path": "",
                "known_object": 0,
                "footprint_coverage": float(shifted.mean()),
                "placement_metadata": {"source_candidate_id": positive_id, "dx": dx, "dy": dy},
                "npz": negative_npz,
            })

    if not rows or not any(int(row["known_object"]) for row in rows):
        raise ObjectMarkerError("marker corpus contains no positive candidates")
    manifest = {
        "schema": SCHEMA,
        "generator": "harvester.v60.object_marker",
        "seed": int(seed),
        "source_sieve_corpus": str(sieve_corpus.resolve()),
        "source_sieve_schema": sieve_manifest["schema"],
        "source_library": provenance,
        "signal_contract": [IMAGE_SIGNAL, FOOTPRINT_SIGNAL, KNOWN_SIGNAL],
        "candidate_count": len(rows),
        "positive_count": sum(int(row["known_object"]) for row in rows),
        "negative_count": sum(int(not row["known_object"]) for row in rows),
        "skipped_instance_count": len(skipped_instances),
        "skipped_instances": skipped_instances,
        "family_splits": dict(sorted(family_splits.items())),
        "rows": rows,
    }
    (partial_output / "object_marker_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    partial_output.replace(output)
    return {
        "schema": SCHEMA,
        "output": str(output),
        "candidate_count": len(rows),
        "skipped_instance_count": len(skipped_instances),
    }


def load_object_marker_manifest(root: str | Path) -> dict[str, Any]:
    path = Path(root) / "object_marker_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"object marker manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ObjectMarkerError(f"expected {SCHEMA}, got {manifest.get('schema')!r}")
    return manifest


def validate_object_marker_corpus(root: str | Path) -> dict[str, Any]:
    root_path = Path(root)
    manifest = load_object_marker_manifest(root_path)
    failures: list[str] = []
    family_splits: dict[str, str] = {}
    split_counts: dict[str, int] = {}
    positive_count = 0
    negative_count = 0
    for index, row in enumerate(manifest.get("rows", [])):
        prefix = f"row[{index}]"
        split = str(row.get("split", ""))
        split_counts[split] = split_counts.get(split, 0) + 1
        library_family = row.get("library_family")
        if library_family:
            prior = family_splits.setdefault(str(library_family), split)
            if prior != split:
                failures.append(f"{prefix}: library family crosses split")
        known = int(row.get("known_object", 0))
        positive_count += known
        negative_count += int(not known)
        if known and not row.get("library_id"):
            failures.append(f"{prefix}: positive row is missing library_id")
        if not known and row.get("library_id") is not None:
            failures.append(f"{prefix}: negative row carries library_id")
        npz_path = root_path / str(row.get("npz", ""))
        if not npz_path.is_file():
            failures.append(f"{prefix}: NPZ not found")
            continue
        try:
            with np.load(npz_path, allow_pickle=False) as payload:
                image = _as_hwc_image(payload[IMAGE_SIGNAL])
                mask = _as_mask(payload[FOOTPRINT_SIGNAL])
                if int(np.asarray(payload[KNOWN_SIGNAL]).item()) != known:
                    failures.append(f"{prefix}: known target mismatch")
                if known and not (mask >= 0.5).any():
                    failures.append(f"{prefix}: positive candidate has empty footprint")
                if not np.isfinite(image).all() or not np.isfinite(mask).all():
                    failures.append(f"{prefix}: non-finite payload")
        except (KeyError, OSError, ValueError, ObjectMarkerError) as exc:
            failures.append(f"{prefix}: invalid payload: {exc}")
    if int(manifest.get("candidate_count", -1)) != len(manifest.get("rows", [])):
        failures.append("manifest candidate_count does not match rows")
    if int(manifest.get("positive_count", -1)) != positive_count:
        failures.append("manifest positive_count does not match rows")
    report = {
        "schema": VALIDATION_SCHEMA,
        "corpus_root": str(root_path),
        "candidate_count": len(manifest.get("rows", [])),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "split_counts": dict(sorted(split_counts.items())),
        "library_family_count": len(family_splits),
        "failures": failures,
        "valid": not failures,
    }
    return report


def marker_input_tensor(image: np.ndarray, footprint: np.ndarray) -> torch.Tensor:
    """Convert one HWC image plus one footprint to a model-ready BCHW tensor."""
    image_array = _as_hwc_image(image)
    mask_array = _as_mask(footprint)
    combined = np.concatenate([np.moveaxis(image_array, -1, 0), mask_array[None]], axis=0)
    return torch.from_numpy(combined.astype(np.float32, copy=False)).unsqueeze(0)


__all__ = [
    "EMBEDDING_DIM",
    "FOOTPRINT_SIGNAL",
    "IMAGE_SIGNAL",
    "KNOWN_SIGNAL",
    "MARKER_SIGNAL",
    "ObjectMarkerError",
    "ObjectMarkerNet",
    "build_marker_map",
    "build_library_gallery_inputs",
    "build_object_marker_corpus",
    "load_object_marker_manifest",
    "marker_input_tensor",
    "marker_loss",
    "retrieve_library_identity",
    "validate_object_marker_corpus",
]
