"""Spec 103 T009 — assemble the synthetic 13-channel training store.

Reads the synthetic manifest from spec103_make_synthetic_adts.py plus captured minimap PNGs,
a clearly-labeled legacy hillshade fallback, opt-in authored time-of-day variants, or a hash-bound
`lit profile`/`light profile` export. The variant path uses only known heights and generated neutral
albedo; it never relights captured
PNGs. It writes a zarr store with the same array names the trainer reads from the real V18
store: minimap_rgb, height_257, normal_xyz, liquid_mask, liquid_height,
object_precise_mask (all zeros here — synthetic tiles have no objects/liquid). Normals are
derived analytically from the known height field. The WDL prior is NOT stored — the assembler
derives it from height_257 (the verified ::16 outer transform) at batch time.

Run from wow-viewer/data-harvester/ (fast, CPU-only):

    uv run python scripts/spec103_build_synthetic_store.py \
        --manifest ../output/spec103/synthetic/synthetic_manifest.json \
        --minimap-dir ../output/spec103/synthetic/captures \
        --output ../output/datasets/spec103/synthetic_v1.zarr

    # before a capture run exists, the loop stays testable with either:
    #   --synthesize-minimaps             (legacy fixed hillshade; labeled in attrs)
    #   --lighting-time 0.25 --lighting-time 0.5
    #       (versioned authored day/night variants; grouped by their source tile)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from PIL import Image

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.spec103.terrain_lighting import (  # noqa: E402
    AUTHORED_MCSH_BAKE_DIRECTION,
    AUTHORED_MCSH_EVIDENCE_STATE,
    AUTHORED_MCSH_MODEL,
    GRID_TO_RENDERER_NORMAL_TRANSFORM,
    LIGHTING_MODEL,
    NEUTRAL_GENERATED_ALBEDO,
    TerrainLightingSample,
    evaluate_authored_day_night,
    grid_normals_to_renderer,
    load_lighting_profile_artifact,
    shade_terrain,
    synthesize_authored_height_shadow,
)

CHUNK_METERS = 533.33333 / 16.0  # world meters per chunk; height grid step = tile/256


def normals_from_height(height_257: np.ndarray) -> np.ndarray:
    """Analytic unit normals from the known height field (finite differences)."""
    step = 533.33333 / 256.0
    gy, gx = np.gradient(height_257.astype(np.float64), step)
    nx, ny, nz = -gx, -gy, np.ones_like(gx)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    return np.stack([nx / norm, ny / norm, nz / norm], axis=-1).astype(np.float32)


def hillshade_minimap(height_257: np.ndarray) -> np.ndarray:
    """Procedural fallback minimap: fixed-light hillshade of the known height, 256×256 u8 RGB."""
    normals = normals_from_height(height_257)
    light = np.array([-0.5, -0.5, 0.72])
    light = light / np.linalg.norm(light)
    shade = np.clip(normals @ light, 0.0, 1.0)
    shade = (0.25 + 0.75 * shade)[:256, :256]
    rgb = np.stack([shade * 0.55, shade * 0.62, shade * 0.42], axis=-1)  # muted terrain green
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _is_declared(value: object) -> bool:
    text = str(value or "").strip()
    return bool(text) and text.upper() != "UNSPECIFIED"


def _require_sha256(value: object, *, tile_name: str, field: str) -> str:
    digest = str(value or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{tile_name}: capture sidecar {field} must be a SHA-256 hex digest")
    return digest


def _require_finite_vector(
    value: object, *, tile_name: str, field: str
) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{tile_name}: capture sidecar {field} must be a 3-vector")
    result = [float(component) for component in value]
    if not np.isfinite(result).all():
        raise ValueError(f"{tile_name}: capture sidecar {field} contains non-finite values")
    return result


def _resolve_capture_adt(
    metadata_input: dict[str, object], *, tile: dict[str, object]
) -> tuple[Path, str]:
    tile_name = str(tile["tile_name"])
    client_root_text = str(metadata_input.get("client_root") or "").strip()
    adt_virtual_path = str(metadata_input.get("adt_path") or "").strip().replace("\\", "/")
    expected_virtual_path = (
        f"World/Maps/{tile['map']}/{tile_name}.adt"
    )
    if not client_root_text or adt_virtual_path.casefold() != expected_virtual_path.casefold():
        raise ValueError(
            f"{tile_name}: capture sidecar ADT path must be {expected_virtual_path!r}"
        )
    client_root = Path(client_root_text).expanduser().resolve()
    if not client_root.is_dir():
        raise ValueError(f"{tile_name}: capture sidecar client_root does not exist: {client_root}")
    adt_path = (client_root / Path(*adt_virtual_path.split("/"))).resolve()
    if not adt_path.is_relative_to(client_root) or not adt_path.is_file():
        raise ValueError(
            f"{tile_name}: capture sidecar ADT is not a readable loose file under client_root: "
            f"{adt_path}"
        )
    expected_sha256 = _require_sha256(
        metadata_input.get("adt_sha256"), tile_name=tile_name, field="input.adt_sha256"
    )
    actual_sha256 = _sha256_file(adt_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"{tile_name}: capture ADT hash does not match its lighting sidecar")
    return adt_path, actual_sha256


def _load_capture_lighting_evidence(
    png_path: Path, tile: dict[str, object]
) -> dict[str, object]:
    """Load and verify the capture sidecar so RGB never loses its lighting lineage."""
    tile_name = str(tile["tile_name"])
    sidecar_path = Path(f"{png_path}.lighting.json")
    if not sidecar_path.is_file():
        raise ValueError(
            f"{tile_name}: captured minimap is missing required lighting sidecar: "
            f"{sidecar_path}"
        )
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != "wowviewer-terrain-capture-lighting-v2":
        raise ValueError(
            f"{tile_name}: capture sidecar must use "
            "wowviewer-terrain-capture-lighting-v2"
        )
    metadata_tile = metadata.get("tile") or {}
    expected_tile = {
        "name": tile_name,
        "map": str(tile["map"]),
        "x": int(tile["tile_x"]),
        "y": int(tile["tile_y"]),
    }
    actual_tile = {
        "name": metadata_tile.get("name"),
        "map": metadata_tile.get("map"),
        "x": metadata_tile.get("x"),
        "y": metadata_tile.get("y"),
    }
    if actual_tile != expected_tile:
        raise ValueError(
            f"{tile_name}: capture sidecar tile identity mismatch; expected "
            f"{expected_tile!r}, got {actual_tile!r}"
        )
    metadata_input = metadata.get("input") or {}
    if not isinstance(metadata_input, dict):
        raise ValueError(f"{tile_name}: capture sidecar input must be an object")
    adt_path, actual_adt_sha256 = _resolve_capture_adt(metadata_input, tile=tile)

    output = metadata.get("output") or {}
    if not isinstance(output, dict):
        raise ValueError(f"{tile_name}: capture sidecar output must be an object")
    expected_png_sha256 = _require_sha256(
        output.get("png_sha256"), tile_name=tile_name, field="output.png_sha256"
    )
    actual_png_sha256 = _sha256_file(png_path)
    if expected_png_sha256 != actual_png_sha256:
        raise ValueError(
            f"{tile_name}: capture PNG hash does not match its lighting sidecar"
        )

    with Image.open(png_path) as image:
        actual_width, actual_height = image.size
    output_width = output.get("width")
    output_height = output.get("height")
    if (
        not isinstance(output_width, int)
        or not isinstance(output_height, int)
        or output_width <= 0
        or output_width != output_height
        or (output_width, output_height) != (actual_width, actual_height)
    ):
        raise ValueError(
            f"{tile_name}: capture output dimensions must describe the square PNG exactly"
        )

    camera = metadata.get("camera") or {}
    if not isinstance(camera, dict):
        raise ValueError(f"{tile_name}: capture sidecar camera must be an object")
    camera_mode = str(camera.get("mode") or "")
    image_axis_contract = str(camera.get("image_axis_contract") or "")
    if camera_mode != "top_down_orthographic_one_adt_tile_v1":
        raise ValueError(f"{tile_name}: capture camera is not the canonical one-ADT view")
    if image_axis_contract != "right=adt_tile_x_positive;down=adt_tile_y_positive":
        raise ValueError(f"{tile_name}: capture image axis/orientation contract is invalid")
    camera_position = _require_finite_vector(
        camera.get("position"), tile_name=tile_name, field="camera.position"
    )
    finite_camera_scalars: dict[str, float] = {}
    for field in ("far_plane", "terrain_min_height", "terrain_max_height"):
        raw_value = camera.get(field)
        if not isinstance(raw_value, (int, float)) or not np.isfinite(raw_value):
            raise ValueError(f"{tile_name}: capture sidecar camera.{field} must be finite")
        finite_camera_scalars[field] = float(raw_value)
    if finite_camera_scalars["far_plane"] <= 0.0:
        raise ValueError(f"{tile_name}: capture camera far_plane must be positive")
    if finite_camera_scalars["terrain_max_height"] < finite_camera_scalars["terrain_min_height"]:
        raise ValueError(f"{tile_name}: capture camera terrain height bounds are inverted")

    lighting = metadata.get("lighting") or {}
    source = metadata.get("lighting_source") or {}
    if not isinstance(lighting, dict) or not isinstance(source, dict):
        raise ValueError(f"{tile_name}: capture sidecar lighting/source must be objects")
    renderer_contract = str(metadata.get("renderer_contract") or "").strip()
    source_kind = str(metadata.get("lighting_source_kind") or "").strip()
    profile_revision = str(metadata.get("lighting_profile_revision") or "")
    evidence_state = str(metadata.get("lighting_evidence_state") or "").strip()
    game_time = lighting.get("game_time")
    if (
        not renderer_contract
        or not profile_revision
        or not evidence_state
        or source_kind not in {"authored_fallback", "client_lit_global_clear"}
        or not isinstance(game_time, (int, float))
        or not np.isfinite(game_time)
        or not 0.0 <= float(game_time) < 1.0
    ):
        raise ValueError(f"{tile_name}: capture sidecar has incomplete lighting identity")

    source_identifier = str(source.get("identifier") or "").strip()
    if not source_identifier:
        raise ValueError(f"{tile_name}: capture lighting source identifier is empty")
    track_ids = source.get("contributing_track_ids")
    if (
        not isinstance(track_ids, list)
        or any(not isinstance(track_id, int) or not 0 <= track_id < 18 for track_id in track_ids)
        or len(track_ids) != len(set(track_ids))
    ):
        raise ValueError(f"{tile_name}: capture contributing_track_ids are invalid")
    declared_source_sha256 = str(source.get("sha256") or "").strip().lower()
    direction_evidence_state = str(source.get("direction_evidence_state") or "").strip()
    mcsh_evidence_state = str(source.get("mcsh_evidence_state") or "").strip()
    if not direction_evidence_state or not mcsh_evidence_state:
        raise ValueError(f"{tile_name}: capture lighting direction/MCSH evidence is incomplete")
    if source_kind == "client_lit_global_clear":
        declared_source_sha256 = _require_sha256(
            declared_source_sha256,
            tile_name=tile_name,
            field="lighting_source.sha256",
        )
        if not {0, 1, 7}.issubset(track_ids):
            raise ValueError(
                f"{tile_name}: client LIT capture must identify direct, ambient, and fog tracks"
            )
        lit_version = str(source.get("lit_version") or "").strip()
        lit_light_index = source.get("lit_light_index")
        lit_group_index = source.get("lit_group_index")
        lit_time = source.get("lit_time")
        if (
            re.fullmatch(r"0x[0-9a-fA-F]{8}", lit_version) is None
            or not isinstance(lit_light_index, int)
            or lit_light_index < 0
            or not isinstance(lit_group_index, int)
            or lit_group_index < 0
            or not isinstance(lit_time, (int, float))
            or not np.isfinite(lit_time)
            or not 0.0 <= float(lit_time) <= 2880.0
        ):
            raise ValueError(f"{tile_name}: client LIT profile selection is incomplete")
        expected_lit_time = float(game_time) * 2880.0
        if not np.isclose(float(lit_time), expected_lit_time, rtol=0.0, atol=1e-3):
            raise ValueError(f"{tile_name}: client LIT time does not match capture game_time")
        digest_state = "declared_client_artifact_sha256"
    else:
        if (
            source_identifier != profile_revision
            or declared_source_sha256
            or track_ids
            or source.get("lit_version") not in {None, ""}
            or source.get("lit_light_index") is not None
            or source.get("lit_group_index") is not None
            or source.get("lit_time") is not None
        ):
            raise ValueError(
                f"{tile_name}: authored capture source must bind only its profile revision"
            )
        digest_state = "authored_profile_revision_no_external_artifact"

    light_direction = _require_finite_vector(
        lighting.get("light_direction"), tile_name=tile_name, field="lighting.light_direction"
    )
    directional_color = _require_finite_vector(
        lighting.get("directional_color"), tile_name=tile_name, field="lighting.directional_color"
    )
    ambient_color = _require_finite_vector(
        lighting.get("ambient_color"), tile_name=tile_name, field="lighting.ambient_color"
    )
    fog_color = _require_finite_vector(
        lighting.get("fog_color"), tile_name=tile_name, field="lighting.fog_color"
    )
    numeric_lighting: dict[str, float] = {}
    for field in ("directional_intensity", "ambient_intensity", "mcsh_shadow_strength"):
        raw_value = lighting.get(field)
        if not isinstance(raw_value, (int, float)) or not np.isfinite(raw_value):
            raise ValueError(f"{tile_name}: capture lighting.{field} must be finite")
        numeric_lighting[field] = float(raw_value)
    if numeric_lighting["directional_intensity"] < 0.0 or numeric_lighting["ambient_intensity"] < 0.0:
        raise ValueError(f"{tile_name}: capture lighting intensities cannot be negative")
    if not 0.0 <= numeric_lighting["mcsh_shadow_strength"] <= 1.0:
        raise ValueError(f"{tile_name}: capture MCSH strength must be within 0..1")

    sidecar_sha256 = _sha256_file(sidecar_path)
    source_identity_payload = {
        "kind": source_kind,
        "identifier": source_identifier,
        "sha256": declared_source_sha256,
        "profile_revision": profile_revision,
        "track_ids": track_ids,
    }
    source_identity_sha256 = hashlib.sha256(
        json.dumps(source_identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    return {
        "lighting_variant_id": (
            f"{profile_revision}:time={float(game_time):.9f}:adt={actual_adt_sha256}:"
            f"png={actual_png_sha256}:sidecar={sidecar_sha256}"
        ),
        "lighting_profile_revision": profile_revision,
        "lighting_evidence_state": evidence_state,
        "lighting_model": renderer_contract,
        "game_time": float(game_time),
        "light_direction_xyz": light_direction,
        "directional_color_rgb": directional_color,
        "directional_intensity": numeric_lighting["directional_intensity"],
        "ambient_color_rgb": ambient_color,
        "ambient_intensity": numeric_lighting["ambient_intensity"],
        "fog_color_rgb": fog_color,
        "mcsh_shadow_strength": numeric_lighting["mcsh_shadow_strength"],
        "capture_lighting_metadata_path": str(sidecar_path.resolve()),
        "capture_lighting_metadata_sha256": sidecar_sha256,
        "capture_png_sha256": actual_png_sha256,
        "capture_adt_path": str(adt_path),
        "capture_adt_sha256": actual_adt_sha256,
        "capture_camera_mode": camera_mode,
        "capture_camera_position_xyz": camera_position,
        "capture_camera_far_plane": finite_camera_scalars["far_plane"],
        "capture_camera_terrain_min_height": finite_camera_scalars["terrain_min_height"],
        "capture_camera_terrain_max_height": finite_camera_scalars["terrain_max_height"],
        "capture_image_axis_contract": image_axis_contract,
        "capture_output_width": output_width,
        "capture_output_height": output_height,
        "capture_lighting_source_kind": source_kind,
        "capture_lighting_source_identifier": source_identifier,
        "capture_lighting_source_sha256": declared_source_sha256,
        "capture_lighting_source_digest_state": digest_state,
        "capture_lighting_source_identity_sha256": source_identity_sha256,
        "capture_lit_version": str(source.get("lit_version") or ""),
        "capture_lit_light_index": source.get("lit_light_index"),
        "capture_lit_light_name": str(source.get("lit_light_name") or ""),
        "capture_lit_group_index": source.get("lit_group_index"),
        "capture_lit_time": source.get("lit_time"),
        "capture_lit_track_ids": track_ids,
        "capture_direction_evidence_state": direction_evidence_state,
        "capture_mcsh_evidence_state": mcsh_evidence_state,
    }


def _verify_height_provenance(
    tile: dict[str, object],
    *,
    manifest_path: Path,
    require_licensed_synthetic: bool,
) -> tuple[Path, str]:
    height_path = _resolve_manifest_path(manifest_path, str(tile["height_npy"]))
    if not height_path.is_file():
        raise ValueError(f"{tile['tile_name']}: height file does not exist: {height_path}")
    actual_sha256 = _sha256_file(height_path)
    expected_sha256 = str(tile.get("height_sha256") or "").strip().lower()
    if _is_declared(expected_sha256) and expected_sha256 != actual_sha256:
        raise ValueError(
            f"{tile['tile_name']}: height_sha256 mismatch; expected {expected_sha256}, "
            f"got {actual_sha256}"
        )

    if require_licensed_synthetic:
        origin = str(tile.get("terrain_source_origin") or "").strip()
        if origin != "analytic_generated":
            raise ValueError(
                f"{tile['tile_name']}: licensed synthetic gate requires "
                "terrain_source_origin='analytic_generated'"
            )
        if not _is_declared(tile.get("terrain_source_license")):
            raise ValueError(
                f"{tile['tile_name']}: licensed synthetic gate requires an explicit "
                "terrain_source_license; UNSPECIFIED is rejected"
            )
        if not _is_declared(tile.get("terrain_source_rights_assertion")):
            raise ValueError(
                f"{tile['tile_name']}: licensed synthetic gate requires an explicit "
                "terrain_source_rights_assertion; UNSPECIFIED is rejected"
            )
        if not _is_declared(expected_sha256):
            raise ValueError(
                f"{tile['tile_name']}: licensed synthetic gate requires height_sha256"
            )
    return height_path, actual_sha256


def _load_mcsh_shadow(
    tile: dict[str, object],
    *,
    manifest_path: Path,
    require_licensed_synthetic: bool,
    synthesize_mcsh: bool,
    height_257: np.ndarray,
) -> tuple[np.ndarray, str, str, str, str]:
    """Load a 0-lit/1-shadow MCSH mask, or return explicit zero/absent evidence."""
    value = str(tile.get("mcsh_shadow_npy") or "").strip()
    if synthesize_mcsh and value:
        raise ValueError(
            f"{tile['tile_name']}: --synthesize-mcsh cannot be combined with "
            "mcsh_shadow_npy"
        )
    if synthesize_mcsh:
        shadow = synthesize_authored_height_shadow(height_257)
        shadow_sha256 = hashlib.sha256(shadow.tobytes(order="C")).hexdigest()
        return (
            shadow,
            AUTHORED_MCSH_EVIDENCE_STATE,
            "",
            shadow_sha256,
            AUTHORED_MCSH_MODEL,
        )
    if not value:
        return (
            np.zeros((256, 256), dtype=np.float32),
            "absent_zero_fill",
            "",
            "",
            "none",
        )
    if require_licensed_synthetic:
        raise ValueError(
            f"{tile['tile_name']}: licensed synthetic gate rejects mcsh_shadow_npy until "
            "the shadow source has its own origin, rights assertion, and content hash"
        )

    path = _resolve_manifest_path(manifest_path, value)
    if not path.is_file():
        raise ValueError(f"{tile['tile_name']}: MCSH shadow file does not exist: {path}")
    raw = np.load(path)
    if raw.ndim == 1 and raw.size == 64 * 64:
        raw = raw.reshape(64, 64)
    if raw.shape == (64, 64):
        shadow = np.repeat(np.repeat(raw, 4, axis=0), 4, axis=1)
    elif raw.shape == (256, 256):
        shadow = raw
    elif raw.shape == (257, 257):
        shadow = raw[:256, :256]
    else:
        raise ValueError(
            f"{tile['tile_name']}: MCSH shadow must be 64x64, 256x256, or 257x257, "
            f"got {raw.shape}"
        )
    shadow = np.asarray(shadow, dtype=np.float32)
    if float(np.nanmax(shadow, initial=0.0)) > 1.0:
        shadow /= 255.0
    if not np.isfinite(shadow).all():
        raise ValueError(f"{tile['tile_name']}: MCSH shadow contains non-finite values")
    return (
        np.clip(shadow, 0.0, 1.0),
        "provided_mcsh_mask",
        str(path),
        _sha256_file(path),
        "provided_mask_bake_model_unknown",
    )


def authored_lighting_minimap(
    height_257: np.ndarray,
    sample: TerrainLightingSample,
    mcsh_shadow: np.ndarray | None = None,
) -> np.ndarray:
    """Render a 256x256 neutral-albedo variant from known height and optional MCSH."""
    grid_normals = normals_from_height(height_257)[:256, :256]
    normals = grid_normals_to_renderer(grid_normals)
    shadow = (
        np.zeros((256, 256), dtype=np.float32)
        if mcsh_shadow is None
        else np.asarray(mcsh_shadow, dtype=np.float32)
    )
    if shadow.shape != (256, 256):
        raise ValueError(f"mcsh_shadow must be (256, 256), got {shadow.shape}")
    albedo = np.asarray(NEUTRAL_GENERATED_ALBEDO, dtype=np.float32)
    rgb = shade_terrain(albedo, normals, shadow, sample)
    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)


def _normalize_lighting_times(values: list[float]) -> list[TerrainLightingSample]:
    samples = [evaluate_authored_day_night(value) for value in values]
    keys = [round(sample.game_time, 9) for sample in samples]
    if len(set(keys)) != len(keys):
        raise ValueError(
            "duplicate --lighting-time values after [0,1) wrapping would create wasteful "
            "duplicate variants"
        )
    return samples


def build_synthetic_store(
    *,
    manifest_path: Path,
    output_path: Path,
    minimap_dir: Path | None = None,
    synthesize_minimaps: bool = False,
    lighting_times: list[float] | None = None,
    lighting_profile: Path | None = None,
    require_licensed_synthetic: bool = False,
    synthesize_mcsh: bool = False,
) -> dict[str, object]:
    """Build the store and return its compact, JSON-safe contract summary."""
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tiles: list[dict[str, object]] = manifest["tiles"]
    if not tiles:
        raise ValueError("manifest has no tiles")

    if lighting_profile is not None and lighting_times:
        raise ValueError("--lighting-profile cannot be combined with --lighting-time")
    lighting_samples = (
        load_lighting_profile_artifact(lighting_profile)
        if lighting_profile is not None
        else _normalize_lighting_times(lighting_times or [])
    )
    contains_client_profile = any(
        sample.color_source_kind.startswith("client_") for sample in lighting_samples
    )
    if lighting_samples and minimap_dir is not None:
        raise ValueError(
            "lighting profile variants cannot be combined with --minimap-dir: variants are "
            "generated from known height and neutral albedo, never from captured PNGs or by "
            "relighting them"
        )
    if lighting_samples and synthesize_minimaps:
        raise ValueError(
            "--lighting-time cannot be combined with legacy --synthesize-minimaps"
        )
    if require_licensed_synthetic and not lighting_samples:
        raise ValueError(
            "--require-licensed-synthetic requires one or more authored --lighting-time variants; "
            "captured and legacy fallback minimaps are rejected"
        )
    if require_licensed_synthetic and contains_client_profile:
        raise ValueError(
            "--require-licensed-synthetic rejects client-derived LIT/DBC color profiles; "
            "use the private-BYOD lane"
        )
    if synthesize_mcsh and not lighting_samples:
        raise ValueError("--synthesize-mcsh requires one or more --lighting-time variants")

    prepared: list[dict[str, object]] = []
    for source_tile_id, tile in enumerate(tiles):
        height_path, height_sha256 = _verify_height_provenance(
            tile,
            manifest_path=manifest_path,
            require_licensed_synthetic=require_licensed_synthetic,
        )
        height = np.load(height_path).astype(np.float32)
        if height.shape != (257, 257):
            raise ValueError(
                f"{tile['tile_name']}: height must be (257, 257), got {height.shape}"
            )
        if not np.isfinite(height).all():
            raise ValueError(f"{tile['tile_name']}: height contains non-finite values")
        capture_evidence: dict[str, object] | None = None
        if not lighting_samples and minimap_dir is not None:
            capture_png = minimap_dir / f"{tile['tile_name']}.png"
            if capture_png.exists():
                capture_evidence = _load_capture_lighting_evidence(capture_png, tile)
        if lighting_samples:
            shadow, shadow_state, shadow_path, shadow_sha256, shadow_model = (
                _load_mcsh_shadow(
                    tile,
                    manifest_path=manifest_path,
                    require_licensed_synthetic=require_licensed_synthetic,
                    synthesize_mcsh=synthesize_mcsh,
                    height_257=height,
                )
            )
        else:
            shadow = np.zeros((256, 256), dtype=np.float32)
            shadow_state, shadow_path, shadow_sha256, shadow_model = (
                "not_used_by_legacy_minimap_path",
                "",
                "",
                "none",
            )
        prepared.append(
            {
                "source_tile_id": source_tile_id,
                "tile": tile,
                "height": height,
                "height_path": height_path,
                "height_sha256": height_sha256,
                "shadow": shadow,
                "shadow_evidence_state": shadow_state,
                "shadow_source_path": shadow_path,
                "shadow_sha256": shadow_sha256,
                "shadow_model": shadow_model,
                "capture_evidence": capture_evidence,
            }
        )

    output = output_path.resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite existing store: {output}")

    variants_per_source = len(lighting_samples) if lighting_samples else 1
    row_count = len(prepared) * variants_per_source
    out = zarr.open_group(str(output), mode="w")
    arrays = {
        "minimap_rgb": out.create_array(
            "minimap_rgb",
            shape=(row_count, 256, 256, 3),
            chunks=(1, 256, 256, 3),
            dtype=np.uint8,
        ),
        "height_257": out.create_array(
            "height_257",
            shape=(row_count, 257, 257),
            chunks=(1, 257, 257),
            dtype=np.float32,
        ),
        "normal_xyz": out.create_array(
            "normal_xyz",
            shape=(row_count, 257, 257, 3),
            chunks=(1, 257, 257, 3),
            dtype=np.float32,
        ),
        "liquid_mask": out.create_array(
            "liquid_mask",
            shape=(row_count, 256, 256),
            chunks=(1, 256, 256),
            dtype=np.float32,
        ),
        "liquid_height": out.create_array(
            "liquid_height",
            shape=(row_count, 256, 256),
            chunks=(1, 256, 256),
            dtype=np.float32,
        ),
        "object_precise_mask": out.create_array(
            "object_precise_mask",
            shape=(row_count, 257, 257),
            chunks=(1, 257, 257),
            dtype=np.float32,
        ),
    }

    rows: list[dict[str, object]] = []
    minimap_sources: dict[str, int] = {}
    for source in prepared:
        tile = source["tile"]
        assert isinstance(tile, dict)
        height = source["height"]
        shadow = source["shadow"]
        assert isinstance(height, np.ndarray) and isinstance(shadow, np.ndarray)
        tile_name = str(tile["tile_name"])
        source_group_id = str(
            tile.get("source_group_id") or f"synthetic:{tile['map']}:{tile_name}"
        )
        samples: list[TerrainLightingSample | None] = (
            list(lighting_samples) if lighting_samples else [None]
        )
        for sample in samples:
            row = len(rows)
            png = (minimap_dir / f"{tile_name}.png") if minimap_dir else None
            if sample is not None:
                minimap = authored_lighting_minimap(height, sample, shadow)
                minimap_source = "synthesized_authored_lighting"
                lighting_variant_id = (
                    f"{sample.profile_revision}:time={sample.game_time:.9f}"
                )
            elif png is not None and png.exists():
                capture_evidence = source["capture_evidence"]
                if not isinstance(capture_evidence, dict):
                    raise AssertionError(
                        f"{tile_name}: capture evidence was not prepared before store creation"
                    )
                minimap = np.asarray(
                    Image.open(png)
                    .convert("RGB")
                    .resize((256, 256), Image.Resampling.BILINEAR),
                    dtype=np.uint8,
                )
                minimap_source = "captured"
                lighting_variant_id = str(capture_evidence["lighting_variant_id"])
            elif synthesize_minimaps:
                minimap = hillshade_minimap(height)
                minimap_source = "synthesized"
                lighting_variant_id = ""
            else:
                raise ValueError(
                    f"{tile_name}: no captured minimap at {png} — run the capture commands, "
                    "pass --synthesize-minimaps, or provide --lighting-time"
                )
            minimap_sources[minimap_source] = minimap_sources.get(minimap_source, 0) + 1

            arrays["minimap_rgb"][row] = minimap
            arrays["height_257"][row] = height
            arrays["normal_xyz"][row] = normals_from_height(height)
            index_row: dict[str, object] = {
                "row": row,
                "map": tile["map"],
                "tile_x": int(tile["tile_x"]),
                "tile_y": int(tile["tile_y"]),
                "tile_id": row,
                "build": "synthetic",
                "pattern": tile["pattern"],
                "amplitude": float(tile["amplitude"]),
                "minimap_source": minimap_source,
                "source_tile_id": int(source["source_tile_id"]),
                "source_tile_name": tile_name,
                "source_group_id": source_group_id,
                "lighting_variant_id": lighting_variant_id,
                "terrain_source_origin": str(
                    tile.get("terrain_source_origin") or "UNSPECIFIED"
                ),
                "terrain_source_license": str(
                    tile.get("terrain_source_license") or "UNSPECIFIED"
                ),
                "terrain_source_rights_assertion": str(
                    tile.get("terrain_source_rights_assertion") or "UNSPECIFIED"
                ),
                "height_source_path": str(source["height_path"]),
                "height_sha256": source["height_sha256"],
                "shadow_evidence_state": source["shadow_evidence_state"],
                "shadow_source_path": source["shadow_source_path"],
                "shadow_sha256": source["shadow_sha256"],
                "shadow_model": source["shadow_model"],
                "shadow_bake_direction_xyz": (
                    list(AUTHORED_MCSH_BAKE_DIRECTION)
                    if source["shadow_model"] == AUTHORED_MCSH_MODEL
                    else []
                ),
                "normal_source": "derived_known_height_finite_difference",
                "lighting_normal_transform": (
                    GRID_TO_RENDERER_NORMAL_TRANSFORM if sample is not None else ""
                ),
                "albedo_source": (
                    "generated_neutral_constant" if sample is not None else "not_applicable"
                ),
                "capture_lighting_metadata_path": "",
                "capture_lighting_metadata_sha256": "",
                "capture_png_sha256": "",
                "capture_adt_path": "",
                "capture_adt_sha256": "",
                "capture_camera_mode": "",
                "capture_camera_position_xyz": [],
                "capture_camera_far_plane": None,
                "capture_camera_terrain_min_height": None,
                "capture_camera_terrain_max_height": None,
                "capture_image_axis_contract": "",
                "capture_output_width": None,
                "capture_output_height": None,
                "capture_lighting_source_kind": "",
                "capture_lighting_source_identifier": "",
                "capture_lighting_source_sha256": "",
                "capture_lighting_source_digest_state": "",
                "capture_lighting_source_identity_sha256": "",
                "capture_lit_version": "",
                "capture_lit_light_index": None,
                "capture_lit_light_name": "",
                "capture_lit_group_index": None,
                "capture_lit_time": None,
                "capture_lit_track_ids": [],
                "capture_direction_evidence_state": "",
                "capture_mcsh_evidence_state": "",
            }
            if sample is not None:
                index_row.update(sample.index_metadata())
            elif minimap_source == "captured":
                index_row.update(capture_evidence)
            rows.append(index_row)

    pq.write_table(pa.Table.from_pylist(rows), output / "index.parquet")
    source_licenses = sorted(
        {str(tile.get("terrain_source_license") or "UNSPECIFIED") for tile in tiles}
    )
    source_rights = sorted(
        {
            str(tile.get("terrain_source_rights_assertion") or "UNSPECIFIED")
            for tile in tiles
        }
    )
    source_origins = sorted(
        {str(tile.get("terrain_source_origin") or "UNSPECIFIED") for tile in tiles}
    )
    lighting_profile_revisions = sorted(
        {
            str(row.get("lighting_profile_revision") or "")
            for row in rows
            if row.get("lighting_profile_revision")
        }
    )
    lighting_evidence_states = sorted(
        {
            str(row.get("lighting_evidence_state") or "")
            for row in rows
            if row.get("lighting_evidence_state")
        }
    )
    lighting_color_source_kinds = sorted(
        {
            str(row.get("lighting_color_source_kind") or "")
            for row in rows
            if row.get("lighting_color_source_kind")
        }
    )
    lighting_profile_artifact_sha256 = sorted(
        {
            str(row.get("lighting_profile_artifact_sha256") or "")
            for row in rows
            if row.get("lighting_profile_artifact_sha256")
        }
    )
    capture_lighting_source_kinds = sorted(
        {
            str(row.get("capture_lighting_source_kind") or "")
            for row in rows
            if row.get("capture_lighting_source_kind")
        }
    )
    contains_client_lighting = contains_client_profile or any(
        source_kind.startswith("client_")
        for source_kind in capture_lighting_source_kinds
    )
    rights_contract: dict[str, object]
    if require_licensed_synthetic:
        rights_contract = {
            "rights_class": "clean_synthetic",
            "contains_raw_game_client_files": False,
            "contains_client_derived_training_data": False,
            "distribution_policy": "operator_declared_license_only",
        }
    elif contains_client_lighting:
        rights_contract = {
            "rights_class": "private_byod",
            "contains_raw_game_client_files": False,
            "contains_client_derived_training_data": True,
            "distribution_policy": "private_operator_workflow_only",
        }
    else:
        rights_contract = {
            "rights_class": "provenance_unverified",
            "distribution_policy": "not_asserted",
        }

    contract: dict[str, object] = {
        "schema": (
            "spec103-lighting-profile-variants-store-v1"
            if lighting_samples
            else "spec103-synthetic-store-v2"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "tile_count": row_count,
        "source_tile_count": len(prepared),
        "variants_per_source": variants_per_source,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256_file(manifest_path),
        "minimap_sources": minimap_sources,
        "signals": sorted(arrays),
        "wdl_prior_policy": (
            "derived at batch time: outer = height_257[::16, ::16]; "
            "wdl_height_33 prohibited"
        ),
        "lighting_profile_revision": (
            lighting_profile_revisions[0] if len(lighting_profile_revisions) == 1 else ""
        ),
        "lighting_profile_revisions": lighting_profile_revisions,
        "lighting_evidence_state": (
            lighting_evidence_states[0] if len(lighting_evidence_states) == 1 else ""
        ),
        "lighting_evidence_states": lighting_evidence_states,
        "lighting_model": LIGHTING_MODEL if lighting_samples else "",
        "lighting_times": [sample.game_time for sample in lighting_samples],
        "lighting_color_source_kinds": lighting_color_source_kinds,
        "lighting_profile_artifact_sha256": lighting_profile_artifact_sha256,
        "capture_lighting_source_kinds": capture_lighting_source_kinds,
        "capture_lighting_sidecar_count": sum(
            bool(row.get("capture_lighting_metadata_path")) for row in rows
        ),
        "variant_split_policy": (
            "all rows sharing source_group_id must remain in one partition"
            if lighting_samples
            else "not_applicable"
        ),
        "source_origin_summary": source_origins,
        "source_license_summary": source_licenses,
        "source_rights_assertion_summary": source_rights,
        "licensed_synthetic_gate": require_licensed_synthetic,
        "synthesize_mcsh": synthesize_mcsh,
        "synthesized_mcsh_model": AUTHORED_MCSH_MODEL if synthesize_mcsh else "",
        "synthesized_mcsh_evidence_state": (
            AUTHORED_MCSH_EVIDENCE_STATE if synthesize_mcsh else ""
        ),
        **rights_contract,
    }
    out.attrs.update(contract)
    (output / "contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )
    return contract


def main() -> int:
    ap = argparse.ArgumentParser(description="Spec 103 synthetic 13-channel store builder")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument(
        "--minimap-dir",
        type=Path,
        default=None,
        help="captured minimap PNGs named <tile_name>.png",
    )
    ap.add_argument(
        "--synthesize-minimaps",
        action="store_true",
        help="use the legacy procedural hillshade fallback for missing captured PNGs",
    )
    ap.add_argument(
        "--lighting-time",
        action="append",
        type=float,
        default=[],
        metavar="FRACTION",
        help=(
            "generate a versioned authored MCNR/MCSH lighting variant at normalized time; "
            "repeat for multiple variants"
        ),
    )
    ap.add_argument(
        "--lighting-profile",
        type=Path,
        default=None,
        help=(
            "hash-bound JSON from `lit profile` or `light profile`; its client colors force "
            "the private-BYOD rights lane"
        ),
    )
    ap.add_argument(
        "--require-licensed-synthetic",
        action="store_true",
        help=(
            "fail closed unless time variants use hash-verified analytic heights with an "
            "explicit operator-supplied license and rights assertion"
        ),
    )
    ap.add_argument(
        "--synthesize-mcsh",
        action="store_true",
        help=(
            "add a deterministic fixed-direction height-ray shadow to lighting variants; "
            "authored augmentation, not client-exact MCSH"
        ),
    )
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    try:
        contract = build_synthetic_store(
            manifest_path=args.manifest,
            output_path=args.output,
            minimap_dir=args.minimap_dir,
            synthesize_minimaps=args.synthesize_minimaps,
            lighting_times=args.lighting_time,
            lighting_profile=args.lighting_profile,
            require_licensed_synthetic=args.require_licensed_synthetic,
            synthesize_mcsh=args.synthesize_mcsh,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"[spec103] wrote {contract['tile_count']} rows -> {args.output.resolve()}")
    print(f"[spec103] minimap sources: {contract['minimap_sources']}")
    print(f"[spec103] rights class: {contract['rights_class']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
