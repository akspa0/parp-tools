"""Versioned authored terrain lighting for Spec 103 synthetic minimaps.

The lighting *mechanics* mirror the recovered 1.0.0 terrain contract: renderer-space
MCNR normals receive clamped Lambert lighting and MCSH modulates only the directional
term.  The time-of-day colors, direction curve, and shadow-strength coefficient below
are an authored fallback.  They are deliberately labeled as such and are not claimed
to be recovered client LIT tracks or Light* DBC records.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

PROFILE_REVISION = "wow-1.0.0-authored-day-night-v1"
EVIDENCE_STATE = "authored_fallback_not_client_light_data"
LIGHTING_MODEL = "mcnr_lambert_plus_mcsh_directional_v1"
DEFAULT_GAME_TIME = 0.35
DEFAULT_MCSH_SHADOW_STRENGTH = 0.60
NEUTRAL_GENERATED_ALBEDO = (0.62, 0.62, 0.62)
AUTHORED_MCSH_MODEL = "authored_height_ray_shadow_256_v1"
AUTHORED_MCSH_EVIDENCE_STATE = "authored_height_ray_shadow_not_client_exact"
AUTHORED_MCSH_BAKE_DIRECTION = (-0.65, -0.35, 0.675)
GRID_TO_RENDERER_NORMAL_TRANSFORM = "grid_xyz_to_renderer_neg_y_neg_x_z_v1"
AUTHORED_DIRECTION_EVIDENCE_STATE = "authored_solar_direction_not_client_light_data"
AUTHORED_MCSH_STRENGTH_EVIDENCE_STATE = "authored_mcsh_strength_not_client_exact"
LIT_PROFILE_SCHEMA = "wowviewer.lit-profile.v1"
DBC_PROFILE_SCHEMA = "wowviewer.light-dbc-profile.v1"
REQUIRED_DBC_TABLES = {
    "Light",
    "LightParams",
    "LightIntBand",
    "LightFloatBand",
    "LightSkybox",
}

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class TerrainLightingSample:
    """One evaluated sample from the authored time-of-day profile."""

    profile_revision: str
    evidence_state: str
    lighting_model: str
    game_time: float
    light_direction: NDArray[np.float32]
    directional_color: NDArray[np.float32]
    directional_intensity: float
    ambient_color: NDArray[np.float32]
    ambient_intensity: float
    fog_color: NDArray[np.float32]
    mcsh_shadow_strength: float
    color_source_kind: str = "authored_fallback"
    color_source_identifier: str = PROFILE_REVISION
    color_source_sha256: str = ""
    profile_artifact_sha256: str = ""
    source_evidence_json: str = ""
    direction_evidence_state: str = AUTHORED_DIRECTION_EVIDENCE_STATE
    mcsh_strength_evidence_state: str = AUTHORED_MCSH_STRENGTH_EVIDENCE_STATE

    def index_metadata(self) -> dict[str, object]:
        """Return Arrow-friendly provenance fields for a synthetic-store row."""
        return {
            "lighting_profile_revision": self.profile_revision,
            "lighting_evidence_state": self.evidence_state,
            "lighting_model": self.lighting_model,
            "game_time": self.game_time,
            "light_direction_xyz": self.light_direction.tolist(),
            "directional_color_rgb": self.directional_color.tolist(),
            "directional_intensity": self.directional_intensity,
            "ambient_color_rgb": self.ambient_color.tolist(),
            "ambient_intensity": self.ambient_intensity,
            "fog_color_rgb": self.fog_color.tolist(),
            "mcsh_shadow_strength": self.mcsh_shadow_strength,
            "lighting_color_source_kind": self.color_source_kind,
            "lighting_color_source_identifier": self.color_source_identifier,
            "lighting_color_source_sha256": self.color_source_sha256,
            "lighting_profile_artifact_sha256": self.profile_artifact_sha256,
            "lighting_source_evidence_json": self.source_evidence_json,
            "direction_evidence_state": self.direction_evidence_state,
            "mcsh_strength_evidence_state": self.mcsh_strength_evidence_state,
        }


def load_lighting_profile_artifact(path: str | Path) -> list[TerrainLightingSample]:
    """Load a hash-bound `lit profile` or `light profile` export.

    Client artifacts supply colors only. Direction and MCSH strength remain the same explicit
    authored gaps used by the renderer bridge; this function never promotes them to client truth.
    """
    artifact_path = Path(path).resolve()
    payload_bytes = artifact_path.read_bytes()
    artifact_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    payload = _as_mapping(json.loads(payload_bytes), "lighting profile artifact")
    schema = str(payload.get("schema") or "")
    if schema == LIT_PROFILE_SCHEMA:
        return _load_lit_profile(payload, artifact_path, artifact_sha256)
    if schema == DBC_PROFILE_SCHEMA:
        return _load_dbc_profile(payload, artifact_path, artifact_sha256)
    raise ValueError(
        f"unsupported lighting profile schema {schema!r}; expected "
        f"{LIT_PROFILE_SCHEMA!r} or {DBC_PROFILE_SCHEMA!r}"
    )


def _load_lit_profile(
    payload: dict[str, Any], artifact_path: Path, artifact_sha256: str
) -> list[TerrainLightingSample]:
    source = _require_mapping(payload, "source")
    selection = _require_mapping(payload, "selection")
    layout = _require_mapping(payload, "lit")
    source_sha256 = _require_sha256(source.get("sha256"), "LIT source sha256")
    identifier = str(
        source.get("virtual_path")
        or source.get("path")
        or source.get("label")
        or artifact_path
    )
    samples: list[TerrainLightingSample] = []
    for raw_sample in _require_list(payload, "samples"):
        sample = _as_mapping(raw_sample, "LIT sample")
        game_time = _normalized_time(sample.get("normalized_time"), "normalized_time")
        direct = _lit_track_rgb(sample, "direct")
        ambient = _lit_track_rgb(sample, "ambient")
        fog = _lit_track_rgb(sample, "fog")
        compact_evidence = {
            "schema": LIT_PROFILE_SCHEMA,
            "artifact_path": str(artifact_path),
            "artifact_sha256": artifact_sha256,
            "source": source,
            "lit": layout,
            "selection": selection,
            "sample": sample,
        }
        samples.append(
            _client_color_sample(
                game_time=game_time,
                directional_color=direct,
                ambient_color=ambient,
                fog_color=fog,
                profile_revision="lit-global-clear-colors-v1",
                evidence_state="client_lit_colors_authored_direction_and_mcsh_strength",
                source_kind="client_lit_profile_artifact",
                source_identifier=identifier,
                source_sha256=source_sha256,
                artifact_sha256=artifact_sha256,
                evidence=compact_evidence,
            )
        )
    return _validate_external_samples(samples)


def _load_dbc_profile(
    payload: dict[str, Any], artifact_path: Path, artifact_sha256: str
) -> list[TerrainLightingSample]:
    source = _require_mapping(payload, "source")
    query = _require_mapping(payload, "query")
    tables = _require_list(source, "tables")
    if len(tables) != len(REQUIRED_DBC_TABLES):
        raise ValueError(
            f"Light DBC profile must bind exactly five tables, found {len(tables)}"
        )
    table_hashes: dict[str, str] = {}
    definition_hashes: dict[str, str] = {}
    for raw_table in tables:
        table = _as_mapping(raw_table, "Light DBC source table")
        name = str(table.get("table") or "")
        if not name or name in table_hashes:
            raise ValueError(f"invalid or duplicate Light DBC table name {name!r}")
        table_hashes[name] = _require_sha256(table.get("dbc_sha256"), f"{name} DBC sha256")
        definition_hashes[name] = _require_sha256(
            table.get("dbd_sha256"), f"{name} DBD sha256"
        )
    if set(table_hashes) != REQUIRED_DBC_TABLES:
        raise ValueError(
            "Light DBC profile table set mismatch: "
            f"expected {sorted(REQUIRED_DBC_TABLES)}, found {sorted(table_hashes)}"
        )
    source_sha256 = hashlib.sha256(
        json.dumps(table_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    exact_build = str(source.get("exact_build") or "")
    map_id = query.get("map_id")
    coordinate = query.get("coordinate")
    if not exact_build or map_id is None or not isinstance(coordinate, dict):
        raise ValueError("Light DBC profile is missing exact build/map/coordinate evidence")
    identifier = f"build={exact_build};map={map_id};world={coordinate.get('world_position')}"
    samples: list[TerrainLightingSample] = []
    for raw_sample in _require_list(payload, "samples"):
        sample = _as_mapping(raw_sample, "Light DBC sample")
        game_time = _normalized_time(
            sample.get("evaluated_normalized0_to1"), "evaluated_normalized0_to1"
        )
        raw_color_bands = _require_list(sample, "color_bands")
        bands: dict[int, dict[str, Any]] = {}
        for raw_band in raw_color_bands:
            band = _as_mapping(raw_band, "color band")
            try:
                index = int(band.get("index"))
            except (TypeError, ValueError) as exc:
                raise ValueError("Light DBC color-band index must be an integer") from exc
            if index in bands:
                raise ValueError(f"Light DBC sample has duplicate color band {index}")
            bands[index] = band
        if len(raw_color_bands) != 18 or set(bands) != set(range(18)):
            raise ValueError("Light DBC sample must contain exactly color bands 0..17")
        float_bands = _require_list(sample, "float_bands")
        float_band_indices: set[int] = set()
        for raw_band in float_bands:
            band = _as_mapping(raw_band, "float band")
            try:
                index = int(band.get("index"))
            except (TypeError, ValueError) as exc:
                raise ValueError("Light DBC float-band index must be an integer") from exc
            if index in float_band_indices:
                raise ValueError(f"Light DBC sample has duplicate float band {index}")
            float_band_indices.add(index)
        if len(float_bands) != 6 or float_band_indices != set(range(6)):
            raise ValueError("Light DBC sample must contain exactly float bands 0..5")
        direct = _dbc_band_rgb(bands, 0, "Direct")
        ambient = _dbc_band_rgb(bands, 1, "Ambient")
        fog = _dbc_band_rgb(bands, 7, "Fog")
        compact_evidence = {
            "schema": DBC_PROFILE_SCHEMA,
            "artifact_path": str(artifact_path),
            "artifact_sha256": artifact_sha256,
            "exact_build": exact_build,
            "map_id": map_id,
            "coordinate": coordinate,
            "table_sha256": table_hashes,
            "definition_sha256": definition_hashes,
            "requested_time": sample.get("requested_time"),
            "spatial_blend": sample.get("spatial_blend"),
            "primary_light_params": sample.get("primary_light_params"),
            "primary_skybox": sample.get("primary_skybox"),
            "selected_color_bands": [bands[index] for index in (0, 1, 7)],
            "float_bands": float_bands,
        }
        samples.append(
            _client_color_sample(
                game_time=game_time,
                directional_color=direct,
                ambient_color=ambient,
                fog_color=fog,
                profile_revision="light-dbc-exact-build-colors-v1",
                evidence_state="client_light_dbc_colors_authored_direction_and_mcsh_strength",
                source_kind="client_light_dbc_profile_artifact",
                source_identifier=identifier,
                source_sha256=source_sha256,
                artifact_sha256=artifact_sha256,
                evidence=compact_evidence,
            )
        )
    return _validate_external_samples(samples)


def _client_color_sample(
    *,
    game_time: float,
    directional_color: NDArray[np.float32],
    ambient_color: NDArray[np.float32],
    fog_color: NDArray[np.float32],
    profile_revision: str,
    evidence_state: str,
    source_kind: str,
    source_identifier: str,
    source_sha256: str,
    artifact_sha256: str,
    evidence: dict[str, Any],
) -> TerrainLightingSample:
    authored = evaluate_authored_day_night(game_time)
    return TerrainLightingSample(
        profile_revision=profile_revision,
        evidence_state=evidence_state,
        lighting_model=LIGHTING_MODEL,
        game_time=game_time,
        light_direction=authored.light_direction,
        directional_color=directional_color,
        directional_intensity=1.0,
        ambient_color=ambient_color,
        ambient_intensity=1.0,
        fog_color=fog_color,
        mcsh_shadow_strength=DEFAULT_MCSH_SHADOW_STRENGTH,
        color_source_kind=source_kind,
        color_source_identifier=source_identifier,
        color_source_sha256=source_sha256,
        profile_artifact_sha256=artifact_sha256,
        source_evidence_json=json.dumps(evidence, sort_keys=True, separators=(",", ":")),
    )


def _validate_external_samples(
    samples: list[TerrainLightingSample],
) -> list[TerrainLightingSample]:
    if not samples:
        raise ValueError("lighting profile artifact contains no samples")
    times = [round(sample.game_time, 9) for sample in samples]
    if len(times) != len(set(times)):
        raise ValueError("lighting profile artifact contains duplicate evaluated times")
    return samples


def _lit_track_rgb(sample: dict[str, Any], name: str) -> NDArray[np.float32]:
    track = _require_mapping(sample, name)
    if track.get("present") is not True:
        raise ValueError(f"LIT sample required track {name!r} is not present")
    return _rgb(_require_mapping(track, "rgb"), f"LIT {name}")


def _dbc_band_rgb(
    bands: dict[int, dict[str, Any]], index: int, expected_name: str
) -> NDArray[np.float32]:
    band = bands.get(index)
    if band is None or str(band.get("name")) != expected_name:
        raise ValueError(f"Light DBC sample is missing color band {index} ({expected_name})")
    return _rgb(_require_mapping(band, "rgb"), f"Light DBC {expected_name}")


def _rgb(value: dict[str, Any], label: str) -> NDArray[np.float32]:
    try:
        result = np.asarray([value["r"], value["g"], value["b"]], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} RGB is malformed") from exc
    if result.shape != (3,) or not np.isfinite(result).all() or np.any(result < 0.0):
        raise ValueError(f"{label} RGB must contain three finite non-negative values")
    return result


def _normalized_time(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not np.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError(f"{label} must be within 0..1")
    return result % 1.0


def _require_sha256(value: Any, label: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{label} must be a 64-character hexadecimal digest")
    return text


def _require_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    return _as_mapping(parent.get(key), key)


def _as_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_list(parent: dict[str, Any], key: str) -> list[Any]:
    value = parent.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be an array")
    return value


def _lerp(start: FloatArray, end: FloatArray, factor: float) -> NDArray[np.float32]:
    return np.asarray(start + ((end - start) * factor), dtype=np.float32)


def evaluate_authored_day_night(game_time: float) -> TerrainLightingSample:
    """Evaluate the authored fallback at a normalized game time, wrapping to [0, 1)."""
    wrapped_time = float(game_time - np.floor(game_time))
    sun_angle = wrapped_time * (2.0 * np.pi)
    sun_height = float(np.sin(sun_angle - (np.pi * 0.5)))
    sun_horizontal = float(np.cos(sun_angle - (np.pi * 0.5)))
    direction = np.asarray(
        [sun_horizontal * 0.5, 0.3, max(sun_height, 0.05)], dtype=np.float64
    )
    direction /= np.linalg.norm(direction)

    day_factor = max(0.0, sun_height)
    directional_color = _lerp(
        np.asarray([0.20, 0.20, 0.35]),
        np.asarray([0.80, 0.78, 0.70]),
        day_factor,
    )
    ambient_color = _lerp(
        np.asarray([0.25, 0.25, 0.35]),
        np.asarray([0.55, 0.55, 0.60]),
        day_factor,
    )
    fog_color = _lerp(
        np.asarray([0.08, 0.08, 0.15]),
        np.asarray([0.60, 0.70, 0.85]),
        day_factor,
    )

    return TerrainLightingSample(
        profile_revision=PROFILE_REVISION,
        evidence_state=EVIDENCE_STATE,
        lighting_model=LIGHTING_MODEL,
        game_time=wrapped_time,
        light_direction=direction.astype(np.float32),
        directional_color=directional_color,
        directional_intensity=1.0,
        ambient_color=ambient_color,
        ambient_intensity=1.0,
        fog_color=fog_color,
        mcsh_shadow_strength=DEFAULT_MCSH_SHADOW_STRENGTH,
    )


def grid_normals_to_renderer(grid_normals: FloatArray) -> NDArray[np.float32]:
    """Map ADT/grid-space (nx, ny, nz) to renderer space (-ny, -nx, nz)."""
    normals = np.asarray(grid_normals, dtype=np.float32)
    if normals.shape[-1:] != (3,):
        raise ValueError(f"grid_normals must end in 3 channels, got {normals.shape}")
    return np.stack(
        [-normals[..., 1], -normals[..., 0], normals[..., 2]], axis=-1
    ).astype(np.float32)


def shade_terrain(
    albedo_rgb: FloatArray,
    renderer_space_normals: FloatArray,
    mcsh_shadow: FloatArray,
    sample: TerrainLightingSample,
) -> NDArray[np.float32]:
    """Apply ambient + MCNR Lambert directional light with MCSH visibility modulation.

    ``mcsh_shadow`` uses 0 for lit and 1 for shadowed.  MCSH does not darken the
    ambient term.  Inputs may be a single RGB/normal or image-shaped arrays whose
    leading dimensions broadcast together.
    """
    albedo = np.asarray(albedo_rgb, dtype=np.float32)
    normals = np.asarray(renderer_space_normals, dtype=np.float32)
    shadow = np.asarray(mcsh_shadow, dtype=np.float32)
    if albedo.shape[-1:] != (3,):
        raise ValueError(f"albedo_rgb must end in 3 channels, got {albedo.shape}")
    if normals.shape[-1:] != (3,):
        raise ValueError(
            f"renderer_space_normals must end in 3 channels, got {normals.shape}"
        )

    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    unit_normals = np.divide(
        normals,
        lengths,
        out=np.broadcast_to(np.asarray([0.0, 0.0, 1.0], dtype=np.float32), normals.shape).copy(),
        where=lengths > 1e-12,
    )
    diffuse = np.maximum(0.0, unit_normals @ sample.light_direction)
    shadow = np.clip(shadow, 0.0, 1.0)
    visibility = 1.0 - (shadow * np.clip(sample.mcsh_shadow_strength, 0.0, 1.0))
    ambient = sample.ambient_color * sample.ambient_intensity
    directional = (
        sample.directional_color
        * sample.directional_intensity
        * diffuse[..., np.newaxis]
        * visibility[..., np.newaxis]
    )
    return np.maximum(0.0, albedo * (ambient + directional)).astype(np.float32)


def synthesize_authored_height_shadow(height_257: FloatArray) -> NDArray[np.float32]:
    """Bake a deterministic binary 256x256 height-ray shadow from known terrain.

    This is an authored data-augmentation seam, not a reconstruction of the client's
    terrain-shadow baking tool and not native 64x64-per-MCNK MCSH evidence.  The fixed
    direction makes the result independent of the time-of-day variant, like baked MCSH.
    """
    height = np.asarray(height_257, dtype=np.float32)
    if height.shape != (257, 257):
        raise ValueError(f"height_257 must be (257, 257), got {height.shape}")
    if not np.isfinite(height).all():
        raise ValueError("height_257 contains non-finite values")

    raster = height[:256, :256]
    direction = np.asarray(AUTHORED_MCSH_BAKE_DIRECTION, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    horizontal = float(np.linalg.norm(direction[:2]))
    unit_xy = direction[:2] / horizontal
    height_gain_per_meter = float(direction[2] / horizontal)
    meters_per_pixel = 533.33333 / 256.0
    shadow = np.zeros((256, 256), dtype=bool)
    visited_offsets: set[tuple[int, int]] = set()

    for step in range(1, 65):
        dx = int(round(float(unit_xy[0]) * step))
        dy = int(round(float(unit_xy[1]) * step))
        if (dx, dy) == (0, 0) or (dx, dy) in visited_offsets:
            continue
        visited_offsets.add((dx, dy))

        target_x0 = max(0, -dx)
        target_x1 = min(256, 256 - dx)
        target_y0 = max(0, -dy)
        target_y1 = min(256, 256 - dy)
        source_x0 = target_x0 + dx
        source_x1 = target_x1 + dx
        source_y0 = target_y0 + dy
        source_y1 = target_y1 + dy
        distance_meters = np.hypot(dx, dy) * meters_per_pixel
        ray_height = (
            raster[target_y0:target_y1, target_x0:target_x1]
            + distance_meters * height_gain_per_meter
        )
        occluded = (
            raster[source_y0:source_y1, source_x0:source_x1] > ray_height + 0.05
        )
        shadow[target_y0:target_y1, target_x0:target_x1] |= occluded

    return shadow.astype(np.float32)
