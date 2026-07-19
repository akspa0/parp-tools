"""Universal raster-to-relief preprocessing and deterministic terrain mesh contract.

This module deliberately owns no neural network. It defines the stable boundary around the Spec
114 universal relief model: any decodable RGB/RGBA/grayscale raster becomes one or more RGB model
tiles, tile relief predictions stitch back to complete source coverage, and a finite relief field
becomes a deterministic grid mesh with source-image UVs.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

SUPPORTED_INPUT_MODES = frozenset({"1", "L", "LA", "I", "I;16", "F", "P", "RGB", "RGBA"})


@dataclass(frozen=True)
class RasterTile:
    """One model-sized RGB tile and its location in the padded source raster."""

    index: int
    x: int
    y: int
    rgb_chw: np.ndarray


@dataclass(frozen=True)
class RasterTransform:
    """Complete reversible coverage transform from source raster to model tiles."""

    original_width: int
    original_height: int
    original_mode: str
    padded_width: int
    padded_height: int
    pad_left: int
    pad_top: int
    tile_size: int
    overlap: int
    alpha_background: tuple[int, int, int]
    tile_origins: tuple[tuple[int, int], ...]

    @property
    def pad_right(self) -> int:
        return self.padded_width - self.original_width - self.pad_left

    @property
    def pad_bottom(self) -> int:
        return self.padded_height - self.original_height - self.pad_top


@dataclass(frozen=True)
class PreparedRaster:
    """RGB source pixels, model tiles, and the coverage transform joining them."""

    source_rgb_hwc: np.ndarray
    tiles: tuple[RasterTile, ...]
    transform: RasterTransform


@dataclass(frozen=True)
class TerrainMesh:
    """Deterministic X/Z grid with Y relief, per-vertex normals, UVs, and triangles."""

    vertices: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    faces: np.ndarray
    grid_width: int
    grid_height: int
    extent_x: float
    extent_z: float
    vertical_scale: float
    vertical_offset: float


def _validate_tile_options(tile_size: int, overlap: int) -> None:
    if tile_size < 16:
        raise ValueError("tile_size must be at least 16 pixels")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be in [0, tile_size)")


def _normalize_scalar_channel(values: np.ndarray) -> np.ndarray:
    channel = np.asarray(values, dtype=np.float32)
    if not np.isfinite(channel).all():
        finite = channel[np.isfinite(channel)]
        fill = float(np.median(finite)) if finite.size else 0.0
        channel = np.nan_to_num(channel, nan=fill, posinf=fill, neginf=fill)
    low = float(channel.min())
    high = float(channel.max())
    if high - low <= 1e-12:
        return np.zeros_like(channel, dtype=np.float32)
    return ((channel - low) / (high - low)).astype(np.float32)


def raster_to_rgb(
    image: Image.Image,
    *,
    alpha_background: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Convert a supported Pillow image to finite uint8 RGB without changing its dimensions."""
    if image.width < 1 or image.height < 1:
        raise ValueError("source raster must have non-zero dimensions")
    if image.mode not in SUPPORTED_INPUT_MODES:
        raise ValueError(f"unsupported raster mode {image.mode!r}")
    if len(alpha_background) != 3 or any(v < 0 or v > 255 for v in alpha_background):
        raise ValueError("alpha_background must contain three uint8 values")

    if image.mode in {"RGBA", "LA", "P"} and ("transparency" in image.info or "A" in image.mode):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (*alpha_background, 255))
        rgb = Image.alpha_composite(background, rgba).convert("RGB")
        return np.asarray(rgb, dtype=np.uint8)

    if image.mode in {"I", "I;16", "F"}:
        scalar = _normalize_scalar_channel(np.asarray(image))
        gray = np.round(scalar * 255.0).astype(np.uint8)
        return np.repeat(gray[..., None], 3, axis=2)

    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_raster_rgb(
    path: str | Path,
    *,
    alpha_background: tuple[int, int, int] = (0, 0, 0),
) -> tuple[np.ndarray, str]:
    """Load one raster and return ``(rgb_hwc_uint8, original_mode)`` with useful errors."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"source raster does not exist: {source}")
    try:
        with Image.open(source) as image:
            image.load()
            original_mode = image.mode
            return raster_to_rgb(image, alpha_background=alpha_background), original_mode
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"source is not a decodable raster: {source}") from exc


def _axis_origins(length: int, tile_size: int, overlap: int) -> tuple[int, ...]:
    if length < tile_size:
        raise ValueError("padded axis must not be smaller than tile_size")
    if length == tile_size:
        return (0,)
    stride = tile_size - overlap
    origins = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if origins[-1] != last:
        origins.append(last)
    return tuple(origins)


def prepare_raster(
    image: Image.Image,
    *,
    tile_size: int = 224,
    overlap: int = 28,
    alpha_background: tuple[int, int, int] = (0, 0, 0),
) -> PreparedRaster:
    """Convert an arbitrary raster into full-coverage overlapping RGB model tiles.

    Small images are symmetrically edge-padded. Large images stay at native resolution and tile;
    they are never stretched to a square. Each tile is float32 CHW in ``[0,1]``.
    """
    _validate_tile_options(tile_size, overlap)
    source_rgb = raster_to_rgb(image, alpha_background=alpha_background)
    original_height, original_width = source_rgb.shape[:2]

    padded_width = max(original_width, tile_size)
    padded_height = max(original_height, tile_size)
    pad_left = (padded_width - original_width) // 2
    pad_right = padded_width - original_width - pad_left
    pad_top = (padded_height - original_height) // 2
    pad_bottom = padded_height - original_height - pad_top
    padded = np.pad(
        source_rgb,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="edge",
    )

    x_origins = _axis_origins(padded_width, tile_size, overlap)
    y_origins = _axis_origins(padded_height, tile_size, overlap)
    tile_origins = tuple((x, y) for y in y_origins for x in x_origins)
    tiles = []
    for index, (x, y) in enumerate(tile_origins):
        tile_hwc = padded[y : y + tile_size, x : x + tile_size]
        tile_chw = np.transpose(tile_hwc.astype(np.float32) / 255.0, (2, 0, 1))
        tiles.append(RasterTile(index=index, x=x, y=y, rgb_chw=tile_chw))

    transform = RasterTransform(
        original_width=original_width,
        original_height=original_height,
        original_mode=image.mode,
        padded_width=padded_width,
        padded_height=padded_height,
        pad_left=pad_left,
        pad_top=pad_top,
        tile_size=tile_size,
        overlap=overlap,
        alpha_background=alpha_background,
        tile_origins=tile_origins,
    )
    return PreparedRaster(source_rgb_hwc=source_rgb, tiles=tuple(tiles), transform=transform)


def _blend_window(tile_size: int, overlap: int) -> np.ndarray:
    if overlap == 0:
        return np.ones((tile_size, tile_size), dtype=np.float32)
    ramp_length = min(overlap, tile_size // 2)
    phase = np.linspace(0.0, np.pi / 2.0, ramp_length + 2, dtype=np.float32)[1:-1]
    ramp = np.sin(phase) ** 2
    axis = np.ones(tile_size, dtype=np.float32)
    axis[:ramp_length] = ramp
    axis[-ramp_length:] = ramp[::-1]
    return np.outer(axis, axis).astype(np.float32)


def stitch_relief(
    tile_predictions: Iterable[np.ndarray],
    transform: RasterTransform,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Blend model-tile relief predictions and crop them to exact source-image coverage."""
    predictions = tuple(np.asarray(prediction, dtype=np.float32) for prediction in tile_predictions)
    if len(predictions) != len(transform.tile_origins):
        raise ValueError(
            f"expected {len(transform.tile_origins)} tile predictions, got {len(predictions)}"
        )
    if not predictions:
        raise ValueError("at least one tile prediction is required")

    accumulated = np.zeros((transform.padded_height, transform.padded_width), dtype=np.float64)
    weights = np.zeros_like(accumulated)
    window = _blend_window(transform.tile_size, transform.overlap).astype(np.float64)
    for prediction, (x, y) in zip(predictions, transform.tile_origins, strict=True):
        if prediction.shape != (transform.tile_size, transform.tile_size):
            raise ValueError(
                f"tile prediction shape must be {(transform.tile_size, transform.tile_size)}, "
                f"got {prediction.shape}"
            )
        if not np.isfinite(prediction).all():
            raise ValueError("tile predictions must contain only finite values")
        accumulated[y : y + transform.tile_size, x : x + transform.tile_size] += prediction * window
        weights[y : y + transform.tile_size, x : x + transform.tile_size] += window

    if np.any(weights <= 0.0):
        raise ValueError("tile coverage left unweighted pixels")
    padded_relief = (accumulated / weights).astype(np.float32)
    top = transform.pad_top
    left = transform.pad_left
    relief = padded_relief[
        top : top + transform.original_height,
        left : left + transform.original_width,
    ]
    return normalize_relief(relief) if normalize else relief


def normalize_relief(relief: np.ndarray) -> np.ndarray:
    """Return finite float32 relief in ``[0,1]``; constant fields become stable zeros."""
    values = np.asarray(relief, dtype=np.float32)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("relief must be a non-empty 2D array")
    if not np.isfinite(values).all():
        raise ValueError("relief must contain only finite values")
    low = float(values.min())
    high = float(values.max())
    if high - low <= 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def build_terrain_mesh(
    relief: np.ndarray,
    *,
    extent_x: float = 1.0,
    extent_z: float | None = None,
    vertical_scale: float = 0.25,
    vertical_offset: float = 0.0,
) -> TerrainMesh:
    """Build a deterministic finite grid mesh whose UVs cover the complete relief/image."""
    normalized = normalize_relief(relief)
    grid_height, grid_width = normalized.shape
    if grid_width < 2 or grid_height < 2:
        raise ValueError("relief grid must be at least 2x2 to form a mesh")
    if not np.isfinite([extent_x, vertical_scale, vertical_offset]).all() or extent_x <= 0.0:
        raise ValueError("extent_x must be positive and all mesh scales must be finite")
    if extent_z is None:
        extent_z = extent_x * (grid_height - 1) / (grid_width - 1)
    if not np.isfinite(extent_z) or extent_z <= 0.0:
        raise ValueError("extent_z must be positive and finite")

    xs = np.linspace(0.0, extent_x, grid_width, dtype=np.float32)
    zs = np.linspace(0.0, extent_z, grid_height, dtype=np.float32)
    x_grid, z_grid = np.meshgrid(xs, zs)
    y_grid = normalized * np.float32(vertical_scale) + np.float32(vertical_offset)
    vertices = np.stack((x_grid, y_grid, z_grid), axis=-1).reshape(-1, 3)

    spacing_z = extent_z / (grid_height - 1)
    spacing_x = extent_x / (grid_width - 1)
    derivative_z, derivative_x = np.gradient(y_grid, spacing_z, spacing_x)
    normals_grid = np.stack((-derivative_x, np.ones_like(y_grid), -derivative_z), axis=-1)
    lengths = np.linalg.norm(normals_grid, axis=-1, keepdims=True)
    normals = (normals_grid / np.maximum(lengths, 1e-12)).astype(np.float32).reshape(-1, 3)

    us = np.linspace(0.0, 1.0, grid_width, dtype=np.float32)
    vs = np.linspace(1.0, 0.0, grid_height, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(us, vs)
    uvs = np.stack((u_grid, v_grid), axis=-1).reshape(-1, 2)

    faces = np.empty(((grid_height - 1) * (grid_width - 1) * 2, 3), dtype=np.int32)
    cursor = 0
    for row in range(grid_height - 1):
        row_start = row * grid_width
        next_start = (row + 1) * grid_width
        for column in range(grid_width - 1):
            top_left = row_start + column
            top_right = top_left + 1
            bottom_left = next_start + column
            bottom_right = bottom_left + 1
            faces[cursor] = (top_left, bottom_left, top_right)
            faces[cursor + 1] = (top_right, bottom_left, bottom_right)
            cursor += 2

    if not np.isfinite(vertices).all() or not np.isfinite(normals).all():
        raise ValueError("mesh construction produced non-finite values")
    return TerrainMesh(
        vertices=vertices.astype(np.float32),
        normals=normals,
        uvs=uvs.astype(np.float32),
        faces=faces,
        grid_width=grid_width,
        grid_height=grid_height,
        extent_x=float(extent_x),
        extent_z=float(extent_z),
        vertical_scale=float(vertical_scale),
        vertical_offset=float(vertical_offset),
    )


def write_obj(mesh: TerrainMesh, output: str | Path, *, texture_filename: str | None = None) -> Path:
    """Write a deterministic OBJ/MTL pair. Faces reference matching vertex/UV/normal indices."""
    output_path = Path(output)
    if output_path.suffix.lower() != ".obj":
        raise ValueError("OBJ output path must end with .obj")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    material_name = "source_image_material"
    mtl_path = output_path.with_suffix(".mtl")

    lines = []
    if texture_filename:
        lines.extend((f"mtllib {mtl_path.name}", f"usemtl {material_name}"))
    lines.extend(f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in mesh.vertices)
    lines.extend(f"vt {u:.9g} {v:.9g}" for u, v in mesh.uvs)
    lines.extend(f"vn {x:.9g} {y:.9g} {z:.9g}" for x, y, z in mesh.normals)
    for face in mesh.faces:
        one_based = face + 1
        lines.append("f " + " ".join(f"{index}/{index}/{index}" for index in one_based))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if texture_filename:
        mtl_path.write_text(
            f"newmtl {material_name}\nKd 1 1 1\nmap_Kd {texture_filename}\n",
            encoding="utf-8",
        )
    return output_path
