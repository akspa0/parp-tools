"""Authentic Development Ground-Truth Dataset Extractor (Spec 264).

Extracts authentic root ADTs, object placements (_obj0.adt), and matching
minimap PNGs directly from the non-museum developer assets under
`wow-viewer/test_data/original_development/World/Maps/development/`.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage, sparse
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components

TILE_WORLD_SIZE = 533.33333  # Yards per ADT tile (1600 / 3)
CHUNK_WORLD_SIZE = TILE_WORLD_SIZE / 16.0  # 33.33333 yards per chunk
MAP_ORIGIN = 32.0 * TILE_WORLD_SIZE  # 17066.666 yards


@dataclass
class M2Placement:
    name: str
    unique_id: int
    pos: Tuple[float, float, float]
    rot: Tuple[float, float, float]
    scale: float


@dataclass
class WmoPlacement:
    name: str
    unique_id: int
    pos: Tuple[float, float, float]
    rot: Tuple[float, float, float]
    bounds_min: Tuple[float, float, float]
    bounds_max: Tuple[float, float, float]
    pixel_box: Tuple[int, int, int, int]  # (min_x, min_y, max_x, max_y) on 256x256 grid


@dataclass
class DevelopmentTileData:
    tile_x: int
    tile_y: int
    height_257: np.ndarray  # Shape (257, 257) float32 in yards
    minimap_rgb: np.ndarray  # Shape (256, 256, 3) float32 in [0, 1]
    m2_placements: List[M2Placement] = field(default_factory=list)
    wmo_placements: List[WmoPlacement] = field(default_factory=list)
    building_mask: np.ndarray = field(default_factory=lambda: np.zeros((256, 256), dtype=bool))
    alpha_mask: Optional[np.ndarray] = None  # Shape (256, 256) float32 in [0, 1] texture splat alpha
    has_sculpted_terrain: bool = False

    @property
    def min_height(self) -> float:
        return float(np.min(self.height_257))

    @property
    def max_height(self) -> float:
        return float(np.max(self.height_257))

    @property
    def relief_span(self) -> float:
        return self.max_height - self.min_height


class DevelopmentGroundTruthExtractor:
    """Extractor for authentic development map tiles."""

    def __init__(
        self,
        maps_dir: Path | str,
        textures_dir: Optional[Path | str] = None,
    ):
        self.maps_dir = Path(maps_dir)
        if textures_dir is not None:
            self.textures_dir = Path(textures_dir)
        else:
            # Default location relative to maps_dir
            cand = self.maps_dir.parent.parent / "Textures" / "Minimap"
            self.textures_dir = cand if cand.is_dir() else self.maps_dir

    def extract_height_257(self, adt_path: Path | str) -> Tuple[np.ndarray, bool]:
        """Extract exact 257x257 elevation lattice (in yards) from root ADT MCNK chunks."""
        data = Path(adt_path).read_bytes()
        total_len = len(data)

        # Locate MCNK chunks
        pos = 0
        mcnk_offsets: List[int] = []
        while pos + 8 <= total_len:
            magic = data[pos : pos + 4][::-1]
            size = struct.unpack_from("<I", data, pos + 4)[0]
            if magic == b"MCNK":
                mcnk_offsets.append(pos)
            pos += 8 + size

        h257 = np.zeros((257, 257), dtype=np.float32)
        valid = np.zeros((257, 257), dtype=bool)
        has_non_zero = False

        for off in mcnk_offsets:
            if off + 8 + 128 > total_len:
                continue
            header = data[off + 8 : off + 8 + 128]
            cx, cy = struct.unpack_from("<II", header, 4)
            if cx >= 16 or cy >= 16:
                continue

            ofs_mcvt = struct.unpack_from("<I", header, 0x14)[0]
            pos_z = struct.unpack_from("<f", header, 0x70)[0]

            if ofs_mcvt == 0 or off + 8 + ofs_mcvt + 145 * 4 > total_len:
                continue

            raw_heights = np.frombuffer(
                data[off + 8 + ofs_mcvt : off + 8 + ofs_mcvt + 145 * 4],
                dtype=np.float32,
            )
            if len(raw_heights) != 145:
                continue

            if np.any(np.abs(raw_heights) > 1e-4) or abs(pos_z) > 1e-4:
                has_non_zero = True

            idx = 0
            for row in range(17):
                inner = (row & 1) != 0
                count = 8 if inner else 9
                for col in range(count):
                    lx = (col * 2 + 1) if inner else (col * 2)
                    ly = row
                    gx = cx * 16 + lx
                    gy = cy * 16 + ly
                    if gx < 257 and gy < 257:
                        h257[gy, gx] = raw_heights[idx] + pos_z
                        valid[gy, gx] = True
                    idx += 1

        # Interpolate unpopulated quincunx lattice nodes
        for y in range(257):
            for x in range(257):
                if not valid[y, x]:
                    nbrs = []
                    if x > 0 and valid[y, x - 1]:
                        nbrs.append(h257[y, x - 1])
                    if x < 256 and valid[y, x + 1]:
                        nbrs.append(h257[y, x + 1])
                    if y > 0 and valid[y - 1, x]:
                        nbrs.append(h257[y - 1, x])
                    if y < 256 and valid[y + 1, x]:
                        nbrs.append(h257[y + 1, x])
                    h257[y, x] = float(np.mean(nbrs)) if nbrs else 0.0

        return h257, has_non_zero

    def extract_objects(
        self,
        obj_path: Path | str,
        tile_x: int,
        tile_y: int,
        minimap_rgb: Optional[np.ndarray] = None,
    ) -> Tuple[List[M2Placement], List[WmoPlacement], np.ndarray]:
        """Extract M2 doodads and WMO buildings from _obj0.adt with tight shape segmentation."""
        p = Path(obj_path)
        if not p.is_file():
            return [], [], np.zeros((256, 256), dtype=bool)

        data = p.read_bytes()
        total_len = len(data)
        pos = 0

        m2_names: List[str] = []
        wmo_names: List[str] = []
        m2_placements: List[M2Placement] = []
        wmo_placements: List[WmoPlacement] = []
        building_mask = np.zeros((256, 256), dtype=bool)

        # Attempt to load minimap for local terrain background estimation if not provided
        if minimap_rgb is None:
            cand_img = self.textures_dir / f"development_{tile_x}_{tile_y}.png"
            if cand_img.is_file():
                try:
                    loaded = Image.open(cand_img).convert("RGB").resize((256, 256))
                    minimap_rgb = np.array(loaded, dtype=np.float32)
                except Exception:
                    minimap_rgb = None

        tile_origin_x = MAP_ORIGIN - tile_x * TILE_WORLD_SIZE
        tile_origin_y = MAP_ORIGIN - tile_y * TILE_WORLD_SIZE

        while pos + 8 <= total_len:
            magic = data[pos : pos + 4][::-1]
            size = struct.unpack_from("<I", data, pos + 4)[0]
            name = magic.decode("ascii", errors="replace")

            if name == "MMDX" and size > 0:
                raw_str = data[pos + 8 : pos + 8 + size]
                m2_names = [s.decode("ascii", errors="replace") for s in raw_str.split(b"\x00") if s]

            elif name == "MWMO" and size > 0:
                raw_str = data[pos + 8 : pos + 8 + size]
                wmo_names = [s.decode("ascii", errors="replace") for s in raw_str.split(b"\x00") if s]

            elif name == "MDDF" and size >= 36:
                count = size // 36
                for i in range(count):
                    e_off = pos + 8 + i * 36
                    name_id, uid = struct.unpack_from("<ii", data, e_off)
                    rx, rz, ry = struct.unpack_from("<fff", data, e_off + 8)
                    rot_x, rot_z, rot_y = struct.unpack_from("<fff", data, e_off + 20)
                    scale = struct.unpack_from("<H", data, e_off + 32)[0] / 1024.0

                    asset_name = m2_names[name_id] if 0 <= name_id < len(m2_names) else f"M2_{name_id}"
                    world_x = MAP_ORIGIN - ry
                    world_y = MAP_ORIGIN - rx
                    m2_placements.append(
                        M2Placement(
                            name=asset_name,
                            unique_id=uid,
                            pos=(world_x, world_y, rz),
                            rot=(rot_x, rot_y, rot_z),
                            scale=scale,
                        )
                    )
                    # Tight realistic footprint for M2 doodads (1px for small, 3x3 for large props)
                    px = int(round(((rx - tile_x * TILE_WORLD_SIZE) / TILE_WORLD_SIZE) * 256.0))
                    py = int(round(((ry - tile_y * TILE_WORLD_SIZE) / TILE_WORLD_SIZE) * 256.0))
                    if 0 <= px < 256 and 0 <= py < 256:
                        if scale > 1.2:
                            r_px = 1
                            building_mask[max(0, py - r_px) : min(256, py + r_px + 1), max(0, px - r_px) : min(256, px + r_px + 1)] = True
                        else:
                            building_mask[py, px] = True

            elif name == "MODF" and size >= 64:
                count = size // 64
                for i in range(count):
                    e_off = pos + 8 + i * 64
                    name_id, uid = struct.unpack_from("<ii", data, e_off)
                    rx, rz, ry = struct.unpack_from("<fff", data, e_off + 8)
                    rot_x, rot_z, rot_y = struct.unpack_from("<fff", data, e_off + 20)
                    bmin_x, bmin_z, bmin_y = struct.unpack_from("<fff", data, e_off + 32)
                    bmax_x, bmax_z, bmax_y = struct.unpack_from("<fff", data, e_off + 44)

                    asset_name = wmo_names[name_id] if 0 <= name_id < len(wmo_names) else f"WMO_{name_id}"
                    world_x = MAP_ORIGIN - ry
                    world_y = MAP_ORIGIN - rx

                    # Project placement center to 256x256 minimap pixel space
                    px = int(round(((rx - tile_x * TILE_WORLD_SIZE) / TILE_WORLD_SIZE) * 256.0))
                    py = int(round(((ry - tile_y * TILE_WORLD_SIZE) / TILE_WORLD_SIZE) * 256.0))

                    name_lower = asset_name.lower()

                    # Vegetation / terrain grass discriminator (minimap albedo)
                    if minimap_rgb is not None:
                        is_grass = (minimap_rgb[..., 1] > minimap_rgb[..., 0] + 10.0) & (
                            minimap_rgb[..., 1] > minimap_rgb[..., 2] + 12.0
                        )
                    else:
                        is_grass = np.zeros((256, 256), dtype=bool)

                    if 0 <= px < 256 and 0 <= py < 256:
                        if "wall" in name_lower:
                            # Wall segment: narrow corridor (~2 px wide, ~12 px long along rot_z)
                            angle = np.radians(rot_z)
                            for t in np.linspace(-6, 6, 13):
                                wx = int(round(px + t * np.cos(angle)))
                                wy = int(round(py + t * np.sin(angle)))
                                if 0 <= wx < 256 and 0 <= wy < 256:
                                    building_mask[max(0, wy - 1) : min(256, wy + 2), max(0, wx - 1) : min(256, wx + 2)] = True
                        elif "tower" in name_lower:
                            # Guard tower: circular footprint radius ~6-7 pixels, non-grass
                            Y, X = np.ogrid[:256, :256]
                            d = np.sqrt((X - px) ** 2 + (Y - py) ** 2)
                            building_mask |= (d <= 7.0) & (~is_grass)
                        elif any(tok in name_lower for tok in ("building", "largebuilding", "inn", "barracks", "house", "keep", "barn")):
                            # Large building: oriented bounding box (~22x32 pixels rotated along rot_z)
                            angle = np.radians(rot_z)
                            cos_a, sin_a = np.cos(angle), np.sin(angle)
                            Y, X = np.ogrid[:256, :256]
                            dx = X - px
                            dy = Y - py
                            lx = dx * cos_a + dy * sin_a
                            ly = -dx * sin_a + dy * cos_a
                            in_obb = (np.abs(lx) <= 22.0) & (np.abs(ly) <= 32.0)
                            building_mask |= in_obb & (~is_grass)
                        elif "gate" in name_lower:
                            Y, X = np.ogrid[:256, :256]
                            d = np.sqrt((X - px) ** 2 + (Y - py) ** 2)
                            building_mask |= (d <= 4.0) & (~is_grass)
                        else:
                            # Generic structure: tight 5px radius, non-grass
                            Y, X = np.ogrid[:256, :256]
                            d = np.sqrt((X - px) ** 2 + (Y - py) ** 2)
                            building_mask |= (d <= 5.0) & (~is_grass)

                    wmo_placements.append(
                        WmoPlacement(
                            name=asset_name,
                            unique_id=uid,
                            pos=(world_x, world_y, rz),
                            rot=(rot_x, rot_y, rot_z),
                            bounds_min=(bmin_x, bmin_y, bmin_z),
                            bounds_max=(bmax_x, bmax_y, bmax_z),
                            pixel_box=(max(0, px - 22), max(0, py - 22), min(255, px + 22), min(255, py + 22)),
                        )
                    )

            pos += 8 + size

        return m2_placements, wmo_placements, building_mask

    def extract_pm4_objects(
        self,
        pm4_path: Path | str,
        tile_x: int,
        tile_y: int,
    ) -> Tuple[List[WmoPlacement], np.ndarray]:
        """PM4 collision extraction disabled: PM4 collision data is not used for minimap object masking."""
        return [], np.zeros((256, 256), dtype=bool)

    def load_tile(self, tile_x: int, tile_y: int) -> Optional[DevelopmentTileData]:
        """Load paired ground truth and minimap for a development tile."""
        adt_path = self.maps_dir / f"development_{tile_x}_{tile_y}.adt"
        if not adt_path.is_file():
            return None

        # Load heightmap
        height_257, is_sculpted = self.extract_height_257(adt_path)

        # Load matching minimap PNG
        png_path = self.textures_dir / f"development_{tile_x}_{tile_y}.png"
        if png_path.is_file():
            img = Image.open(png_path).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
            minimap_rgb = np.asarray(img, dtype=np.float32) / 255.0
        else:
            minimap_rgb = np.zeros((256, 256, 3), dtype=np.float32)

        # Load object placements from _obj0.adt
        obj_path = self.maps_dir / f"development_{tile_x}_{tile_y}_obj0.adt"
        if obj_path.is_file():
            m2s, wmos, bmask = self.extract_objects(obj_path, tile_x, tile_y, minimap_rgb=minimap_rgb * 255.0)
        else:
            m2s, wmos, bmask = [], [], np.zeros((256, 256), dtype=bool)

        # Load texture alpha splats from _tex0.adt if available
        tex_path = self.maps_dir / f"development_{tile_x}_{tile_y}_tex0.adt"
        alpha_mask = self.extract_alpha_splats(tex_path)

        return DevelopmentTileData(
            tile_x=tile_x,
            tile_y=tile_y,
            height_257=height_257,
            minimap_rgb=minimap_rgb,
            m2_placements=m2s,
            wmo_placements=wmos,
            building_mask=bmask,
            alpha_mask=alpha_mask,
            has_sculpted_terrain=is_sculpted,
        )

    @staticmethod
    def _decode_mcal_alpha(data: bytes, offset: int, flags: int, length: int) -> np.ndarray:
        """Decode single layer MCAL alpha map (64x64) handling RLE compression, 4096 uncompressed, and 2048 4-bit."""
        amap = bytearray(4096)
        read_pos = offset
        end_pos = min(len(data), offset + length)
        is_compressed = bool(flags & 0x200)

        if is_compressed:
            write_pos = 0
            while write_pos < 4096 and read_pos < end_pos:
                header = data[read_pos]
                read_pos += 1
                count = header & 0x7F
                is_fill = bool(header & 0x80)
                if count == 0:
                    continue
                if write_pos + count > 4096:
                    count = 4096 - write_pos
                if is_fill:
                    val = data[read_pos] if read_pos < end_pos else 0
                    read_pos += 1
                    amap[write_pos : write_pos + count] = bytes([val]) * count
                    write_pos += count
                else:
                    take = min(count, end_pos - read_pos)
                    amap[write_pos : write_pos + take] = data[read_pos : read_pos + take]
                    read_pos += take
                    write_pos += count
        elif length >= 4096:
            take = min(4096, end_pos - read_pos)
            amap[:take] = data[read_pos : read_pos + take]
        elif length >= 2048:
            write_pos = 0
            for b in data[read_pos : min(end_pos, read_pos + 2048)]:
                low = (b & 0x0F) * 17
                high = ((b >> 4) & 0x0F) * 17
                amap[write_pos] = low
                amap[write_pos + 1] = high
                write_pos += 2
        return np.frombuffer(amap, dtype=np.uint8).reshape((64, 64))

    def extract_alpha_splats(self, tex0_path: Path | str) -> Optional[np.ndarray]:
        """Extract continuous 256x256 composite texture splat alpha map from _tex0.adt."""
        path = Path(tex0_path)
        if not path.is_file():
            return None

        data = path.read_bytes()
        total_len = len(data)
        pos = 0
        mcnk_offsets: List[Tuple[int, int]] = []
        while pos + 8 <= total_len:
            magic = data[pos : pos + 4][::-1]
            size = struct.unpack_from("<I", data, pos + 4)[0]
            if magic == b"MCNK":
                mcnk_offsets.append((pos + 8, size))
            pos += 8 + size

        if len(mcnk_offsets) < 256:
            return None

        from scipy import ndimage

        grid = np.zeros((1024, 1024), dtype=np.float32)
        has_any_alpha = False

        for cy in range(16):
            for cx in range(16):
                off, sz = mcnk_offsets[cy * 16 + cx]
                body = data[off : off + sz]
                sp = 0
                mcly_data: Optional[bytes] = None
                mcal_data: Optional[bytes] = None
                while sp + 8 <= len(body):
                    sm = body[sp : sp + 4][::-1]
                    ss = struct.unpack_from("<I", body, sp + 4)[0]
                    if sm == b"MCLY":
                        mcly_data = body[sp + 8 : sp + 8 + ss]
                    elif sm == b"MCAL":
                        mcal_data = body[sp + 8 : sp + 8 + ss]
                    sp += 8 + ss

                if mcly_data and mcal_data:
                    num_layers = len(mcly_data) // 16
                    chunk_blend = np.zeros((64, 64), dtype=np.float32)
                    for li in range(1, num_layers):
                        l_entry = mcly_data[li * 16 : (li + 1) * 16]
                        tex_id, flags, a_off, eff_id = struct.unpack_from("<IIII", l_entry, 0)
                        next_off = len(mcal_data)
                        if li + 1 < num_layers:
                            next_off = struct.unpack_from("<IIII", mcly_data[(li + 1) * 16 : (li + 2) * 16], 0)[2]
                        a_len = max(0, next_off - a_off)
                        a_map = self._decode_mcal_alpha(mcal_data, a_off, flags, a_len)
                        chunk_blend += (a_map.astype(np.float32) / 255.0)
                        has_any_alpha = True
                    grid[cy * 64 : (cy + 1) * 64, cx * 64 : (cx + 1) * 64] = np.clip(chunk_blend, 0.0, 1.0)

        if not has_any_alpha:
            return None

        # Downsample 1024x1024 composite alpha to 256x256 matching minimap grid
        alpha_256 = ndimage.zoom(grid, (256.0 / 1024.0, 256.0 / 1024.0), order=1)
        return np.clip(alpha_256, 0.0, 1.0).astype(np.float32)

    def is_tile_sculpted(self, adt_path: Path | str) -> bool:
        """Fast check whether an ADT has sculpted non-flat terrain without building full lattice."""
        data = Path(adt_path).read_bytes()
        pos = 0
        total_len = len(data)
        while pos + 8 <= total_len:
            magic = data[pos : pos + 4][::-1]
            size = struct.unpack_from("<I", data, pos + 4)[0]
            if magic == b"MCNK":
                if pos + 8 + 128 <= total_len:
                    header = data[pos + 8 : pos + 8 + 128]
                    ofs_mcvt = struct.unpack_from("<I", header, 0x14)[0]
                    pos_z = struct.unpack_from("<f", header, 0x70)[0]
                    if abs(pos_z) > 1e-3:
                        return True
                    if ofs_mcvt > 0 and pos + 8 + ofs_mcvt + 145 * 4 <= total_len:
                        heights = np.frombuffer(
                            data[pos + 8 + ofs_mcvt : pos + 8 + ofs_mcvt + 145 * 4],
                            dtype=np.float32,
                        )
                        if np.any(np.abs(heights) > 1e-3):
                            return True
            pos += 8 + size
        return False

    def scan_all_sculpted_tiles(self) -> List[Tuple[int, int]]:
        """Fast scan of all development tiles and return list of (tx, ty) for sculpted tiles."""
        results: List[Tuple[int, int]] = []
        for p in sorted(self.maps_dir.glob("development_*_*.adt")):
            # Ignore obj0 / tex0 files
            if "_obj" in p.name or "_tex" in p.name:
                continue
            parts = p.stem.split("_")
            if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                tx, ty = int(parts[1]), int(parts[2])
                if self.is_tile_sculpted(p):
                    results.append((tx, ty))
        return results
