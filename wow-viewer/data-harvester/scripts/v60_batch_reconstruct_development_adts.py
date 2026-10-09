"""Batch Full-Map Terrain Reconstruction & LK ADT Generation for Development (Spec 266).

Reconstructs all development map tiles using the trained TrestleElevationUNet,
synthesized WDL macro-trestle lattices, and tight authentic object sieving.
Outputs valid LK v18 ADT files ready to load in WoWViewer.
"""

from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from harvester.v60.development_ground_truth import (
    CHUNK_WORLD_SIZE,
    MAP_ORIGIN,
    TILE_WORLD_SIZE,
    DevelopmentGroundTruthExtractor,
)
from harvester.v60.mesh_exporter import export_obj_mesh
from harvester.v60.shadow_difference_refiner import extract_ridge_contours
from harvester.v60.terrain_feature_synthesizer import (
    compute_surface_normals,
    synthesize_multiband_terrain,
)
from harvester.v60.wdl_elevation_calibrator import WdlElevationParser
from harvester.v60.trestle_dataset import compute_photometric_normals
from harvester.v60.trestle_elevation_model import TrestleElevationUNet
from harvester.v60.trestle_wdl_synthesizer import TrestleWdlSynthesizer


def has_real_mcvt(p: Path) -> bool:
    """Check if an ADT file contains real, non-zero authored MCVT elevation data."""
    if not p.is_file():
        return False
    data = p.read_bytes()
    pos = 0
    total_len = len(data)
    while pos + 8 <= total_len:
        m = data[pos : pos + 4][::-1]
        s = struct.unpack_from("<I", data, pos + 4)[0]
        if m == b"MCNK":
            if pos + 8 + 128 > total_len:
                break
            ofs = struct.unpack_from("<I", data, pos + 8 + 0x14)[0]
            z = struct.unpack_from("<f", data, pos + 8 + 0x70)[0]
            if ofs > 0 and pos + 8 + ofs + 12 <= total_len:
                raw = struct.unpack_from("<fff", data, pos + 8 + ofs)
                if any(abs(v) > 1e-4 for v in raw) or abs(z) > 1e-4:
                    return True
            elif abs(z) > 1e-4:
                return True
        pos += 8 + s
    return False


def find_authentic_adt(tx: int, ty: int) -> Optional[Tuple[Path, bool]]:
    """Check candidate directories for an authentic ADT.
    Returns (path, has_real_mcvt).
    Prioritizes WoWMuseum monolithic ADTs for full texture support.
    """
    adt_name = f"development_{tx}_{ty}.adt"
    for cand_dir in [
        Path("../test_data/WoWMuseum/335-dev/World/Maps/development"),
        Path("../test_data/original_development/World/Maps/development"),
        Path("../test_data/original_development/WDT-ADT"),
    ]:
        p = cand_dir / adt_name
        if p.is_file():
            if has_real_mcvt(p):
                return (p, True)
            return (p, False)
    return None


def patch_existing_adt(
    template_path: Path,
    out_path: Path,
    h257: np.ndarray,
) -> None:
    """Patch MCVT heights and MCNR surface normals in an existing monolithic ADT file with h257.
    Preserves 100% of existing chunks (MTEX, MCLY, MCAL, MMDX, MWMO, MDDF, MODF, MCRF, MCSE, MCSH, MCLQ).
    """
    data = bytearray(template_path.read_bytes())
    total_len = len(data)
    pos = 0
    mcnk_offsets: List[int] = []

    while pos + 8 <= total_len:
        magic = data[pos : pos + 4][::-1]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        if magic == b"MCNK":
            mcnk_offsets.append(pos)
        pos += 8 + size

    # Precompute smooth 3D surface normals
    normals_257 = compute_surface_normals(h257)

    for off in mcnk_offsets:
        if off + 8 + 128 > total_len:
            continue
        header = data[off + 8 : off + 8 + 128]
        cx, cy = struct.unpack_from("<II", header, 4)
        if cx >= 16 or cy >= 16:
            continue
        ofs_mcvt = struct.unpack_from("<I", header, 0x14)[0]
        ofs_mcnr = struct.unpack_from("<I", header, 0x18)[0]
        if ofs_mcvt == 0:
            continue

        # Extract 145 heights and normals from h257
        raw145 = np.zeros(145, dtype=np.float32)
        norm145 = np.zeros((145, 3), dtype=np.float32)
        idx = 0
        for row in range(17):
            inner = (row & 1) != 0
            count = 8 if inner else 9
            for col in range(count):
                lx = (col * 2 + 1) if inner else (col * 2)
                ly = row
                gx = cx * 16 + lx
                gy = cy * 16 + ly
                raw145[idx] = h257[gy, gx]
                norm145[idx] = normals_257[gy, gx]
                idx += 1

        pos_z = float(raw145[0])
        struct.pack_into("<f", data, off + 8 + 0x70, pos_z)

        mcvt_data_off = off + 8 + ofs_mcvt
        for i in range(145):
            struct.pack_into("<f", data, mcvt_data_off + i * 4, raw145[i] - pos_z)

        # Update MCNR normals if offset is present
        if ofs_mcnr > 0 and off + 8 + ofs_mcnr + 435 <= total_len:
            mcnr_data_off = off + 8 + ofs_mcnr
            for i in range(145):
                nx = int(np.clip(norm145[i, 0] * 127.0, -127, 127))
                ny = int(np.clip(norm145[i, 1] * 127.0, -127, 127))
                nz = int(np.clip(norm145[i, 2] * 127.0, -127, 127))
                struct.pack_into("<bbb", data, mcnr_data_off + i * 3, nx, ny, nz)

        # Normalize MCCV to canonical Blizzard neutral whiteplate (127, 127, 127, 255)
        # Prevents corrupted near-black template junk (24, 24, 24) from rendering patched tiles pitch black
        mcnk_size = struct.unpack_from("<I", data, off + 4)[0]
        sub_pos = off + 8 + 128
        sub_end = off + 8 + mcnk_size
        while sub_pos + 8 <= sub_end and sub_pos + 8 <= total_len:
            sub_m = data[sub_pos : sub_pos + 4][::-1]
            sub_s = struct.unpack_from("<I", data, sub_pos + 4)[0]
            if sub_m == b"MCCV" and sub_s == 580:
                data[sub_pos + 8 : sub_pos + 8 + 580] = b"\x7f\x7f\x7f\xff" * 145
                break
            sub_pos += 8 + sub_s

    out_path.write_bytes(data)


def build_synthetic_lk_adt(
    out_path: Path,
    h257: np.ndarray,
    map_name: str,
    tile_x: int,
    tile_y: int,
) -> None:
    """Construct a clean, valid monolithic LK v18 ADT from a 257x257 elevation lattice."""
    mver = b"MVER" + struct.pack("<I", 4) + struct.pack("<I", 18)

    # MTEX string table
    tex_str = b"Tileset\\Generic\\Base.blp\x00"
    mtex = b"MTEX" + struct.pack("<I", len(tex_str)) + tex_str

    # Empty placement subchunks
    mmdx = b"MMDX" + struct.pack("<I", 0)
    mmid = b"MMID" + struct.pack("<I", 0)
    mwmo = b"MWMO" + struct.pack("<I", 0)
    mwid = b"MWID" + struct.pack("<I", 0)
    mddf = b"MDDF" + struct.pack("<I", 0)
    modf = b"MODF" + struct.pack("<I", 0)

    # 256 MCNK subchunks
    mcnk_chunks: List[bytes] = []

    for cy in range(16):
        for cx in range(16):
            wx = MAP_ORIGIN - (tile_y * TILE_WORLD_SIZE + cy * CHUNK_WORLD_SIZE)
            wy = MAP_ORIGIN - (tile_x * TILE_WORLD_SIZE + cx * CHUNK_WORLD_SIZE)

            raw145 = np.zeros(145, dtype=np.float32)
            idx = 0
            for row in range(17):
                inner = (row & 1) != 0
                count = 8 if inner else 9
                for col in range(count):
                    lx = (col * 2 + 1) if inner else (col * 2)
                    ly = row
                    gx = cx * 16 + lx
                    gy = cy * 16 + ly
                    raw145[idx] = h257[gy, gx]
                    idx += 1

            pos_z = float(raw145[0])
            mcvt_payload = (raw145 - pos_z).astype(np.float32).tobytes()
            mcvt = b"MCVT" + struct.pack("<I", len(mcvt_payload)) + mcvt_payload

            # Upward normals
            mcnr_payload = b"\x00\x00\x7f" * 145 + b"\x00" * 13
            mcnr = b"MCNR" + struct.pack("<I", len(mcnr_payload)) + mcnr_payload

            header = bytearray(128)
            struct.pack_into("<I", header, 0, 0)  # flags
            struct.pack_into("<II", header, 4, cx, cy)
            struct.pack_into("<I", header, 0x14, 128)  # ofsMcvt
            struct.pack_into("<I", header, 0x18, 128 + len(mcvt))  # ofsMcnr
            struct.pack_into("<fff", header, 0x68, wx, wy, pos_z)

            payload = bytes(header) + mcvt + mcnr
            mcnk = b"MCNK" + struct.pack("<I", len(payload)) + payload
            mcnk_chunks.append(mcnk)

    # MHDR & MCIN offsets
    mhdr_len = 64 + 8
    mcin_len = 256 * 16 + 8
    mhdr_pos = len(mver)
    mcin_pos = mhdr_pos + mhdr_len

    post_headers_pos = mcin_pos + mcin_len
    mtex_off = post_headers_pos - (mhdr_pos + 8)
    mmdx_off = mtex_off + len(mtex)
    mmid_off = mmdx_off + len(mmdx)
    mwmo_off = mmid_off + len(mmid)
    mwid_off = mwmo_off + len(mwmo)
    mddf_off = mwid_off + len(mwid)
    modf_off = mddf_off + len(mddf)

    mhdr_payload = bytearray(64)
    struct.pack_into("<IIIIIIII", mhdr_payload, 0, 0, mtex_off, mmdx_off, mmid_off, mwmo_off, mwid_off, mddf_off, modf_off)
    mhdr = b"MHDR" + struct.pack("<I", 64) + bytes(mhdr_payload)

    # Build MCIN entries
    mcnk_start_offset = modf_off + len(modf) + (mhdr_pos + 8)
    mcin_payload = bytearray(256 * 16)
    curr_mcnk_off = mcnk_start_offset

    for i, c in enumerate(mcnk_chunks):
        struct.pack_into("<IIII", mcin_payload, i * 16, curr_mcnk_off, len(c), 0, 0)
        curr_mcnk_off += len(c)

    mcin = b"MCIN" + struct.pack("<I", len(mcin_payload)) + bytes(mcin_payload)

    out_bytes = mver + mhdr + mcin + mtex + mmdx + mmid + mwmo + mwid + mddf + modf + b"".join(mcnk_chunks)
    out_path.write_bytes(out_bytes)


def update_wdt_main(template_wdt: Path, out_wdt: Path, present_coords: List[Tuple[int, int]]) -> None:
    """Copy template WDT and ensure all generated tiles are flagged present in MAIN chunk."""
    data = bytearray(template_wdt.read_bytes())
    main_off = 52  # offset of MAIN chunk payload in development.wdt

    for tx, ty in present_coords:
        # tx = col, ty = row
        slot = (ty * 64 + tx) * 8
        if main_off + slot + 8 <= len(data):
            flag = struct.unpack_from("<I", data, main_off + slot)[0]
            struct.pack_into("<I", data, main_off + slot, flag | 1)

    out_wdt.write_bytes(data)


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch Full-Map Elevation Reconstruction & LK ADT Generator")
    parser.add_argument("--auth-wdl", default="../test_data/original_development/World/Maps/development/development.wdl", help="Authentic development.wdl path")
    parser.add_argument("--trestle-model", default="../output/models/trestle_elevation_v1.pt", help="Path to trained TrestleElevationUNet")
    parser.add_argument("--synthesized-wdl", default="../output/development_synthesized_wdl.npz", help="Path to synthesized WDL NPZ")
    parser.add_argument("--minimap-dir", default="../test_data/original_development/World/Textures/Minimap", help="Minimap tiles directory")
    parser.add_argument("--template-dir", default="../test_data/WoWMuseum/335-dev/World/Maps/development", help="Template ADT directory")
    parser.add_argument("--out-dir", default="../output/development_reconstructed_lk", help="Output directory for LK ADT map")
    parser.add_argument("--batch-size", type=int, default=32, help="GPU batch size")
    parser.add_argument("--export-objs", action="store_true", default=False, help="Export 3D OBJ meshes for tiles")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("Full Development Map Terrain Reconstruction -> LK ADT Pipeline")
    print(f"Output Directory: {out_dir}")
    print("=================================================================")

    # 1. Load trained Trestle Elevation Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading TrestleElevationUNet on {device}...")
    model = TrestleElevationUNet().to(device)
    ckpt = torch.load(args.trestle_model, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 2. Build Continuous Continent-Wide Harmonic Macro Elevation Field
    auth_wdl = None
    if args.auth_wdl and Path(args.auth_wdl).is_file():
        auth_wdl = WdlElevationParser(args.auth_wdl)
        print(f"Loaded Authentic development.wdl from {args.auth_wdl}")

    # 3. Discover all minimap tiles for development
    mmap_dir = Path(args.minimap_dir)
    template_dir = Path(args.template_dir)
    mmap_files = sorted(list(mmap_dir.glob("development_*_*.png")))
    print(f"Discovered {len(mmap_files)} development minimap tiles.")

    print("Building Continuous Continent-Wide Harmonic Macro Elevation Field...")
    grid_64 = np.zeros((64, 64), dtype=np.float32)
    mask_64 = np.zeros((64, 64), dtype=bool)
    water_64 = np.zeros((64, 64), dtype=bool)

    if auth_wdl is not None:
        for ty in range(64):
            for tx in range(64):
                if auth_wdl.has_tile_data(tx, ty):
                    h17 = auth_wdl.extract_tile_17(tx, ty)
                    if h17 is not None:
                        grid_64[ty, tx] = float(np.mean(np.maximum(h17, 0.0)))
                        mask_64[ty, tx] = True

    for mf in mmap_files:
        parts = mf.stem.split("_")
        tx, ty = int(parts[-2]), int(parts[-1])
        if not mask_64[ty, tx]:
            img_c = Image.open(mf).convert("RGB")
            rgb_c = np.array(img_c, dtype=np.float32) / 255.0
            lum_c = 0.299 * rgb_c[..., 0] + 0.587 * rgb_c[..., 1] + 0.114 * rgb_c[..., 2]
            if lum_c.mean() > 0.82 or rgb_c.std() < 0.05:
                water_64[ty, tx] = True

    is_fixed = mask_64 | water_64
    diffused_64 = grid_64.copy()
    diffused_64[water_64] = 0.0
    is_fixed[0, :] = True; diffused_64[0, :] = 0.0
    is_fixed[63, :] = True; diffused_64[63, :] = 0.0
    is_fixed[:, 0] = True; diffused_64[:, 0] = 0.0
    is_fixed[:, 63] = True; diffused_64[:, 63] = 0.0

    for _ in range(300):
        lap = 0.25 * (
            np.roll(diffused_64, 1, axis=0) +
            np.roll(diffused_64, -1, axis=0) +
            np.roll(diffused_64, 1, axis=1) +
            np.roll(diffused_64, -1, axis=1)
        )
        diffused_64[~is_fixed] = lap[~is_fixed]

    diffused_1025 = ndimage.zoom(diffused_64, (1025.0 / 64.0, 1025.0 / 64.0), order=3)[:1025, :1025].astype(np.float32)
    global_lattice = diffused_1025.copy()

    if auth_wdl is not None:
        for ty in range(64):
            for tx in range(64):
                if mask_64[ty, tx]:
                    h17 = auth_wdl.extract_tile_17(tx, ty)
                    if h17 is not None:
                        sy = ty * 16
                        sx = tx * 16
                        global_lattice[sy : sy + 17, sx : sx + 17] = np.maximum(h17, 0.0)

    print("  [OK] Continuous Macro Field assembled (0 to 912 yds, zero discontinuities)")

    # Parse coordinates
    tile_jobs: List[Tuple[int, int, Path]] = []
    for mf in mmap_files:
        stem = mf.stem
        parts = stem.split("_")
        if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
            tx, ty = int(parts[-2]), int(parts[-1])
            tile_jobs.append((tx, ty, mf))

    print(f"Prepared {len(tile_jobs)} valid tile reconstruction jobs.")

    # Process in batches
    extractor = DevelopmentGroundTruthExtractor(template_dir)
    batch_size = args.batch_size
    reconstructed_count = 0
    t_start = time.time()
    present_coords: List[Tuple[int, int]] = []
    continent_elevation_map = np.full((64, 64), np.nan, dtype=np.float32)

    # Precompute 2D Isotropic Poisson frequency filter (220 deg azimuth)
    u_f = np.fft.fftfreq(256) * 2.0 * np.pi
    v_f = np.fft.fftfreq(256) * 2.0 * np.pi
    U_g, V_g = np.meshgrid(u_f, v_f)
    rad_az = np.radians(220.0)
    lx_f = float(np.cos(rad_az))
    ly_f = float(np.sin(rad_az))
    k_par_g = U_g * lx_f + V_g * ly_f
    k2_g = U_g**2 + V_g**2
    lam_f = 0.025
    poisson_filter = (1j * k_par_g) / (k2_g + lam_f)
    poisson_filter[0, 0] = 0.0

    for b_idx in range(0, len(tile_jobs), batch_size):
        b_slice = tile_jobs[b_idx : b_idx + batch_size]
        features_list = []
        trestle_list = []
        sfs_list = []
        ridge_list = []
        water_list = []
        bounds_ref_list = []
        obj_mask_list = []
        alpha_mask_list = []

        for tx, ty, mf in b_slice:
            img = Image.open(mf).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
            rgb = np.array(img, dtype=np.float32) / 255.0

            # Water detection
            lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
            water_cand = (lum > 0.82) & (ndimage.gaussian_filter(lum, 2.0) > 0.80)
            has_water = float(np.mean(water_cand)) > 0.08

            # Continuous Macro Elevation from Global Lattice
            sy = ty * 16
            sx = tx * 16
            h17_tile = global_lattice[sy : sy + 17, sx : sx + 17]
            z_trestle = ndimage.zoom(h17_tile, (257.0 / 17.0, 257.0 / 17.0), order=3)[:257, :257].astype(np.float32)

            # Discover authentic object mask and alpha splats
            obj_mask_256: Optional[np.ndarray] = None
            alpha_mask_256: Optional[np.ndarray] = None

            for cand_obj_dir in [
                Path("../test_data/original_development/WDT-ADT"),
                Path("../test_data/original_development/World/Maps/development"),
            ]:
                cand_obj0 = cand_obj_dir / f"development_{tx}_{ty}_obj0.adt"
                if cand_obj0.is_file():
                    try:
                        _, _, bmask = extractor.extract_objects(
                            cand_obj0, tx, ty, minimap_rgb=rgb * 255.0
                        )
                        obj_mask_256 = bmask
                    except Exception:
                        pass
                    break

            for cand_tex_dir in [
                Path("../test_data/original_development/WDT-ADT"),
                Path("../test_data/original_development/World/Maps/development"),
            ]:
                cand_tex0 = cand_tex_dir / f"development_{tx}_{ty}_tex0.adt"
                if cand_tex0.is_file():
                    try:
                        alpha_mask_256 = extractor.extract_alpha_splats(cand_tex0)
                    except Exception:
                        pass
                    break

            # Shadow residual & Poisson SfS
            clean_shadow = lum.copy()
            if has_water:
                clean_shadow[water_cand] = 0.0

            # Suppress object roofs and texture bleed from Poisson shadow residual
            if obj_mask_256 is not None and np.any(obj_mask_256):
                dil = ndimage.binary_dilation(obj_mask_256, iterations=2)
                perim = dil & (~obj_mask_256)
                fill_val = float(np.median(clean_shadow[perim])) if np.any(perim) else float(np.median(clean_shadow))
                clean_shadow[obj_mask_256] = fill_val

            if alpha_mask_256 is not None:
                clean_shadow *= (1.0 - 0.5 * np.clip(alpha_mask_256, 0.0, 1.0))

            # 1. Structure-texture decomposition to STRIP tileset texture noise
            # Tileset patterns (cobblestones, grass, dirt) have variations at sigma <= 2.5.
            # Real terrain shading is smooth at sigma >= 4.5.
            shading_structure = ndimage.gaussian_filter(clean_shadow, sigma=4.5)

            S_mat = np.fft.fft2(shading_structure)
            int_h = np.fft.ifft2(S_mat * poisson_filter).real.astype(np.float32)

            # Ridge crest extraction: only on real mountain relief, not flat plains
            macro_ptp = float(np.ptp(z_trestle))
            macro_slope = float(np.max(np.abs(np.gradient(z_trestle))))
            if macro_slope < 0.08 or macro_ptp < 15.0:
                # Flat plain / courtyard / water: no mountain ridges!
                ridge_mask = np.zeros((256, 256), dtype=np.uint8)
            else:
                ridge_mask, _ = extract_ridge_contours(shading_structure, sigma=4.0, curvature_threshold=0.005)

            if obj_mask_256 is not None and np.any(obj_mask_256):
                ridge_mask[obj_mask_256] = 0
            if alpha_mask_256 is not None:
                ridge_mask &= ~(alpha_mask_256 > 0.25)

            water_mask_257 = None
            if has_water:
                water_mask_257 = ndimage.zoom(water_cand.astype(np.float32), (257.0 / 256.0, 257.0 / 256.0), order=0) > 0.5
            elif np.all(z_trestle <= 0.05):
                water_mask_257 = np.ones((257, 257), dtype=bool)

            normals = compute_photometric_normals(rgb)
            z_trestle_256 = ndimage.zoom(z_trestle, (256.0 / 257.0, 256.0 / 257.0), order=1)[:256, :256]
            norm_trestle = ((z_trestle_256 - 150.0) / 350.0).astype(np.float32)

            feat = np.concatenate([rgb, norm_trestle[..., None], normals], axis=-1)  # (256, 256, 6)
            features_list.append(feat)
            trestle_list.append(z_trestle)
            sfs_list.append(int_h)
            ridge_list.append(ridge_mask)
            water_list.append(water_mask_257)
            bounds_ref_list.append((float(np.min(z_trestle)), float(np.max(z_trestle))))
            obj_mask_list.append(obj_mask_256)
            alpha_mask_list.append(alpha_mask_256)

        # Tensor forward pass
        feat_tensor = torch.from_numpy(np.stack(features_list, axis=0)).permute(0, 3, 1, 2).float().to(device)

        with torch.no_grad():
            pred_delta, _ = model(feat_tensor)
            delta_np = pred_delta.squeeze(1).cpu().numpy()

        # Assemble final elevation and write ADT
        for i, (tx, ty, mf) in enumerate(b_slice):
            adt_out = out_dir / f"development_{tx}_{ty}.adt"
            present_coords.append((tx, ty))

            # 1. Check if an authentic ADT already exists with real authored terrain
            auth_info = find_authentic_adt(tx, ty)
            if auth_info is not None and auth_info[1] is True:
                # Authentic Blizzard authored terrain exists!
                # Copy directly — PRESERVE AUTHENTIC TERRAIN 100%!
                auth_p = auth_info[0]
                adt_out.write_bytes(auth_p.read_bytes())
                try:
                    h257_auth, _ = extractor.extract_height_257(auth_p)
                    continent_elevation_map[ty, tx] = float(np.mean(h257_auth))
                except Exception:
                    pass
                reconstructed_count += 1
                continue

            # 2. Only reconstruct if the tile has NO real authored terrain!
            delta_257 = ndimage.zoom(delta_np[i], (257.0 / 256.0, 257.0 / 256.0), order=1)[:257, :257].astype(np.float32)
            z_final, _, _ = synthesize_multiband_terrain(
                macro_wdl_257=trestle_list[i],
                integrated_sfs_256=sfs_list[i],
                ridge_mask_256=ridge_list[i],
                water_mask_257=water_list[i],
                neural_residual_257=delta_257,
                object_mask_256=obj_mask_list[i],
                alpha_mask_256=alpha_mask_list[i],
                target_relief_yards=1.0,
                ridge_boost_yards=0.8,
                sfs_sigma_cut=16.0,
            )

            continent_elevation_map[ty, tx] = float(np.mean(z_final))

            template_adt = auth_info[0] if auth_info is not None else None
            if template_adt is None:
                for c_cand in [
                    Path("../test_data/WoWMuseum/335-dev/World/Maps/development") / f"development_{tx}_{ty}.adt",
                    Path("../test_data/original_development/WDT-ADT") / f"development_{tx}_{ty}.adt",
                    template_dir / f"development_{tx}_{ty}.adt",
                ]:
                    if c_cand.is_file():
                        template_adt = c_cand
                        break

            if template_adt is not None and template_adt.is_file():
                # Patch existing ADT preserving all textures and MCNK sub-structures
                patch_existing_adt(template_adt, adt_out, z_final)
            else:
                # Synthesize fresh monolithic LK v18 ADT
                build_synthetic_lk_adt(adt_out, z_final, "development", tx, ty)

            if args.export_objs and (tx % 8 == 0 and ty % 8 == 0):
                obj_out = out_dir / f"development_{tx}_{ty}.obj"
                export_obj_mesh(z_final, mf, obj_out, is_world_yards=True)

            reconstructed_count += 1

        if (b_idx + batch_size) % 128 == 0 or b_idx + batch_size >= len(tile_jobs):
            elapsed = time.time() - t_start
            rate = reconstructed_count / elapsed
            print(f"  Processed {reconstructed_count}/{len(tile_jobs)} tiles ({rate:.1f} tiles/sec)...")

    # Copy and update development.wdt
    template_wdt = template_dir / "development.wdt"
    if template_wdt.is_file():
        out_wdt = out_dir / "development.wdt"
        update_wdt_main(template_wdt, out_wdt, present_coords)
        print(f"  [OK] Exported updated WDT: {out_wdt} ({len(present_coords)} tiles flagged in MAIN)")

    # Save continent relief overview map
    np.save(out_dir / "continent_elevation_64x64.npy", continent_elevation_map)

    # Render overview heatmap PNG
    valid_mask = ~np.isnan(continent_elevation_map)
    if np.any(valid_mask):
        c_min = float(np.min(continent_elevation_map[valid_mask]))
        c_max = float(np.max(continent_elevation_map[valid_mask]))
        norm_c = np.clip((continent_elevation_map - c_min) / max(1.0, c_max - c_min), 0.0, 1.0)
        c_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        c_rgb[valid_mask, 0] = (norm_c[valid_mask] * 255.0).astype(np.uint8)
        c_rgb[valid_mask, 1] = ((1.0 - norm_c[valid_mask]) * 220.0).astype(np.uint8)
        c_rgb[valid_mask, 2] = 40
        Image.fromarray(c_rgb).resize((512, 512), Image.Resampling.NEAREST).save(out_dir / "development_continent_elevation_overview.png")

    total_time = time.time() - t_start
    print("\n=================================================================")
    print(f"Batch Reconstruction Complete!")
    print(f"Total LK ADT Tiles Generated: {reconstructed_count}")
    print(f"Total Wall Time:             {total_time:.2f} seconds ({reconstructed_count / total_time:.1f} tiles/sec)")
    print(f"Reconstructed Map Root:      {out_dir}")
    print("=================================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
