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
from harvester.v60.trestle_dataset import compute_photometric_normals
from harvester.v60.trestle_elevation_model import TrestleElevationUNet
from harvester.v60.trestle_wdl_synthesizer import TrestleWdlSynthesizer


def patch_existing_adt(
    template_path: Path,
    out_path: Path,
    h257: np.ndarray,
) -> None:
    """Patch MCVT heights in an existing monolithic/root ADT file with h257."""
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

    for off in mcnk_offsets:
        if off + 8 + 128 > total_len:
            continue
        header = data[off + 8 : off + 8 + 128]
        cx, cy = struct.unpack_from("<II", header, 4)
        if cx >= 16 or cy >= 16:
            continue
        ofs_mcvt = struct.unpack_from("<I", header, 0x14)[0]
        if ofs_mcvt == 0:
            continue

        # Extract 145 heights from h257
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
        struct.pack_into("<f", data, off + 8 + 0x70, pos_z)

        mcvt_data_off = off + 8 + ofs_mcvt
        for i in range(145):
            struct.pack_into("<f", data, mcvt_data_off + i * 4, raw145[i] - pos_z)

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
    parser.add_argument("--trestle-model", default="../output/models/trestle_elevation_v1.pt", help="Path to trained TrestleElevationUNet")
    parser.add_argument("--synthesized-wdl", default="../output/development_synthesized_wdl.npz", help="Path to synthesized WDL NPZ")
    parser.add_argument("--minimap-dir", default="../test_data/original_development/World/Textures/Minimap", help="Minimap tiles directory")
    parser.add_argument("--template-dir", default="../test_data/original_development/World/Maps/development", help="Template ADT directory")
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

    # 2. Load Synthesized WDL Lattice
    print(f"Loading Synthesized WDL Lattices from {args.synthesized_wdl}...")
    wdl_synth = TrestleWdlSynthesizer.load_npz(args.synthesized_wdl)

    # 3. Discover all minimap tiles for development
    mmap_dir = Path(args.minimap_dir)
    template_dir = Path(args.template_dir)
    mmap_files = sorted(list(mmap_dir.glob("development_*_*.png")))
    print(f"Discovered {len(mmap_files)} development minimap tiles.")

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

    for b_idx in range(0, len(tile_jobs), batch_size):
        b_slice = tile_jobs[b_idx : b_idx + batch_size]
        features_list = []
        trestle_list = []
        bounds_ref_list = []

        for tx, ty, mf in b_slice:
            img = Image.open(mf).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
            rgb = np.array(img, dtype=np.float32) / 255.0

            # WDL Trestle Prior (257x257 yards)
            h17 = wdl_synth.get_trestle_17(tx, ty)
            if h17 is not None and not np.all(np.abs(h17) < 1e-4):
                z_trestle = ndimage.zoom(h17, (257.0 / 17.0, 257.0 / 17.0), order=3)[:257, :257].astype(np.float32)
            else:
                z_trestle = np.zeros((257, 257), dtype=np.float32)

            normals = compute_photometric_normals(rgb)
            z_trestle_256 = ndimage.zoom(z_trestle, (256.0 / 257.0, 256.0 / 257.0), order=1)[:256, :256]
            norm_trestle = ((z_trestle_256 - 150.0) / 350.0).astype(np.float32)

            feat = np.concatenate([rgb, norm_trestle[..., None], normals], axis=-1)  # (256, 256, 6)
            features_list.append(feat)
            trestle_list.append(z_trestle)
            bounds_ref_list.append((float(np.min(z_trestle)), float(np.max(z_trestle))))

        # Tensor forward pass
        feat_tensor = torch.from_numpy(np.stack(features_list, axis=0)).permute(0, 3, 1, 2).float().to(device)

        with torch.no_grad():
            pred_delta, _ = model(feat_tensor)
            delta_np = pred_delta.squeeze(1).cpu().numpy()

        # Assemble final elevation and write ADT
        for i, (tx, ty, mf) in enumerate(b_slice):
            delta_257 = ndimage.zoom(delta_np[i], (257.0 / 256.0, 257.0 / 256.0), order=1)[:257, :257].astype(np.float32)
            z_final = trestle_list[i] + delta_257

            continent_elevation_map[ty, tx] = float(np.mean(z_final))
            present_coords.append((tx, ty))

            # ADT Destination: development_{tx}_{ty}.adt
            adt_out = out_dir / f"development_{tx}_{ty}.adt"
            template_adt = template_dir / f"development_{tx}_{ty}.adt"

            if template_adt.is_file():
                # Patch existing ADT preserving textures and MCNK sub-structures
                patch_existing_adt(template_adt, adt_out, z_final)
                # Copy companions if present
                for suffix in ["_obj0.adt", "_obj1.adt", "_tex0.adt", "_tex1.adt"]:
                    cand_comp = template_dir / f"development_{tx}_{ty}{suffix}"
                    if cand_comp.is_file():
                        comp_out = out_dir / f"development_{tx}_{ty}{suffix}"
                        comp_out.write_bytes(cand_comp.read_bytes())
            else:
                # Synthesize fresh monolithic LK v18 ADT
                build_synthetic_lk_adt(adt_out, z_final, "development", tx, ty)

            if args.export_objs and (tx % 8 == 0 and ty % 8 == 0):
                obj_out = out_dir / f"development_{tx}_{ty}.obj"
                export_heightmap_obj(z_final, obj_out)

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
