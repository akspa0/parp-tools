"""Patch V20 signals directly into existing Zarr datasets in-place.

Appends:
- liquid_type_256: broadcasted 16x16 liquid flags masked by liquid_mask.
- ground_intent_height_257: heightmap inpainted under structures.
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import zarr
from scipy.interpolate import griddata

# Add src to python path
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


from harvester.patch_utils import process_single_tile



def main() -> None:
    parser = argparse.ArgumentParser(description="Patch V20 signals directly into Zarr stores in-place.")
    parser.add_argument("--dataset-dir", type=str, default="wow-viewer/output/datasets/v18")
    parser.add_argument("--builds", nargs="*", default=["0_5_3_3368", "3_3_5_12340"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    dataset_base = Path(args.dataset_dir)
    if not dataset_base.exists():
        # Let's resolve robustly relative to the script file location
        script_dir = Path(__file__).resolve().parent
        # Try to find wow-viewer root (parent of data-harvester)
        wow_viewer_root = None
        for p in [script_dir] + list(script_dir.parents):
            if p.name == "wow-viewer":
                wow_viewer_root = p
                break
            elif (p / "wow-viewer").exists():
                wow_viewer_root = p / "wow-viewer"
                break
        
        resolved = False
        if wow_viewer_root is not None:
            # Try relative to wow-viewer root directly
            p_alt = wow_viewer_root / args.dataset_dir
            if p_alt.exists():
                dataset_base = p_alt
                resolved = True
            else:
                # Try stripping 'wow-viewer/' prefix from args.dataset_dir if present
                clean_dir = args.dataset_dir
                if clean_dir.startswith("wow-viewer/"):
                    clean_dir = clean_dir[len("wow-viewer/"):]
                elif clean_dir.startswith("wow-viewer\\"):
                    clean_dir = clean_dir[len("wow-viewer\\"):]
                
                p_alt2 = wow_viewer_root / clean_dir
                if p_alt2.exists():
                    dataset_base = p_alt2
                    resolved = True
                else:
                    # Try output/datasets/v18 directly under wow-viewer root
                    p_direct = wow_viewer_root / "output" / "datasets" / "v18"
                    if p_direct.exists():
                        dataset_base = p_direct
                        resolved = True

        if not resolved:
            # Try absolute project roots
            alt_absolute = Path("i:/parp/parp-tools/wow-viewer") / args.dataset_dir
            if alt_absolute.exists():
                dataset_base = alt_absolute
            else:
                alt_absolute_2 = Path("i:/parp/parp-tools") / args.dataset_dir
                if alt_absolute_2.exists():
                    dataset_base = alt_absolute_2
                else:
                    # Fallback to direct absolute paths
                    fb1 = Path("i:/parp/parp-tools/wow-viewer/output/datasets/v18")
                    if fb1.exists():
                        dataset_base = fb1
                    else:
                        print(f"Error: dataset dir not found at {args.dataset_dir} or relative to {wow_viewer_root}")
                        sys.exit(1)


    print(f"Dataset root: {dataset_base}")
    print(f"Target builds: {args.builds}")

    for build in args.builds:
        zarr_path = dataset_base / f"{build}.zarr"
        if not zarr_path.exists():
            print(f"Build directory {zarr_path} does not exist, skipping.")
            continue

        print(f"\nProcessing build: {build}...")
        group = zarr.open_group(str(zarr_path), mode="r+" if not args.dry_run else "r")

        # Resolve object mask array name (prioritize precise 3D silhouette mask)
        obj_key = None
        for k in ["object_precise_mask", "object_roof_mask", "object_filtered_mask", "object_mask"]:
            if k in group:
                obj_key = k
                break
        if not obj_key:
            print(f"Error: no object mask array found in {build}.zarr, skipping.")
            continue

        # Load input arrays (read-only views/handles)
        height_arr = group["height_257"]
        obj_arr = group[obj_key]
        liq_mask_arr = group["liquid_mask"]
        mcnk_flags_arr = group["mcnk_flags_16"] if "mcnk_flags_16" in group else None

        num_tiles = height_arr.shape[0]
        print(f"Total tiles to process: {num_tiles}")
        print(f"Using object mask array: '{obj_key}'")

        if args.dry_run:
            print("[DRY-RUN] Would compute and patch arrays.")
            continue

        # Pre-allocate output arrays in Zarr if they do not exist
        compressors = getattr(height_arr, "compressors", None)
        if compressors is not None:
            create_kwargs = {"compressors": compressors}
        else:
            try:
                create_kwargs = {"compressor": height_arr.compressor}
            except Exception:
                create_kwargs = {}

        if "liquid_type_256" not in group:
            liquid_type_arr = group.create_array(
                "liquid_type_256",
                shape=(num_tiles, 256, 256),
                chunks=(64, 256, 256),
                dtype=np.uint8,
                **create_kwargs,
            )
            print("Created liquid_type_256 array.")
        else:
            liquid_type_arr = group["liquid_type_256"]
            print("Using existing liquid_type_256 array.")

        if "ground_intent_height_257" not in group:
            ground_height_arr = group.create_array(
                "ground_intent_height_257",
                shape=(num_tiles, 257, 257),
                chunks=(64, 257, 257),
                dtype=np.float32,
                **create_kwargs,
            )
            print("Created ground_intent_height_257 array.")
        else:
            ground_height_arr = group["ground_intent_height_257"]
            print("Using existing ground_intent_height_257 array.")


        # If mcnk_flags_16 is not present, we will fallback to a default zero array
        if mcnk_flags_arr is None:
            print("Warning: mcnk_flags_16 not found in Zarr, fallback to all water class.")

        # Process in batches to avoid loading the whole dataset into RAM at once
        batch_size = args.batch_size
        start_time = time.time()

        with Pool(processes=args.workers) as pool:
            for start_idx in range(0, num_tiles, batch_size):
                end_idx = min(start_idx + batch_size, num_tiles)
                print(f"  Batch {start_idx}-{end_idx}...")

                # Read batch into memory
                h_batch = height_arr[start_idx:end_idx]
                obj_batch = obj_arr[start_idx:end_idx]
                if obj_batch.ndim == 3 and obj_batch.shape[1] == 256:
                    obj_batch = np.pad(obj_batch, ((0, 0), (0, 1), (0, 1)), mode="edge")
                liq_batch = liq_mask_arr[start_idx:end_idx]
                
                if mcnk_flags_arr is not None:
                    flags_batch = mcnk_flags_arr[start_idx:end_idx]
                else:
                    flags_batch = np.zeros((end_idx - start_idx, 16, 16), dtype=np.int32)

                # Prepare worker task list
                tasks = []
                for j in range(end_idx - start_idx):
                    idx = start_idx + j
                    tasks.append((idx, h_batch[j], obj_batch[j], liq_batch[j], flags_batch[j]))

                # Run in parallel
                results = pool.map(process_single_tile, tasks)

                # Write batch results back to Zarr sequentially
                liq_types_np = np.zeros((end_idx - start_idx, 256, 256), dtype=np.uint8)
                ground_h_np = np.zeros((end_idx - start_idx, 257, 257), dtype=np.float32)

                for idx, liq_t, gr_h in results:
                    local_idx = idx - start_idx
                    liq_types_np[local_idx] = liq_t
                    ground_h_np[local_idx] = gr_h

                liquid_type_arr[start_idx:end_idx] = liq_types_np
                ground_height_arr[start_idx:end_idx] = ground_h_np

        elapsed = time.time() - start_time
        print(f"Finished patching build '{build}' in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
