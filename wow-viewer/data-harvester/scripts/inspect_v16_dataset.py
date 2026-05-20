from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
from PIL import Image as _PILImage
from PIL import ImageDraw as _PILImageDraw
import pyarrow.compute as pc
import pyarrow.parquet as pq
import zarr
import zarr.storage

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATASET_ROOT = _PROJECT_ROOT / "output" / "datasets" / "v16"


def _open_store(zarr_path: Path):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Object at .* is not recognized as a component of a Zarr hierarchy\.",
            category=UserWarning,
        )
        store = zarr.storage.LocalStore(str(zarr_path), read_only=True)
        root = zarr.open_group(store=store, mode="r")
    return store, root


def _select_sample_ids(
    table,
    sample_count: int,
    seed: int,
    mode: str,
) -> list[int]:
    tile_count = int(table.num_rows)
    if tile_count <= 0 or sample_count <= 0:
        return []
    if tile_count <= sample_count:
        return list(range(tile_count))
    if mode == "linspace":
        return sorted({int(round(v)) for v in np.linspace(0, tile_count - 1, num=sample_count)})

    rng = np.random.RandomState(int(seed))
    all_ids = np.arange(tile_count, dtype=np.int64)

    if mode == "liquid_focus" and "has_liquid_mask" in table.column_names:
        liquid_col = table.column("has_liquid_mask")
        liquid_ids = np.array([i for i in range(tile_count) if bool(liquid_col[i].as_py())], dtype=np.int64)
        non_liquid_ids = np.array([i for i in range(tile_count) if not bool(liquid_col[i].as_py())], dtype=np.int64)
        target_liquid = min(len(liquid_ids), max(1, sample_count // 2))
        chosen_liquid = rng.choice(liquid_ids, size=target_liquid, replace=False) if target_liquid > 0 else np.array([], dtype=np.int64)
        remaining = sample_count - len(chosen_liquid)
        pool = np.setdiff1d(all_ids, chosen_liquid, assume_unique=False)
        chosen_other = rng.choice(pool, size=remaining, replace=False) if remaining > 0 else np.array([], dtype=np.int64)
        out = np.concatenate([chosen_liquid, chosen_other])
        rng.shuffle(out)
        return [int(v) for v in out.tolist()]

    chosen = rng.choice(all_ids, size=sample_count, replace=False)
    return [int(v) for v in chosen.tolist()]


def _table_rows(table, tile_ids: list[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for tile_id in tile_ids:
        rows.append({col: table.column(col)[tile_id].as_py() for col in table.column_names})
    return rows


def _build_summary(
    build: str,
    zarr_path: Path,
    *,
    sample_count: int,
    sample_seed: int,
    sample_mode: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    store, root = _open_store(zarr_path)
    try:
        index_path = zarr_path / "index.parquet"
        if not index_path.exists():
            raise RuntimeError(f"{zarr_path} is missing index.parquet")
        table = pq.read_table(str(index_path))
        tile_count = int(table.num_rows)

        placements_path = zarr_path / "placements.parquet"
        placement_rows = 0
        if placements_path.exists():
            placement_rows = int(pq.read_metadata(str(placements_path)).num_rows)

        signal_counts: dict[str, int] = {}
        for col in table.column_names:
            if not col.startswith("has_"):
                continue
            count_scalar = pc.sum(table.column(col))
            signal_counts[col] = 0 if count_scalar is None else int(count_scalar.as_py() or 0)

        maps: dict[str, dict[str, int]] = {}
        if "map" in table.column_names:
            map_values = table.column("map")
            mddf_values = table.column("n_mddf") if "n_mddf" in table.column_names else None
            modf_values = table.column("n_modf") if "n_modf" in table.column_names else None
            for i in range(tile_count):
                map_name = str(map_values[i].as_py())
                entry = maps.setdefault(map_name, {"tiles": 0, "n_mddf": 0, "n_modf": 0})
                entry["tiles"] += 1
                if mddf_values is not None:
                    entry["n_mddf"] += int(mddf_values[i].as_py() or 0)
                if modf_values is not None:
                    entry["n_modf"] += int(modf_values[i].as_py() or 0)

        array_info = {
            key: {
                "shape": list(root[key].shape),
                "dtype": str(root[key].dtype),
            }
            for key in sorted(root.array_keys())
        }

        sample_ids = _select_sample_ids(table, sample_count=sample_count, seed=sample_seed, mode=sample_mode)
        sample_rows = _table_rows(table, sample_ids)

        summary = {
            "build": build,
            "zarr_path": str(zarr_path),
            "tile_count": tile_count,
            "placement_rows": placement_rows,
            "array_info": array_info,
            "signal_counts": signal_counts,
            "maps": maps,
            "sample_tile_ids": sample_ids,
            "sample_count": int(sample_count),
            "sample_seed": int(sample_seed),
            "sample_mode": sample_mode,
        }
        return summary, sample_rows
    finally:
        store.close()


def _make_contact_sheet(images: list[np.ndarray], *, fill_value: int = 0) -> np.ndarray:
    if not images:
        raise ValueError("No images provided")
    # Ensure all images are RGB
    images = [np.repeat(x[:, :, None], 3, axis=2) if x.ndim == 2 else x for x in images]
    height, width = images[0].shape[:2]
    channels = images[0].shape[2]
    cols = math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / cols)
    sheet = np.full((rows * height, cols * width, channels), fill_value, dtype=images[0].dtype)

    for idx, image in enumerate(images):
        y = (idx // cols) * height
        x = (idx % cols) * width
        sheet[y:y + height, x:x + width, ...] = image
    return sheet


def _write_contact_sheets(zarr_path: Path, build: str, tile_ids: list[int], output_dir: Path) -> None:
    try:
        from matplotlib import pyplot as plt
    except Exception as ex:
        raise RuntimeError("matplotlib is required for --write-images") from ex

    store, root = _open_store(zarr_path)
    try:
        minimaps: list[np.ndarray] = []
        object_masks: list[np.ndarray] = []
        liquid_masks: list[np.ndarray] = []
        instance_masks: list[np.ndarray] = []
        filtered_masks: list[np.ndarray] = []
        mddf_masks: list[np.ndarray] = []
        modf_masks: list[np.ndarray] = []

        for tile_id in tile_ids:
            minimaps.append(root["minimap_rgb"][tile_id].astype(np.uint8))
            object_masks.append((root["object_mask"][tile_id].astype(np.float32) * 255).astype(np.uint8))
            liquid_masks.append((np.clip(root["liquid_mask"][tile_id].astype(np.float32), 0.0, 1.0) * 255).astype(np.uint8))

            instance = root["object_instance_mask"][tile_id].astype(np.int32)
            nonzero = instance > 0
            if np.any(nonzero):
                scaled = np.zeros_like(instance, dtype=np.uint8)
                max_val = int(instance[nonzero].max())
                scaled[nonzero] = np.clip((instance[nonzero] / max(max_val, 1)) * 255, 0, 255).astype(np.uint8)
                instance_masks.append(scaled)
            else:
                instance_masks.append(np.zeros_like(instance, dtype=np.uint8))

            if "object_filtered_mask" in root:
                filtered_masks.append((root["object_filtered_mask"][tile_id].astype(np.float32) * 255).astype(np.uint8))
            if "mddf_mask" in root:
                mddf_masks.append((root["mddf_mask"][tile_id].astype(np.float32) * 255).astype(np.uint8))
            if "modf_mask" in root:
                modf_masks.append((root["modf_mask"][tile_id].astype(np.float32) * 255).astype(np.uint8))

        # One combined grid: all signals for all tiles
        def _to_rgb(img: np.ndarray) -> np.ndarray:
            if img.ndim == 2:
                return np.repeat(img[:, :, None], 3, axis=2)
            return img

        all_images = [_to_rgb(x) for x in minimaps + liquid_masks + object_masks + instance_masks]
        if filtered_masks:
            all_images += [_to_rgb(x) for x in filtered_masks + mddf_masks + modf_masks]
        combined = _make_contact_sheet(all_images)

        plt.imsave(output_dir / f"{build}.contact_sheet.png", combined)
    finally:
        store.close()


def _resize_u8(arr: np.ndarray, size: int) -> np.ndarray:
    img = _PILImage.fromarray(arr)
    img = img.resize((size, size), _PILImage.Resampling.BILINEAR)
    return np.asarray(img)


def _to_gray_u8(arr: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    x = arr.astype(np.float32)
    if lo is None:
        lo = float(np.min(x))
    if hi is None:
        hi = float(np.max(x))
    rng = hi - lo
    if abs(rng) < 1e-8:
        y = np.zeros_like(x, dtype=np.float32)
    else:
        y = np.clip((x - lo) / rng, 0.0, 1.0)
    return (y * 255.0).astype(np.uint8)


def _draw_label_rgb(img_u8: np.ndarray, text: str) -> np.ndarray:
    if img_u8.ndim == 2:
        rgb = np.repeat(img_u8[:, :, None], 3, axis=2)
    else:
        rgb = img_u8
    img = _PILImage.fromarray(rgb, "RGB")
    drw = _PILImageDraw.Draw(img)
    drw.rectangle([(0, 0), (img.width, 18)], fill=(0, 0, 0))
    drw.text((4, 3), text, fill=(255, 255, 255))
    return np.asarray(img)


def _write_labeled_visual_audit(
    zarr_path: Path,
    build: str,
    tile_ids: list[int],
    sample_rows: list[dict[str, object]],
    output_dir: Path,
    *,
    panel_size: int = 256,
    overview_columns: int = 2,
) -> None:
    store, root = _open_store(zarr_path)
    try:
        strips: list[np.ndarray] = []
        for i, tile_id in enumerate(tile_ids):
            row = sample_rows[i] if i < len(sample_rows) else {}
            minimap = root["minimap_rgb"][tile_id].astype(np.uint8)
            height = root["height_257"][tile_id].astype(np.float32)
            liquid = root["liquid_mask"][tile_id].astype(np.float32)
            obj = root["object_mask"][tile_id].astype(np.float32)

            h_u8 = _to_gray_u8(height)
            l_u8 = _to_gray_u8(np.clip(liquid, 0.0, 1.0), lo=0.0, hi=1.0)
            o_u8 = _to_gray_u8(np.clip(obj, 0.0, 1.0), lo=0.0, hi=1.0)

            panels = [
                _draw_label_rgb(_resize_u8(minimap, panel_size), "input/minimap"),
                _draw_label_rgb(_resize_u8(h_u8, panel_size), "height"),
                _draw_label_rgb(_resize_u8(l_u8, panel_size), "liquid mask"),
                _draw_label_rgb(_resize_u8(o_u8, panel_size), "object mask"),
            ]

            # New arrays from spec 003
            if "object_filtered_mask" in root:
                filtered = root["object_filtered_mask"][tile_id].astype(np.float32)
                f_u8 = _to_gray_u8(np.clip(filtered, 0.0, 1.0), lo=0.0, hi=1.0)
                panels.append(_draw_label_rgb(_resize_u8(f_u8, panel_size), "filtered mask"))

            if "mddf_mask" in root:
                mddf = root["mddf_mask"][tile_id].astype(np.float32)
                mddf_u8 = _to_gray_u8(np.clip(mddf, 0.0, 1.0), lo=0.0, hi=1.0)
                panels.append(_draw_label_rgb(_resize_u8(mddf_u8, panel_size), "mddf mask"))

            if "modf_mask" in root:
                modf = root["modf_mask"][tile_id].astype(np.float32)
                modf_u8 = _to_gray_u8(np.clip(modf, 0.0, 1.0), lo=0.0, hi=1.0)
                panels.append(_draw_label_rgb(_resize_u8(modf_u8, panel_size), "modf mask"))

            if "mcnk_flags_16" in root:
                flags = root["mcnk_flags_16"][tile_id].astype(np.float32)
                flags_u8 = _to_gray_u8(flags, lo=0.0, hi=float(max(int(flags.max()), 1)))
                flags_up = _resize_u8(flags_u8, panel_size)
                panels.append(_draw_label_rgb(flags_up, "mcnk flags"))

            strip = np.concatenate(panels, axis=1)

            map_name = str(row.get("map", "unknown"))
            tile_x = int(row["tile_x"]) if row.get("tile_x") is not None else -1
            tile_y = int(row["tile_y"]) if row.get("tile_y") is not None else -1
            liq_src = "none"
            for src in ("mcnk", "mh2o", "mclq", "unified", "wl"):
                if bool(row.get(f"has_liquid_source_{src}", False)):
                    liq_src = src
                    break
            title = f"{build} sample={i:02d} tile_id={tile_id} map={map_name} xy=({tile_x},{tile_y}) liquid_src={liq_src}"
            img = _PILImage.fromarray(strip, "RGB")
            drw = _PILImageDraw.Draw(img)
            drw.rectangle([(0, 0), (img.width, 18)], fill=(20, 20, 20))
            drw.text((4, 3), title, fill=(240, 240, 240))
            strips.append(np.asarray(img))

        if not strips:
            return

        strip_h, strip_w = strips[0].shape[0], strips[0].shape[1]
        cols = max(1, int(overview_columns))
        rows = math.ceil(len(strips) / cols)
        canvas = _PILImage.new("RGB", (cols * strip_w, rows * strip_h), (12, 12, 12))
        for i, strip in enumerate(strips):
            x = (i % cols) * strip_w
            y = (i // cols) * strip_h
            canvas.paste(_PILImage.fromarray(strip, "RGB"), (x, y))
        canvas.save(output_dir / f"{build}.validation_audit_overview.png")
    finally:
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect V16 Zarr datasets and backfill human-friendly summaries")
    parser.add_argument("--build", type=str, help="Single build key")
    parser.add_argument("--builds", nargs="+", help="Multiple build keys")
    parser.add_argument("--sample-count", type=int, default=16, help="Number of sample tiles to record")
    parser.add_argument("--sample-seed", type=int, default=42, help="Seed for sample tile selection")
    parser.add_argument(
        "--sample-mode",
        choices=["random", "linspace", "liquid_focus"],
        default="random",
        help="How sample tiles are selected",
    )
    parser.add_argument("--output-dir", type=Path, default=_DATASET_ROOT / "inspection", help="Directory for summary/sample outputs")
    parser.add_argument("--write-images", action="store_true", help="Write combined contact-sheet PNG with all signals")
    parser.add_argument(
        "--write-overview",
        action="store_true",
        help="Write labeled visual audit overview with minimap/height/liquid/object panels",
    )
    parser.add_argument("--overview-columns", type=int, default=2, help="Column count for labeled overview grid")
    parser.add_argument("--backfill-summary", action="store_true", help="Write _dataset_summary.json into each build store")
    args = parser.parse_args()

    builds = args.builds or ([args.build] if args.build else [])
    if not builds:
        parser.error("Provide --build or --builds")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for build in builds:
        zarr_path = _DATASET_ROOT / f"{build}.zarr"
        if not zarr_path.exists():
            print(f"SKIP {build}: no final store at {zarr_path}")
            continue

        summary, sample_rows = _build_summary(
            build,
            zarr_path,
            sample_count=int(args.sample_count),
            sample_seed=int(args.sample_seed),
            sample_mode=str(args.sample_mode),
        )
        sample_ids = summary["sample_tile_ids"]

        summary_path = args.output_dir / f"{build}.summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        sample_path = args.output_dir / f"{build}.samples.json"
        sample_path.write_text(json.dumps(sample_rows, indent=2), encoding="utf-8")

        if args.backfill_summary:
            (zarr_path / "_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        if args.write_images:
            _write_contact_sheets(zarr_path, build, list(sample_ids), args.output_dir)
        if args.write_overview:
            _write_labeled_visual_audit(
                zarr_path,
                build,
                list(sample_ids),
                sample_rows,
                args.output_dir,
                overview_columns=int(args.overview_columns),
            )

        print(f"Wrote {summary_path}")
        print(f"Wrote {sample_path}")
        if args.backfill_summary:
            print(f"Wrote {zarr_path / '_dataset_summary.json'}")
        if args.write_images:
            print(f"Wrote sample sheets for {build} into {args.output_dir}")
        if args.write_overview:
            print(f"Wrote labeled visual audit overview for {build} into {args.output_dir}")


if __name__ == "__main__":
    main()
