"""Generate image-layout probes from map-derived WAV or raw byte payloads."""

from __future__ import annotations

import argparse
import json
import math
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw


DEFAULT_WIDTHS = (64, 128, 192, 256, 257, 320, 384, 512, 514, 768, 1024, 1028, 2048)
DEFAULT_STRIDES = (2, 4, 8)
DEFAULT_TOP = 36
DEFAULT_MAX_PIXELS = 1_048_576


@dataclass(frozen=True)
class Candidate:
    name: str
    path: str
    mode: str
    width: int
    original_height: int
    displayed_height: int
    entropy: float
    contrast: float
    corr_x: float
    corr_y: float
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe WAV/raw bytes as image layouts to expose repeatable structure."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input .wav or raw byte file.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for PNG/JSON output.")
    parser.add_argument(
        "--widths",
        nargs="*",
        type=int,
        default=list(DEFAULT_WIDTHS),
        help="Candidate image widths. Defaults include power-of-two and 257/514 terrain widths.",
    )
    parser.add_argument(
        "--strides",
        nargs="*",
        type=int,
        default=list(DEFAULT_STRIDES),
        help="Byte deinterleave strides for phase views.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP,
        help="Number of top-ranked candidates to include in the contact sheet.",
    )
    parser.add_argument(
        "--max-pixels-per-image",
        type=int,
        default=DEFAULT_MAX_PIXELS,
        help="Maximum displayed pixels per output image. Tall images are row-sampled.",
    )
    parser.add_argument(
        "--limit-bytes",
        type=int,
        default=0,
        help="Optional byte cap for quick experiments. Zero means analyze the full payload.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=257,
        help="Sample tile size for reversing flattened heightmap audio.",
    )
    parser.add_argument(
        "--index-parquet",
        type=Path,
        default=None,
        help="Optional dataset index parquet for map-coordinate tile mosaics.",
    )
    parser.add_argument(
        "--map",
        default="",
        help="Map name to filter in --index-parquet, for example Azeroth.",
    )
    parser.add_argument(
        "--mosaic-tile-pixels",
        type=int,
        default=64,
        help="Displayed tile size for generated mosaics.",
    )
    return parser.parse_args()


def read_payload(path: Path) -> tuple[bytes, dict[str, object]]:
    metadata: dict[str, object] = {
        "input_path": str(path),
        "input_size_bytes": path.stat().st_size,
        "input_kind": "raw",
    }
    if path.suffix.lower() != ".wav":
        return path.read_bytes(), metadata

    try:
        with wave.open(str(path), "rb") as wav:
            metadata.update(
                {
                    "input_kind": "wav",
                    "channels": wav.getnchannels(),
                    "sample_width_bytes": wav.getsampwidth(),
                    "sample_rate": wav.getframerate(),
                    "frames": wav.getnframes(),
                    "compression": wav.getcomptype(),
                }
            )
            return wav.readframes(wav.getnframes()), metadata
    except wave.Error as exc:
        metadata.update({"input_kind": "raw_wav_fallback", "wav_error": str(exc)})
        return path.read_bytes(), metadata


def normalize_to_u8(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)

    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.0))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        lo = float(finite.min())
        hi = float(finite.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr - lo) / (hi - lo)
    return np.clip(scaled * 255.0, 0.0, 255.0).astype(np.uint8)


def decode_samples(payload: bytes, metadata: dict[str, object]) -> np.ndarray | None:
    sample_width = int(metadata.get("sample_width_bytes", 0) or 0)
    channels = int(metadata.get("channels", 1) or 1)
    if sample_width == 1:
        samples = np.frombuffer(payload, dtype=np.uint8).astype(np.int16) - 128
    elif sample_width == 2:
        samples = np.frombuffer(payload[: len(payload) - (len(payload) % 2)], dtype="<i2")
    elif sample_width == 4:
        samples = np.frombuffer(payload[: len(payload) - (len(payload) % 4)], dtype="<i4")
    else:
        return None

    if channels > 1:
        usable = samples.size - (samples.size % channels)
        samples = samples[:usable].reshape(-1, channels)[:, 0]
    return samples


def reshape_view(values: np.ndarray, width: int, max_pixels: int) -> tuple[np.ndarray, int, int]:
    if width <= 0 or values.size < width:
        raise ValueError("not enough values for requested width")

    original_height = values.size // width
    trimmed = values[: original_height * width]
    image = trimmed.reshape(original_height, width)
    displayed = sample_rows(image, max_pixels)
    return displayed, original_height, displayed.shape[0]


def sample_rows(image: np.ndarray, max_pixels: int) -> np.ndarray:
    max_rows = max(1, max_pixels // max(1, image.shape[1]))
    if image.shape[0] <= max_rows:
        return image
    row_idx = np.linspace(0, image.shape[0] - 1, max_rows).astype(np.int64)
    return image[row_idx]


def image_metrics(gray: np.ndarray) -> tuple[float, float, float, float, float]:
    data = gray.astype(np.float32, copy=False)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    probs = hist[hist > 0] / max(1, gray.size)
    entropy = float(-(probs * np.log2(probs)).sum())
    contrast = float(data.std() / 255.0)
    corr_x = adjacent_corr(data[:, :-1], data[:, 1:]) if data.shape[1] > 1 else 0.0
    corr_y = adjacent_corr(data[:-1, :], data[1:, :]) if data.shape[0] > 1 else 0.0
    structure = (abs(corr_x) + abs(corr_y)) * 0.5
    entropy_shape = 1.0 - abs(entropy - 5.5) / 5.5
    score = float((contrast * 0.45) + (structure * 0.45) + (max(0.0, entropy_shape) * 0.10))
    return entropy, contrast, corr_x, corr_y, score


def adjacent_corr(left: np.ndarray, right: np.ndarray) -> float:
    a = left.ravel()
    b = right.ravel()
    if a.size < 2:
        return 0.0
    a_std = float(a.std())
    b_std = float(b.std())
    if a_std == 0.0 or b_std == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def save_gray_candidate(
    candidates: list[Candidate],
    out_dir: Path,
    name: str,
    mode: str,
    values: np.ndarray,
    width: int,
    max_pixels: int,
) -> None:
    if values.size < width:
        return
    gray_values = values if values.dtype == np.uint8 else normalize_to_u8(values)
    image, original_height, displayed_height = reshape_view(gray_values, width, max_pixels)
    entropy, contrast, corr_x, corr_y, score = image_metrics(image)
    filename = safe_name(f"{name}_w{width}.png")
    path = out_dir / filename
    Image.fromarray(image, mode="L").save(path)
    candidates.append(
        Candidate(
            name=filename.removesuffix(".png"),
            path=str(path),
            mode=mode,
            width=width,
            original_height=original_height,
            displayed_height=displayed_height,
            entropy=entropy,
            contrast=contrast,
            corr_x=corr_x,
            corr_y=corr_y,
            score=score,
        )
    )


def save_rgb_candidate(
    candidates: list[Candidate],
    out_dir: Path,
    name: str,
    rgb_values: np.ndarray,
    width: int,
    max_pixels: int,
) -> None:
    pixels = rgb_values.size // 3
    if pixels < width:
        return
    original_height = pixels // width
    trimmed = rgb_values[: original_height * width * 3]
    image = trimmed.reshape(original_height, width, 3)
    image = sample_rows(image, max_pixels)
    gray = np.asarray(Image.fromarray(image, mode="RGB").convert("L"))
    entropy, contrast, corr_x, corr_y, score = image_metrics(gray)
    filename = safe_name(f"{name}_w{width}.png")
    path = out_dir / filename
    Image.fromarray(image, mode="RGB").save(path)
    candidates.append(
        Candidate(
            name=filename.removesuffix(".png"),
            path=str(path),
            mode="rgb_triplets",
            width=width,
            original_height=original_height,
            displayed_height=image.shape[0],
            entropy=entropy,
            contrast=contrast,
            corr_x=corr_x,
            corr_y=corr_y,
            score=score,
        )
    )


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def generate_candidates(
    payload: bytes,
    out_dir: Path,
    widths: list[int],
    strides: list[int],
    max_pixels: int,
) -> list[Candidate]:
    raw = np.frombuffer(payload, dtype=np.uint8)
    candidates: list[Candidate] = []
    if raw.size == 0:
        return candidates

    delta = np.abs(np.diff(raw.astype(np.int16), prepend=int(raw[0]))).astype(np.uint8)
    for width in widths:
        save_gray_candidate(candidates, out_dir, "bytes_u8_row_major", "bytes_u8", raw, width, max_pixels)
        save_gray_candidate(candidates, out_dir, "bytes_u8_delta", "bytes_delta", delta, width, max_pixels)
        save_rgb_candidate(candidates, out_dir, "bytes_rgb_triplets", raw, width, max_pixels)

        for bit in range(8):
            bitplane = (((raw >> bit) & 1) * 255).astype(np.uint8)
            save_gray_candidate(
                candidates,
                out_dir,
                f"bytes_bitplane_{bit}",
                f"bitplane_{bit}",
                bitplane,
                width,
                max_pixels,
            )

        for stride in strides:
            if stride <= 1:
                continue
            for phase in range(stride):
                save_gray_candidate(
                    candidates,
                    out_dir,
                    f"bytes_stride{stride}_phase{phase}",
                    f"bytes_stride{stride}_phase{phase}",
                    raw[phase::stride],
                    width,
                    max_pixels,
                )

    if raw.size >= 2:
        even_bytes = raw[: raw.size - (raw.size % 2)]
        for dtype_name, dtype in (
            ("i16_le", "<i2"),
            ("i16_be", ">i2"),
            ("u16_le", "<u2"),
            ("u16_be", ">u2"),
        ):
            samples = even_bytes.view(np.dtype(dtype))
            for width in widths:
                save_gray_candidate(
                    candidates,
                    out_dir,
                    f"samples_{dtype_name}",
                    dtype_name,
                    samples,
                    width,
                    max_pixels,
                )

    if raw.size >= 4:
        float_bytes = raw[: raw.size - (raw.size % 4)]
        floats = float_bytes.view(np.dtype("<f4"))
        floats = floats[np.isfinite(floats)]
        if floats.size:
            for width in widths:
                save_gray_candidate(
                    candidates,
                    out_dir,
                    "samples_f32_le",
                    "f32_le",
                    floats,
                    width,
                    max_pixels,
                )

    return candidates


def write_contact_sheet(candidates: list[Candidate], out_path: Path, top: int) -> None:
    chosen = candidates[:top]
    if not chosen:
        return

    thumb_w = 220
    thumb_h = 180
    label_h = 44
    cols = 3
    rows = math.ceil(len(chosen) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)

    for index, candidate in enumerate(chosen):
        col = index % cols
        row = index // cols
        x = col * thumb_w
        y = row * (thumb_h + label_h)
        with Image.open(candidate.path) as src:
            thumb = src.convert("RGB")
            thumb.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            sheet.paste(thumb, (x, y))
        label = f"{index + 1}. {candidate.mode} w={candidate.width}\nscore={candidate.score:.3f} h={candidate.original_height}"
        draw.text((x + 4, y + thumb_h + 4), label, fill="black")

    sheet.save(out_path)


def write_tile_unswizzle(
    payload: bytes,
    metadata: dict[str, object],
    out_dir: Path,
    tile_size: int,
    index_parquet: Path | None,
    map_name: str,
    mosaic_tile_pixels: int,
) -> dict[str, object] | None:
    samples = decode_samples(payload, metadata)
    if samples is None or tile_size <= 0:
        return None

    tile_samples = tile_size * tile_size
    tile_count = samples.size // tile_samples
    remainder = samples.size % tile_samples
    if tile_count == 0:
        return None

    tiles = normalize_to_u8(samples[: tile_count * tile_samples]).reshape(
        tile_count, tile_size, tile_size
    )
    tile_dir = out_dir / "tile_unswizzle"
    tile_dir.mkdir(parents=True, exist_ok=True)

    stream_path = tile_dir / f"stream_order_tiles_{tile_size}.png"
    write_stream_mosaic(tiles, stream_path, mosaic_tile_pixels)

    tile_summary: dict[str, object] = {
        "tile_size": tile_size,
        "tile_sample_count": tile_samples,
        "decoded_sample_count": int(samples.size),
        "tile_count": int(tile_count),
        "remainder_samples": int(remainder),
        "stream_order_mosaic": str(stream_path),
        "note": "This reverses the heightmap flattening used by the map-to-audio script.",
    }

    if index_parquet is not None:
        indexed = write_indexed_mosaic(
            tiles=tiles,
            index_parquet=index_parquet,
            map_name=map_name,
            out_dir=tile_dir,
            mosaic_tile_pixels=mosaic_tile_pixels,
        )
        tile_summary.update(indexed)

    (tile_dir / "tile_unswizzle_summary.json").write_text(
        json.dumps(tile_summary, indent=2), encoding="utf-8"
    )
    return tile_summary


def write_stream_mosaic(tiles: np.ndarray, out_path: Path, mosaic_tile_pixels: int) -> None:
    count = tiles.shape[0]
    cols = max(1, math.ceil(math.sqrt(count)))
    rows = math.ceil(count / cols)
    tile_px = max(8, mosaic_tile_pixels)
    mosaic = Image.new("L", (cols * tile_px, rows * tile_px), 0)
    for index, tile in enumerate(tiles):
        tile_img = Image.fromarray(tile, mode="L").resize(
            (tile_px, tile_px), Image.Resampling.BILINEAR
        )
        x = (index % cols) * tile_px
        y = (index // cols) * tile_px
        mosaic.paste(tile_img, (x, y))
    mosaic.save(out_path)


def write_indexed_mosaic(
    tiles: np.ndarray,
    index_parquet: Path,
    map_name: str,
    out_dir: Path,
    mosaic_tile_pixels: int,
) -> dict[str, object]:
    table = pq.read_table(str(index_parquet))
    rows = []
    for i in range(table.num_rows):
        row = {col: table.column(col)[i].as_py() for col in table.column_names}
        if map_name and str(row.get("map", "")).lower() != map_name.lower():
            continue
        rows.append(row)

    rows = sorted(rows, key=lambda row: (row["tile_y"], row["tile_x"]))
    usable = min(len(rows), tiles.shape[0])
    if usable == 0:
        return {
            "indexed_mosaic": None,
            "index_rows": len(rows),
            "index_note": "No index rows matched the requested map.",
        }

    selected = rows[:usable]
    min_x = min(int(row["tile_x"]) for row in selected)
    max_x = max(int(row["tile_x"]) for row in selected)
    min_y = min(int(row["tile_y"]) for row in selected)
    max_y = max(int(row["tile_y"]) for row in selected)
    tile_px = max(8, mosaic_tile_pixels)
    mosaic = Image.new("L", ((max_x - min_x + 1) * tile_px, (max_y - min_y + 1) * tile_px), 0)

    for index, row in enumerate(selected):
        tile_img = Image.fromarray(tiles[index], mode="L").resize(
            (tile_px, tile_px), Image.Resampling.BILINEAR
        )
        x = (int(row["tile_x"]) - min_x) * tile_px
        y = (int(row["tile_y"]) - min_y) * tile_px
        mosaic.paste(tile_img, (x, y))

    suffix = safe_name(map_name or "indexed")
    path = out_dir / f"{suffix}_coordinate_mosaic_{tile_px}px_tiles.png"
    mosaic.save(path)
    return {
        "indexed_mosaic": str(path),
        "index_rows": len(rows),
        "indexed_tiles_used": usable,
        "tile_x_range": [min_x, max_x],
        "tile_y_range": [min_y, max_y],
        "mosaic_tile_pixels": tile_px,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload, metadata = read_payload(args.input)
    if args.limit_bytes and args.limit_bytes > 0:
        payload = payload[: args.limit_bytes]
        metadata["limit_bytes"] = args.limit_bytes

    tile_summary = None
    if not args.limit_bytes:
        tile_summary = write_tile_unswizzle(
            payload=payload,
            metadata=metadata,
            out_dir=args.output_dir,
            tile_size=args.tile_size,
            index_parquet=args.index_parquet,
            map_name=args.map,
            mosaic_tile_pixels=args.mosaic_tile_pixels,
        )

    widths = sorted({width for width in args.widths if width > 0})
    candidates_dir = args.output_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    candidates = generate_candidates(
        payload=payload,
        out_dir=candidates_dir,
        widths=widths,
        strides=args.strides,
        max_pixels=args.max_pixels_per_image,
    )
    candidates.sort(key=lambda item: item.score, reverse=True)

    summary = {
        "metadata": metadata | {"payload_bytes_analyzed": len(payload)},
        "note": "Ranked image layouts are hypotheses. Structure here is not proof of hidden payloads.",
        "widths": widths,
        "strides": args.strides,
        "tile_unswizzle": tile_summary,
        "candidate_count": len(candidates),
        "candidates": [asdict(candidate) for candidate in candidates],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_contact_sheet(candidates, args.output_dir / "contact_sheet.png", args.top)

    print(f"Analyzed payload bytes: {len(payload):,}")
    print(f"Candidates written: {len(candidates):,}")
    print(f"Summary: {args.output_dir / 'summary.json'}")
    print(f"Contact sheet: {args.output_dir / 'contact_sheet.png'}")
    if tile_summary:
        print(f"Tile unswizzle: {args.output_dir / 'tile_unswizzle'}")


if __name__ == "__main__":
    main()
