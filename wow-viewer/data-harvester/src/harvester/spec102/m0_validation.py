"""Self-describing validation panels for the Spec 102 M0 mask model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class M0ValidationSample:
    row: int
    build: str
    map_name: str
    tile_x: int
    tile_y: int
    source_rgb: np.ndarray
    probability: np.ndarray
    target: np.ndarray


def _binary_metrics(probability: np.ndarray, target: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = np.asarray(probability) >= threshold
    truth = np.asarray(target) >= 0.5
    intersection = int((predicted & truth).sum())
    union = int((predicted | truth).sum())
    predicted_count = int(predicted.sum())
    target_count = int(truth.sum())
    return {
        "iou": intersection / max(union, 1),
        "dice": (2.0 * intersection) / max(predicted_count + target_count, 1),
        "predicted_pixels": predicted_count,
        "target_pixels": target_count,
    }


def _agreement_rgb(probability: np.ndarray, target: np.ndarray, threshold: float) -> np.ndarray:
    predicted = np.asarray(probability) >= threshold
    truth = np.asarray(target) >= 0.5
    result = np.zeros((*predicted.shape, 3), dtype=np.uint8)
    result[predicted & truth] = (40, 220, 90)   # true positive
    result[predicted & ~truth] = (240, 65, 65)  # false positive
    result[~predicted & truth] = (70, 135, 255)  # false negative
    return result


def render_m0_validation_panel(
    samples: list[M0ValidationSample],
    *,
    split: str,
    epoch: int,
    threshold: float = 0.5,
    checkpoint_label: str = "current epoch",
) -> Image.Image:
    """Render a panel whose meaning is embedded directly in the PNG."""
    if not samples:
        raise ValueError("M0 validation panel requires at least one sample")
    tile_size = 256
    header_height = 76
    row_label_height = 30
    row_height = tile_size + row_label_height
    canvas = Image.new("RGB", (tile_size * 4, header_height + row_height * len(samples)), (18, 18, 22))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (8, 7),
        f"Spec 102 M0 validation | split={split} | epoch={epoch} | threshold={threshold:.2f}",
        fill=(255, 255, 255), font=font,
    )
    draw.text(
        (8, 23),
        f"checkpoint={checkpoint_label} | white in probability/target means object footprint",
        fill=(205, 205, 210), font=font,
    )
    draw.text(
        (8, 39),
        "Agreement legend: GREEN=true positive  RED=false positive  BLUE=false negative  BLACK=true negative",
        fill=(235, 215, 150), font=font,
    )
    headings = (
        "INPUT: raw minimap RGB",
        "PREDICTION: probability",
        "TARGET: precise 257->256",
        "AGREEMENT: TP / FP / FN",
    )
    for column, heading in enumerate(headings):
        draw.text((column * tile_size + 8, 59), heading, fill=(255, 255, 255), font=font)

    for index, sample in enumerate(samples):
        y = header_height + index * row_height
        source = np.asarray(sample.source_rgb, dtype=np.uint8)
        probability = np.clip(np.asarray(sample.probability) * 255.0, 0, 255).astype(np.uint8)
        target = (np.asarray(sample.target) >= 0.5).astype(np.uint8) * 255
        agreement = _agreement_rgb(sample.probability, sample.target, threshold)
        canvas.paste(Image.fromarray(source, "RGB"), (0, y))
        canvas.paste(Image.fromarray(probability, "L").convert("RGB"), (tile_size, y))
        canvas.paste(Image.fromarray(target, "L").convert("RGB"), (tile_size * 2, y))
        canvas.paste(Image.fromarray(agreement, "RGB"), (tile_size * 3, y))
        metrics = _binary_metrics(sample.probability, sample.target, threshold)
        label = (
            f"row={sample.row}  {sample.build}/{sample.map_name} tile=({sample.tile_x},{sample.tile_y})  "
            f"IoU={metrics['iou']:.4f} Dice={metrics['dice']:.4f}  "
            f"pred_px={metrics['predicted_pixels']} target_px={metrics['target_pixels']}"
        )
        draw.rectangle((0, y + tile_size, canvas.width, y + row_height), fill=(18, 18, 22))
        draw.text((8, y + tile_size + 8), label, fill=(235, 235, 235), font=font)
    return canvas
