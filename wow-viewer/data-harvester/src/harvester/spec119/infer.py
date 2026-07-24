"""Spec 119 loose-image inference (T022/T023, FR-013).

Runs a frozen classifier or segmenter checkpoint on loose PNGs with NO store or ground truth
present. Architectures are reconstructed from the checkpoint's ``base`` alone (D-02); a
checkpoint missing ``base`` is refused, never guessed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from harvester.spec119.object_library_contract import ObjectLibraryContractError


def _load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    if not Path(path).is_file():
        raise ObjectLibraryContractError(f"{path}: checkpoint does not exist")
    return torch.load(str(path), map_location="cpu", weights_only=False)


def _require_base(checkpoint: dict[str, Any], path: Path) -> int:
    architecture = checkpoint.get("architecture") or {}
    base = architecture.get("base")
    if not isinstance(base, int) or base < 1:
        raise ObjectLibraryContractError(
            f"{path}: checkpoint architecture block is missing 'base'; refusing to guess the "
            "architecture (D-02)"
        )
    return int(base)


def load_classifier_checkpoint(path: Path):
    """Rebuild an ObjectClassifier from its checkpoint; returns ``(model, class_index)``."""
    import torch  # noqa: F401  (keeps the torch import local, consistent with trainers)

    from harvester.spec119.classifier_model import ObjectClassifier

    checkpoint = _load_checkpoint(path)
    if checkpoint.get("kind") != "classifier":
        raise ObjectLibraryContractError(f"{path}: not a classifier checkpoint")
    architecture = checkpoint.get("architecture") or {}
    backbone = architecture.get("backbone", "scratch")
    base = architecture.get("base", 16)
    class_index = dict(architecture.get("class_index") or {})
    if not class_index:
        raise ObjectLibraryContractError(f"{path}: classifier checkpoint is missing class_index")
    model = ObjectClassifier(backbone=backbone, base=base, num_classes=len(class_index))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, class_index


def load_segmenter_checkpoint(path: Path):
    """Rebuild an ObjectSegmenter from its checkpoint."""
    from harvester.spec119.segmenter_model import ObjectSegmenter

    checkpoint = _load_checkpoint(path)
    if checkpoint.get("kind") != "segmenter":
        raise ObjectLibraryContractError(f"{path}: not a segmenter checkpoint")
    base = _require_base(checkpoint, path)
    model = ObjectSegmenter(base=base)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _read_loose_image(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def infer_classifier(model, class_index: dict[str, int], image: np.ndarray) -> dict[str, Any]:
    """Class + confidence + per-class probabilities for one HWC float image."""
    import torch

    index_class = {index: name for name, index in class_index.items()}
    with torch.no_grad():
        x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        probs = torch.softmax(model(x).squeeze(0), dim=0).numpy()
    predicted = int(probs.argmax())
    return {
        "predicted_class": index_class[predicted],
        "confidence": float(probs[predicted]),
        "per_class_probs": {index_class[i]: float(probs[i]) for i in range(len(probs))},
    }


def infer_segmenter(model, image: np.ndarray) -> np.ndarray:
    """Binary mask (uint8 255 foreground / 0 background) for one HWC float image."""
    import torch

    with torch.no_grad():
        x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        prediction = model(x).squeeze(0).squeeze(0).numpy()
    return ((prediction > 0.5).astype(np.uint8)) * 255


def main() -> int:
    """CLI per contracts/cli-contract.md §4 (loose PNGs, no store/ground truth)."""
    ap = argparse.ArgumentParser(
        description="Spec 119 loose-image inference (FR-013: no store/ground truth needed)"
    )
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--inputs", required=True, nargs="+", type=Path, help="loose PNG paths")
    ap.add_argument("--output", required=True, type=Path,
                    help="classifier: predictions JSON path; segmenter: mask-PNG output dir")
    args = ap.parse_args()

    try:
        raw = _load_checkpoint(args.checkpoint)
        kind = raw.get("kind")
        if kind == "classifier":
            model, class_index = load_classifier_checkpoint(args.checkpoint)
        elif kind == "segmenter":
            model = load_segmenter_checkpoint(args.checkpoint)
        else:
            raise ObjectLibraryContractError(
                f"{args.checkpoint}: unknown checkpoint kind {kind!r}"
            )
    except ObjectLibraryContractError as exc:
        raise SystemExit(str(exc)) from exc

    if kind == "classifier":
        results = []
        for image_path in args.inputs:
            image = _read_loose_image(image_path)
            results.append({"input": str(image_path), **infer_classifier(model, class_index, image)})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2), flush=True)
        print(f"wrote {args.output}", flush=True)
    else:
        from PIL import Image

        args.output.mkdir(parents=True, exist_ok=True)
        for image_path in args.inputs:
            mask = infer_segmenter(model, _read_loose_image(image_path))
            target = args.output / f"{image_path.stem}_mask.png"
            Image.fromarray(mask).save(target)
            print(f"wrote {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "infer_classifier",
    "infer_segmenter",
    "load_classifier_checkpoint",
    "load_segmenter_checkpoint",
    "main",
]
