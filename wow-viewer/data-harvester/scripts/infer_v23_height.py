"""Run deterministic inference for the Spec 089 V23 height predictor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.v23.checkpoint import load_checkpoint
from harvester.v23.dataset import V23HeightDataset
from harvester.v23.inference import run_cai_inference
from harvester.v23.model import V23HeightPredictor
from transformers import DepthAnythingConfig


def _resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but CUDA is unavailable.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _effective_seed(args: argparse.Namespace) -> int:
    return 0 if bool(args.deterministic) else int(args.seed)


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[V23HeightPredictor, dict[str, Any]]:
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    config_dict = checkpoint.config.get("model_config")
    model_config = DepthAnythingConfig.from_dict(config_dict) if isinstance(config_dict, dict) else None
    model = V23HeightPredictor(
        in_channels=int(checkpoint.config.get("in_channels", 15)),
        config=model_config,
        load_pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint.model_state)
    model.eval()
    return model, checkpoint.config


def _save_preview(array: np.ndarray, output_path: Path) -> None:
    arr = np.asarray(array, dtype=np.float32)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo > 1e-8:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)
    Image.fromarray((arr * 255.0).astype(np.uint8), mode="L").save(output_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infer V23 terrain height.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--v22-store", type=Path, required=True)
    parser.add_argument("--build", required=True)
    parser.add_argument("--tiles", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cai-r", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--save-preview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    device = _resolve_device(args.device)
    effective_seed = _effective_seed(args)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)
    if bool(args.deterministic):
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    model, checkpoint_config = _load_model(args.checkpoint, device)
    dataset = V23HeightDataset(
        args.v22_store,
        build=args.build,
        input_mode=checkpoint_config.get("input_mode", "full"),
        tileset_prune_table=checkpoint_config.get("tileset_prune_table"),
    )
    inputs = [dataset[int(tile_index)]["input"] for tile_index in args.tiles]
    use_fp16 = bool(args.fp16) and device.type == "cuda"
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
        if int(args.cai_r) <= 1:
            prediction = model(inputs[0].unsqueeze(0).to(device)).metric_height.detach().cpu()[0, 0]
        else:
            prediction = run_cai_inference(model, inputs, cai_r=int(args.cai_r), device=device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_dir / "prediction.npz", metric_height=prediction.numpy())
    if bool(args.save_preview):
        _save_preview(prediction.numpy(), args.output_dir / "prediction.png")
    return {"output_dir": args.output_dir, "shape": tuple(prediction.shape)}


if __name__ == "__main__":
    main()
