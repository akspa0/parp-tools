"""FR-013: Stage A inference — one tile (or row) -> WDL prior NPZ.

Deterministic: eval mode + torch.use_deterministic_algorithms(True); the seed
argument exists only to prove it does not change the output (FR-014).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harvester.v24 import stage_a, train_common  # noqa: E402
from harvester.v24.tiles import HEIGHT_SCALE, TileSource  # noqa: E402


def resolve_row(source: TileSource, args: argparse.Namespace) -> int:
    if args.row is not None:
        return args.row
    for i in range(len(source)):
        if (
            source.index["map"][i].lower() == args.map.lower()
            and source.index["tile_x"][i] == args.tile_x
            and source.index["tile_y"][i] == args.tile_y
        ):
            return i
    raise SystemExit(f"tile {args.map} ({args.tile_x}, {args.tile_y}) not in the V24 store")


def load_model(checkpoint_path: str, device: torch.device) -> stage_a.StageAModel:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = stage_a.StageAModel(base=checkpoint["config"]["base"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--v24-store", required=True)
    parser.add_argument("--v18-store", default=None)
    parser.add_argument("--row", type=int, default=None)
    parser.add_argument("--map", default=None)
    parser.add_argument("--tile-x", type=int, default=None)
    parser.add_argument("--tile-y", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-synth", action="store_true", help="minimap-only regime")
    parser.add_argument("--seed", type=int, default=94)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    train_common.set_determinism(args.seed)
    device = train_common.pick_device(args.device)

    source = TileSource(args.v24_store, args.v18_store)
    row = resolve_row(source, args)
    record = source.load(row)

    if record.audit_empty:
        mean = float(record.height.mean())
        np.savez(
            args.output,
            outer=np.full((17, 17), mean, np.float32),
            inner=np.full((16, 16), mean, np.float32),
            prior_unavailable=np.array(True),
        )
        print("prior_unavailable=True (audit-empty tile)")
        return 0

    model = load_model(args.checkpoint, device)
    x_np, q_np = stage_a.build_input(record, include_synth=not args.no_synth)
    x = torch.from_numpy(x_np)[None].to(device)
    q = torch.from_numpy(q_np)[None].to(device)

    started = time.time()
    with torch.no_grad():
        outer, inner = model(x, q)
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.time() - started

    np.savez(
        args.output,
        outer=(outer[0].cpu().numpy() * HEIGHT_SCALE).astype(np.float32),
        inner=(inner[0].cpu().numpy() * HEIGHT_SCALE).astype(np.float32),
        prior_unavailable=np.array(False),
    )
    vram = train_common.peak_vram_gb()
    print(
        f"wrote {args.output} (row={row}, wall={wall * 1000:.1f} ms"
        + (f", peak_vram={vram:.3f} GB" if vram is not None else "")
        + ")"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
