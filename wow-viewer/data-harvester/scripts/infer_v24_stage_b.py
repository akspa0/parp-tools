"""FR-018: full V24 inference pipeline (Stage A -> prior -> Stage B -> height).

Single entry point: final height_257 = upsample(stage_a_prior, 257) + residual.
Deterministic (FR-019/SC-004): eval mode + torch.use_deterministic_algorithms;
--seed exists only to prove it does not affect the output.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harvester.v24 import stage_a, stage_b, train_common  # noqa: E402
from harvester.v24.tiles import HEIGHT_SCALE, RESIDUAL_SCALE, TileSource  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a-checkpoint", required=True)
    parser.add_argument("--stage-b-checkpoint", required=True)
    parser.add_argument("--v24-store", required=True)
    parser.add_argument("--v18-store", default=None)
    parser.add_argument("--row", type=int, default=None)
    parser.add_argument("--map", default=None)
    parser.add_argument("--tile-x", type=int, default=None)
    parser.add_argument("--tile-y", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-synth", action="store_true", help="minimap-only Stage A regime")
    parser.add_argument("--seed", type=int, default=94)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    train_common.set_determinism(args.seed)
    device = train_common.pick_device(args.device)

    source = TileSource(args.v24_store, args.v18_store)
    if args.row is not None:
        row = args.row
    else:
        from infer_v24_stage_a import resolve_row  # same directory

        row = resolve_row(source, args)
    record = source.load(row)
    if record.audit_empty:
        print("prior_unavailable=True (audit-empty tile); Stage B does not run")
        np.savez(args.output, prior_unavailable=np.array(True))
        return 0

    ckpt_a = torch.load(args.stage_a_checkpoint, map_location=device, weights_only=True)
    model_a = stage_a.StageAModel(base=ckpt_a["config"]["base"]).to(device)
    model_a.load_state_dict(ckpt_a["model_state"])
    model_a.eval()

    ckpt_b = torch.load(args.stage_b_checkpoint, map_location=device, weights_only=True)
    model_b = stage_b.StageBModel(base=ckpt_b["config"]["base"]).to(device)
    model_b.load_state_dict(ckpt_b["model_state"])
    model_b.eval()

    started = time.time()
    with torch.no_grad():
        xa_np, qa_np = stage_a.build_input(record, include_synth=not args.no_synth)
        xa = torch.from_numpy(xa_np)[None].to(device)
        qa = torch.from_numpy(qa_np)[None].to(device)
        outer, inner = model_a(xa, qa)
        prior_outer = outer[0].cpu().numpy() * HEIGHT_SCALE
        prior_inner = inner[0].cpu().numpy() * HEIGHT_SCALE

        xb_np, prior_up = stage_b.build_input(record, prior_outer, prior_inner)
        xb = torch.from_numpy(xb_np)[None].to(device)
        residual = model_b(xb)[0].cpu().numpy() * RESIDUAL_SCALE

    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.time() - started
    final = (prior_up + residual).astype(np.float32)

    np.savez(
        args.output,
        height_257=final,
        prior_outer=prior_outer.astype(np.float32),
        prior_inner=prior_inner.astype(np.float32),
        prior_upsampled=prior_up.astype(np.float32),
        residual=residual.astype(np.float32),
        prior_unavailable=np.array(False),
    )
    vram = train_common.peak_vram_gb()
    print(
        f"wrote {args.output} (row={row}, wall={wall:.3f} s"
        + (f", peak_vram={vram:.3f} GB" if vram is not None else "")
        + ")"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
