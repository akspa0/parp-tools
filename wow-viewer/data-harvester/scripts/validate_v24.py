"""FR-025: V24 validation report (SC-001 .. SC-005).

Checks coverage + confidence bounds on the V24 store, Stage A / Stage B
quality against the trivial baselines, pipeline determinism across different
seeds (bit-identical outputs), and the 6 GB envelope (peak VRAM + wall time).
Emits output/v24_validation/<run_id>/report.json and a side-by-side preview PNG.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harvester.v24 import lattice, stage_a, stage_b, train_common  # noqa: E402
from harvester.v24 import store as v24_store  # noqa: E402
from harvester.v24.tiles import HEIGHT_SCALE, RESIDUAL_SCALE, TileSource  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent


def _stage_a_eval(model, records, device):
    """Returns dict with cheat/minimap-only L1 and the block_reduce baseline."""
    losses = {"cheat": [], "minimap_only": [], "baseline": []}
    real_num = real_den = synth_num = synth_den = 0.0
    with torch.no_grad():
        for record in records:
            targets = stage_a.build_target(record)
            to, ti, wo, wi = (torch.from_numpy(t)[None] for t in targets)

            for regime in ("cheat", "minimap_only"):
                x_np, q_np = stage_a.build_input(record, include_synth=regime == "cheat")
                x = torch.from_numpy(x_np)[None].to(device)
                q = torch.from_numpy(q_np)[None].to(device)
                po, pi = model(x, q)
                po, pi = po.cpu(), pi.cpu()
                loss = stage_a.weighted_l1(po, pi, to, ti, wo, wi).item() * HEIGHT_SCALE
                losses[regime].append(loss)

                if regime == "cheat":
                    for pred, tgt, weight, source in (
                        (po[0], to[0], wo[0], record.source_outer),
                        (pi[0], ti[0], wi[0], record.source_inner),
                    ):
                        diff = (pred - tgt).abs().numpy() * weight.numpy()
                        real_mask = (source == 0) & (weight.numpy() > 0)
                        synth_mask = (source == 1) & (weight.numpy() > 0)
                        real_num += float(diff[real_mask].sum())
                        real_den += float(weight.numpy()[real_mask].sum())
                        synth_num += float(diff[synth_mask].sum())
                        synth_den += float(weight.numpy()[synth_mask].sum())

            # Baseline: the float lattice sample of height_257 vs the prior.
            bo = torch.from_numpy(record.synth_outer / HEIGHT_SCALE)[None]
            bi = torch.from_numpy(record.synth_inner / HEIGHT_SCALE)[None]
            losses["baseline"].append(
                stage_a.weighted_l1(bo, bi, to, ti, wo, wi).item() * HEIGHT_SCALE
            )

    return {
        "val_l1_cheat": float(np.mean(losses["cheat"])),
        "val_l1_minimap_only": float(np.mean(losses["minimap_only"])),
        "block_reduce_baseline_l1": float(np.mean(losses["baseline"])),
        "val_l1_real_cells": real_num / real_den * HEIGHT_SCALE if real_den else None,
        "val_l1_synth_cells": synth_num / synth_den * HEIGHT_SCALE if synth_den else None,
    }


def _pipeline_eval(model_a, model_b, records, device):
    """Full pipeline final-height L1 vs the two Stage B baselines + timing."""
    final_l1, prior_l1, block_l1, walls = [], [], [], []
    preview = None
    with torch.no_grad():
        for record in records:
            started = time.time()
            x_np, q_np = stage_a.build_input(record, include_synth=True)
            outer, inner = model_a(
                torch.from_numpy(x_np)[None].to(device),
                torch.from_numpy(q_np)[None].to(device),
            )
            prior_outer = outer[0].cpu().numpy() * HEIGHT_SCALE
            prior_inner = inner[0].cpu().numpy() * HEIGHT_SCALE
            xb_np, prior_up = stage_b.build_input(record, prior_outer, prior_inner)
            residual = model_b(torch.from_numpy(xb_np)[None].to(device))[0].cpu().numpy()
            if device.type == "cuda":
                torch.cuda.synchronize()
            walls.append(time.time() - started)

            final = prior_up + residual * RESIDUAL_SCALE
            _, valid = stage_b.build_target(record, prior_up)
            mask = valid > 0.5
            final_l1.append(float(np.abs(record.height - final)[mask].mean()))
            prior_l1.append(float(np.abs(record.height - prior_up)[mask].mean()))
            br_up = lattice.upsample_prior_257(record.synth_outer, record.synth_inner)
            block_l1.append(float(np.abs(record.height - br_up)[mask].mean()))

            if preview is None:
                preview = (record, prior_up, final)

    return {
        "final_l1": float(np.mean(final_l1)),
        "upsampled_prior_l1": float(np.mean(prior_l1)),
        "block_reduce_bilinear_l1": float(np.mean(block_l1)),
        "mean_wall_s_per_tile": float(np.mean(walls)),
        "max_wall_s_per_tile": float(np.max(walls)),
    }, preview


def _determinism_check(args, run_dir: Path, row: int) -> bool:
    outputs = []
    for seed in (11, 22):
        out = run_dir / f"determinism_seed{seed}.npz"
        proc = subprocess.run(
            [
                sys.executable, str(SCRIPTS / "infer_v24_stage_b.py"),
                "--stage-a-checkpoint", args.stage_a_checkpoint,
                "--stage-b-checkpoint", args.stage_b_checkpoint,
                "--v24-store", args.v24_store,
                *(["--v18-store", args.v18_store] if args.v18_store else []),
                "--row", str(row),
                "--seed", str(seed),
                "--output", str(out),
            ],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            print(f"determinism run failed: {proc.stderr}", file=sys.stderr)
            return False
        with np.load(out) as data:
            outputs.append(data["height_257"].copy())
    return bool(np.array_equal(outputs[0], outputs[1]))


def _write_preview(preview, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    record, prior_up, final = preview
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    vmin, vmax = record.height.min(), record.height.max()
    for ax, (title, image) in zip(
        axes,
        [
            ("V18 height_257 (GT)", record.height),
            ("Stage A prior (upsampled)", prior_up),
            ("V24 final height", final),
            ("abs error (final)", np.abs(record.height - final)),
        ],
        strict=True,
    ):
        im = ax.imshow(image, vmin=vmin if "error" not in title else None,
                       vmax=vmax if "error" not in title else None,
                       cmap="terrain" if "error" not in title else "magma")
        ax.set_title(f"{title}\n{record.map_name} ({record.tile_x},{record.tile_y})")
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _minimap_only_eval(model_minimap, records, device):
    """Evaluate a StageAMinimapOnly model on the held-out val rows.

    Reuses the same world-unit weighted L1 as the cheat regime so the
    minimap-only number is directly comparable to the cheat regime's
    val_l1_cheat and the block_reduce baseline.
    """
    losses = {"minimap_only": [], "baseline": []}
    with torch.no_grad():
        for record in records:
            targets = stage_a.build_target(record)
            to, ti, wo, wi = (torch.from_numpy(t)[None] for t in targets)

            x_np = stage_a.build_minimap_only_input(record.cleaned_minimap)
            x = torch.from_numpy(x_np)[None].to(device)
            po, pi = model_minimap(x)
            po, pi = po.cpu(), pi.cpu()
            losses["minimap_only"].append(
                stage_a.weighted_l1(po, pi, to, ti, wo, wi).item() * HEIGHT_SCALE
            )

            bo = torch.from_numpy(record.synth_outer / HEIGHT_SCALE)[None]
            bi = torch.from_numpy(record.synth_inner / HEIGHT_SCALE)[None]
            losses["baseline"].append(
                stage_a.weighted_l1(bo, bi, to, ti, wo, wi).item() * HEIGHT_SCALE
            )

    return {
        "val_l1_minimap_only": float(np.mean(losses["minimap_only"])),
        "block_reduce_baseline_l1": float(np.mean(losses["baseline"])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v24-store", required=True)
    parser.add_argument("--v18-store", default=None)
    parser.add_argument("--stage-a-checkpoint", required=True)
    parser.add_argument("--stage-b-checkpoint", required=True)
    parser.add_argument(
        "--minimap-only-checkpoint", default=None,
        help="optional path to a StageAMinimapOnly checkpoint; adds the "
             "stage_a_minimap_only block + SC-002-MINIMAP gate to the report",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=94)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    train_common.set_determinism(args.seed, strict=False)
    device = train_common.pick_device(args.device)
    run_dir = Path(__file__).resolve().parents[2] / "output" / "v24_validation" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    source = TileSource(args.v24_store, args.v18_store)
    group = source.v24
    stats = v24_store.coverage_stats(group)

    # SC-001 confidence bound: conf >= 0.9 on >= 80% of real-WDL-available cells.
    real_available = np.asarray(group["wdl_prior_real_available"][:])
    high_conf = total_real = 0
    for name in ("wdl_prior_confidence_outer", "wdl_prior_confidence_inner"):
        conf = np.asarray(group[name][:])[real_available]
        high_conf += int((conf >= 0.9).sum())
        total_real += conf.size
    conf_bound = high_conf / total_real if total_real else None

    rows = source.usable_rows()
    _, val_rows = train_common.split_rows(rows, args.val_fraction, args.seed)
    val_records = [source.load(r) for r in val_rows]

    ckpt_a = torch.load(args.stage_a_checkpoint, map_location=device, weights_only=True)
    model_a = stage_a.StageAModel(base=ckpt_a["config"]["base"]).to(device)
    model_a.load_state_dict(ckpt_a["model_state"])
    model_a.eval()
    ckpt_b = torch.load(args.stage_b_checkpoint, map_location=device, weights_only=True)
    model_b = stage_b.StageBModel(base=ckpt_b["config"]["base"]).to(device)
    model_b.load_state_dict(ckpt_b["model_state"])
    model_b.eval()

    stage_a_report = _stage_a_eval(model_a, val_records, device)
    pipeline_report, preview = _pipeline_eval(model_a, model_b, val_records, device)
    deterministic = _determinism_check(args, run_dir, row=val_rows[0])
    peak_vram = train_common.peak_vram_gb()

    # Spec 096: optional minimap-only Stage A evaluation.
    stage_a_minimap_report = None
    minimap_only_beats_baseline = None
    if args.minimap_only_checkpoint is not None:
        ckpt_m = torch.load(args.minimap_only_checkpoint, map_location=device,
                            weights_only=True)
        model_m = stage_a.StageAMinimapOnly(base=ckpt_m["config"]["base"]).to(device)
        model_m.load_state_dict(ckpt_m["model_state"])
        model_m.eval()
        stage_a_minimap_report = _minimap_only_eval(model_m, val_records, device)
        stage_a_minimap_report["params"] = stage_a.parameter_count(model_m)
        minimap_only_beats_baseline = (
            stage_a_minimap_report["val_l1_minimap_only"]
            < stage_a_minimap_report["block_reduce_baseline_l1"]
        )

    checks = {
        "SC-001_coverage": stats["real_plus_synthetic_ratio_of_non_empty"] >= 0.95,
        "SC-001_confidence_bound": conf_bound is None or conf_bound >= 0.80,
        "SC-002_stage_a_beats_baseline": (
            stage_a_report["val_l1_cheat"] < stage_a_report["block_reduce_baseline_l1"]
        ),
        "SC-002_params": stage_a.parameter_count(model_a) <= 1_000_000,
        "SC-003_final_beats_prior": (
            pipeline_report["final_l1"] < pipeline_report["upsampled_prior_l1"]
        ),
        "SC-003_final_beats_block_reduce": (
            pipeline_report["final_l1"] < pipeline_report["block_reduce_bilinear_l1"]
        ),
        "SC-003_params": stage_b.parameter_count(model_b) <= 2_000_000,
        "SC-004_deterministic": deterministic,
        "SC-005_vram_under_4gb": peak_vram is None or peak_vram < 4.0,
        "SC-005_wall_under_3s": pipeline_report["max_wall_s_per_tile"] < 3.0,
    }
    if minimap_only_beats_baseline is not None:
        checks["SC-002-MINIMAP_minimap_only_beats_baseline"] = minimap_only_beats_baseline

    report = {
        "run_id": args.run_id,
        "v24_store": args.v24_store,
        "val_tiles": len(val_rows),
        "coverage": stats,
        "confidence_bound_real_cells": conf_bound,
        "stage_a": {**stage_a_report, "params": stage_a.parameter_count(model_a)},
        "pipeline": {**pipeline_report, "stage_b_params": stage_b.parameter_count(model_b)},
        "peak_vram_gb": peak_vram,
        "checks": checks,
        "all_pass": all(checks.values()),
    }
    if stage_a_minimap_report is not None:
        report["stage_a_minimap_only"] = stage_a_minimap_report
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if preview is not None:
        _write_preview(preview, run_dir / "preview.png")

    print(json.dumps({"checks": checks, "all_pass": report["all_pass"]}, indent=2))
    print(f"report: {report_path}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
