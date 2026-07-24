"""Spec 121 US1: Stage A trainer — minimap RGB -> 545-point WDL lattice (USER runs CUDA).

Mirrors ``harvester/spec117/lattice_train.py`` (same masked target contract, same required
``--held-out-split``, same tile-mean baseline, same warmup-aware stopping, same preview sheets)
with the Spec 121 additions:

- ``--architecture {lattice_net, mit_b0_lattice}``: the SegFormer-B0 backbone lane (default) or
  Spec 117's from-scratch fallback.
- ``--pretrained``: load ``--pretrained-hub-id``/``--pretrained-revision`` encoder weights
  (mit_b0_lattice only; download happens ONLY inside a confirmed user run — never in a dry-run).
- ``--object-mask-weight``: tile-coverage loss weighting from
  ``object_geometry_visible_mask_257`` (D-05; default 0.0 = parity; missing array = warn +
  disable + recorded, never a crash).
- Param band (SC-004): ``mit_b0_lattice`` runs refuse ``--confirm-run`` outside 3–30M params;
  the dry-run plan flags the violation either way.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from harvester.spec117.lattice_contract import (
    INNER_DIM,
    OUTER_DIM,
    architecture_identity,
    build_lattice_stage_run,
    identity_for_path,
)
from harvester.spec117.lattice_evaluate import (
    relief_stratified_metrics,
    tile_relief_and_baseline,
)
from harvester.spec117.lattice_model import (
    compute_lattice_tile_mean_baseline,
    encode_lattice_target,
    lattice_gradient_loss,
    lattice_loss,
    select_lattice_rows,
)
from harvester.spec121.lattice_backbone_model import (
    ARCHITECTURE_IDS,
    LATTICE_NET_ID,
    MIT_B0_HUB_ID,
    MIT_B0_LATTICE_ID,
    MIT_B0_LICENSE,
    build_stage_a_model,
    load_pretrained_lattice_encoder,
    parameter_band_ok,
)
from harvester.spec121.object_mask_tile_loss import (
    load_tile_coverages,
    touched_untouched_mae,
    weighted_lattice_loss,
)
from harvester.spec121.store_check import missing_required, object_mask_present, report
from harvester.v50.contracts import release_identity, require_store_release, validate_release
from harvester.v50.direct_geometry_train import apply_held_out_split
from harvester.v50.height_relative_evaluate import (
    compute_row_metrics,
    render_validation_sheet,
    select_fixed_preview_rows,
)
from harvester.v50.height_relative_train import (
    SOURCE_CHOICES,
    TrainerContractError,
    curriculum_identity,
    require_new_output,
    select_training_rows,
    validate_curriculum_contract,
    validate_source_selection,
)
from harvester.v50.lr_schedule import (
    make_onecycle_scheduler,
    warmup_complete,
    warmup_epochs_for,
)
from harvester.v50.model_stage_contract import sha256_file

STAGE = "lattice_prior"
LR_SCHEDULES = frozenset({"constant", "onecycle"})


def build_stage_a_plan(
    *,
    architecture: dict,
    architecture_id: str,
    source: str,
    train_rows: int,
    val_rows: int,
    excluded_train: int,
    excluded_val: int,
    batch_size: int,
    epochs: int,
    seed: int,
    lr: float,
    lr_schedule: str,
    object_mask_weight: float,
    object_mask_signal_present: bool,
    pretrained: dict | None,
    parameter_band: bool | None,
) -> dict:
    """Machine-readable no-training preview printed before CUDA allocation."""
    if batch_size < 1 or epochs < 1:
        raise TrainerContractError("batch size and epochs must both be positive")
    if lr_schedule not in LR_SCHEDULES:
        raise TrainerContractError(f"lr schedule must be one of {sorted(LR_SCHEDULES)}")
    if not 0.0 <= object_mask_weight <= 1.0:
        raise TrainerContractError(
            f"--object-mask-weight must be in [0, 1], got {object_mask_weight}"
        )
    return {
        "schema": "v121-stage-a-plan-v1",
        "stage": STAGE,
        "architecture": architecture,
        "architecture_id": architecture_id,
        "pretrained": pretrained,
        "parameter_band_ok": parameter_band,
        "source_filter": source,
        "split_counts": {"train": train_rows, "val": val_rows},
        "excluded_no_present_lattice": {"train": excluded_train, "val": excluded_val},
        "batch_size": batch_size,
        "epochs": epochs,
        "seed": seed,
        "optimizer": {"name": "AdamW", "learning_rate": lr, "weight_decay": 1e-4},
        "lr_schedule": lr_schedule,
        "train_steps_per_epoch": math.ceil(max(train_rows, 1) / batch_size),
        "deployment_inputs": ["minimap_rgb"],
        "training_target": "wdl_outer_17+wdl_inner_16 -> wdl_lattice_545",
        "object_mask_weight": object_mask_weight,
        "object_mask_signal_present": object_mask_signal_present,
        "no_gan_no_adversarial_no_generative_image": True,
    }


def _dense_lattice_field(outer: np.ndarray, inner: np.ndarray, size: int = 256) -> np.ndarray:
    """Bilinear-upsample both grids and average — identical to the downstream bridge so a
    preview shows exactly the dense field Stage B will consume."""
    import torch
    import torch.nn.functional as functional

    o = functional.interpolate(
        torch.from_numpy(np.asarray(outer, dtype=np.float32))[None, None],
        size=(size, size), mode="bilinear", align_corners=True,
    )[0, 0].numpy()
    i = functional.interpolate(
        torch.from_numpy(np.asarray(inner, dtype=np.float32))[None, None],
        size=(size, size), mode="bilinear", align_corners=True,
    )[0, 0].numpy()
    return (o + i) / 2.0


def _lattice_val_samples(model, group, rows: list[int], index, device) -> list[dict]:
    import torch

    model.eval()
    outer_count = OUTER_DIM * OUTER_DIM
    samples: list[dict] = []
    with torch.no_grad():
        for row in rows:
            rgb = np.asarray(group["minimap_rgb"][row], dtype=np.uint8)
            target, _mask, _tmin, _tmax = encode_lattice_target(
                np.asarray(group["wdl_outer_17"][row]),
                np.asarray(group["wdl_inner_16"][row]),
                np.asarray(group["wdl_outer_present"][row]),
                np.asarray(group["wdl_inner_present"][row]),
            )
            rgb_t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            pred = model(rgb_t).squeeze(0).cpu().numpy()
            dense_pred = _dense_lattice_field(
                pred[:outer_count].reshape(OUTER_DIM, OUTER_DIM),
                pred[outer_count:].reshape(INNER_DIM, INNER_DIM),
            )
            dense_truth = _dense_lattice_field(
                target[:outer_count].reshape(OUTER_DIM, OUTER_DIM),
                target[outer_count:].reshape(INNER_DIM, INNER_DIM),
            )
            metrics = compute_row_metrics(dense_pred, dense_truth)
            row_meta = index[row]
            samples.append({
                "rgb": rgb, "target": dense_truth, "predicted": dense_pred,
                "label": f"row {row}  {row_meta.get('map', '?')}",
                "metrics": metrics,
            })
    return samples


def main() -> int:
    import pyarrow.parquet as pq
    import torch
    import zarr
    from torch.utils.data import DataLoader, Dataset

    ap = argparse.ArgumentParser(description="Spec 121 Stage A: minimap -> WDL lattice (USER runs CUDA)")
    ap.add_argument("--store", required=True, type=Path, help="v50 Full-profile store carrying the WDL lattice arrays")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--run-id", required=True, help="immutable run identity, e.g. lattice-mit_b0-v1")
    ap.add_argument("--source", required=True, choices=sorted(SOURCE_CHOICES))
    ap.add_argument(
        "--held-out-split", required=True, type=Path,
        help="REQUIRED (FR-004): a v50-held-out-split-v1 directory. Refuses a leaky split "
             "(verified_violation_count != 0). No --val-key/--val-value fallback exists.",
    )
    ap.add_argument("--architecture", default=MIT_B0_LATTICE_ID, choices=sorted(ARCHITECTURE_IDS),
                    help="mit_b0_lattice = SegFormer-B0 backbone (default, the Spec 121 lane); "
                         "lattice_net = Spec 117 from-scratch fallback (known plateau risk)")
    ap.add_argument("--base", type=int, default=64, help="lattice_net encoder width (fallback arch only)")
    ap.add_argument("--pretrained", action="store_true",
                    help="mit_b0_lattice only: initialize the encoder from --pretrained-hub-id "
                         "(download happens ONLY in this confirmed user run, never in a dry-run)")
    ap.add_argument("--pretrained-hub-id", default=MIT_B0_HUB_ID)
    ap.add_argument("--pretrained-revision", default="main")
    ap.add_argument("--object-mask-weight", type=float, default=0.0,
                    help="tile-coverage loss weight from object_geometry_visible_mask_257 "
                         "(1 - w * coverage per tile; 0 = parity). Missing array: warn + disable "
                         "+ record, never crash.")
    ap.add_argument("--confirm-run", action="store_true",
                    help="launch CUDA training; without this flag only print the validated plan")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-schedule", default="constant", choices=sorted(LR_SCHEDULES))
    ap.add_argument("--patience", type=int, default=30,
                    help="TRAINING-loss plateau patience (post-warmup). val_mae NEVER stops a run. "
                         "0 = run the full --epochs.")
    ap.add_argument("--pct-start", type=float, default=0.1,
                    help="OneCycleLR warmup fraction; the early-stopper ignores stale epochs until "
                         "warmup completes.")
    ap.add_argument("--gradient-weight", type=float, default=0.25,
                    help="Loss-only 2D finite-difference gradient term on the lattice grids "
                         "(V7-ported; 0 = disable).")
    ap.add_argument("--init-weights", type=Path, default=None,
                    help="Optional same-architecture checkpoint to continue from (fails closed on "
                         "shape mismatch).")
    ap.add_argument("--seed", type=int, default=121)
    ap.add_argument("--release", default="v50.2", type=validate_release,
                    help="store release identity. v50.2 = the v50.1 signals PLUS the Spec 117 "
                         "WDL lattice arrays and Spec 118 object_geometry_visible_* arrays "
                         "(this lane's required substrate).")
    ap.add_argument("--workers", type=int, choices=[0], default=0)
    args = ap.parse_args()

    group = zarr.open_group(str(args.store), mode="r")
    try:
        require_store_release(group, args.release, store=args.store)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    index = pq.read_table(args.store / "index.parquet").to_pylist()
    try:
        array_lengths = {name: int(group[name].shape[0]) for name in group.array_keys()}
        validate_curriculum_contract(attrs=dict(group.attrs), array_lengths=array_lengths, index_rows=index)
        validate_source_selection(attrs=dict(group.attrs), source=args.source)
        missing = missing_required(group)
        if missing:
            raise TrainerContractError(
                f"store is missing required arrays {missing}; rebuild the store (Full profile) "
                "before training Stage A — see quickstart step 0"
            )
        selected_rows = select_training_rows(index, args.source)
    except TrainerContractError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        train_rows_all, val_rows_all, split_manifest = apply_held_out_split(
            index_rows=index, selected_rows=selected_rows, split_dir=args.held_out_split,
        )
    except TrainerContractError as exc:
        raise SystemExit(str(exc)) from exc

    train_rows, excluded_train = select_lattice_rows(group, train_rows_all)
    val_rows, excluded_val = select_lattice_rows(group, val_rows_all)
    if len(train_rows) < 32 or len(val_rows) < 8:
        raise SystemExit(
            f"insufficient rows with a present lattice sample after --source {args.source}: "
            f"train={len(train_rows)} val={len(val_rows)} "
            f"(excluded_no_present_lattice train={excluded_train} val={excluded_val})"
        )

    # Object-mask signal: warn + disable when the store lacks it (never crash).
    mask_signal = object_mask_present(group)
    object_mask_weight = float(args.object_mask_weight)
    if object_mask_weight > 0.0 and not mask_signal:
        print(
            "[object-mask] store lacks object_geometry_visible_mask_257; disabling "
            f"--object-mask-weight {object_mask_weight} (recorded as signal absent)",
            flush=True,
        )
        object_mask_weight = 0.0

    if args.pretrained and args.architecture != MIT_B0_LATTICE_ID:
        raise SystemExit("--pretrained only applies to --architecture mit_b0_lattice")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, config_payload = build_stage_a_model(
        args.architecture, base=args.base,
    )
    identity = architecture_identity(
        model, architecture_id=args.architecture, config=config_payload,
    )
    band_ok = parameter_band_ok(model) if args.architecture == MIT_B0_LATTICE_ID else None
    pretrained_record = (
        {"hub_id": args.pretrained_hub_id, "revision": args.pretrained_revision,
         "license": MIT_B0_LICENSE, "optional": True}
        if args.pretrained else None
    )
    plan = build_stage_a_plan(
        architecture=identity,
        architecture_id=args.architecture,
        source=args.source,
        train_rows=len(train_rows), val_rows=len(val_rows),
        excluded_train=excluded_train, excluded_val=excluded_val,
        batch_size=args.batch, epochs=args.epochs, seed=args.seed,
        lr=args.lr, lr_schedule=args.lr_schedule,
        object_mask_weight=object_mask_weight,
        object_mask_signal_present=mask_signal,
        pretrained=pretrained_record,
        parameter_band=band_ok,
    )
    plan["pct_start"] = args.pct_start if args.lr_schedule == "onecycle" else None
    plan["warmup_epochs"] = (
        warmup_epochs_for(args.pct_start, args.epochs, plan["train_steps_per_epoch"])
        if args.lr_schedule == "onecycle" else 0
    )
    plan["gradient_weight"] = args.gradient_weight
    plan["held_out_split"] = {
        "path": str((args.held_out_split / "split.json").resolve()),
        "sha256": sha256_file(args.held_out_split / "split.json"),
        "verified_violation_count": int(split_manifest["verified_violation_count"]),
        "absolute_comparison_to_prior_runs_invalid": True,
    }
    plan["store_check"] = report(group)
    print(json.dumps(plan, indent=2), flush=True)
    if not args.confirm_run:
        print("DRY RUN ONLY: add --confirm-run to launch user-owned CUDA training.", flush=True)
        return 0
    if band_ok is False:
        raise SystemExit(
            f"mit_b0_lattice parameter count {identity['parameter_count']} is outside the "
            "3-30M band (SC-004); refusing --confirm-run"
        )
    try:
        require_new_output(args.output)
    except TrainerContractError as exc:
        raise SystemExit(str(exc)) from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; user-run training refuses CPU.")
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_coverages = (
        load_tile_coverages(group, train_rows) if mask_signal else np.zeros(len(train_rows), dtype=np.float32)
    )
    val_coverages = (
        load_tile_coverages(group, val_rows) if mask_signal else np.zeros(len(val_rows), dtype=np.float32)
    )

    class RowDataset(Dataset):
        def __init__(self, rows: list[int], coverages: np.ndarray) -> None:
            self.rows = rows
            self.coverages = coverages

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, i: int):
            row = self.rows[i]
            rgb = np.asarray(group["minimap_rgb"][row], dtype=np.float32) / 255.0
            target, mask, _tile_min, _tile_max = encode_lattice_target(
                np.asarray(group["wdl_outer_17"][row]),
                np.asarray(group["wdl_inner_16"][row]),
                np.asarray(group["wdl_outer_present"][row]),
                np.asarray(group["wdl_inner_present"][row]),
            )
            return (
                torch.from_numpy(rgb).permute(2, 0, 1),
                torch.from_numpy(target),
                torch.from_numpy(mask),
                torch.tensor(float(self.coverages[i]), dtype=torch.float32),
            )

    val_targets_masks = []
    for row in val_rows:
        target, mask, _tile_min, _tile_max = encode_lattice_target(
            np.asarray(group["wdl_outer_17"][row]),
            np.asarray(group["wdl_inner_16"][row]),
            np.asarray(group["wdl_outer_present"][row]),
            np.asarray(group["wdl_inner_present"][row]),
        )
        val_targets_masks.append((target, mask))
    tile_mean_baseline = compute_lattice_tile_mean_baseline(val_targets_masks)

    device = torch.device("cuda")
    if args.pretrained:
        load_pretrained_lattice_encoder(
            model, hub_id=args.pretrained_hub_id, revision=args.pretrained_revision,
        )
        print(f"[pretrained] encoder initialized from {args.pretrained_hub_id}@"
              f"{args.pretrained_revision} ({MIT_B0_LICENSE})", flush=True)
    if args.init_weights is not None:
        try:
            init_ckpt = torch.load(args.init_weights, map_location="cpu", weights_only=False)
            model.load_state_dict(init_ckpt["model"])
        except (RuntimeError, KeyError) as exc:
            raise SystemExit(
                f"--init-weights {args.init_weights} is incompatible with "
                f"--architecture {args.architecture} (shape mismatch): {exc}"
            ) from exc
        print(f"[init-weights] initialized model from {args.init_weights} "
              f"(source epoch {init_ckpt.get('epoch', '?')})", flush=True)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        RowDataset(train_rows, train_coverages), batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, generator=train_generator,
    )
    val_loader = DataLoader(
        RowDataset(val_rows, val_coverages), batch_size=args.batch,
        num_workers=args.workers, pin_memory=True,
    )
    scheduler = None
    warmup_epochs = 0
    if args.lr_schedule == "onecycle":
        scheduler, warmup_epochs = make_onecycle_scheduler(
            opt, max_lr=args.lr, epochs=args.epochs,
            steps_per_epoch=len(train_loader), pct_start=args.pct_start,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    curriculum_id = curriculum_identity(args.store)
    # config_payload travels in plain form alongside its hash so downstream reconstruction
    # (bridge/infer) needs no Hub access — the Spec 117 lattice_config.base lesson.
    backbone_config = (
        {"architecture": args.architecture, "base": args.base}
        if args.architecture == LATTICE_NET_ID
        else {"architecture": args.architecture, "segformer_config": config_payload}
    )
    run_identity = {
        **release_identity(args.release),
        "stage": STAGE,
        "architecture": identity,
        "backbone_config": backbone_config,
        "pretrained": pretrained_record,
        "source_filter": args.source,
        "init_weights": str(args.init_weights.resolve()) if args.init_weights else None,
        "store": str(args.store.resolve()),
        "optimizer": plan["optimizer"],
        "schedule": {
            "max_epochs": args.epochs, "batch_size": args.batch,
            "patience": args.patience, "workers": args.workers, "seed": args.seed,
            "lr_schedule": args.lr_schedule,
            "pct_start": args.pct_start if args.lr_schedule == "onecycle" else None,
            "warmup_epochs": warmup_epochs,
        },
        "gradient_weight": args.gradient_weight,
        "object_mask_weight": object_mask_weight,
        "object_mask_signal_present": mask_signal,
        "held_out_split": plan["held_out_split"],
    }
    (args.output / "training_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (args.output / "run_identity.json").write_text(json.dumps(run_identity, indent=2), encoding="utf-8")

    fixed_preview_rows = select_fixed_preview_rows(val_rows, 8)
    per_epoch: list[dict] = []
    best_train = float("inf")
    best_val = float("inf")
    train_stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y, mask, coverage in train_loader:
            opt.zero_grad(set_to_none=True)
            x_device = x.to(device)
            y_device = y.to(device)
            mask_device = mask.to(device)
            predicted = model(x_device)
            if object_mask_weight > 0.0:
                loss = weighted_lattice_loss(
                    predicted, y_device, mask_device, coverage.to(device), object_mask_weight,
                )
            else:
                loss = lattice_loss(predicted, y_device, mask_device)
            if args.gradient_weight > 0:
                loss = loss + args.gradient_weight * lattice_gradient_loss(
                    predicted, y_device, mask_device
                )
            loss.backward()
            opt.step()
            if scheduler is not None:
                scheduler.step()
            train_losses.append(float(loss.detach().item()))
        model.eval()
        val_absolute_error = 0.0
        val_present = 0.0
        with torch.no_grad():
            for x, y, mask, _coverage in val_loader:
                y_device = y.to(device)
                mask_device = mask.to(device)
                predicted = model(x.to(device))
                val_absolute_error += float((torch.abs(predicted - y_device) * mask_device).sum().item())
                val_present += float(mask_device.sum().item())
        val_mae = val_absolute_error / max(val_present, 1.0)
        train_loss = float(np.mean(train_losses))
        per_epoch.append({"epoch": epoch, "train_loss": train_loss, "val_mae": val_mae})
        checkpoint = {
            **run_identity, "model": model.state_dict(), "epoch": epoch,
            "val_mae": val_mae, "curriculum_identity": curriculum_id,
        }
        torch.save(checkpoint, args.output / "checkpoint_last.pt")
        best_val = min(best_val, val_mae)
        # checkpoint_best.pt = best TRAINING loss (same rationale as Spec 117: no model in this
        # series has over-fit; val is a diagnostic, never the stopper or the selector).
        if train_loss < best_train - 1e-4:
            best_train = train_loss
            train_stale = 0
            torch.save(checkpoint, args.output / "checkpoint_best.pt")
            preview_dir = args.output / "validation" / "best_previews"
            preview_dir.mkdir(parents=True, exist_ok=True)
            render_validation_sheet(
                _lattice_val_samples(model, group, fixed_preview_rows, index, device),
                preview_dir / f"epoch_{epoch:04d}.png",
                title=f"stage A lattice fixed validation | epoch {epoch} | train {train_loss:.6f} | val {val_mae:.6f}",
            )
        elif warmup_complete(epoch, warmup_epochs):
            train_stale += 1
        warmup_tag = " (warmup)" if not warmup_complete(epoch, warmup_epochs) else ""
        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.6f} val_mae={val_mae:.6f} "
            f"tile_mean={tile_mean_baseline:.6f} best_train={best_train:.6f} best_val={best_val:.6f} "
            f"train_stale={train_stale}/{args.patience}{warmup_tag}",
            flush=True,
        )
        if args.patience > 0 and train_stale >= args.patience:
            print(f"[stop] training loss plateaued for {train_stale} epochs "
                  "(val is a diagnostic, never the stopper)", flush=True)
            break

    torch.save(checkpoint, args.output / "checkpoint_final.pt")
    best_train_record = min(per_epoch, key=lambda e: e["train_loss"])
    best_val_record = min(per_epoch, key=lambda e: e["val_mae"])
    best_checkpoint = torch.load(args.output / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    model.eval()
    beats_tile_mean = bool(best_val_record["val_mae"] < tile_mean_baseline)

    # Relief-stratified + outer/inner + object-touched honest evaluation (native masked MAE).
    outer_count = OUTER_DIM * OUTER_DIM
    per_tile_relief: list[dict] = []
    per_tile_mae: list[float] = []
    outer_abs = outer_n = inner_abs = inner_n = 0.0
    with torch.no_grad():
        for row in val_rows:
            rgb = np.asarray(group["minimap_rgb"][row], dtype=np.float32) / 255.0
            target, mask, tile_min, tile_max = encode_lattice_target(
                np.asarray(group["wdl_outer_17"][row]),
                np.asarray(group["wdl_inner_16"][row]),
                np.asarray(group["wdl_outer_present"][row]),
                np.asarray(group["wdl_inner_present"][row]),
            )
            rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
            pred = model(rgb_t).squeeze(0).cpu().numpy()
            present = mask > 0
            model_mae = float(np.abs(pred[present] - target[present]).mean())
            per_tile_mae.append(model_mae)
            relief, tile_mean_mae = tile_relief_and_baseline(target, mask, tile_min, tile_max)
            per_tile_relief.append(
                {"row": int(row), "model_mae": model_mae,
                 "tile_mean_mae": tile_mean_mae, "relief": relief}
            )
            o_mask = present[:outer_count]
            i_mask = present[outer_count:]
            if o_mask.any():
                outer_abs += float(np.abs(pred[:outer_count][o_mask] - target[:outer_count][o_mask]).sum())
                outer_n += float(o_mask.sum())
            if i_mask.any():
                inner_abs += float(np.abs(pred[outer_count:][i_mask] - target[outer_count:][i_mask]).sum())
                inner_n += float(i_mask.sum())
    relief_strata = relief_stratified_metrics(per_tile_relief, n_strata=4)
    relief_top = relief_strata["relief_subset"]
    touched_split = (
        touched_untouched_mae(np.asarray(per_tile_mae), val_coverages)
        if mask_signal else None
    )

    eval_dir = args.output / "validation" / "final_best"
    eval_dir.mkdir(parents=True, exist_ok=True)
    fixed_samples = _lattice_val_samples(model, group, fixed_preview_rows, index, device)
    render_validation_sheet(
        fixed_samples, eval_dir / "fixed_rows.png",
        title=f"stage A lattice fixed validation | best epoch {best_checkpoint['epoch']}",
    )
    all_val_samples = _lattice_val_samples(model, group, val_rows, index, device)
    worst_samples = sorted(
        all_val_samples, key=lambda s: (-float(s["metrics"]["mae"]), s["label"])
    )[:8]
    render_validation_sheet(
        worst_samples, eval_dir / "worst_cases.png",
        title=f"stage A lattice worst held-out rows | best epoch {best_checkpoint['epoch']}",
    )

    stage_run = build_lattice_stage_run(
        run_id=args.run_id,
        architecture=identity,
        curriculum=identity_for_path(args.store / "index.parquet", display_path=str(args.store.resolve())),
        checkpoint={
            **identity_for_path(args.output / "checkpoint_best.pt"),
            "best_epoch": int(best_checkpoint["epoch"]),
        },
        baselines={"tile_mean": {"val_mae": tile_mean_baseline}},
        visual_evidence={
            "fixed_rows": "validation/final_best/fixed_rows.png",
            "worst_cases": "validation/final_best/worst_cases.png",
            "best_previews": "validation/best_previews/",
        },
        metrics={
            "selection": "best_training_loss",
            "checkpoint_epoch": int(best_checkpoint["epoch"]),
            "best_train_epoch": best_train_record["epoch"],
            "best_train_loss": best_train_record["train_loss"],
            "final_epoch": per_epoch[-1]["epoch"],
            "final_train_loss": per_epoch[-1]["train_loss"],
            "final_val_mae": per_epoch[-1]["val_mae"],
            "best_val_epoch": best_val_record["epoch"],
            "best_val_mae": best_val_record["val_mae"],
            "beats_tile_mean_baseline": beats_tile_mean,
            # SC-001 gate: >= 15% below tile-mean.
            "sc001_margin_vs_tile_mean": (
                (tile_mean_baseline - best_val_record["val_mae"]) / tile_mean_baseline
                if tile_mean_baseline > 0 else 0.0
            ),
            "sc001_pass": bool(
                tile_mean_baseline > 0
                and (tile_mean_baseline - best_val_record["val_mae"]) / tile_mean_baseline >= 0.15
            ),
            "outer_mae": outer_abs / max(outer_n, 1.0),
            "inner_mae": inner_abs / max(inner_n, 1.0),
            "relief_stratified": relief_strata,
            "object_mask_weight": object_mask_weight,
            "object_mask_signal_present": mask_signal,
            "object_touched_split": touched_split,
            "pretrained": pretrained_record,
            "held_out_split": plan["held_out_split"],
            "excluded_no_present_lattice": plan["excluded_no_present_lattice"],
            "structural_failure_epoch1_best": best_train_record["epoch"] == 1,
        },
    )
    (args.output / "model_stage_run.json").write_text(json.dumps(stage_run, indent=2), encoding="utf-8")
    (args.output / "training_summary.json").write_text(
        json.dumps({"per_epoch_metrics": per_epoch, "model_stage_run": stage_run["run_id"]}, indent=2),
        encoding="utf-8",
    )
    print(
        f"[relief-stratified] highest-relief stratum ({relief_top['n_tiles']} tiles, "
        f"relief {relief_top['relief_min']:.1f}..{relief_top['relief_max']:.1f}): "
        f"model_mae={relief_top['model_mae']:.6f} vs tile_mean={relief_top['tile_mean_mae']:.6f} "
        f"beats_tile_mean={relief_top['model_beats_tile_mean']}",
        flush=True,
    )
    if stage_run["metrics"]["structural_failure_epoch1_best"]:
        print("STRUCTURAL FAILURE: training loss never improved past epoch 1; run is not a success.",
              flush=True)
        return 1
    print(
        f"selection=best_training_loss checkpoint_epoch={best_checkpoint['epoch']} "
        f"best_val_mae={best_val_record['val_mae']:.6f}(epoch {best_val_record['epoch']}) "
        f"tile_mean_baseline={tile_mean_baseline:.6f} "
        f"sc001_margin={stage_run['metrics']['sc001_margin_vs_tile_mean']:.4f} "
        f"sc001_pass={stage_run['metrics']['sc001_pass']} "
        f"promotion=pending(user visual gate; G1 fail = lane stops with recorded negative)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
