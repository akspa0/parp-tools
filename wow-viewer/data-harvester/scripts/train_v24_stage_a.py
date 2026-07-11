"""FR-012: train Stage A (minimap -> WDL prior correlation).

Target: the merged WDL prior. Sample weight: per-cell confidence, with
learned-fill cells excluded. The synthetic-WDL cheat channel is dropped with
probability --synth-dropout so the model also learns the minimap-only regime.
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
from harvester.v24.tiles import HEIGHT_SCALE, MultiTileSource, TileSource  # noqa: E402


def _build_source(v24_stores: list[str], v18_stores: list[str] | None) -> TileSource | MultiTileSource:
    if v18_stores and len(v18_stores) != len(v24_stores):
        raise ValueError(
            f"--v18-store count ({len(v18_stores)}) must match --v24-store count ({len(v24_stores)}) when given"
        )
    if len(v24_stores) == 1:
        return TileSource(v24_stores[0], (v18_stores[0] if v18_stores else None))
    pairs = [
        (v24, v18_stores[i] if v18_stores else None) for i, v24 in enumerate(v24_stores)
    ]
    return MultiTileSource(pairs)


def _load_tensors(source: TileSource | MultiTileSource, rows: list[int], label: str = "load",
                  minimap_only: bool = False, guided: bool = False, dav2: bool = False):
    inputs, quincunxes, t_outer, t_inner = [], [], [], []
    w_outer, w_inner, s_outer, s_inner = [], [], [], []
    n = len(rows)
    step = max(1, n // 20)
    started = time.time()
    for i, row in enumerate(rows):
        record = source.load(row)
        if dav2:
            # DA-V2 uses 256×256 input (native minimap resolution).
            if guided:
                x = stage_a.build_dav2_input(record.cleaned_minimap, normal=record.normal)
            else:
                x = stage_a.build_dav2_input(record.cleaned_minimap)
            q = np.zeros((33, 33), dtype=np.float32)
        elif minimap_only and guided:
            # 9-channel input: minimap + normal + normal-Sobel.
            x = stage_a.build_guided_input(
                record.cleaned_minimap, normal=record.normal,
            )
            q = np.zeros((33, 33), dtype=np.float32)
        elif minimap_only:
            x = stage_a.build_minimap_only_input(record.cleaned_minimap)
            q = np.zeros((33, 33), dtype=np.float32)
        else:
            x, q = stage_a.build_input(record, include_synth=True)
        inputs.append(x)
        quincunxes.append(q)
        outer, inner, wo, wi = stage_a.build_target(record)
        t_outer.append(outer)
        t_inner.append(inner)
        w_outer.append(wo)
        w_inner.append(wi)
        s_outer.append(record.source_outer)
        s_inner.append(record.source_inner)
        if (i + 1) % step == 0 or (i + 1) == n:
            elapsed = time.time() - started
            pct = 100.0 * (i + 1) / n
            eta = elapsed / (i + 1) * (n - i - 1) if i else 0.0
            print(
                f"[{label}] {i + 1}/{n} tiles ({pct:.0f}%) "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
    return (
        torch.from_numpy(np.stack(inputs)),
        torch.from_numpy(np.stack(quincunxes)),
        torch.from_numpy(np.stack(t_outer)),
        torch.from_numpy(np.stack(t_inner)),
        torch.from_numpy(np.stack(w_outer)),
        torch.from_numpy(np.stack(w_inner)),
        torch.from_numpy(np.stack(s_outer)),
        torch.from_numpy(np.stack(s_inner)),
    )


def _drop_synth(
    x: torch.Tensor, q: torch.Tensor, drop_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero the synth channel (11), presence flag (12), and quincunx anchor."""
    drop_mask = drop_mask.to(x.device)
    x_out = x.clone()
    x_out[drop_mask, 11] = 0.0
    x_out[drop_mask, 12] = 0.0
    q_out = q.clone()
    q_out[drop_mask] = 0.0
    return x_out, q_out


def _eval_split(model, x, q, to, ti, wo, wi, so, si, device, include_synth: bool,
                minimap_only: bool = False, use_tta: int = 0):
    """Weighted L1 plus real-vs-synthetic cell split (world units)."""
    model.eval()
    with torch.no_grad():
        if minimap_only:
            xb = x
        elif include_synth:
            xb, qb = x, q
        else:
            xb, qb = _drop_synth(x, q, torch.ones(x.shape[0], dtype=torch.bool))
        # TTA: if --use-tta N was given, run N augmented passes and average.
        if use_tta and use_tta > 1 and minimap_only:
            po, pi = stage_a.tta_predict(
                model, xb.to(device), n_aug=use_tta,
            )
        elif use_tta and use_tta > 1:
            po, pi = stage_a.tta_predict(
                model, xb.to(device), n_aug=use_tta,
            )
        else:
            po, pi = (model(xb.to(device)) if minimap_only
                      else model(xb.to(device), qb.to(device)))
        # to/wo/etc. may be GPU-resident (train_common.gpu_resident) or CPU;
        # match whichever device val_data actually lives on.
        po, pi = po.to(to.device), pi.to(to.device)

    loss = stage_a.weighted_l1(po, pi, to, ti, wo, wi).item() * HEIGHT_SCALE

    def cell_l1(source_value: int) -> float | None:
        mask_o = (so == source_value).float() * wo.sign()
        mask_i = (si == source_value).float() * wi.sign()
        denom = mask_o.sum() + mask_i.sum()
        if denom.item() == 0:
            return None
        num = (mask_o * (po - to).abs()).sum() + (mask_i * (pi - ti).abs()).sum()
        return (num / denom).item() * HEIGHT_SCALE

    return loss, cell_l1(0), cell_l1(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v24-store", required=True, nargs="+",
                         help="one or more V24 stores; multiple stores (e.g. different builds) are concatenated")
    parser.add_argument("--v18-store", default=None, nargs="*",
                         help="matching V18 store overrides, one per --v24-store (omit to use each store's own v18_store_path attr)")
    parser.add_argument("--output", required=True, help="run directory (created)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=94)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--minimap-only", action="store_true", default=False,
                        help="train 3-channel minimap-only Stage A (no alpha/normal/mcnr/quincunx)")
    parser.add_argument("--guided", action="store_true", default=False,
                        help="train 9-channel minimap+normal+Sobel Stage A "
                             "(StageAMinimapOnlyGuided). Requires the V18 store to have "
                             "normal_xyz and object_precise_mask populated. Combine "
                             "with --minimap-only. Uses RAdam + OneCycleLR by default.")
    parser.add_argument("--dav2", action="store_true", default=False,
                        help="train V24.1 Stage A with DA-V2-Small pretrained encoder + "
                             "LoRA + DPT head (Spec 101). Uses 256x256 input, SiLogLoss "
                             "(or hybrid), and a fixed scheduler. Combine with "
                             "--minimap-only (3ch) or --guided (9ch). Default lr=5e-6. "
                             "DA-V2 uses 256x256 inputs (16x more RAM than 64x64); "
                             "use --limit 500 --no-gpu-resident-data on 12GB GPUs.")
    parser.add_argument("--loss-type", choices=["l1", "silog", "hybrid"], default="l1",
                        help="loss function: l1 (weighted L1), silog (scale-invariant log), "
                             "hybrid (0.7*silog + 0.3*l1). Default for --dav2 is hybrid.")
    parser.add_argument("--scheduler", choices=["onecycle", "cosine"], default="cosine",
                        help="LR scheduler: onecycle (OneCycleLR, per-batch stepping) or "
                             "cosine (CosineAnnealingLR, per-epoch stepping). Default: cosine.")
    parser.add_argument("--silog-weight", type=float, default=0.7,
                        help="weight for SiLogLoss in hybrid mode (default 0.7)")
    parser.add_argument("--l1-weight", type=float, default=0.3,
                        help="weight for L1 in hybrid mode (default 0.3)")
    parser.add_argument("--silog-shift", type=float, default=10.0,
                        help="constant added before log in SiLogLoss (normalized space; "
                             "default 10.0 = 1000 world units)")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA adapter rank for DA-V2 encoder (default 16; "
                             "try 32 or 64 for more capacity if overfitting)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="optimizer weight decay (default 1e-4; try 1e-3 to reduce overfitting)")
    parser.add_argument("--8bit-optimizer", dest="use_8bit_optimizer",
                        action="store_true", default=False,
                        help="use bitsandbytes 8-bit Adam optimizer (4x less optimizer "
                             "state RAM). Requires bitsandbytes installed.")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False,
                        help="enable gradient checkpointing on the DA-V2 encoder "
                             "(trades compute for ~2-3x less activation VRAM)")
    parser.add_argument("--gan", action="store_true", default=False,
                        help="enable PatchGAN discriminator adversarial loss (Spec 101 Slice 7). "
                             "Lambda ramps from 0 to --adv-lambda-max over epochs 5-30.")
    parser.add_argument("--adv-lambda-max", type=float, default=0.1,
                        help="maximum adversarial loss weight (default 0.1)")
    parser.add_argument("--adv-lambda-ramp-epochs", type=int, default=25,
                        help="number of epochs to ramp lambda from 0 to max (default 25, "
                             "starting at epoch 5)")
    parser.add_argument("--use-tta", type=int, default=0,
                        help="if >0, evaluate the validation set with N-augmentation "
                             "TTA (1=no TTA, 5=full TTA). Doubles validation wall-time but "
                             "reduces variance by sqrt(N).")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--synth-dropout", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-interval", type=int, default=1,
                        help="print per-batch progress every N batches (1 = every batch)")
    parser.add_argument("--patience", type=int, default=0,
                        help="early-stop after N epochs without val improvement (0 = run all epochs)")
    parser.add_argument("--cudnn-benchmark", dest="cudnn_benchmark", action="store_true", default=True,
                        help="fast cuDNN kernel autotune + TF32 for training (default on; no bearing on inference determinism)")
    parser.add_argument("--no-cudnn-benchmark", dest="cudnn_benchmark", action="store_false")
    parser.add_argument("--gpu-resident-data", dest="gpu_resident_data", action="store_true", default=True,
                        help="keep the whole train/val tensor set resident on the GPU when it fits (default on)")
    parser.add_argument("--no-gpu-resident-data", dest="gpu_resident_data", action="store_false")
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--autotune-batch-size", action="store_true", default=False,
                        help="probe --autotune-batch-candidates and pick the largest that fits before training")
    parser.add_argument("--autotune-batch-candidates", nargs="+", type=int,
                        default=[16, 32, 64, 96, 128, 192, 256, 384, 512])
    parser.add_argument("--autotune-safety-factor", type=float, default=0.85)
    args = parser.parse_args()

    # DA-V2 defaults: hybrid loss, cosine scheduler, lr=5e-6.
    if args.dav2:
        if args.loss_type == "l1":
            args.loss_type = "hybrid"
        if args.lr == 1e-3:
            args.lr = 1e-4  # LoRA needs higher LR than full fine-tune (DA-V2 uses 5e-6 for full).
        if args.batch_size == 64:
            args.batch_size = 8  # DA-V2 is 25M params; start conservative.

    train_common.set_determinism(args.seed, strict=False)
    train_common.configure_perf(args.cudnn_benchmark)
    device = train_common.pick_device(args.device)
    run_dir = Path(args.output)
    logger = train_common.RunLogger(run_dir)
    print(f"stage A training: device={device} run_dir={run_dir}")

    source = _build_source(args.v24_store, args.v18_store)
    rows = source.usable_rows()
    if args.limit:
        rows = rows[: args.limit]
    train_rows, val_rows = train_common.split_rows(rows, args.val_fraction, args.seed)
    print(f"tiles: {len(rows)} usable -> {len(train_rows)} train / {len(val_rows)} val")

    started_load = time.time()
    # Preload V18 data in one sequential Zarr pass (no per-tile random-access seeks).
    source.preload(train_rows + val_rows)
    train_data = _load_tensors(source, train_rows, label="load train",
                               minimap_only=args.minimap_only,
                               guided=args.guided,
                               dav2=args.dav2)
    val_data = _load_tensors(source, val_rows, label="load val",
                             minimap_only=args.minimap_only,
                             guided=args.guided,
                             dav2=args.dav2)
    # Free the V18 preload cache to reclaim RAM (tensors are already extracted).
    source._v18_cache.clear()
    import gc
    gc.collect()
    print(f"loaded tensors in {time.time() - started_load:.1f}s (V18 cache freed)", flush=True)

    if args.gpu_resident_data:
        train_data = train_common.gpu_resident(train_data, device)
        val_data = train_common.gpu_resident(val_data, device)
        print(f"train/val tensors resident={train_data[0].is_cuda if device.type == 'cuda' else False}", flush=True)

    if args.dav2:
        # V24.1: DA-V2-Small pretrained encoder + LoRA + DPT head.
        in_ch = 9 if args.guided else 3
        model = stage_a.StageADAV2(
            in_channels=in_ch,
            load_pretrained=True,
            local_files_only=True,
            lora_rank=args.lora_rank,
        ).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params = total_params  # for metrics dict
        print(f"stage A (DA-V2) params: {total_params} total, {trainable} trainable "
              f"({'9ch guided' if args.guided else '3ch minimap-only'})")
        # Only train LoRA + patch proj + head (backbone is frozen).
        # Enable gradient checkpointing if requested (before optimizer).
        if args.gradient_checkpointing and hasattr(model.encoder, 'gradient_checkpointing_enable'):
            model.encoder.gradient_checkpointing_enable()
            print("gradient checkpointing enabled on DA-V2 encoder")

        if args.use_8bit_optimizer:
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(
                    model.trainable_parameters(), lr=args.lr,
                    weight_decay=args.weight_decay,
                )
                print("optimizer: bitsandbytes 8-bit AdamW")
            except ImportError:
                print("WARNING: bitsandbytes not installed; falling back to AdamW")
                optimizer = torch.optim.AdamW(
                    model.trainable_parameters(), lr=args.lr,
                    weight_decay=args.weight_decay,
                )
        else:
            optimizer = torch.optim.AdamW(
                model.trainable_parameters(), lr=args.lr,
                weight_decay=args.weight_decay,
            )
    elif args.minimap_only and args.guided:
        model = stage_a.StageAMinimapOnlyGuided().to(device)
        params = stage_a.parameter_count(model)
        print(f"stage A params: {params} (minimap-only 9ch guided)")
        optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.minimap_only:
        model = stage_a.StageAMinimapOnly().to(device)
        params = stage_a.parameter_count(model)
        print(f"stage A params: {params} (minimap-only 3ch)")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        model = stage_a.StageAModel().to(device)
        params = stage_a.parameter_count(model)
        print(f"stage A params: {params} (13ch full)")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Unpack training data before scheduler setup (n is needed for n_batches).
    minimap_only = args.minimap_only
    x, q, to, ti, wo, wi, so, si = train_data
    n = x.shape[0]

    # Scheduler: cosine (per-epoch) or onecycle (per-batch, fixed total_steps).
    n_batches = (n + args.batch_size - 1) // args.batch_size
    if args.scheduler == "onecycle" or args.guided:
        if args.scheduler == "onecycle":
            # Fixed: total_steps = n_batches * epochs (not just epochs).
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=args.lr,
                total_steps=n_batches * args.epochs, pct_start=0.05,
            )
            print(f"optimizer: AdamW + OneCycleLR (max_lr={args.lr}, "
                  f"total_steps={n_batches * args.epochs})")
        else:
            # Guided default: RAdam + OneCycleLR (fixed total_steps).
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=args.lr,
                total_steps=n_batches * args.epochs, pct_start=0.05,
            )
            print(f"optimizer: RAdam + OneCycleLR (max_lr={args.lr}, "
                  f"total_steps={n_batches * args.epochs})")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs,
        )
        print(f"optimizer: AdamW + CosineAnnealingLR (lr={args.lr})")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)

    # GAN discriminator setup (Spec 101 Slice 7).
    model_D = None
    opt_D = None
    if args.gan:
        from harvester.v24 import discriminator as v24_disc
        model_D = v24_disc.WDLDiscriminator(in_channels=1, base=32, n_layers=3).to(device)
        opt_D = torch.optim.Adam(model_D.parameters(), lr=2e-4, betas=(0.5, 0.999))
        print(f"GAN enabled: discriminator params={v24_disc.parameter_count(model_D)}, "
              f"lambda_max={args.adv_lambda_max}, ramp_epochs={args.adv_lambda_ramp_epochs}")

    if args.autotune_batch_size:
        candidates = [c for c in sorted(args.autotune_batch_candidates) if c <= n]
        if candidates:
            snapshot = train_common.snapshot_state(model, optimizer)

            def _try_step(bs: int) -> None:
                idx = torch.randperm(n)[:bs].to(x.device)
                if minimap_only:
                    xb = x[idx].to(device, non_blocking=True)
                else:
                    drop = torch.rand(len(idx)) < args.synth_dropout
                    xb, qb = _drop_synth(x[idx], q[idx], drop)
                    xb, qb = xb.to(device, non_blocking=True), qb.to(device, non_blocking=True)
                tob, tib = to[idx].to(device, non_blocking=True), ti[idx].to(device, non_blocking=True)
                wob, wib = wo[idx].to(device, non_blocking=True), wi[idx].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                    po, pi = model(xb) if minimap_only else model(xb, qb)
                    loss = stage_a.weighted_l1(po, pi, tob, tib, wob, wib)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            args.batch_size = train_common.autotune_batch_size(
                device, candidates, _try_step, safety_fraction=args.autotune_safety_factor
            )
            train_common.restore_state(model, optimizer, snapshot)
            scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)
            print(f"autotune selected batch_size={args.batch_size}", flush=True)

    n_batches = (n + args.batch_size - 1) // args.batch_size
    generator = torch.Generator().manual_seed(args.seed)
    best_val = float("inf")
    stopper = train_common.EarlyStopping(patience=args.patience, min_delta=1e-4)
    epoch_started = time.time()

    for epoch in range(1, args.epochs + 1):
        # GAN lambda schedule: 0 for first 5 epochs, ramp to max over next ramp_epochs.
        if args.gan and epoch >= 5:
            ramp_progress = min(1.0, (epoch - 5) / max(1, args.adv_lambda_ramp_epochs))
            lambda_adv = args.adv_lambda_max * ramp_progress
        else:
            lambda_adv = 0.0
        model.train()
        if model_D is not None:
            model_D.train()
        perm = torch.randperm(n, generator=generator)
        epoch_loss, batches = 0.0, 0
        d_loss_epoch = 0.0
        g_adv_loss_epoch = 0.0
        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size].to(x.device)
            if args.dav2 or minimap_only:
                xb = x[idx].to(device, non_blocking=True)
            else:
                drop = torch.rand(len(idx), generator=generator) < args.synth_dropout
                xb, qb = _drop_synth(x[idx], q[idx], drop)
                xb, qb = xb.to(device, non_blocking=True), qb.to(device, non_blocking=True)
            tob, tib = to[idx].to(device, non_blocking=True), ti[idx].to(device, non_blocking=True)
            wob, wib = wo[idx].to(device, non_blocking=True), wi[idx].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                if args.dav2 or minimap_only:
                    po, pi = model(xb)
                else:
                    po, pi = model(xb, qb)
                # Loss: L1, SiLogLoss, or hybrid.
                if args.loss_type == "silog":
                    loss = stage_a.SiLogLoss(shift=args.silog_shift)(
                        po, pi, tob, tib, wob, wib,
                    )
                elif args.loss_type == "hybrid":
                    loss = stage_a.hybrid_loss(
                        po, pi, tob, tib, wob, wib,
                        silog_weight=args.silog_weight,
                        l1_weight=args.l1_weight,
                        silog_shift=args.silog_shift,
                    )
                else:
                    loss = stage_a.weighted_l1(po, pi, tob, tib, wob, wib)
                # GAN adversarial loss (if enabled and lambda > 0).
                if model_D is not None and lambda_adv > 0:
                    from harvester.v24.discriminator import _render_quincunx_33
                    gen_prior = _render_quincunx_33(po, pi).unsqueeze(1)
                    g_adv_logits = model_D(gen_prior)
                    bce = torch.nn.BCEWithLogitsLoss()
                    real_label = torch.ones(xb.shape[0], 1, device=device)
                    g_adv = bce(g_adv_logits.mean(dim=[2, 3]), real_label)
                    loss = loss + lambda_adv * g_adv
                    g_adv_loss_epoch += g_adv.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # Discriminator step (if GAN is enabled).
            if model_D is not None and lambda_adv > 0:
                opt_D.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                    with torch.no_grad():
                        gen_prior = _render_quincunx_33(po, pi).unsqueeze(1)
                    # Real prior: render from targets.
                    real_prior = _render_quincunx_33(tob, tib).unsqueeze(1)
                    d_real = model_D(real_prior)
                    d_fake = model_D(gen_prior.detach())
                    bce = torch.nn.BCEWithLogitsLoss()
                    real_label = torch.ones(xb.shape[0], 1, device=device)
                    fake_label = torch.zeros(xb.shape[0], 1, device=device)
                    d_loss = (bce(d_real.mean(dim=[2, 3]), real_label) +
                              bce(d_fake.mean(dim=[2, 3]), fake_label)) * 0.5
                scaler.scale(d_loss).backward()
                scaler.step(opt_D)
                scaler.update()
                d_loss_epoch += d_loss.item()
            # OneCycleLR steps per batch; CosineAnnealingLR steps per epoch.
            if args.scheduler == "onecycle":
                scheduler.step()
            epoch_loss += loss.item()
            batches += 1

            if batches % args.log_interval == 0 or start + args.batch_size >= n:
                elapsed = time.time() - epoch_started
                pct = 100.0 * batches / n_batches
                eta = elapsed / batches * (n_batches - batches) if batches else 0.0
                print(
                    f"epoch {epoch} batch {batches}/{n_batches} ({pct:.0f}%) "
                    f"loss={loss.item() * HEIGHT_SCALE:.4f} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )

        # CosineAnnealingLR steps per epoch; OneCycleLR already stepped per batch.
        if args.scheduler != "onecycle":
            scheduler.step()
        if minimap_only:
            val_loss, val_real, val_synth = _eval_split(
                model, *val_data, device=device, include_synth=True,
                minimap_only=True, use_tta=args.use_tta,
            )
        else:
            val_loss, val_real, val_synth = _eval_split(
                model, *val_data, device=device, include_synth=True
            )
            val_nosynth, _, _ = _eval_split(
                model, *val_data, device=device, include_synth=False
            )

        train_loss_world = epoch_loss / max(1, batches) * HEIGHT_SCALE
        log_kw = dict(
            epoch=epoch,
            train_loss=train_loss_world,
            val_l1=val_loss,
            val_l1_real_cells=val_real if val_real is not None else -1.0,
            val_l1_synth_cells=val_synth if val_synth is not None else -1.0,
            lr=scheduler.get_last_lr()[0],
        )
        if model_D is not None:
            log_kw["d_loss"] = d_loss_epoch / max(1, batches)
            log_kw["g_adv_loss"] = g_adv_loss_epoch / max(1, batches)
            log_kw["lambda_adv"] = lambda_adv
        if not minimap_only:
            log_kw["val_l1_minimap_only"] = val_nosynth
        logger.log_epoch(**log_kw)

        if val_loss < best_val:
            best_val = val_loss
            if args.dav2:
                in_channels = 9 if args.guided else 3
                model_type = "dav2"
            elif args.minimap_only:
                in_channels = stage_a.IN_CHANNELS_MINIMAP_ONLY
                model_type = "minimap_only"
            else:
                in_channels = stage_a.IN_CHANNELS
                model_type = "full"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "base": 28,
                        "in_channels": in_channels,
                        "minimap_only": bool(args.minimap_only),
                        "model_type": model_type,
                        "loss_type": args.loss_type,
                        "scheduler_type": args.scheduler,
                        "guided": bool(args.guided),
                        "dav2": bool(args.dav2),
                        "silog_shift": args.silog_shift,
                        "silog_weight": args.silog_weight,
                        "l1_weight": args.l1_weight,
                    },
                    "height_scale": HEIGHT_SCALE,
                    "seed": args.seed,
                    "epoch": epoch,
                    "val_l1": val_loss,
                },
                run_dir / "stage_a.pt",
            )

        if stopper.step(epoch, val_loss, train_loss_world):
            print(
                f"early stopping at epoch {epoch}: no val improvement for "
                f"{args.patience} epochs (best val_l1={stopper.best:.4f} @ epoch {stopper.best_epoch})"
                + (" [overtraining detected: train falling, val rising]" if stopper.overtraining else ""),
                flush=True,
            )
            break

    metrics = {
        "params": params,
        "best_val_l1": best_val,
        "best_epoch": stopper.best_epoch,
        "epochs_run": epoch,
        "early_stopped": stopper.stopped,
        "overtraining_detected": stopper.overtraining,
        "train_tiles": len(train_rows),
        "val_tiles": len(val_rows),
        "epochs": args.epochs,
        "patience": args.patience,
        "peak_vram_gb": train_common.peak_vram_gb(),
        "model_type": "dav2" if args.dav2 else ("minimap_only" if args.minimap_only else "full"),
        "loss_type": args.loss_type,
        "scheduler_type": args.scheduler,
        "guided": bool(args.guided),
        "dav2": bool(args.dav2),
    }
    logger.write_json("stage_a_metrics.json", metrics)
    logger.close()
    print(
        f"done: best_val_l1={best_val:.4f} world units @ epoch {stopper.best_epoch} "
        f"(ran {epoch}/{args.epochs} epochs"
        + (", early stopped" if stopper.stopped else "")
        + (", overtraining detected" if stopper.overtraining else "")
        + f"); checkpoint={run_dir / 'stage_a.pt'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
