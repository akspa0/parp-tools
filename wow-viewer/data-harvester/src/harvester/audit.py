"""Signal dropout auditor for NPZ shard datasets.

Scans shards and produces a per-build, per-signal inventory so missing
signals (dropouts) are never hand-waved away.  The audit is the first
guard in every training run — if required signals are missing, the
training script refuses to proceed until the harvest is fixed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# NPZ keys D1 requires for every training sample.
D1_REQUIRED_KEYS = frozenset({"minimap_rgb_256", "mcly_texture_ids"})
D1_ALPHA_KEYS = frozenset({"mcal_alpha_pack_256", "mcal_alpha_pack"})
D1_OPTIONAL_KEYS = frozenset(
    {
        "height_257",
        "height_65",
        "height_17",
        "mcnr_normal_xyz",
        "mcsh_shadow_mask_256",
        "shadow_residual_mask_256",
        "hole_mask_16",
        "mclq_surface_height",
        "mclq_type_mask",
        "mcly_layer_mask",
        "object_mask_257",
        "object_precise_mask_257",
        "placement_mddf_data",
        "placement_modf_data",
        "metadata.json",
    }
)

# Regex to pull build / map / tile from a shard path.
_PATH_RE = re.compile(
    r"[\\/](?P<build>\d+_\d+_\d+_\d+)[\\/]"
    r"(?P<map>[^\\/]+)[\\/]"
    r"(?P<map2>[^\\/]+)_(?P<tx>\d+)_(?P<ty>\d+)_harvest\.(?:npz|zarr)$"
)


@dataclass
class ShardSignalInventory:
    """Per-shard signal presence report."""

    path: str
    build: str
    map_name: str
    tile_x: int
    tile_y: int
    present_keys: list[str]
    missing_required: list[str]
    missing_alpha: list[str]
    is_d1_eligible: bool
    error: str | None = None


@dataclass
class BuildAuditSummary:
    """Aggregated signal stats for one client build."""

    build: str
    total_shards: int
    d1_eligible: int
    missing_minimap: int
    missing_mcly: int
    missing_alpha: int
    error_shards: int
    dropout_paths: list[str] = field(default_factory=list)


@dataclass
class AuditReport:
    """Full signal audit across all scanned shards."""

    scanned: int
    train_count: int
    val_count: int
    d1_eligible_train: int
    d1_eligible_val: int
    dropouts_train: list[ShardSignalInventory]
    dropouts_val: list[ShardSignalInventory]
    by_build: list[BuildAuditSummary]
    all_shards: list[ShardSignalInventory] = field(default_factory=list)


def _parse_shard_path(path: Path) -> tuple[str, str, int, int]:
    """Extract (build, map_name, tile_x, tile_y) from a shard path."""
    m = _PATH_RE.search(str(path))
    if m is None:
        raise ValueError(f"Cannot parse shard path: {path}")
    return m.group("build"), m.group("map"), int(m.group("tx")), int(m.group("ty"))


def signal_inventory(npz_path: Path) -> ShardSignalInventory:
    """Produce a per-shard inventory of present and missing D1 signals.

    Does NOT eagerly load array data — only checks key presence and
    basic shape sanity for alpha.
    """
    try:
        build, map_name, tx, ty = _parse_shard_path(npz_path)
        all_keys: set[str]
        if npz_path.suffix == ".zarr" and npz_path.is_dir():
            import zarr as _zarr

            store = _zarr.storage.LocalStore(str(npz_path), read_only=True)
            root = _zarr.open_group(store, mode="r")
            all_keys = set(root.array_keys())
        else:
            with np.load(npz_path) as data:
                all_keys = set(data.files)
        present = sorted(all_keys)
        missing_req = sorted(D1_REQUIRED_KEYS - all_keys)
        missing_alpha = [] if (all_keys & D1_ALPHA_KEYS) else ["mcal_alpha_pack_256"]
        is_eligible = len(missing_req) == 0 and len(missing_alpha) == 0
        return ShardSignalInventory(
            path=str(npz_path),
            build=build,
            map_name=map_name,
            tile_x=tx,
            tile_y=ty,
            present_keys=present,
            missing_required=missing_req,
            missing_alpha=missing_alpha,
            is_d1_eligible=is_eligible,
        )
    except Exception as exc:
        return ShardSignalInventory(
            path=str(npz_path),
            build="unknown",
            map_name="unknown",
            tile_x=-1,
            tile_y=-1,
            present_keys=[],
            missing_required=[],
            missing_alpha=[],
            is_d1_eligible=False,
            error=str(exc),
        )


def audit_shards(
    train_paths: list[Path],
    val_paths: list[Path],
) -> AuditReport:
    """Full signal audit across train + val shard lists.

    Returns an AuditReport with per-shard inventories, dropout lists,
    and per-build summaries.
    """
    train_inv: list[ShardSignalInventory] = []
    val_inv: list[ShardSignalInventory] = []
    train_eligible = 0
    val_eligible = 0
    dropouts_train: list[ShardSignalInventory] = []
    dropouts_val: list[ShardSignalInventory] = []

    for p in train_paths:
        inv = signal_inventory(p)
        train_inv.append(inv)
        if inv.is_d1_eligible:
            train_eligible += 1
        else:
            dropouts_train.append(inv)

    for p in val_paths:
        inv = signal_inventory(p)
        val_inv.append(inv)
        if inv.is_d1_eligible:
            val_eligible += 1
        else:
            dropouts_val.append(inv)

    # Per-build summaries
    by_build: dict[str, BuildAuditSummary] = {}
    for inv in train_inv + val_inv:
        b = inv.build
        if b not in by_build:
            by_build[b] = BuildAuditSummary(
                build=b,
                total_shards=0,
                d1_eligible=0,
                missing_minimap=0,
                missing_mcly=0,
                missing_alpha=0,
                error_shards=0,
            )
        s = by_build[b]
        s.total_shards += 1
        if inv.is_d1_eligible:
            s.d1_eligible += 1
        if "minimap_rgb_256" in inv.missing_required:
            s.missing_minimap += 1
        if "mcly_texture_ids" in inv.missing_required:
            s.missing_mcly += 1
        if inv.missing_alpha:
            s.missing_alpha += 1
        if inv.error:
            s.error_shards += 1
        if not inv.is_d1_eligible:
            s.dropout_paths.append(inv.path)

    return AuditReport(
        scanned=len(train_inv) + len(val_inv),
        train_count=len(train_inv),
        val_count=len(val_inv),
        d1_eligible_train=train_eligible,
        d1_eligible_val=val_eligible,
        dropouts_train=dropouts_train,
        dropouts_val=dropouts_val,
        by_build=sorted(by_build.values(), key=lambda x: x.build),
        all_shards=train_inv + val_inv,
    )


def format_audit_terminal(report: AuditReport, model_name: str = "D1") -> str:
    """Human-readable terminal summary of an audit report."""
    lines: list[str] = []
    lines.append(f"=== {model_name} SIGNAL AUDIT ===")
    lines.append(
        f"  Scanned: {report.scanned} shards  (train={report.train_count}, val={report.val_count})"
    )
    lines.append(f"  D1-eligible train: {report.d1_eligible_train}  val: {report.d1_eligible_val}")
    lines.append(f"  Dropouts train: {len(report.dropouts_train)}  val: {len(report.dropouts_val)}")

    if report.dropouts_train or report.dropouts_val:
        lines.append("")
        lines.append("  --- DROPOUTS DETECTED ---")
    if report.dropouts_train:
        lines.append(f"  Training dropouts ({len(report.dropouts_train)}):")
        for d in report.dropouts_train:
            missing = d.missing_required + d.missing_alpha
            lines.append(f"    [{d.build}] {d.path}  missing={missing}")
    if report.dropouts_val:
        lines.append(f"  Validation dropouts ({len(report.dropouts_val)}):")
        for d in report.dropouts_val:
            missing = d.missing_required + d.missing_alpha
            lines.append(f"    [{d.build}] {d.path}  missing={missing}")

    lines.append("")
    lines.append("  --- PER-BUILD SUMMARY ---")
    for b in report.by_build:
        pct = b.d1_eligible / max(b.total_shards, 1) * 100
        lines.append(
            f"    {b.build}: {b.total_shards} total, {b.d1_eligible} D1-ok ({pct:.1f}%), "
            f"missing_mm={b.missing_minimap}, missing_mcly={b.missing_mcly}, "
            f"missing_alpha={b.missing_alpha}, errors={b.error_shards}"
        )
        if b.dropout_paths:
            lines.append(f"      {len(b.dropout_paths)} dropout shard(s)")

    return "\n".join(lines)


def audit_to_json(report: AuditReport) -> dict:
    """Serialize an AuditReport to a JSON-safe dict."""
    return {
        "scanned": report.scanned,
        "train_count": report.train_count,
        "val_count": report.val_count,
        "d1_eligible_train": report.d1_eligible_train,
        "d1_eligible_val": report.d1_eligible_val,
        "dropouts_train": [
            {
                "path": d.path,
                "build": d.build,
                "map": d.map_name,
                "tile_x": d.tile_x,
                "tile_y": d.tile_y,
                "missing_required": d.missing_required,
                "missing_alpha": d.missing_alpha,
                "error": d.error,
            }
            for d in report.dropouts_train
        ],
        "dropouts_val": [
            {
                "path": d.path,
                "build": d.build,
                "map": d.map_name,
                "tile_x": d.tile_x,
                "tile_y": d.tile_y,
                "missing_required": d.missing_required,
                "missing_alpha": d.missing_alpha,
                "error": d.error,
            }
            for d in report.dropouts_val
        ],
        "by_build": [
            {
                "build": b.build,
                "total_shards": b.total_shards,
                "d1_eligible": b.d1_eligible,
                "missing_minimap": b.missing_minimap,
                "missing_mcly": b.missing_mcly,
                "missing_alpha": b.missing_alpha,
                "error_shards": b.error_shards,
                "dropout_count": len(b.dropout_paths),
            }
            for b in report.by_build
        ],
    }
