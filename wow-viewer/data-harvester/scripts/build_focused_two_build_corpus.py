"""Focused two-build V18 corpus wrapper.

Scoping:
    This wrapper only orchestrates the focused two-build lane (0_5_3_3368 +
    3_3_5_12340) for the V18 distill corpus work. It does NOT introduce
    Plan B (synthesizer, distillation, student, open-source release). The
    other four builds (0_5_5_3494, 0_7_0_3694, 3_0_1_8303, 4_0_0_11927)
    stay where they are - this script never deletes or rewrites them.

Pipeline:
    1. generate-viewer-stubs  - write per-tile JSON stubs + capture ledger
       from the existing V18 index.parquet (X_Y naming).
    2. reconcile-captures     - if a build already has legacy MdxViewer
       captures in objectsonly/<tile>.png (Y_X naming), rename them to
       X_Y and place a copy at images/<tile>_object_visibility_mask.png.
       Drop tiles whose object_visibility_mask is all-zero (bogus renders
       for tiles with no data).
    3. capture-renderer-truth - run WowViewer.Tool.ValidationCapture
       capture-batch over the ledger with the existing WoWViewer renderer
       and the artifact-producing variants needed by the spec
       (primary,no-objects,objects-only). Skips tiles marked captured_complete.
    4. patch-renderer-truth   - land object_visibility_mask into the V18
       Zarr store. no_object_minimap is no longer emitted.
    5. clear-renderer-truth   - remove renderer-truth flags for a build when
       the current capture source is not trusted.
    6. validate-signals       - run signal validation on the patched store.

Idempotency:
    Each step is safe to re-run. generate-viewer-stubs overwrites stubs;
    reconcile-captures only writes images that are missing; capture-batch
    skips captured_complete rows; patch-renderer-truth backs up the
    index.parquet before mutating.

Usage:
    cd wow-viewer/data-harvester
    uv run python scripts/build_focused_two_build_corpus.py all
    uv run python scripts/build_focused_two_build_corpus.py reconcile-and-patch --builds 0_5_3_3368
    uv run python scripts/build_focused_two_build_corpus.py validate
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import PIL.Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BUILD_SCRIPT = _PROJECT_ROOT / "data-harvester" / "scripts" / "build_v18_dataset.py"
_CAPTURE_ROOT = _PROJECT_ROOT.parent / "output" / "tmp" / "mdxviewer_validation_smoke"
_DATASET_ROOT = _PROJECT_ROOT / "output" / "datasets" / "v18"
_EVIDENCE_ROOT = _DATASET_ROOT / "focused_build_evidence"
FOCUSED_BUILDS = ("0_5_3_3368", "3_3_5_12340")

# Bogus-tile filter: an object_visibility_mask whose mean intensity is below
# this threshold is treated as a render of an empty/no-data tile and is
# skipped from the V18 patch.
BOGUS_MASK_MEAN_THRESHOLD = 1.0

# Map of legacy capture variants to their X_Y-aligned derived suffix. Only
# the ObjectsOnly variant is consumed; no_object_minimap is dropped.
_LEGACY_VARIANT_DIRS = {
    "objectsonly": ("_object_visibility_mask.png", "visibility_mask"),
}


def _run(cmd: list[str]) -> int:
    print(f"\n>>> {' '.join(cmd)}")
    return subprocess.run(cmd, check=False).returncode


def _write_evidence(build: str, payload: dict) -> Path:
    build_dir = _EVIDENCE_ROOT / build
    build_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = build_dir / f"{payload['step']}_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _rewrite_yx_to_xy_in_name(stem: str) -> str | None:
    """If `stem` matches `<map>_<y>_<x>`, swap the trailing two numbers to
    produce `<map>_<x>_<y>`. Returns None if the stem is not in the
    `<map>_<int>_<int>` shape (so X_Y files pass through unchanged)."""
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    head, a, b = parts[0], parts[-2], parts[-1]
    try:
        int(a)
        int(b)
    except ValueError:
        return None
    middle = "_".join(parts[1:-2])
    rebuilt = head if not middle else f"{head}_{middle}"
    return f"{rebuilt}_{b}_{a}"


def _is_bogus_object_mask(path: Path) -> bool:
    """Detect a render of an empty/no-data tile. The object_visibility_mask
    for such a tile is uniformly near-zero."""
    try:
        with PIL.Image.open(str(path)) as img:
            arr = np.asarray(img.convert("L"), dtype=np.float32)
    except Exception:
        return True
    return float(arr.mean()) < BOGUS_MASK_MEAN_THRESHOLD


def step_generate_stubs(builds: list[str]) -> int:
    cmd = [
        sys.executable,
        str(_BUILD_SCRIPT),
        "generate-viewer-stubs",
        "--builds",
        *builds,
        "--capture-root",
        str(_CAPTURE_ROOT),
    ]
    return _run(cmd)


def step_reconcile_captures(builds: list[str]) -> dict[str, dict]:
    """Reconcile legacy MdxViewer captures into the X_Y-aligned images/ layout.

    For each focused build:
      - scan legacy <build>/objectsonly/*.png (Y_X names)
      - for each PNG, compute the X_Y name; if the source name is already
        in X_Y, leave it as-is
      - if images/<x_y>_object_visibility_mask.png does not already exist,
        copy the source
      - drop any tile whose object_visibility_mask has mean intensity below
        BOGUS_MASK_MEAN_THRESHOLD (treats empty renders as bogus)
    """
    report: dict[str, dict] = {}
    for build in builds:
        build_capture_dir = _CAPTURE_ROOT / build
        images_dir = build_capture_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        skipped_bogus = 0
        skipped_existing = 0
        rewrites = 0
        for variant_dir_name, (suffix, key) in _LEGACY_VARIANT_DIRS.items():
            variant_dir = build_capture_dir / variant_dir_name
            if not variant_dir.exists():
                continue
            for src in sorted(variant_dir.glob("*_viewer_validation.png")):
                stem = src.stem[: -len("_viewer_validation")]
                rewritten = _rewrite_yx_to_xy_in_name(stem)
                if rewritten is None:
                    tile_name = stem
                else:
                    tile_name = rewritten
                    rewrites += 1
                dst = images_dir / f"{tile_name}{suffix}"
                if dst.exists():
                    skipped_existing += 1
                    continue
                if _is_bogus_object_mask(src):
                    skipped_bogus += 1
                    continue
                shutil.copy2(src, dst)
                copied += 1
        payload = {
            "step": "reconcile_captures",
            "build": build,
            "capture_dir": str(build_capture_dir),
            "images_dir": str(images_dir),
            "copied": copied,
            "rewrites_yx_to_xy": rewrites,
            "skipped_existing": skipped_existing,
            "skipped_bogus": skipped_bogus,
            "bogus_threshold_mean_intensity": BOGUS_MASK_MEAN_THRESHOLD,
        }
        report[build] = payload
        _write_evidence(build, payload)
        print(
            f"  {build}: reconciled {copied} captures (rewrote {rewrites} Y_X -> X_Y, "
            f"skipped {skipped_bogus} bogus, {skipped_existing} already-present) into {images_dir}"
        )
    return report


def step_capture_renderer_truth(builds: list[str], mode_flags: list[str], variants: str) -> int:
    cmd = [
        sys.executable,
        str(_BUILD_SCRIPT),
        "capture-renderer-truth",
        "--builds",
        *builds,
        "--capture-root",
        str(_CAPTURE_ROOT),
        "--variants",
        variants,
        *mode_flags,
    ]
    return _run(cmd)


def step_patch_renderer_truth(builds: list[str]) -> int:
    cmd = [
        sys.executable,
        str(_BUILD_SCRIPT),
        "patch-renderer-truth",
        "--builds",
        *builds,
        "--capture-root",
        str(_CAPTURE_ROOT),
        "--no-backup",
    ]
    return _run(cmd)


def step_clear_renderer_truth(builds: list[str], reason: str) -> int:
    cmd = [
        sys.executable,
        str(_BUILD_SCRIPT),
        "clear-renderer-truth",
        "--builds",
        *builds,
        "--no-backup",
        "--reason",
        reason,
    ]
    return _run(cmd)


def step_validate_signals(builds: list[str]) -> int:
    cmd = [
        sys.executable,
        str(_BUILD_SCRIPT),
        "validate-signals",
        "--builds",
        *builds,
    ]
    return _run(cmd)


def cmd_all(args: argparse.Namespace) -> int:
    builds = list(args.builds) if args.builds else list(FOCUSED_BUILDS)
    mode_key = args.capture_mode or "dry-run"
    mode_flag = "--renderer" if mode_key == "renderer" else f"--{mode_key}"
    mode_flags = [mode_flag]
    print(f"Focused two-build V18 corpus wrapper")
    print(f"  builds      : {builds}")
    print(f"  capture_root: {_CAPTURE_ROOT}")
    print(f"  dataset_root: {_DATASET_ROOT}")
    print(f"  capture_mode: {mode_flags[0]}")
    print(f"  variants    : {args.variants}")

    rc = step_generate_stubs(builds)
    if rc != 0:
        return rc
    step_reconcile_captures(builds)
    rc = step_capture_renderer_truth(builds, mode_flags, args.variants)
    if rc != 0:
        return rc
    rc = step_patch_renderer_truth(builds)
    if rc != 0:
        return rc
    rc = step_validate_signals(builds)
    return rc


def cmd_reconcile_and_patch(args: argparse.Namespace) -> int:
    builds = list(args.builds) if args.builds else list(FOCUSED_BUILDS)
    step_reconcile_captures(builds)
    return step_patch_renderer_truth(builds)


def cmd_reconcile(args: argparse.Namespace) -> int:
    builds = list(args.builds) if args.builds else list(FOCUSED_BUILDS)
    step_reconcile_captures(builds)
    return 0


def cmd_patch(args: argparse.Namespace) -> int:
    builds = list(args.builds) if args.builds else list(FOCUSED_BUILDS)
    return step_patch_renderer_truth(builds)


def cmd_validate(args: argparse.Namespace) -> int:
    builds = list(args.builds) if args.builds else list(FOCUSED_BUILDS)
    return step_validate_signals(builds)


def cmd_clear(args: argparse.Namespace) -> int:
    builds = list(args.builds) if args.builds else list(FOCUSED_BUILDS)
    return step_clear_renderer_truth(builds, args.reason)


def main() -> None:
    parser = argparse.ArgumentParser(description="Focused two-build V18 corpus wrapper (spec 047 Plan A)")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--builds", nargs="+", default=None, help="Builds to operate on (default: focused two-build lane)")

    all_p = sub.add_parser("all", parents=[common], help="Full focused pipeline: stubs -> reconcile -> capture -> patch -> validate")
    all_p.add_argument(
        "--capture-mode",
        choices=["dry-run", "real-scene-dry-run", "renderer", "native-renderer", "gpu-viewer-style", "stub-scene"],
        default="dry-run",
        help="Capture-batch run mode (default: dry-run; use renderer for the existing WoWViewer renderer path)",
    )
    all_p.add_argument(
        "--variants",
        default="primary,no-objects,objects-only",
        help="Capture variants to render (default: primary,no-objects,objects-only). Pass 'all' for the full QA set.",
    )
    all_p.set_defaults(func=cmd_all)

    recon_p = sub.add_parser("reconcile", parents=[common], help="Reconcile legacy MdxViewer capture layout into the new X_Y-aligned images/ layout (drops bogus empty renders)")
    recon_p.set_defaults(func=cmd_reconcile)

    patch_p = sub.add_parser("patch", parents=[common], help="Patch renderer-truth captures into the V18 Zarr store")
    patch_p.set_defaults(func=cmd_patch)

    rp_p = sub.add_parser("reconcile-and-patch", parents=[common], help="Reconcile legacy captures and patch the Zarr store")
    rp_p.set_defaults(func=cmd_reconcile_and_patch)

    val_p = sub.add_parser("validate", parents=[common], help="Validate signal coverage on the patched Zarr store")
    val_p.set_defaults(func=cmd_validate)

    clear_p = sub.add_parser("clear", parents=[common], help="Clear renderer-truth signals from the V18 store when the current capture source is not trusted")
    clear_p.add_argument(
        "--reason",
        default="untrusted renderer-truth source",
        help="Reason recorded in renderer_truth_reset_report.json",
    )
    clear_p.set_defaults(func=cmd_clear)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
