from __future__ import annotations

import sys
from pathlib import Path

import train_v18 as _train_v18

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATASET_DIR = _PROJECT_ROOT / "output" / "datasets" / "v18"
_DEFAULT_CURATION_ROOT = _DEFAULT_DATASET_DIR / "curation"
_DEFAULT_BUILDS = ("0_5_3_3368", "3_3_5_12340")
_DEFAULT_BUCKET_ROTATION_FRACTION = "0.10"
_DEFAULT_EARLY_STOP_PATIENCE = "8"


def _has_flag(name: str) -> bool:
    prefix = f"{name}="
    return any(arg == name or arg.startswith(prefix) for arg in sys.argv[1:])


def _value_after(name: str) -> str | None:
    for idx, arg in enumerate(sys.argv[1:], start=1):
        if arg == name and idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
        if arg.startswith(f"{name}="):
            return arg.split("=", 1)[1]
    return None


def _latest_manifest() -> Path | None:
    candidates = sorted(
        _DEFAULT_CURATION_ROOT.glob("*/kept_tiles.parquet"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _ensure_defaults() -> None:
    if not _has_flag("--dataset-dir"):
        sys.argv.extend(["--dataset-dir", str(_DEFAULT_DATASET_DIR)])

    if not _has_flag("--build") and not _has_flag("--builds"):
        sys.argv.extend(["--builds", *_DEFAULT_BUILDS])

    if not _has_flag("--curation-manifest"):
        latest = _latest_manifest()
        if latest is not None:
            sys.argv.extend(["--curation-manifest", str(latest)])

    if not _has_flag("--target-vram-gb"):
        sys.argv.extend(["--target-vram-gb", "8"])

    if not _has_flag("--autotune-batch-size") and not _has_flag("--no-autotune-batch-size"):
        sys.argv.append("--autotune-batch-size")

    if not _has_flag("--strict-build-balance") and not _has_flag("--no-strict-build-balance"):
        sys.argv.append("--strict-build-balance")

    if not _has_flag("--train-bucket-rotation-fraction") and not _has_flag("--train-epoch-tiles"):
        sys.argv.extend(["--train-bucket-rotation-fraction", _DEFAULT_BUCKET_ROTATION_FRACTION])

    if not _has_flag("--early-stop-patience"):
        sys.argv.extend(["--early-stop-patience", _DEFAULT_EARLY_STOP_PATIENCE])

    rotation_fraction = _value_after("--train-bucket-rotation-fraction")
    if (
        not _has_flag("--bucket-sampling-profile")
        and rotation_fraction is not None
        and float(rotation_fraction) > 0.0
    ):
        sys.argv.extend(["--bucket-sampling-profile", "uniform"])


def main() -> None:
    _ensure_defaults()
    _train_v18.main()


if __name__ == "__main__":
    main()
