from __future__ import annotations

import sys
from pathlib import Path

import infer_v16_1 as _infer_v16_1

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATASET_DIR = _PROJECT_ROOT / "output" / "datasets" / "v18"
_DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "output" / "datasets" / "v18_inference"


def _has_flag(name: str) -> bool:
    prefix = f"{name}="
    return any(arg == name or arg.startswith(prefix) for arg in sys.argv[1:])


def _ensure_defaults() -> None:
    if not _has_flag("--dataset-dir"):
        sys.argv.extend(["--dataset-dir", str(_DEFAULT_DATASET_DIR)])
    if not _has_flag("--output-root"):
        sys.argv.extend(["--output-root", str(_DEFAULT_OUTPUT_ROOT)])


def main() -> None:
    _ensure_defaults()
    _infer_v16_1.main()


if __name__ == "__main__":
    main()
