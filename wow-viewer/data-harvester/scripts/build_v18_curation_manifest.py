from __future__ import annotations

import sys
from pathlib import Path

import build_v16_curation_manifest as _legacy

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATASET_DIR = _PROJECT_ROOT / "output" / "datasets" / "v18"
_DEFAULT_OUTPUT_ROOT = _DEFAULT_DATASET_DIR / "curation"
_DEFAULT_BUILDS = ("0_5_3_3368", "3_3_5_12340")
_DEFAULT_PROFILE = "v18_focus_terrain_v1"
_DEFAULT_RUN_NAME = "v18_focus_terrain_v1"


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


def _ensure_defaults() -> None:
    if not _has_flag("--dataset-dir"):
        sys.argv.extend(["--dataset-dir", str(_DEFAULT_DATASET_DIR)])

    if not _has_flag("--build") and not _has_flag("--builds"):
        sys.argv.extend(["--builds", *_DEFAULT_BUILDS])

    if not _has_flag("--profile"):
        sys.argv.extend(["--profile", _DEFAULT_PROFILE])

    run_name = _value_after("--run-name") or _DEFAULT_RUN_NAME
    if not _has_flag("--run-name"):
        sys.argv.extend(["--run-name", run_name])

    if not _has_flag("--output-dir"):
        sys.argv.extend(["--output-dir", str(_DEFAULT_OUTPUT_ROOT / run_name)])


def main() -> None:
    _ensure_defaults()
    _legacy.main()


if __name__ == "__main__":
    main()
