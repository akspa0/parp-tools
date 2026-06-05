"""Unified V18 training entrypoint.

Usage:
    uv run python -u scripts/train_v18.py normal [--batch-size 8 ...]
    uv run python -u scripts/train_v18.py height [...]
    uv run python -u scripts/train_v18.py holes [...]
    uv run python -u scripts/train_v18.py liquid [...]
    uv run python -u scripts/train_v18.py texcomp [...]

All args after the task name pass through to the common trainer.
"""

from pathlib import Path
import sys

from train_v16_1_common import run_task

_V18_DATASET_DIR = str((Path(__file__).resolve().parent.parent.parent / "output" / "datasets" / "v18"))


def _ensure_v18_dataset_dir() -> None:
    if any(arg == "--dataset-dir" or arg.startswith("--dataset-dir=") for arg in sys.argv[1:]):
        return
    sys.argv.extend(["--dataset-dir", _V18_DATASET_DIR])


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        print("Usage: train_v18.py <task> [args...]", file=sys.stderr)
        print("  task: normal | height | holes | liquid | texcomp", file=sys.stderr)
        sys.exit(1)

    task_name = sys.argv.pop(1)
    valid_tasks = {"normal", "height", "holes", "liquid", "texcomp"}
    if task_name not in valid_tasks:
        print(f"Unknown task: {task_name}", file=sys.stderr)
        print(f"Valid: {', '.join(sorted(valid_tasks))}", file=sys.stderr)
        sys.exit(1)

    _ensure_v18_dataset_dir()
    run_task(task_name)


if __name__ == "__main__":
    main()
