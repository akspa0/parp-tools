"""Unified V18 training entrypoint.

Usage:
    uv run python -u scripts/train_v18.py normal [--batch-size 8 ...]
    uv run python -u scripts/train_v18.py height [...]
    uv run python -u scripts/train_v18.py holes [...]
    uv run python -u scripts/train_v18.py liquid [...]
    uv run python -u scripts/train_v18.py texcomp [...]

All args after the task name pass through to the common trainer.
"""

import sys

from train_v16_1_common import run_task


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

    run_task(task_name)


if __name__ == "__main__":
    main()
