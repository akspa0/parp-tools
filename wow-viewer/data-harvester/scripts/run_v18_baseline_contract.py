from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V18 baseline profile contract and comparison report.")
    parser.add_argument("--refined-manifest", type=Path, required=True, help="Refined manifest directory with kept_tiles.parquet.")
    parser.add_argument("--builds", nargs="+", required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("../output/datasets/v16"))
    parser.add_argument("--output-dir", type=Path, default=Path("../output/tmp/v18_baseline_contract"))
    parser.add_argument("--profile", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--run-prefix", type=str, default="v18_baseline")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--skip-nonref", action="store_true")
    return parser.parse_args()


def _profiles() -> dict[str, dict[str, int]]:
    return {
        "small": {"train_max_tiles": 64, "train_epoch_tiles": 16, "val_max_tiles": 16, "val_epoch_tiles": 8, "epochs": 1, "batch_size": 2},
        "medium": {"train_max_tiles": 256, "train_epoch_tiles": 64, "val_max_tiles": 48, "val_epoch_tiles": 24, "epochs": 2, "batch_size": 4},
        "large": {"train_max_tiles": 512, "train_epoch_tiles": 128, "val_max_tiles": 96, "val_epoch_tiles": 48, "epochs": 4, "batch_size": 8},
    }


def _run(cmd: list[str], cwd: Path) -> None:
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_epoch_metrics(run_dir: Path) -> dict[str, Any]:
    log_path = run_dir / "training_log.json"
    payload = _load_json(log_path)
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"Unexpected or empty training log at {log_path}")
    return dict(payload[-1])


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _md_report(path: Path, report: dict[str, Any]) -> None:
    refined = report["refined_run"]["metrics"]
    nonref = report.get("nonref_run", {}).get("metrics")
    lines: list[str] = []
    lines.append("# V18 Baseline Comparison Report")
    lines.append("")
    lines.append(f"- profile: `{report['profile']}`")
    lines.append(f"- refined run: `{report['refined_run']['run_name']}`")
    if nonref is not None:
        lines.append(f"- non-ref run: `{report['nonref_run']['run_name']}`")
    lines.append("")
    lines.append("## Refined Metrics")
    lines.append("")
    lines.append(f"- train_loss: `{refined.get('train_loss')}`")
    lines.append(f"- val_loss: `{refined.get('val_loss')}`")
    lines.append(f"- elapsed_s: `{refined.get('elapsed_s')}`")
    lines.append(f"- optimizer_steps: `{refined.get('optimizer_steps')}`")
    if nonref is not None:
        lines.append("")
        lines.append("## Delta vs Non-Ref")
        lines.append("")
        for key in ("train_loss", "val_loss", "elapsed_s"):
            rv = float(refined.get(key, 0.0))
            nv = float(nonref.get(key, 0.0))
            lines.append(f"- {key}_delta: `{rv - nv:+.6f}` (refined - nonref)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    profiles = _profiles()
    profile = dict(profiles[args.profile])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_harvester_dir = Path(__file__).resolve().parent.parent

    # T026: baseline profile contract artifact
    profile_payload = {
        "profile_selected": args.profile,
        "profiles": profiles,
        "builds": list(args.builds),
        "dataset_dir": str(args.dataset_dir),
        "refined_manifest": str(args.refined_manifest),
    }
    (output_dir / "baseline_profiles.json").write_text(json.dumps(profile_payload, indent=2), encoding="utf-8")

    refined_run_name = f"{args.run_prefix}_{args.profile}_refined"
    nonref_run_name = f"{args.run_prefix}_{args.profile}_nonref"

    if not args.skip_run:
        refined_cmd = [
            sys.executable,
            "-u",
            "scripts/train_v16_1_normal.py",
            "--dataset-dir",
            str(args.dataset_dir),
            "--builds",
            *[str(b) for b in args.builds],
            "--curation-manifest",
            str(args.refined_manifest),
            "--device",
            "cpu",
            "--epochs",
            str(profile["epochs"]),
            "--batch-size",
            str(profile["batch_size"]),
            "--train-max-tiles",
            str(profile["train_max_tiles"]),
            "--train-epoch-tiles",
            str(profile["train_epoch_tiles"]),
            "--val-max-tiles",
            str(profile["val_max_tiles"]),
            "--rotate-val-tiles",
            "--val-epoch-tiles",
            str(profile["val_epoch_tiles"]),
            "--num-workers",
            "0",
            "--no-compile",
            "--run-name",
            refined_run_name,
        ]
        _run(refined_cmd, cwd=data_harvester_dir)

    refined_run_dir = (data_harvester_dir.parent / "models" / "v16_1" / "normal" / "runs" / f"v17_1_{refined_run_name}")
    refined_metrics = _latest_epoch_metrics(refined_run_dir)

    nonref_payload: dict[str, Any] | None = None
    if not args.skip_nonref:
        nonref_manifest_dir = output_dir / "nonref_manifest"
        if not args.skip_run:
            nonref_manifest_cmd = [
                sys.executable,
                "-u",
                "scripts/build_v16_curation_manifest.py",
                "--dataset-dir",
                str(args.dataset_dir),
                "--builds",
                *[str(b) for b in args.builds],
                "--profile",
                "normal_terrain_v16_1_1",
                "--max-tiles-per-build",
                str(max(64, profile["train_max_tiles"])),
                "--workers",
                "1",
                "--chunk-size",
                "64",
                "--output-dir",
                str(nonref_manifest_dir),
                "--run-name",
                "v18_nonref_baseline_manifest",
            ]
            _run(nonref_manifest_cmd, cwd=data_harvester_dir)

            nonref_cmd = [
                sys.executable,
                "-u",
                "scripts/train_v16_1_normal.py",
                "--dataset-dir",
                str(args.dataset_dir),
                "--builds",
                *[str(b) for b in args.builds],
                "--curation-manifest",
                str(nonref_manifest_dir),
                "--device",
                "cpu",
                "--epochs",
                str(profile["epochs"]),
                "--batch-size",
                str(profile["batch_size"]),
                "--train-max-tiles",
                str(profile["train_max_tiles"]),
                "--train-epoch-tiles",
                str(profile["train_epoch_tiles"]),
                "--val-max-tiles",
                str(profile["val_max_tiles"]),
                "--rotate-val-tiles",
                "--val-epoch-tiles",
                str(profile["val_epoch_tiles"]),
                "--num-workers",
                "0",
                "--no-compile",
                "--run-name",
                nonref_run_name,
            ]
            _run(nonref_cmd, cwd=data_harvester_dir)

        nonref_run_dir = (data_harvester_dir.parent / "models" / "v16_1" / "normal" / "runs" / f"v17_1_{nonref_run_name}")
        nonref_metrics = _latest_epoch_metrics(nonref_run_dir)
        nonref_payload = {
            "run_name": f"v17_1_{nonref_run_name}",
            "run_dir": str(nonref_run_dir),
            "manifest_dir": str(nonref_manifest_dir),
            "metrics": nonref_metrics,
        }

    report = {
        "profile": args.profile,
        "profile_settings": profile,
        "refined_run": {
            "run_name": f"v17_1_{refined_run_name}",
            "run_dir": str(refined_run_dir),
            "manifest_dir": str(args.refined_manifest),
            "metrics": refined_metrics,
        },
    }
    if nonref_payload is not None:
        report["nonref_run"] = nonref_payload

    _write_report(output_dir / "comparison_report.json", report)
    _md_report(output_dir / "comparison_report.md", report)
    print(json.dumps(report, indent=2))
    print(f"Wrote {output_dir / 'baseline_profiles.json'}")
    print(f"Wrote {output_dir / 'comparison_report.md'}")


if __name__ == "__main__":
    main()
