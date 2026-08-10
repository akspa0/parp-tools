from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from harvester.v60.clean_signal_corpus import array_sha256, build_clean_signal_corpus
from harvester.v60.control_corpus import CONTROL_FAMILY_BUCKETS


def _write_control_corpus(root: Path) -> None:
    rows = []
    root.mkdir(parents=True, exist_ok=True)
    for index, family in enumerate(("flat", "ridge")):
        shadow = np.full((256, 256), 0.2 + index * 0.3, dtype=np.float32)
        yy, xx = np.mgrid[:257, :257]
        height = (xx + yy + index).astype(np.float32)
        name = f"{family}-{index:02d}.npz"
        np.savez(root / name, terrain_shadow_256=shadow, height_257=height)
        rows.append(
            {
                "row_id": f"{family}-v{index:02d}",
                "control_family": family,
                "complexity_bucket": CONTROL_FAMILY_BUCKETS[family],
                "source_group_id": f"group-{index}",
                "variant": index,
                "split": "train" if index == 0 else "validation",
                "npz": name,
                "input_sha256": array_sha256(shadow),
                "target_sha256": array_sha256(height),
            }
        )
    (root / "control_manifest.json").write_text(
        json.dumps({"schema": "v60-control-corpus-v1", "row_count": len(rows), "rows": rows}),
        encoding="utf-8",
    )


def test_train_cli_dry_run_writes_nothing_and_reports_matrix(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "control"
    _write_control_corpus(source)
    corpus = tmp_path / "clean"
    build_clean_signal_corpus(source, corpus)
    output = tmp_path / "run"

    from scripts.v60_train_clean_signal import main

    result = main(
        [
            "--corpus",
            str(corpus),
            "--output",
            str(output),
            "--architectures",
            "unet_lite_v2",
            "--loss-profiles",
            "parity",
            "--split",
            "complete_family",
            "--train-size",
            "1",
        ]
    )

    captured = capsys.readouterr().out
    assert result == 0
    assert "DRY RUN ONLY" in captured
    assert '"split"' in captured
    assert not output.exists()


def test_train_cli_confirm_refuses_nonempty_output_before_training(tmp_path: Path) -> None:
    source = tmp_path / "control"
    _write_control_corpus(source)
    corpus = tmp_path / "clean"
    build_clean_signal_corpus(source, corpus)
    output = tmp_path / "run"
    output.mkdir()
    (output / "existing.txt").write_text("keep", encoding="utf-8")

    from scripts.v60_train_clean_signal import main

    with pytest.raises(SystemExit, match="overwrite"):
        main(
            [
                "--corpus",
                str(corpus),
                "--output",
                str(output),
                "--architectures",
                "unet_lite_v2",
                "--loss-profiles",
                "parity",
                "--split",
                "complete_family",
                "--train-size",
                "1",
                "--confirm-run",
            ]
        )
