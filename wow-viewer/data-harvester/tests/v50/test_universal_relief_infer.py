from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from harvester.v50.universal_relief_infer import (
    UniversalInferenceError,
    build_inference_plan,
    main,
    resize_relief_for_mesh,
)
from harvester.v50.universal_relief_model import student_identity_dict


def _checkpoint(path, *, valid_student: bool = True) -> None:
    student = student_identity_dict() if valid_student else {"hub_id": "wrong/model"}
    torch.save(
        {
            "schema": "v114-universal-relief-checkpoint-v1",
            "epoch": 4,
            "student": student,
            "model": {},
            "freeze_backbone": True,
        },
        path,
    )


def test_mesh_resize_preserves_aspect_and_bounds() -> None:
    relief = np.linspace(0.0, 1.0, 400 * 100, dtype=np.float32).reshape(100, 400)
    resized = resize_relief_for_mesh(relief, 101)
    assert resized.shape == (25, 101)
    assert resized.min() >= 0.0
    assert resized.max() <= 1.0


def test_inference_plan_accepts_grayscale_arbitrary_aspect_and_writes_nothing(tmp_path) -> None:
    image = tmp_path / "wide.png"
    Image.fromarray(np.zeros((31, 97), dtype=np.uint8), mode="L").save(image)
    checkpoint = tmp_path / "checkpoint.pt"
    _checkpoint(checkpoint)
    output = tmp_path / "output"

    plan = build_inference_plan(
        image_path=image,
        checkpoint_path=checkpoint,
        output=output,
        overlap=28,
        mesh_max_resolution=257,
        extent_x=100.0,
        vertical_scale=20.0,
    )

    assert not output.exists()
    assert plan["source_mode"] == "L"
    assert plan["source_width"] == 97
    assert plan["source_height"] == 31
    assert plan["tile_count"] == 1
    assert plan["semantics"] == "view_axis_relief"


def test_inference_plan_refuses_wrong_student_identity(tmp_path) -> None:
    image = tmp_path / "source.png"
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8), mode="RGB").save(image)
    checkpoint = tmp_path / "checkpoint.pt"
    _checkpoint(checkpoint, valid_student=False)
    with pytest.raises(UniversalInferenceError, match="student identity"):
        build_inference_plan(
            image_path=image,
            checkpoint_path=checkpoint,
            output=tmp_path / "output",
            overlap=28,
            mesh_max_resolution=257,
            extent_x=1.0,
            vertical_scale=1.0,
        )


def test_cli_defaults_to_dry_run(tmp_path, capsys) -> None:
    image = tmp_path / "source.png"
    Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8), mode="RGB").save(image)
    checkpoint = tmp_path / "checkpoint.pt"
    _checkpoint(checkpoint)
    output = tmp_path / "output"
    assert (
        main(
            [
                "--image",
                str(image),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert not output.exists()
    assert "DRY RUN" in capsys.readouterr().out
