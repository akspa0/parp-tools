"""Unit tests for ComfyUI orchestrator."""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from harvester.v60.comfyui_orchestrator import (
    ComfyUIConnectionError,
    ComfyUIError,
    ComfyUIOrchestrator,
)


def test_build_sam_segmentation_workflow():
    workflow = ComfyUIOrchestrator.build_sam_segmentation_workflow(
        image_name="test_tile.png",
        threshold=0.65,
        refine_iterations=3,
        prefix="minimap_mask",
    )
    assert "1" in workflow
    assert workflow["1"]["class_type"] == "LoadImage"
    assert workflow["1"]["inputs"]["image"] == "test_tile.png"

    assert "2" in workflow
    assert workflow["2"]["class_type"] == "CRTAutoDLSAM3BodyCheckpoint"

    assert "3" in workflow
    assert workflow["3"]["class_type"] == "SAM3_Detect"
    assert workflow["3"]["inputs"]["threshold"] == 0.65
    assert workflow["3"]["inputs"]["refine_iterations"] == 3

    assert "4" in workflow
    assert workflow["4"]["class_type"] == "MaskToImage"

    assert "5" in workflow
    assert workflow["5"]["class_type"] == "SaveImage"
    assert workflow["5"]["inputs"]["filename_prefix"] == "minimap_mask"

    assert "6" in workflow
    assert workflow["6"]["class_type"] == "CLIPTextEncode"
    assert "conditioning" in workflow["3"]["inputs"]


def test_build_mothership_vlm_workflow():
    workflow = ComfyUIOrchestrator.build_mothership_vlm_workflow(
        image_name="doodad_crop.png",
        prompt="Identify whether this object is a tree or building.",
        system_prompt="Top-down terrain segmentation expert.",
        slot=2,
        temperature=0.2,
    )
    assert workflow["1"]["inputs"]["image"] == "doodad_crop.png"
    assert workflow["2"]["class_type"] == "MothershipGemmaVLM"
    assert workflow["2"]["inputs"]["prompt"] == "Identify whether this object is a tree or building."
    assert workflow["2"]["inputs"]["slot"] == 2
    assert workflow["2"]["inputs"]["max_tokens"] == 1024
    assert workflow["3"]["class_type"] == "MothershipStringSink"


def test_upload_image_array(monkeypatch):
    client = ComfyUIOrchestrator(base_url="http://127.0.0.1:8199")

    fake_resp = json.dumps({"name": "uploaded_123.png", "subfolder": "", "type": "input"}).encode("utf-8")

    def mock_request(endpoint, data=None, headers=None, method=None):
        assert endpoint == "upload/image"
        assert b"multipart/form-data" in headers["Content-Type"].encode("utf-8")
        assert b"uploaded_123.png" in data
        return fake_resp

    monkeypatch.setattr(client, "_request", mock_request)

    dummy_arr = np.zeros((64, 64, 3), dtype=np.uint8)
    name = client.upload_image(dummy_arr, filename="uploaded_123.png")
    assert name == "uploaded_123.png"


def test_segment_objects_mocked(monkeypatch):
    client = ComfyUIOrchestrator(base_url="http://127.0.0.1:8199")

    # Mock upload
    monkeypatch.setattr(client, "upload_image", lambda data, **kwargs: "tile.png")

    # Mock queue_prompt
    monkeypatch.setattr(client, "queue_prompt", lambda wf: "prompt-abc")

    # Mock wait_for_completion
    fake_history = {
        "status": {"status_str": "success"},
        "outputs": {
            "5": {
                "images": [{"filename": "mask_001.png", "subfolder": "", "type": "output"}]
            }
        },
    }
    monkeypatch.setattr(client, "wait_for_completion", lambda pid, **kwargs: fake_history)

    # Mock get_output_image to return a 64x64 white circle mask
    mask_img = Image.new("L", (64, 64), color=0)
    for y in range(20, 44):
        for x in range(20, 44):
            mask_img.putpixel((x, y), 255)
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    mask_bytes = buf.getvalue()

    monkeypatch.setattr(client, "get_output_image", lambda filename, **kwargs: mask_bytes)

    mask = client.segment_objects(np.zeros((64, 64, 3), dtype=np.uint8))
    assert mask.shape == (64, 64)
    assert mask.dtype == np.uint8
    assert mask[30, 30] == 255
    assert mask[0, 0] == 0


def test_analyze_vlm_mocked(monkeypatch):
    client = ComfyUIOrchestrator(base_url="http://127.0.0.1:8199")
    monkeypatch.setattr(client, "upload_image", lambda data, **kwargs: "crop.png")
    monkeypatch.setattr(client, "queue_prompt", lambda wf: "prompt-xyz")

    fake_history = {
        "status": {"status_str": "success"},
        "outputs": {
            "3": {
                "text": ["Detected: Goldshire Inn roof (WMO)."]
            }
        },
    }
    monkeypatch.setattr(client, "wait_for_completion", lambda pid, **kwargs: fake_history)

    res = client.analyze_vlm(np.zeros((64, 64, 3), dtype=np.uint8), prompt="Describe object")
    assert "Goldshire Inn roof" in res
