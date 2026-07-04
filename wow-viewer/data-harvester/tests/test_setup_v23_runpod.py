from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import setup_spec077_runpod  # noqa: E402


def test_requested_gpu_types_prefers_consumer_cards_when_requested() -> None:
    original = setup_spec077_runpod._gpu_full_info
    try:
        setup_spec077_runpod._gpu_full_info = lambda: {
            "NVIDIA RTX A4500": {
                "vram_gb": 20,
                "price_per_hour": 0.31,
                "available": True,
                "community_cloud": True,
            },
            "NVIDIA GeForce RTX 3090": {
                "vram_gb": 24,
                "price_per_hour": 0.34,
                "available": True,
                "community_cloud": True,
            },
            "NVIDIA GeForce RTX 4090": {
                "vram_gb": 24,
                "price_per_hour": 0.40,
                "available": True,
                "community_cloud": True,
            },
            "NVIDIA GeForce RTX 5090": {
                "vram_gb": 32,
                "price_per_hour": 0.50,
                "available": True,
                "community_cloud": True,
            },
        }
        args = SimpleNamespace(
            gpu_types=None,
            no_cost_target=False,
            min_gpu_vram_gb=20,
            cloud_type="COMMUNITY",
            max_cost_per_hour=1.00,
            gpu_type="NVIDIA GeForce RTX 4090",
            gpu_fallback=False,
            preferred_gpu_ids=[
                "NVIDIA GeForce RTX 3090",
                "NVIDIA GeForce RTX 4090",
                "NVIDIA GeForce RTX 5090",
            ],
        )

        requested = setup_spec077_runpod._requested_gpu_types(args)
        assert requested[:3] == [
            "NVIDIA GeForce RTX 3090",
            "NVIDIA GeForce RTX 4090",
            "NVIDIA GeForce RTX 5090",
        ]
        assert requested[3] == "NVIDIA RTX A4500"
    finally:
        setup_spec077_runpod._gpu_full_info = original
