"""parp-tools ComfyUI Custom Node Extension Package."""

import sys
from pathlib import Path

# Automatically inject data-harvester/src into sys.path when loaded inside ComfyUI
_src_dir = Path(__file__).resolve().parent.parent / "src"
if _src_dir.is_dir() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from .nodes import (
    WoW_AdtLoader,
    WoW_HardZCalibrator,
    WoW_MeshExporter,
    WoW_ObjectSieveConditioner,
)

NODE_CLASS_MAPPINGS = {
    "WoW_AdtLoader": WoW_AdtLoader,
    "WoW_ObjectSieveConditioner": WoW_ObjectSieveConditioner,
    "WoW_HardZCalibrator": WoW_HardZCalibrator,
    "WoW_MeshExporter": WoW_MeshExporter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WoW_AdtLoader": "WoW ADT Loader",
    "WoW_ObjectSieveConditioner": "WoW Object Sieve Conditioner",
    "WoW_HardZCalibrator": "WoW Hard-Z Calibrator",
    "WoW_MeshExporter": "WoW Mesh Exporter",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
