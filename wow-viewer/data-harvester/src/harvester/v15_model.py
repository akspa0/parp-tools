"""Compatibility shim for legacy V15 imports.

Canonical current model module: harvester.v16_model
"""

from __future__ import annotations

from .v16_model import V16Model

V15Model = V16Model

__all__ = ["V15Model", "V16Model"]
