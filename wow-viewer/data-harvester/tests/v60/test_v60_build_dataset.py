#!/usr/bin/env python3
"""Unit tests for v60 dataset builder CLI and discovery logic (Spec 269)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from v60_build_dataset import _discover_clients, _normalize_build_id


def test_normalize_build_id():
    """Verify normalization of build folder names."""
    assert _normalize_build_id("3.X_Retail_Windows_enUS_3.3.5.12340") == "3_3_5_12340"
    assert _normalize_build_id("0.5.3.3368") == "0_5_3_3368"
    assert _normalize_build_id("4.0.0.11927") == "4_0_0_11927"
    assert _normalize_build_id("custom_build") == "custom_build"


def test_discover_clients_direct_root(tmp_path: Path):
    """Verify discovery when client root directly contains Data folder."""
    client_dir = tmp_path / "3.X_Retail_Windows_enUS_3.3.5.12340" / "World of Warcraft"
    (client_dir / "Data").mkdir(parents=True)

    clients = _discover_clients(str(client_dir))
    assert len(clients) == 1
    build_id, path = clients[0]
    assert build_id == "3_3_5_12340"
    assert Path(path).resolve() == client_dir.resolve()


def test_discover_clients_override_build_id(tmp_path: Path):
    """Verify override_build_id parameter."""
    client_dir = tmp_path / "some_folder"
    (client_dir / "Data").mkdir(parents=True)

    clients = _discover_clients(str(client_dir), override_build_id="3_3_5_12340")
    assert len(clients) == 1
    build_id, path = clients[0]
    assert build_id == "3_3_5_12340"


def test_discover_clients_nested_library(tmp_path: Path):
    """Verify discovery when client root is a library with subfolders."""
    root = tmp_path / "CLIENTS"
    client1 = root / "WoW335" / "3.X_Retail_Windows_enUS_3.3.5.12340" / "World of Warcraft"
    (client1 / "Data").mkdir(parents=True)

    clients = _discover_clients(str(root))
    assert len(clients) == 1
    assert clients[0][0] == "3_3_5_12340"
    assert Path(clients[0][1]).resolve() == client1.resolve()
