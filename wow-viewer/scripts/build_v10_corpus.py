#!/usr/bin/env python3
"""
v10.1 Corpus Builder Orchestrator

Scans source tiles from all staged WoW clients, fingerprints for deduplication,
selects a curated set of ~750-1500 unique tiles, and builds v10 NPZ shards.

Usage:
    python build_v10_corpus.py [--config <config.json>] [--dry-run]
    python build_v10_corpus.py --extract-only  # stop after source discovery
    python build_v10_corpus.py --fingerprint-only  # stop after fingerprinting
    python build_v10_corpus.py --deduplicate-only  # stop after dedup
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# ── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # i:/parp/parp-tools
CONVERTER_PROJECT = (
    REPO_ROOT
    / "wow-viewer"
    / "tools"
    / "converter"
    / "WowViewer.Tool.Converter"
    / "WowViewer.Tool.Converter.csproj"
)
CONVERTER_EXE = (
    REPO_ROOT
    / "wow-viewer"
    / "tools"
    / "converter"
    / "WowViewer.Tool.Converter"
    / "bin"
    / "Debug"
    / "net10.0"
    / "WowViewer.Tool.Converter.exe"
)
STAGED_CLIENTS_ROOT = REPO_ROOT / "output" / "tmp" / "wowarchive-clients"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "v10_1_corpus"
ORIGINAL_DEVELOPMENT_ADT_DIR = (
    REPO_ROOT
    / "gillijimproject_refactor"
    / "test_data"
    / "original_development"
    / "World"
    / "Maps"
    / "development"
)
ORIGINAL_DEVELOPMENT_MINIMAP_ROOT = REPO_ROOT / "datasets" / "original_development" / "development"

# ── Default map list ─────────────────────────────────────────────────────────
# Maps that exist across most WoW client versions (0.5.3 through 4.0.0).
# The user can override this via --config.

DEFAULT_MAPS: list[str] = [
    # ── Azeroth (Eastern Kingdoms + Kalimdor) ──
    "Azeroth",
    "Kalimdor",
    "EasternKingdoms",
    # ── Classic dungeons/raids ──
    "Deadmines",
    "WailingCaverns",
    "RazorfenDowns",
    "RazorfenKraul",
    "ScarletMonastery",
    "Scholomance",
    "Stratholme",
    "SunkenTemple",
    "Uldaman",
    "ZulFarrak",
    "BlackrockDepths",
    "BlackrockSpire",
    "DireMaul",
    "LowerBlackrockSpire",
    # ── Outland (TBC) ──
    "Hellfire",
    "Zangarmarsh",
    "Terokkar",
    "Nagrand",
    "BladesEdge",
    "Netherstorm",
    "Shadowmoon",
    # ── Northrend (WotLK) ──
    "BoreanTundra",
    "HowlingFjord",
    "Dragonblight",
    "GrizzlyHills",
    "ZulDrak",
    "Gundrak",
    "SholazarBasin",
    "StormPeaks",
    "Icecrown",
    "CrystalsongForest",
    "Wintergrasp",
    # ── Cataclysm zones ──
    "MountHyjal",
    "Deepholm",
    "Uldum",
    "TwilightHighlands",
]

# ── Client definitions ───────────────────────────────────────────────────────

CLIENTS: list[dict[str, Any]] = [
    {
        "client_id": "0.5.3.3368",
        "version": "0.5.3.3368",
        "client_path": str(STAGED_CLIENTS_ROOT / "0_5_3_3368" / "World of Warcraft"),
        "era": "alpha",
        "terrain_source_kind": "embedded_wdt_alpha",
        "maps": ["Azeroth", "Kalimdor"],
    },
    {
        "client_id": "0.5.5.3494",
        "version": "0.5.5.3494",
        "client_path": str(STAGED_CLIENTS_ROOT / "0_5_5_3494" / "World of Warcraft"),
        "era": "alpha",
        "terrain_source_kind": "embedded_wdt_alpha",
        "maps": ["Azeroth", "Kalimdor", "EmeraldDream"],
    },
    {
        "client_id": "0.7.0.3694",
        "version": "0.7.0.3694",
        "client_path": str(STAGED_CLIENTS_ROOT / "0_7_0_3694" / "World of Warcraft"),
        "era": "alpha",
        "terrain_source_kind": "embedded_wdt_alpha",
        "maps": ["Azeroth", "Kalimdor", "EmeraldDream"],
    },
    {
        "client_id": "3.0.1.8303",
        "version": "3.0.1.8303",
        "client_path": str(STAGED_CLIENTS_ROOT / "3_0_1_8303" / "World of Warcraft"),
        "era": "wotlk",
        "terrain_source_kind": "loose_adt",
        "maps": ["Northrend"],
    },
    {
        "client_id": "3.3.5.12340",
        "version": "3.3.5.12340",
        "client_path": str(STAGED_CLIENTS_ROOT / "3_3_5_12340" / "World of Warcraft"),
        "era": "wotlk",
        "terrain_source_kind": "loose_adt",
        "maps": ["Azeroth", "Kalimdor", "EmeraldDream", "Northrend", "PVPZone01", "PVPZone02", "PVPZone03", "PVPZone04"],
    },
    {
        "client_id": "4.0.0.11927",
        "version": "4.0.0.11927",
        "client_path": str(STAGED_CLIENTS_ROOT / "4_0_0_11927" / "World of Warcraft"),
        "era": "cata",
        "terrain_source_kind": "loose_adt",
        "maps": ["Azeroth", "Kalimdor", "EmeraldDream", "Deepholm", "LostIsles", "LostIslesPhase1", "LostIslesPhase2"],
    },
    {
        "client_id": "original_development",
        "version": "original_development",
        "client_path": str(ORIGINAL_DEVELOPMENT_ADT_DIR),
        "era": "development",
        "terrain_source_kind": "filesystem_adt_dir",
        "maps": ["development"],
        "minimap_root": str(ORIGINAL_DEVELOPMENT_MINIMAP_ROOT),
    },
]


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class CorpusConfig:
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    max_tiles: int = 1500
    min_tiles: int = 750
    discover_maps: bool = False
    require_minimap: bool = True
    overwrite_shards: bool = False
    run_curation: bool = True
    curation_max_selected_fraction: float = 0.0
    curation_max_per_era: int = 0
    shard_batch_size: int = 32
    shard_batch_timeout_seconds: int = 300
    maps: list[str] = field(default_factory=lambda: DEFAULT_MAPS[:])
    clients: list[dict[str, Any]] = field(default_factory=lambda: CLIENTS[:])
    skip_clients: list[str] = field(default_factory=list)
    skip_maps: list[str] = field(default_factory=list)


@dataclass
class FingerprintEntry:
    tile_name: str
    source_path: str
    file_size: int
    fingerprint: str
    mcnk_count: int
    total_layer_count: int
    max_layer_count: int
    chunks_with_mcvt: int
    chunks_with_holes: int
    chunks_with_liquid_flags: int
    unique_texture_ids: list[int]
    global_min_height: float
    global_max_height: float
    # Enrichment fields (added after fingerprinting)
    client_id: str = ""
    map_name: str = ""
    era: str = ""


# ── Helpers ──────────────────────────────────────────────────────────────────


DISALLOWED_CONVERTER_COMMANDS: set[str] = {
    "extract-map",
    "dataset-build-v10-stage1",
}


def run_dotnet(args: list[str], cwd: str | None = None, timeout: int = 600) -> subprocess.CompletedProcess:
    """Run the converter tool with the given arguments (NPZ-only mode)."""
    if not args:
        raise ValueError("run_dotnet requires at least one converter command argument")

    command_name = str(args[0]).strip().lower()
    if command_name in DISALLOWED_CONVERTER_COMMANDS:
        raise RuntimeError(
            f"Blocked converter command '{command_name}'. This pipeline is NPZ-only and must not dump ADT files to disk."
        )

    if CONVERTER_EXE.exists():
        cmd = [str(CONVERTER_EXE)] + args
    else:
        cmd = [
            "dotnet", "run", "--project", str(CONVERTER_PROJECT),
            "-c", "Debug", "--",
        ] + args
    print(f"  [cmd] {' '.join(str(a) for a in cmd)}")
    return subprocess.run(
        cmd,
        cwd=cwd or str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def check_client_exists(client: dict[str, Any]) -> bool:
    """Check if the client root directory exists."""
    return Path(client["client_path"]).exists()


def client_source_kind(client: dict[str, Any]) -> str:
    return str(client.get("terrain_source_kind") or "loose_adt").lower()


def client_uses_loose_adts(client: dict[str, Any]) -> bool:
    return client_source_kind(client) == "loose_adt"


def client_uses_embedded_alpha(client: dict[str, Any]) -> bool:
    return client_source_kind(client) == "embedded_wdt_alpha"


def describe_unsupported_client_source(client: dict[str, Any]) -> str:
    source_kind = client_source_kind(client)
    if source_kind == "embedded_wdt_alpha":
        return "embedded alpha WDT terrain requires archive scan extraction"
    return f"unsupported terrain_source_kind={source_kind}"


def sanitize_segment(value: str) -> str:
    keep: list[str] = []
    for char in value:
        if char.isalnum() or char in ("-", "_"):
            keep.append(char)
        elif char in (".", " "):
            keep.append("_")
    result = "".join(keep).strip("_")
    return result or "unknown"


def normalize_era_tag(value: str) -> str:
    return sanitize_segment(value.replace(".", "_").replace("-", "_").lower())


def try_parse_tile_coordinates(tile_name: str) -> tuple[int, int] | None:
    parts = tile_name.rsplit("_", 2)
    if len(parts) < 3:
        return None

    try:
        tile_x = int(parts[-2])
        tile_y = int(parts[-1])
    except ValueError:
        return None

    return tile_x, tile_y


def get_field(entry: dict[str, Any], snake: str, pascal: str, default: Any = None) -> Any:
    return entry.get(snake, entry.get(pascal, default))


def normalize_fingerprint_entry(entry: dict[str, Any], client_id: str, map_name: str, era: str) -> dict[str, Any]:
    return {
        "tile_name": get_field(entry, "tile_name", "TileName", ""),
        "source_path": get_field(entry, "source_path", "SourcePath", ""),
        "file_size": int(get_field(entry, "file_size", "FileSize", 0) or 0),
        "fingerprint": get_field(entry, "fingerprint", "Fingerprint", ""),
        "mcnk_count": int(get_field(entry, "mcnk_count", "McnkCount", 0) or 0),
        "total_layer_count": int(get_field(entry, "total_layer_count", "TotalLayerCount", 0) or 0),
        "max_layer_count": int(get_field(entry, "max_layer_count", "MaxLayerCount", 0) or 0),
        "chunks_with_mcvt": int(get_field(entry, "chunks_with_mcvt", "ChunksWithMcvt", 0) or 0),
        "chunks_with_holes": int(get_field(entry, "chunks_with_holes", "ChunksWithHoles", 0) or 0),
        "chunks_with_liquid_flags": int(get_field(entry, "chunks_with_liquid_flags", "ChunksWithLiquidFlags", 0) or 0),
        "unique_texture_ids": list(get_field(entry, "unique_texture_ids", "UniqueTextureIds", []) or []),
        "global_min_height": float(get_field(entry, "global_min_height", "GlobalMinHeight", 0.0) or 0.0),
        "global_max_height": float(get_field(entry, "global_max_height", "GlobalMaxHeight", 0.0) or 0.0),
        "client_id": client_id,
        "map_name": map_name,
        "era": era,
        "era_tag": normalize_era_tag(client_id),
    }


def normalize_scan_fingerprint_entry(entry: dict[str, Any], client_id: str, map_name: str, era: str) -> dict[str, Any]:
    metrics = get_field(entry, "metrics", "Metrics", {}) or {}
    signals = get_field(entry, "signals", "Signals", {}) or {}
    tile_name = str(get_field(entry, "tile_name", "TileName", "") or "")
    source_path = str(get_field(entry, "root_adt_path", "RootAdtPath", "") or "")
    tile_x = int(get_field(entry, "tile_x", "TileX", 0) or 0)
    tile_y = int(get_field(entry, "tile_y", "TileY", 0) or 0)
    texture_layers = int(get_field(metrics, "texture_layer_count", "TextureLayerCount", 0) or 0)
    hole_coverage = float(get_field(metrics, "hole_coverage", "HoleCoverage", 0.0) or 0.0)
    liquid_coverage = float(get_field(metrics, "liquid_coverage", "LiquidCoverage", 0.0) or 0.0)
    has_root = bool(get_field(signals, "has_root_adt", "HasRootAdt", False))
    fingerprint_base = f"scan|{client_id}|{map_name}|{tile_x}|{tile_y}|{source_path}"
    fingerprint = hashlib.sha256(fingerprint_base.encode("utf-8")).hexdigest()

    return {
        "tile_name": tile_name,
        "source_path": source_path,
        "file_size": 0,
        "fingerprint": fingerprint,
        "mcnk_count": 256 if has_root else 0,
        "total_layer_count": texture_layers,
        "max_layer_count": texture_layers,
        "chunks_with_mcvt": 256 if has_root else 0,
        "chunks_with_holes": int(max(0.0, min(1.0, hole_coverage)) * 256),
        "chunks_with_liquid_flags": int(max(0.0, min(1.0, liquid_coverage)) * 256),
        "unique_texture_ids": [],
        "global_min_height": float(get_field(metrics, "height_min", "HeightMin", 0.0) or 0.0),
        "global_max_height": float(get_field(metrics, "height_max", "HeightMax", 0.0) or 0.0),
        "client_id": client_id,
        "map_name": map_name,
        "era": era,
        "era_tag": normalize_era_tag(client_id),
    }


def write_filesystem_scan_manifest(client: dict[str, Any], client_id: str, map_name: str, scan_manifest_path: Path) -> int:
    adt_dir = Path(str(client.get("client_path") or ""))
    entries: list[dict[str, Any]] = []
    for adt_path in sorted(adt_dir.glob(f"{map_name}_*.adt")):
        if any(adt_path.name.endswith(suffix) for suffix in ["_tex0.adt", "_obj0.adt", "_lod.adt"]):
            continue

        tile_name = adt_path.stem
        tile_x = 0
        tile_y = 0
        parsed_coords = try_parse_tile_coordinates(tile_name)
        if parsed_coords is not None:
            tile_x, tile_y = parsed_coords

        entries.append({
            "SampleId": f"{client_id}:{tile_name}",
            "SourceKind": "ClientRoot",
            "BuildLabel": client_id,
            "MapName": map_name,
            "TileX": tile_x,
            "TileY": tile_y,
            "SourceRoot": str(adt_dir),
            "RootAdtPath": str(adt_path),
            "ObjAdtPath": str(adt_path.with_name(tile_name + "_obj0.adt")) if adt_path.with_name(tile_name + "_obj0.adt").exists() else None,
            "TexAdtPath": str(adt_path.with_name(tile_name + "_tex0.adt")) if adt_path.with_name(tile_name + "_tex0.adt").exists() else None,
            "LodAdtPath": str(adt_path.with_name(tile_name + "_lod.adt")) if adt_path.with_name(tile_name + "_lod.adt").exists() else None,
            "TileName": tile_name,
        })

    payload = {
        "schema_version": "v10-filesystem-scan-manifest.v1",
        "client_id": client_id,
        "map_name": map_name,
        "source_kind": "filesystem_adt_dir",
        "filesystem_input_dir": str(adt_dir),
        "entries": entries,
    }
    save_json(scan_manifest_path, payload)
    return len(entries)


def client_by_id(config: CorpusConfig, client_id: str) -> dict[str, Any] | None:
    for client in config.clients:
        if client.get("client_id") == client_id:
            return client
    return None


def maps_for_client(config: CorpusConfig, client: dict[str, Any], output_root: Path) -> list[str]:
    client_maps = client.get("maps")
    if client_source_kind(client) == "filesystem_adt_dir":
        if isinstance(client_maps, list) and client_maps:
            return [str(value) for value in client_maps if str(value).strip()]
        return config.maps[:]

    if isinstance(client_maps, list) and client_maps and not config.discover_maps:
        return [str(value) for value in client_maps if str(value).strip()]

    if not config.discover_maps:
        return config.maps[:]

    client_id = str(client["client_id"])
    maps_dir = ensure_dir(output_root / "map-lists")
    map_list_path = maps_dir / f"{sanitize_segment(client_id)}_maps.json"
    if map_list_path.exists():
        payload = load_json(map_list_path)
    else:
        print(f"  Discovering maps for {client_id}...", end=" ", flush=True)
        result = run_dotnet([
            "list-maps",
            "--client-root", str(client["client_path"]),
            "--output", str(map_list_path),
        ], timeout=600)
        if result.returncode != 0:
            print(f"FAILED (exit {result.returncode}); falling back to configured maps")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-3:]:
                    print(f"      {line}")
            return config.maps[:]
        print("done")
        payload = load_json(map_list_path)

    maps = [str(value) for value in payload.get("maps", []) if str(value).strip()]
    return maps or config.maps[:]


# ── Pipeline stages ──────────────────────────────────────────────────────────


def stage_extract(config: CorpusConfig) -> dict[str, Path]:
    """
    Stage 1: Materialize source tile-scan manifests for all clients/maps.
    Returns a dict mapping "{client_id}/{map_name}" to scan manifest path.
    """
    print("=" * 72)
    print("Stage 1: Source Tile Discovery")
    print("=" * 72)

    output_root = Path(config.output_root)
    source_catalog_root = ensure_dir(output_root / "source_catalog")
    extract_dirs: dict[str, Path] = {}

    for client in config.clients:
        client_id = client["client_id"]
        if client_id in config.skip_clients:
            print(f"  Skipping client {client_id} (in skip list)")
            continue

        if not check_client_exists(client):
            print(f"  Skipping client {client_id}: client path not found")
            continue

        client_root = client["client_path"]
        print(f"\n  Client: {client_id} ({client['era']})")
        print(f"    Root: {client_root}")
        source_kind = client_source_kind(client)
        print(f"    Source kind: {source_kind}")
        client_maps = maps_for_client(config, client, output_root)
        print(f"    Maps: {len(client_maps)}")

        for map_name in client_maps:
            if map_name in config.skip_maps:
                continue

            map_key = f"{client_id}/{map_name}"
            scan_manifest_path = source_catalog_root / f"{sanitize_segment(client_id)}__{sanitize_segment(map_name)}.scan.json"
            extract_marker_path = source_catalog_root / f"{sanitize_segment(client_id)}__{sanitize_segment(map_name)}.meta.json"

            if client_source_kind(client) == "filesystem_adt_dir":
                if not Path(client_root).exists():
                    print(f"    {map_name}: filesystem source missing ({client_root})")
                    continue

                entry_count = write_filesystem_scan_manifest(client, client_id, map_name, scan_manifest_path)
                print(f"    {map_name}: filesystem scan ({entry_count} tiles)")
                save_json(extract_marker_path, {
                    "client_id": client_id,
                    "map_name": map_name,
                    "source_kind": "filesystem_adt_dir",
                    "scan_entry_count": entry_count,
                    "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                })
                if entry_count > 0:
                    extract_dirs[map_key] = scan_manifest_path
                continue

            if scan_manifest_path.exists():
                report = load_json(scan_manifest_path)
                existing_entries = report.get("Entries") or report.get("entries") or []
                if len(existing_entries) > 0:
                    if not report.get("client_id"):
                        report["client_id"] = client_id
                    if not report.get("map_name"):
                        report["map_name"] = map_name
                    if not report.get("source_kind"):
                        report["source_kind"] = client_source_kind(client)
                    save_json(scan_manifest_path, report)
                    print(f"    {map_name}: scan exists ({len(existing_entries)} tiles), skipping")
                    extract_dirs[map_key] = scan_manifest_path
                    continue

            print(f"    {map_name}: scanning source tiles...", end=" ", flush=True)
            try:
                result = run_dotnet([
                    "dataset-scan",
                    "--client-root", client_root,
                    "--map", map_name,
                    "--build", client_id,
                    "--output", str(scan_manifest_path),
                ], timeout=900)
                if result.returncode != 0:
                    print(f"FAILED (exit {result.returncode})")
                    if result.stderr:
                        for line in result.stderr.strip().split("\n")[-3:]:
                            print(f"      {line}")
                    continue

                report = load_json(scan_manifest_path)
                report["client_id"] = client_id
                report["map_name"] = map_name
                report["source_kind"] = client_source_kind(client)
                save_json(scan_manifest_path, report)
                entry_count = len(report.get("Entries") or report.get("entries") or [])
                print(f"{entry_count} tiles")
                save_json(extract_marker_path, {
                    "client_id": client_id,
                    "map_name": map_name,
                    "source_kind": client_source_kind(client),
                    "scan_entry_count": entry_count,
                    "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                })
                if entry_count > 0:
                    extract_dirs[map_key] = scan_manifest_path

            except subprocess.TimeoutExpired:
                print("TIMEOUT")
            except Exception as e:
                print(f"ERROR: {e}")

    print(f"\nSource discovery complete: {len(extract_dirs)} client/map combinations")
    return extract_dirs


def stage_fingerprint(config: CorpusConfig, extract_dirs: dict[str, Path]) -> Path:
    """
    Stage 2: Build dedup fingerprints from scan manifests.
    Returns the path to the merged fingerprint report.
    """
    print("\n" + "=" * 72)
    print("Stage 2: Tile Fingerprinting")
    print("=" * 72)

    output_root = Path(config.output_root)
    fingerprint_dir = ensure_dir(output_root / "fingerprints")
    all_entries: list[dict[str, Any]] = []

    for map_key, source_path in sorted(extract_dirs.items()):
        client_id, map_name = map_key.split("/", 1)

        # Find the era for this client
        era = ""
        for c in config.clients:
            if c["client_id"] == client_id:
                era = c.get("era", "")
                break

        fingerprint_path = fingerprint_dir / f"{client_id}__{map_name}.json"

        if source_path.is_file() and source_path.suffix.lower() == ".json":
            scan_report = load_json(source_path)
            filesystem_input_dir = str(scan_report.get("filesystem_input_dir") or "").strip()
            if filesystem_input_dir:
                if fingerprint_path.exists():
                    print(f"  {map_key}: fingerprint exists, loading")
                    report = load_json(fingerprint_path)
                else:
                    print(f"  {map_key}: fingerprinting filesystem tiles...", end=" ", flush=True)
                    result = run_dotnet([
                        "adt-fingerprint",
                        "--input-dir", filesystem_input_dir,
                        "--output", str(fingerprint_path),
                    ])
                    if result.returncode != 0:
                        print(f"FAILED (exit {result.returncode})")
                        continue
                    print("done")
                    report = load_json(fingerprint_path)

                for entry in report.get("entries") or report.get("Entries", []):
                    if isinstance(entry, dict):
                        all_entries.append(normalize_fingerprint_entry(entry, client_id, map_name, era))
                continue

            print(f"  {map_key}: adapting archive scan metrics")
            scan_entries = scan_report.get("Entries") or scan_report.get("entries") or []
            normalized_entries: list[dict[str, Any]] = []
            for entry in scan_entries:
                if isinstance(entry, dict):
                    normalized_entries.append(normalize_scan_fingerprint_entry(entry, client_id, map_name, era))

            scan_fingerprint_report = {
                "schema_version": "adt-fingerprint.v1.scan-adapter",
                "source": str(source_path),
                "entries": normalized_entries,
            }
            save_json(fingerprint_path, scan_fingerprint_report)
            all_entries.extend(normalized_entries)
            continue

        if fingerprint_path.exists():
            print(f"  {map_key}: fingerprint exists, loading")
            report = load_json(fingerprint_path)
        else:
            print(f"  {map_key}: fingerprinting...", end=" ", flush=True)
            try:
                result = run_dotnet([
                    "adt-fingerprint",
                    "--input-dir", str(source_path),
                    "--output", str(fingerprint_path),
                ])
                if result.returncode != 0:
                    print(f"FAILED (exit {result.returncode})")
                    continue
                print("done")
                report = load_json(fingerprint_path)
            except Exception as e:
                print(f"ERROR: {e}")
                continue

        # Normalize the C# fingerprint report into the snake_case shape used
        # by the dedup scorer. C# writes PascalCase by default.
        for entry in report.get("entries") or report.get("Entries", []):
            if isinstance(entry, dict):
                all_entries.append(normalize_fingerprint_entry(entry, client_id, map_name, era))

    # Write merged fingerprint report
    merged_report = {
        "schema_version": "1.0",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_entries": len(all_entries),
        "clients_processed": len(set(e.get("client_id", "") for e in all_entries)),
        "maps_processed": len(set(e.get("map_name", "") for e in all_entries)),
        "entries": all_entries,
    }
    merged_path = fingerprint_dir / "merged_fingerprints.json"
    save_json(merged_path, merged_report)
    print(f"\nMerged fingerprint report: {len(all_entries)} entries")
    print(f"  Written to: {merged_path}")
    return merged_path


def stage_deduplicate(config: CorpusConfig, merged_fingerprint_path: Path) -> Path:
    """
    Stage 3: Deduplicate based on SHA256 fingerprints.
    Selects the best ~750-1500 unique tiles, preferring tiles with
    more texture layers, MCVT data, and diverse era coverage.
    Returns the path to the deduplicated manifest.
    """
    print("\n" + "=" * 72)
    print("Stage 3: Deduplication")
    print("=" * 72)

    report = load_json(merged_fingerprint_path)
    entries = report.get("entries", [])
    print(f"  Total entries: {len(entries)}")

    # Group by fingerprint
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        fp = entry.get("fingerprint", "")
        groups[fp].append(entry)

    print(f"  Unique fingerprints: {len(groups)}")

    # Filter out blank/empty tiles (height near zero, no layers, no MCVT)
    def is_blank_tile(entry: dict[str, Any]) -> bool:
        """Check if a tile is essentially blank (no real terrain data)."""
        return (
            entry.get("total_layer_count", 0) == 0
            and entry.get("chunks_with_mcvt", 0) == 0
            and abs(entry.get("global_min_height", 0)) < 0.001
            and abs(entry.get("global_max_height", 0)) < 0.001
        )

    blank_count = sum(1 for g in groups.values() for e in g if is_blank_tile(e))
    print(f"  Blank/empty tiles: {blank_count}")

    # Score each fingerprint group and pick the best representative
    def score_entry(entry: dict[str, Any]) -> float:
        """Score a tile for selection preference.
        Higher score = more valuable for training.
        """
        score = 0.0

        # Prefer tiles with texture layers (more interesting terrain)
        score += entry.get("total_layer_count", 0) * 0.5

        # Prefer tiles with MCVT height data
        if entry.get("chunks_with_mcvt", 0) > 0:
            score += 10.0

        # Prefer tiles with height variation
        height_range = entry.get("global_max_height", 0) - entry.get("global_min_height", 0)
        if height_range > 1.0:
            score += min(height_range * 0.1, 20.0)

        # Prefer tiles with holes (more complex terrain)
        score += entry.get("chunks_with_holes", 0) * 2.0

        # Prefer tiles with liquid flags
        score += entry.get("chunks_with_liquid_flags", 0) * 1.0

        # Prefer tiles with unique texture IDs
        unique_tex = entry.get("unique_texture_ids", [])
        score += len(unique_tex) * 3.0

        # Era bonus: prefer later clients (more data signals)
        era = entry.get("era", "")
        if era == "development":
            score += 8.0
        elif era == "cata":
            score += 5.0
        elif era == "wotlk":
            score += 3.0
        elif era == "alpha":
            score += 1.0

        return score

    # Select best representative per fingerprint group
    selected: list[dict[str, Any]] = []
    for fp, group_entries in groups.items():
        # Skip blank tiles
        non_blank = [e for e in group_entries if not is_blank_tile(e)]
        if not non_blank:
            continue

        # Pick the best entry from this group
        best = max(non_blank, key=score_entry)
        selected.append(best)

    print(f"  Non-blank unique tiles: {len(selected)}")

    # Sort by score descending
    selected.sort(key=score_entry, reverse=True)

    # Guarantee at least one representative per client before global trimming.
    guaranteed_by_client: list[dict[str, Any]] = []
    seen_clients: set[str] = set()
    for entry in selected:
        client_id = str(entry.get("client_id") or "")
        if not client_id or client_id in seen_clients:
            continue
        seen_clients.add(client_id)
        guaranteed_by_client.append(entry)

    if guaranteed_by_client:
        guaranteed_keys = {entry.get("fingerprint") for entry in guaranteed_by_client}
        remainder = [entry for entry in selected if entry.get("fingerprint") not in guaranteed_keys]
        selected = guaranteed_by_client + remainder

    # Apply tile budget
    max_tiles = config.max_tiles
    min_tiles = config.min_tiles

    if len(selected) > max_tiles:
        print(f"  Trimming to {max_tiles} tiles (from {len(selected)})")
        selected = selected[:max_tiles]
    elif len(selected) < min_tiles:
        print(f"  WARNING: only {len(selected)} unique tiles (min {min_tiles})")

    # Build deduplicated manifest
    dedup_manifest = {
        "schema_version": "v10.1-dedup-v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_unique_tiles": len(selected),
        "max_tiles_budget": max_tiles,
        "min_tiles_budget": min_tiles,
        "source_fingerprint_report": str(merged_fingerprint_path),
        "entries": selected,
    }

    output_root = Path(config.output_root)
    dedup_dir = ensure_dir(output_root / "deduplicated")
    dedup_path = dedup_dir / "deduplicated_manifest.json"
    save_json(dedup_path, dedup_manifest)
    print(f"\nDeduplicated manifest: {len(selected)} tiles")
    print(f"  Written to: {dedup_path}")

    # Print summary by era
    era_counts: dict[str, int] = defaultdict(int)
    for entry in selected:
        era_counts[entry.get("era", "unknown")] += 1
    print(f"\n  Era breakdown:")
    for era, count in sorted(era_counts.items()):
        print(f"    {era}: {count} tiles")

    return dedup_path


def stage_build_shards(config: CorpusConfig, dedup_path: Path) -> Path:
    """
    Stage 4: Build v10 NPZ shards from the deduplicated tile set.
    Returns the merged native Stage 1 manifest.
    """
    print("\n" + "=" * 72)
    print("Stage 4: v10 NPZ Shard Building")
    print("=" * 72)

    dedup_manifest = load_json(dedup_path)
    entries = dedup_manifest.get("entries", [])

    output_root = Path(config.output_root)
    shard_output_root = ensure_dir(output_root / "v10_shards")
    manifest_path = output_root / "v10_full_native_stage1_manifest.json"
    manifest_entries: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    def write_progress_manifest() -> None:
        manifest = {
            "schema_version": "v10-full-native-stage1-manifest.v1",
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "deduplicated_manifest": str(dedup_path),
            "output_root": str(shard_output_root),
            "max_tiles_budget": config.max_tiles,
            "require_minimap": config.require_minimap,
            "shard_batch_size": config.shard_batch_size,
            "shard_batch_timeout_seconds": config.shard_batch_timeout_seconds,
            "written_or_existing": len(manifest_entries),
            "skipped_count": len(skipped),
            "entries": manifest_entries,
            "skipped": skipped,
        }
        save_json(manifest_path, manifest)

    def append_group_manifest(group_manifest_path: Path, dataset_key: str, client_id: str, map_name: str) -> None:
        group_manifest = load_json(group_manifest_path)
        for shard_entry in group_manifest.get("entries") or group_manifest.get("Entries", []):
            if not isinstance(shard_entry, dict):
                continue
            tile_name = str(get_field(shard_entry, "tile_name", "TileName", ""))
            shard_path = str(get_field(shard_entry, "shard_path", "ShardPath", ""))
            placement_path = str(get_field(shard_entry, "placement_path", "PlacementPath", "") or "")
            available_signals = get_field(shard_entry, "available_signals", "AvailableSignals", []) or []
            manifest_entries.append({
                "dataset_key": dataset_key,
                "era_tag": normalize_era_tag(client_id),
                "client_id": client_id,
                "map_name": map_name,
                "tile_name": tile_name,
                "source_adt_path": str(get_field(shard_entry, "source_adt_path", "SourceAdtPath", "")),
                "shard_path": shard_path,
                "placement_path": placement_path,
                "build_key": client_id,
                "available_signals": available_signals,
                "status": "written",
            })

        for skip_entry in group_manifest.get("skipped") or group_manifest.get("Skipped", []):
            if isinstance(skip_entry, dict):
                skipped.append({
                    "client_id": client_id,
                    "map_name": map_name,
                    "tile_name": str(get_field(skip_entry, "tile_name", "TileName", "")),
                    "reason": str(get_field(skip_entry, "reason", "Reason", "stage1_skipped")),
                    "source_path": str(get_field(skip_entry, "source_adt_path", "SourceAdtPath", "")),
                })

    print(f"  Selected tiles: {len(entries)}")
    print("  NPZ-only mode: enabled (no ADT dump stage)")
    print(f"  Shard output root: {shard_output_root}")

    total_entries = len(entries)
    for index, entry in enumerate(entries, start=1):
        client_id = str(entry.get("client_id") or "")
        map_name = str(entry.get("map_name") or "")
        source_path = str(entry.get("source_path") or "")
        tile_name = str(entry.get("tile_name") or "")
        if not tile_name:
            tile_name = Path(source_path.split("#", 1)[0]).stem

        client = client_by_id(config, client_id)
        if client is None:
            skipped.append({
                "tile_name": tile_name,
                "reason": "missing_client_config",
                "client_id": client_id,
                "map_name": map_name,
                "source_path": source_path,
            })
            continue

        dataset_key = f"{sanitize_segment(client_id)}__{sanitize_segment(map_name)}"
        shard_dir = ensure_dir(shard_output_root / dataset_key)
        output_npz = shard_dir / f"{tile_name}_v10.npz"
        output_placements = shard_dir / f"{tile_name}_v10_placements.json"

        if output_npz.exists() and not config.overwrite_shards:
            manifest_entries.append({
                "dataset_key": dataset_key,
                "era_tag": normalize_era_tag(client_id),
                "client_id": client_id,
                "map_name": map_name,
                "tile_name": tile_name,
                "source_adt_path": source_path,
                "shard_path": str(output_npz),
                "placement_path": str(output_placements) if output_placements.exists() else "",
                "build_key": client_id,
                "available_signals": [],
                "status": "existing",
            })
            continue

        args = [
            "extract-v10-tensors",
            "--input", source_path,
            "--output", str(output_npz),
            "--build-key", client_id,
            "--map-name", map_name,
        ]

        if client_source_kind(client) == "filesystem_adt_dir":
            minimap_root = str(client.get("minimap_root") or "")
            if minimap_root:
                args.extend(["--minimap-root", minimap_root])
        else:
            args.extend(["--client-root", str(client["client_path"])])

        if config.require_minimap:
            args.append("--require-minimap")

        print(f"  [{index}/{total_entries}] {client_id}/{map_name} {tile_name}", end=" ", flush=True)
        try:
            result = run_dotnet(args, timeout=max(1, int(config.shard_batch_timeout_seconds)))
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            skipped.append({
                "tile_name": tile_name,
                "reason": "extract_timeout",
                "client_id": client_id,
                "map_name": map_name,
                "source_path": source_path,
            })
            write_progress_manifest()
            continue

        if result.returncode != 0:
            print(f"FAILED (exit {result.returncode})")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-3:]:
                    print(f"      {line}")
            skipped.append({
                "tile_name": tile_name,
                "reason": "extract_failed",
                "client_id": client_id,
                "map_name": map_name,
                "source_path": source_path,
                "stderr_tail": result.stderr.strip().split("\n")[-6:] if result.stderr else [],
            })
            write_progress_manifest()
            continue

        print("ok")
        manifest_entries.append({
            "dataset_key": dataset_key,
            "era_tag": normalize_era_tag(client_id),
            "client_id": client_id,
            "map_name": map_name,
            "tile_name": tile_name,
            "source_adt_path": source_path,
            "shard_path": str(output_npz),
            "placement_path": str(output_placements) if output_placements.exists() else "",
            "build_key": client_id,
            "available_signals": [],
            "status": "written",
        })
        write_progress_manifest()

    write_progress_manifest()

    print(f"\nShard building complete: {len(manifest_entries)} usable, {len(skipped)} skipped")
    print(f"  Manifest: {manifest_path}")
    return manifest_path


def stage_curate(config: CorpusConfig, stage1_manifest_path: Path) -> Path:
    """Run the trainer-facing curation pass, capped to config.max_tiles."""
    print("\n" + "=" * 72)
    print("Stage 5: Trainer Input Curation")
    print("=" * 72)

    output_root = Path(config.output_root)
    curated_dir = ensure_dir(output_root / "curated")
    curated_manifest = curated_dir / "v10_full_native_curated_manifest.json"
    report_path = curated_dir / "v10_full_native_curated_report.json"
    script_path = REPO_ROOT / "wow-viewer" / "scripts" / "curate_v10_training_shards.py"

    cmd = [
        sys.executable,
        str(script_path),
        str(stage1_manifest_path),
        "--output", str(curated_manifest),
        "--report", str(report_path),
        "--max-total", str(config.max_tiles),
        "--max-selected-fraction", str(config.curation_max_selected_fraction),
        "--max-per-era", str(config.curation_max_per_era),
    ]
    print(f"  [cmd] {' '.join(str(arg) for arg in cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"curation failed with exit code {result.returncode}")

    print(result.stdout.strip())
    print(f"  Curated manifest: {curated_manifest}")
    print(f"  Curation report: {report_path}")
    return curated_manifest


# ── Main ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="v10.1 Corpus Builder - Farm, deduplicate, and build v10 shards"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to config JSON (overrides defaults)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Print what would be done without executing",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=0,
        help="Maximum native v10 shards to pass into training curation. Overrides config.max_tiles.",
    )
    parser.add_argument(
        "--no-discover-maps",
        action="store_true",
        help="Use the configured map list instead of discovering maps from each client.",
    )
    parser.add_argument(
        "--allow-missing-minimap",
        action="store_true",
        help="Build shards even when an archive-backed minimap cannot be found. Trainer curation may reject them.",
    )
    parser.add_argument(
        "--overwrite-shards",
        action="store_true",
        help="Rewrite existing native v10 NPZ shards.",
    )
    parser.add_argument(
        "--shard-batch-size",
        type=int,
        default=0,
        help="Selected tiles per extract-v10-tensors batch. Default comes from config.",
    )
    parser.add_argument(
        "--shard-batch-timeout-seconds",
        type=int,
        default=0,
        help="Timeout for each extract-v10-tensors batch before skipping. Default comes from config.",
    )
    parser.add_argument(
        "--no-curate",
        action="store_true",
        help="Stop after building the merged native Stage 1 manifest.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Stop after source tile discovery",
    )
    parser.add_argument(
        "--fingerprint-only",
        action="store_true",
        help="Stop after fingerprinting",
    )
    parser.add_argument(
        "--deduplicate-only",
        action="store_true",
        help="Stop after deduplication",
    )
    parser.add_argument(
        "--shards-only",
        action="store_true",
        help="Build shards from an existing deduplicated manifest and stop before curation.",
    )
    parser.add_argument(
        "--curate-only",
        action="store_true",
        help="Run only trainer-facing curation from an existing native Stage 1 manifest.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from a specific stage: extract, fingerprint, deduplicate, shards (extract = source discovery)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    config = CorpusConfig()
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            overrides = load_json(config_path)
            for key, value in overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            print(f"Loaded config from {config_path}")
        else:
            print(f"Warning: config file not found: {config_path}")

    if args.max_tiles > 0:
        config.max_tiles = args.max_tiles
    if args.no_discover_maps:
        config.discover_maps = False
    if args.allow_missing_minimap:
        config.require_minimap = False
    if args.overwrite_shards:
        config.overwrite_shards = True
    if args.shard_batch_size > 0:
        config.shard_batch_size = args.shard_batch_size
    if args.shard_batch_timeout_seconds > 0:
        config.shard_batch_timeout_seconds = args.shard_batch_timeout_seconds
    if args.no_curate:
        config.run_curation = False

    print("=" * 72)
    print("v10.1 Corpus Builder")
    print("=" * 72)
    print(f"Output root: {config.output_root}")
    print(f"Clients: {len(config.clients)}")
    print(f"Maps: {'discover per client' if config.discover_maps else len(config.maps)}")
    print(f"Tile budget: {config.min_tiles}-{config.max_tiles}")
    print(f"Require minimap: {config.require_minimap}")
    print(f"Trainer curation cap: {config.max_tiles}")
    print(f"Shard batch size: {config.shard_batch_size}")
    print(f"Shard batch timeout: {config.shard_batch_timeout_seconds}s")
    print()

    if args.dry_run:
        print("DRY RUN - no commands will be executed")
        for client in config.clients:
            client_id = client["client_id"]
            exists = check_client_exists(client)
            print(f"  Client {client_id}: {'EXISTS' if exists else 'MISSING'}")
            if exists:
                source_kind = client_source_kind(client)
                print(f"    Source kind: {source_kind}")
                map_list_path = Path(config.output_root) / "map-lists" / f"{sanitize_segment(client_id)}_maps.json"
                if config.discover_maps and not map_list_path.exists():
                    print("    Would discover maps from client archive/catalog")
                    continue
                client_maps = maps_for_client(config, client, Path(config.output_root))
                for map_name in client_maps[:5]:  # show first 5
                    if client_source_kind(client) == "filesystem_adt_dir":
                        print(f"    Would use filesystem ADT dir: {map_name}")
                    else:
                        print(f"    Would scan (client-root): {map_name}")
                if len(client_maps) > 5:
                    print(f"    ... and {len(client_maps) - 5} more maps")
        return

    output_root = Path(config.output_root)
    if args.curate_only:
        stage1_manifest = output_root / "v10_full_native_stage1_manifest.json"
        if not stage1_manifest.exists():
            print(f"ERROR: native Stage 1 manifest not found: {stage1_manifest}")
            return
        stage_curate(config, stage1_manifest)
        return

    # ── Stage 1: Source Discovery ────────────────────────────────────────
    resume_from = "shards" if args.shards_only else (args.resume or "extract")
    extract_dirs: dict[str, Path] = {}

    if resume_from in ("extract", "all"):
        extract_dirs = stage_extract(config)
        if args.extract_only:
            print("\n--extract-only: stopping after source discovery")
            return
    else:
        # Load from previous run
        source_catalog_root = Path(config.output_root) / "source_catalog"
        if source_catalog_root.exists():
            for scan_file in sorted(source_catalog_root.glob("*.scan.json")):
                payload = load_json(scan_file)
                client_id = str(payload.get("client_id") or "").strip()
                map_name = str(payload.get("map_name") or "").strip()
                if not client_id or not map_name:
                    entries = payload.get("Entries") or payload.get("entries") or []
                    if entries and isinstance(entries[0], dict):
                        first = entries[0]
                        client_id = str(first.get("BuildLabel") or first.get("build_label") or "").strip()
                        map_name = str(first.get("MapName") or first.get("map_name") or "").strip()
                if not client_id or not map_name:
                    continue
                key = f"{client_id}/{map_name}"
                extract_dirs[key] = scan_file
            print(f"Resumed: {len(extract_dirs)} source manifests")

    # ── Stage 2: Fingerprint ─────────────────────────────────────────────
    merged_fingerprint_path: Path | None = None

    if resume_from in ("extract", "fingerprint", "all"):
        if not extract_dirs:
            # Try to load from previous source discovery output
            source_catalog_root = Path(config.output_root) / "source_catalog"
            if source_catalog_root.exists():
                for scan_file in sorted(source_catalog_root.glob("*.scan.json")):
                    payload = load_json(scan_file)
                    client_id = str(payload.get("client_id") or "").strip()
                    map_name = str(payload.get("map_name") or "").strip()
                    if not client_id or not map_name:
                        entries = payload.get("Entries") or payload.get("entries") or []
                        if entries and isinstance(entries[0], dict):
                            first = entries[0]
                            client_id = str(first.get("BuildLabel") or first.get("build_label") or "").strip()
                            map_name = str(first.get("MapName") or first.get("map_name") or "").strip()
                    if not client_id or not map_name:
                        continue
                    key = f"{client_id}/{map_name}"
                    extract_dirs[key] = scan_file
                print(f"Loaded {len(extract_dirs)} source manifests from disk")

        merged_fingerprint_path = stage_fingerprint(config, extract_dirs)
        if args.fingerprint_only:
            print("\n--fingerprint-only: stopping after fingerprinting")
            return
    else:
        # Load from previous fingerprint
        fingerprint_dir = Path(config.output_root) / "fingerprints"
        merged_path = fingerprint_dir / "merged_fingerprints.json"
        if merged_path.exists():
            merged_fingerprint_path = merged_path
            print(f"Resumed fingerprint: {merged_fingerprint_path}")

    # ── Stage 3: Deduplicate ─────────────────────────────────────────────
    dedup_path: Path | None = None

    if resume_from in ("extract", "fingerprint", "deduplicate", "all"):
        if merged_fingerprint_path is None:
            print("ERROR: no fingerprint data available for deduplication")
            return

        dedup_path = stage_deduplicate(config, merged_fingerprint_path)
        if args.deduplicate_only:
            print("\n--deduplicate-only: stopping after deduplication")
            return
    else:
        # Load from previous dedup
        dedup_dir = Path(config.output_root) / "deduplicated"
        dedup_manifest = dedup_dir / "deduplicated_manifest.json"
        if dedup_manifest.exists():
            dedup_path = dedup_manifest
            print(f"Resumed dedup: {dedup_path}")

    # ── Stage 4: Build shards ────────────────────────────────────────────
    stage1_manifest_path: Path | None = None
    if resume_from in ("extract", "fingerprint", "deduplicate", "shards", "all"):
        if dedup_path is None:
            print("ERROR: no deduplicated manifest available for shard building")
            return

        stage1_manifest_path = stage_build_shards(config, dedup_path)
        if args.shards_only:
            print("\n--shards-only: stopping after shard building")
            return

    if config.run_curation:
        if stage1_manifest_path is None:
            stage1_manifest_path = Path(config.output_root) / "v10_full_native_stage1_manifest.json"
        if not stage1_manifest_path.exists():
            print(f"ERROR: native Stage 1 manifest not found for curation: {stage1_manifest_path}")
            return
        stage_curate(config, stage1_manifest_path)

    print("\n" + "=" * 72)
    print("v10.1 Corpus Builder complete!")
    print("=" * 72)


if __name__ == "__main__":
    main()
