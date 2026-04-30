#!/usr/bin/env python3
"""
v10.1 Corpus Builder Orchestrator

Farms ADT data from all staged WoW clients, fingerprints for deduplication,
selects a curated set of ~750-1500 unique tiles, and builds v10 NPZ shards.

Usage:
    python build_v10_corpus.py [--config <config.json>] [--dry-run]
    python build_v10_corpus.py --extract-only  # stop after extraction
    python build_v10_corpus.py --fingerprint-only  # stop after fingerprinting
    python build_v10_corpus.py --deduplicate-only  # stop after dedup
"""

from __future__ import annotations

import argparse
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
STAGED_CLIENTS_ROOT = REPO_ROOT / "output" / "tmp" / "wowarchive-clients"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "v10_1_corpus"

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
    },
    {
        "client_id": "0.5.5.3494",
        "version": "0.5.5.3494",
        "client_path": str(STAGED_CLIENTS_ROOT / "0_5_5_3494" / "World of Warcraft"),
        "era": "alpha",
    },
    {
        "client_id": "0.7.0.3694",
        "version": "0.7.0.3694",
        "client_path": str(STAGED_CLIENTS_ROOT / "0_7_0_3694" / "World of Warcraft"),
        "era": "alpha",
    },
    {
        "client_id": "3.0.1.8303",
        "version": "3.0.1.8303",
        "client_path": str(STAGED_CLIENTS_ROOT / "3_0_1_8303" / "World of Warcraft"),
        "era": "wotlk",
    },
    {
        "client_id": "3.3.5.12340",
        "version": "3.3.5.12340",
        "client_path": str(STAGED_CLIENTS_ROOT / "3_3_5_12340" / "World of Warcraft"),
        "era": "wotlk",
    },
    {
        "client_id": "4.0.0.11927",
        "version": "4.0.0.11927",
        "client_path": str(STAGED_CLIENTS_ROOT / "4_0_0_11927" / "World of Warcraft"),
        "era": "cata",
    },
]


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class CorpusConfig:
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    max_tiles: int = 1500
    min_tiles: int = 750
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


def run_dotnet(args: list[str], cwd: str | None = None, timeout: int = 600) -> subprocess.CompletedProcess:
    """Run the converter tool with the given arguments."""
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


# ── Pipeline stages ──────────────────────────────────────────────────────────


def stage_extract(config: CorpusConfig) -> dict[str, Path]:
    """
    Stage 1: Extract ADTs from all clients for all maps.
    Returns a dict mapping "{client_id}/{map_name}" to the extraction output directory.
    """
    print("=" * 72)
    print("Stage 1: ADT Extraction")
    print("=" * 72)

    output_root = Path(config.output_root)
    raw_adt_root = ensure_dir(output_root / "raw_adts")
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

        for map_name in config.maps:
            if map_name in config.skip_maps:
                continue

            map_key = f"{client_id}/{map_name}"
            map_output_dir = ensure_dir(raw_adt_root / client_id / map_name)

            # Check if already extracted
            existing_adts = list(map_output_dir.glob("*.adt"))
            root_adts = [p for p in existing_adts if not any(
                p.name.endswith(s) for s in ["_tex0.adt", "_obj0.adt", "_lod.adt"]
            )]
            if len(root_adts) > 0:
                print(f"    {map_name}: already extracted ({len(root_adts)} tiles), skipping")
                extract_dirs[map_key] = map_output_dir
                continue

            print(f"    {map_name}: extracting...", end=" ", flush=True)
            try:
                result = run_dotnet([
                    "extract-map",
                    "--client-root", client_root,
                    "--map", map_name,
                    "--output-dir", str(map_output_dir),
                ])
                if result.returncode != 0:
                    print(f"FAILED (exit {result.returncode})")
                    if result.stderr:
                        for line in result.stderr.strip().split("\n")[-3:]:
                            print(f"      {line}")
                    continue

                # Count extracted tiles
                extracted = list(map_output_dir.glob("*.adt"))
                root_count = len([p for p in extracted if not any(
                    p.name.endswith(s) for s in ["_tex0.adt", "_obj0.adt", "_lod.adt"]
                )])
                print(f"{root_count} tiles")
                extract_dirs[map_key] = map_output_dir

            except subprocess.TimeoutExpired:
                print("TIMEOUT")
            except Exception as e:
                print(f"ERROR: {e}")

    print(f"\nExtraction complete: {len(extract_dirs)} client/map combinations")
    return extract_dirs


def stage_fingerprint(config: CorpusConfig, extract_dirs: dict[str, Path]) -> Path:
    """
    Stage 2: Fingerprint all extracted ADTs.
    Returns the path to the merged fingerprint report.
    """
    print("\n" + "=" * 72)
    print("Stage 2: ADT Fingerprinting")
    print("=" * 72)

    output_root = Path(config.output_root)
    fingerprint_dir = ensure_dir(output_root / "fingerprints")
    all_entries: list[dict[str, Any]] = []

    for map_key, adt_dir in sorted(extract_dirs.items()):
        client_id, map_name = map_key.split("/", 1)

        # Find the era for this client
        era = ""
        for c in config.clients:
            if c["client_id"] == client_id:
                era = c.get("era", "")
                break

        fingerprint_path = fingerprint_dir / f"{client_id}__{map_name}.json"

        if fingerprint_path.exists():
            print(f"  {map_key}: fingerprint exists, loading")
            report = load_json(fingerprint_path)
        else:
            print(f"  {map_key}: fingerprinting...", end=" ", flush=True)
            try:
                result = run_dotnet([
                    "adt-fingerprint",
                    "--input-dir", str(adt_dir),
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

        # Enrich entries with client/map/era metadata
        for entry in report.get("Entries", []):
            entry["client_id"] = client_id
            entry["map_name"] = map_name
            entry["era"] = era
            all_entries.append(entry)

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
        if era == "cata":
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


def stage_build_shards(config: CorpusConfig, dedup_path: Path) -> None:
    """
    Stage 4: Build v10 NPZ shards from the deduplicated tile set.
    """
    print("\n" + "=" * 72)
    print("Stage 4: v10 NPZ Shard Building")
    print("=" * 72)

    dedup_manifest = load_json(dedup_path)
    entries = dedup_manifest.get("entries", [])

    output_root = Path(config.output_root)
    shard_output_dir = ensure_dir(output_root / "v10_shards")
    minimap_root = ensure_dir(output_root / "minimaps")

    # Group entries by client/map for batch processing
    # Each entry has source_path pointing to the ADT file
    # We need to run dataset-build-v10-stage1 on the ADT directories

    # Collect unique ADT directories
    adt_dirs: set[Path] = set()
    for entry in entries:
        source_path = Path(entry.get("source_path", ""))
        if source_path.exists():
            adt_dirs.add(source_path.parent)

    print(f"  Processing {len(adt_dirs)} ADT directories")

    for adt_dir in sorted(adt_dirs):
        # Count ADTs in this directory
        adt_files = list(adt_dir.glob("*.adt"))
        root_adts = [p for p in adt_files if not any(
            p.name.endswith(s) for s in ["_tex0.adt", "_obj0.adt", "_lod.adt"]
        )]
        if not root_adts:
            continue

        print(f"\n  Directory: {adt_dir}")
        print(f"    Root ADTs: {len(root_adts)}")

        try:
            result = run_dotnet([
                "dataset-build-v10-stage1",
                "--input-dir", str(adt_dir),
                "--output-dir", str(shard_output_dir),
                "--minimap-root", str(minimap_root),
            ])
            if result.returncode != 0:
                print(f"    FAILED (exit {result.returncode})")
                if result.stderr:
                    for line in result.stderr.strip().split("\n")[-3:]:
                        print(f"      {line}")
            else:
                # Parse output for summary
                for line in result.stdout.strip().split("\n"):
                    if "report" in line.lower() or "written" in line.lower() or "complete" in line.lower():
                        print(f"    {line}")

        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT (may still be processing)")
        except Exception as e:
            print(f"    ERROR: {e}")

    print(f"\nShard building complete. Output: {shard_output_dir}")


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
        "--extract-only",
        action="store_true",
        help="Stop after ADT extraction",
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
        "--resume",
        type=str,
        help="Resume from a specific stage: extract, fingerprint, deduplicate, shards",
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

    print("=" * 72)
    print("v10.1 Corpus Builder")
    print("=" * 72)
    print(f"Output root: {config.output_root}")
    print(f"Clients: {len(config.clients)}")
    print(f"Maps: {len(config.maps)}")
    print(f"Tile budget: {config.min_tiles}-{config.max_tiles}")
    print()

    if args.dry_run:
        print("DRY RUN - no commands will be executed")
        for client in config.clients:
            client_id = client["client_id"]
            exists = check_client_exists(client)
            print(f"  Client {client_id}: {'EXISTS' if exists else 'MISSING'}")
            if exists:
                for map_name in config.maps[:5]:  # show first 5
                    print(f"    Would extract: {map_name}")
                if len(config.maps) > 5:
                    print(f"    ... and {len(config.maps) - 5} more maps")
        return

    # ── Stage 1: Extract ─────────────────────────────────────────────────
    resume_from = args.resume or "extract"
    extract_dirs: dict[str, Path] = {}

    if resume_from in ("extract", "all"):
        extract_dirs = stage_extract(config)
        if args.extract_only:
            print("\n--extract-only: stopping after extraction")
            return
    else:
        # Load from previous run
        raw_adt_root = Path(config.output_root) / "raw_adts"
        if raw_adt_root.exists():
            for client_dir in sorted(raw_adt_root.iterdir()):
                if client_dir.is_dir():
                    for map_dir in sorted(client_dir.iterdir()):
                        if map_dir.is_dir():
                            key = f"{client_dir.name}/{map_dir.name}"
                            extract_dirs[key] = map_dir
            print(f"Resumed: {len(extract_dirs)} extraction directories")

    # ── Stage 2: Fingerprint ─────────────────────────────────────────────
    merged_fingerprint_path: Path | None = None

    if resume_from in ("extract", "fingerprint", "all"):
        if not extract_dirs:
            # Try to load from previous extraction
            raw_adt_root = Path(config.output_root) / "raw_adts"
            if raw_adt_root.exists():
                for client_dir in sorted(raw_adt_root.iterdir()):
                    if client_dir.is_dir():
                        for map_dir in sorted(client_dir.iterdir()):
                            if map_dir.is_dir():
                                key = f"{client_dir.name}/{map_dir.name}"
                                extract_dirs[key] = map_dir
                print(f"Loaded {len(extract_dirs)} extraction directories from disk")

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
    if resume_from in ("extract", "fingerprint", "deduplicate", "shards", "all"):
        if dedup_path is None:
            print("ERROR: no deduplicated manifest available for shard building")
            return

        stage_build_shards(config, dedup_path)

    print("\n" + "=" * 72)
    print("v10.1 Corpus Builder complete!")
    print("=" * 72)


if __name__ == "__main__":
    main()
