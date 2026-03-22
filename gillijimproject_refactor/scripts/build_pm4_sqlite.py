from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, TextIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a SQLite database from compact PM4 object-hypothesis reports."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ndjson", type=Path, help="Existing NDJSON produced by scan-hypotheses-ndjson")
    source.add_argument("--pm4-dir", type=Path, help="PM4 directory to scan via the CLI and ingest directly")
    parser.add_argument(
        "--cli-project",
        type=Path,
        default=Path("src/Pm4Research.Cli/Pm4Research.Cli.csproj"),
        help="Path to the Pm4Research.Cli project when --pm4-dir is used",
    )
    parser.add_argument(
        "--sqlite",
        type=Path,
        required=True,
        help="Output SQLite database path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    connection = sqlite3.connect(args.sqlite)
    try:
        configure_database(connection)
        clear_existing_data(connection)
        if args.ndjson is not None:
            with args.ndjson.open("r", encoding="utf-8") as handle:
                ingest_reports(connection, iter_reports(handle))
        else:
            ingest_reports(connection, iter_reports_from_cli(args.pm4_dir, args.cli_project))
        finalize_database(connection)
    finally:
        connection.close()

    return 0


def configure_database(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS tiles (
            id INTEGER PRIMARY KEY,
            source_path TEXT,
            tile_x INTEGER,
            tile_y INTEGER,
            version INTEGER NOT NULL,
            ck24_group_count INTEGER NOT NULL,
            total_hypothesis_count INTEGER NOT NULL,
            diagnostic_count INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS family_summaries (
            tile_id INTEGER NOT NULL,
            family TEXT NOT NULL,
            object_count INTEGER NOT NULL,
            max_surface_count INTEGER NOT NULL,
            max_index_count INTEGER NOT NULL,
            total_linked_mprl_count INTEGER NOT NULL,
            total_linked_in_bounds_count INTEGER NOT NULL,
            FOREIGN KEY(tile_id) REFERENCES tiles(id)
        );

        CREATE TABLE IF NOT EXISTS objects (
            tile_id INTEGER NOT NULL,
            family TEXT NOT NULL,
            family_object_index INTEGER NOT NULL,
            ck24 INTEGER NOT NULL,
            ck24_type INTEGER NOT NULL,
            ck24_object_id INTEGER NOT NULL,
            surface_count INTEGER NOT NULL,
            total_index_count INTEGER NOT NULL,
            mdos_count INTEGER NOT NULL,
            group_key_count INTEGER NOT NULL,
            mslk_group_object_id_count INTEGER NOT NULL,
            mslk_ref_index_count INTEGER NOT NULL,
            bounds_min_x REAL,
            bounds_min_y REAL,
            bounds_min_z REAL,
            bounds_max_x REAL,
            bounds_max_y REAL,
            bounds_max_z REAL,
            mprl_tile_ref_count INTEGER NOT NULL,
            mprl_linked_ref_count INTEGER NOT NULL,
            mprl_linked_normal_count INTEGER NOT NULL,
            mprl_linked_terminator_count INTEGER NOT NULL,
            mprl_tile_in_bounds_count INTEGER NOT NULL,
            mprl_tile_near_bounds_count INTEGER NOT NULL,
            mprl_linked_in_bounds_count INTEGER NOT NULL,
            mprl_linked_near_bounds_count INTEGER NOT NULL,
            mprl_linked_floor_min INTEGER,
            mprl_linked_floor_max INTEGER,
            FOREIGN KEY(tile_id) REFERENCES tiles(id)
        );

        CREATE INDEX IF NOT EXISTS idx_tiles_xy ON tiles(tile_x, tile_y);
        CREATE INDEX IF NOT EXISTS idx_family_summaries_tile ON family_summaries(tile_id, family);
        CREATE INDEX IF NOT EXISTS idx_objects_tile_family ON objects(tile_id, family);
        CREATE INDEX IF NOT EXISTS idx_objects_ck24 ON objects(ck24, family);
        CREATE INDEX IF NOT EXISTS idx_objects_mprl ON objects(mprl_linked_ref_count, mprl_linked_in_bounds_count);
        """
    )


def clear_existing_data(connection: sqlite3.Connection) -> None:
    connection.execute("DELETE FROM objects")
    connection.execute("DELETE FROM family_summaries")
    connection.execute("DELETE FROM tiles")
    connection.commit()


def ingest_reports(connection: sqlite3.Connection, reports: Iterable[dict[str, Any]]) -> None:
    tile_insert = (
        "INSERT INTO tiles (source_path, tile_x, tile_y, version, ck24_group_count, total_hypothesis_count, diagnostic_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)"
    )
    family_insert = (
        "INSERT INTO family_summaries (tile_id, family, object_count, max_surface_count, max_index_count, total_linked_mprl_count, total_linked_in_bounds_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)"
    )
    object_insert = (
        "INSERT INTO objects ("
        "tile_id, family, family_object_index, ck24, ck24_type, ck24_object_id, surface_count, total_index_count, mdos_count, group_key_count, "
        "mslk_group_object_id_count, mslk_ref_index_count, bounds_min_x, bounds_min_y, bounds_min_z, bounds_max_x, bounds_max_y, bounds_max_z, "
        "mprl_tile_ref_count, mprl_linked_ref_count, mprl_linked_normal_count, mprl_linked_terminator_count, mprl_tile_in_bounds_count, "
        "mprl_tile_near_bounds_count, mprl_linked_in_bounds_count, mprl_linked_near_bounds_count, mprl_linked_floor_min, mprl_linked_floor_max) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )

    processed = 0
    for report in reports:
        diagnostics = report.get("Diagnostics", [])
        cursor = connection.execute(
            tile_insert,
            (
                report.get("SourcePath"),
                report.get("TileX"),
                report.get("TileY"),
                report.get("Version", 0),
                report.get("Ck24GroupCount", 0),
                report.get("TotalHypothesisCount", 0),
                len(diagnostics) if isinstance(diagnostics, list) else 0,
            ),
        )
        tile_id = cursor.lastrowid

        for family in report.get("Families", []):
            connection.execute(
                family_insert,
                (
                    tile_id,
                    family.get("Family"),
                    family.get("ObjectCount", 0),
                    family.get("MaxSurfaceCount", 0),
                    family.get("MaxIndexCount", 0),
                    family.get("TotalLinkedMprlCount", 0),
                    family.get("TotalLinkedInBoundsCount", 0),
                ),
            )

        for obj in report.get("Objects", []):
            bounds = obj.get("Bounds") or {}
            min_point = bounds.get("Min") or {}
            max_point = bounds.get("Max") or {}
            footprint = obj.get("MprlFootprint") or {}
            connection.execute(
                object_insert,
                (
                    tile_id,
                    obj.get("Family"),
                    obj.get("FamilyObjectIndex", 0),
                    obj.get("Ck24", 0),
                    obj.get("Ck24Type", 0),
                    obj.get("Ck24ObjectId", 0),
                    obj.get("SurfaceCount", 0),
                    obj.get("TotalIndexCount", 0),
                    obj.get("MdosCount", 0),
                    obj.get("GroupKeyCount", 0),
                    obj.get("MslkGroupObjectIdCount", 0),
                    obj.get("MslkRefIndexCount", 0),
                    min_point.get("X"),
                    min_point.get("Y"),
                    min_point.get("Z"),
                    max_point.get("X"),
                    max_point.get("Y"),
                    max_point.get("Z"),
                    footprint.get("TileRefCount", 0),
                    footprint.get("LinkedRefCount", 0),
                    footprint.get("LinkedNormalCount", 0),
                    footprint.get("LinkedTerminatorCount", 0),
                    footprint.get("TileInBoundsCount", 0),
                    footprint.get("TileNearBoundsCount", 0),
                    footprint.get("LinkedInBoundsCount", 0),
                    footprint.get("LinkedNearBoundsCount", 0),
                    footprint.get("LinkedFloorMin"),
                    footprint.get("LinkedFloorMax"),
                ),
            )

        processed += 1
        if processed % 25 == 0:
            connection.commit()
            print(f"ingested {processed} tiles", file=sys.stderr)

    connection.commit()


def iter_reports(handle: TextIO) -> Iterable[dict[str, Any]]:
    for line in handle:
        stripped = line.strip()
        if not stripped:
            continue
        yield json.loads(stripped)


def iter_reports_from_cli(pm4_dir: Path, cli_project: Path) -> Iterable[dict[str, Any]]:
    command = [
        "dotnet",
        "run",
        "--project",
        str(cli_project),
        "--",
        "scan-hypotheses-ndjson",
        "--input",
        str(pm4_dir),
    ]

    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
    ) as process:
        assert process.stdout is not None
        assert process.stderr is not None

        try:
            for line in process.stdout:
                stripped = line.strip()
                if not stripped:
                    continue
                yield json.loads(stripped)
        finally:
            stderr_text = process.stderr.read()
            return_code = process.wait()
            if stderr_text:
                sys.stderr.write(stderr_text)
            if return_code != 0:
                raise RuntimeError(f"CLI scan-hypotheses-ndjson failed with exit code {return_code}")


def finalize_database(connection: sqlite3.Connection) -> None:
    connection.execute("ANALYZE")
    connection.commit()


if __name__ == "__main__":
    raise SystemExit(main())