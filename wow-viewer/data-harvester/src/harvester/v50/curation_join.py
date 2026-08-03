"""Join a Spec 122 curation manifest onto a curriculum being built.

WHY THIS EXISTS
---------------
The curation layer classifies every tile by terrain regime (steep / rolling / flat), object
coverage, RGB contrast, and height relief. Curriculum builders that glob tiles off disk and join
against the raw store index bypass all of it, and the cost is measurable: the first residual-extractor
run scored mean per-tile correlation 0.317, but split by terrain it was 0.518 on textured tiles and
0.117 on flat ones. A third of the Azeroth corpus is classified flat, and flat terrain carries no
shading signal to extract -- those tiles contributed loss and gradient without contributing anything
learnable, and they dragged the reported aggregate down while hiding the real result.

THE RULE THIS FOLLOWS
---------------------
Curation **partitions, it never filters**. Every tile stays in the curriculum store and stays
queryable. The regime lands in ``index.parquet`` as a column, and *selection* happens at training
time where it is explicit, recorded, and reversible -- not by silently dropping rows at build time.

Metrics are then reported per regime, so a strong regime can never mask a dead one.
"""

from __future__ import annotations

from pathlib import Path

CURATION_FIELDS = (
    "keep",
    "reason",
    "height_regime",
    "bucket",
    "object_coverage",
    "rgb_std",
    "height_range",
    "height_std",
    "normal_relief",
)

UNCURATED = "uncurated"


def load_curation_manifest(curation_dir: Path) -> dict[tuple[str, int, int], dict]:
    """Map (map, tile_x, tile_y) -> curation record.

    ``curation_dir`` is a Spec 122 curation output directory containing ``curation_manifest.parquet``.
    """
    import pyarrow.parquet as pq

    manifest = Path(curation_dir) / "curation_manifest.parquet"
    if not manifest.exists():
        raise SystemExit(
            f"curation manifest not found: {manifest}\n"
            "Pass the curation output directory (the one holding curation_manifest.parquet), "
            "or omit --curation to build without regime tagging."
        )
    table = pq.read_table(manifest)
    missing = [f for f in ("map", "tile_x", "tile_y") if f not in table.column_names]
    if missing:
        raise SystemExit(f"curation manifest is missing key columns {missing}: {manifest}")

    records: dict[tuple[str, int, int], dict] = {}
    for row in table.to_pylist():
        key = (str(row.get("map", "")), int(row.get("tile_x", -1)), int(row.get("tile_y", -1)))
        records.setdefault(key, {field: row.get(field) for field in CURATION_FIELDS if field in table.column_names})
    return records


def curation_columns(
    keys: list[tuple[str, int, int]],
    records: dict[tuple[str, int, int], dict],
) -> dict[str, list]:
    """Build index.parquet columns for the given rows, in row order.

    Tiles with no curation record are tagged ``uncurated`` rather than dropped or guessed, so a
    partial manifest is visible in the data instead of silently degrading the split.
    """
    columns: dict[str, list] = {field: [] for field in CURATION_FIELDS}
    for key in keys:
        record = records.get(key) or {}
        for field in CURATION_FIELDS:
            value = record.get(field)
            if field in ("height_regime", "bucket", "reason") and value is None:
                value = UNCURATED
            elif field == "keep" and value is None:
                value = True
            columns[field].append(value)
    return columns


def regime_counts(regimes: list) -> dict[str, int]:
    counts: dict[str, int] = {}
    for regime in regimes:
        name = str(regime) if regime is not None else UNCURATED
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))
