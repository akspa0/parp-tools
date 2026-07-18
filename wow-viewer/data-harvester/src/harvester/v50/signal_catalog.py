"""Parse the frozen v50 signal catalog table (Spec 112 T002).

The markdown table under "## Frozen Signal Catalog" in
``docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md`` is the single human-authored
source of truth for what v50 signals exist (constitution: Spec Docs Are Source of Truth). This
module is its ONLY programmatic reader: both the manifest-template generator and its tests import
``parse_catalog_table``, so the template can never silently diverge from the catalog — the exact
drift that shipped four dead signals and an era-impossible ``mccv_rgb`` into the v50.1 corpus
(Spec 112 research.md Decision 3).

Era availability is resolved from an explicit allow-list of known era-restricted signals
(``_ERA_RESTRICTED``), never inferred from free Notes prose — a signal absent from that list is
available for every build era.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

CATALOG_HEADING = "## Frozen Signal Catalog"
_HEADER_COLUMNS = ("Signal", "dtype", "Shape", "V50 Policy", "Required", "Notes")

# Known era-restricted signals: name -> (first build-era prefix that has it, human reason).
# MCCV vertex colors were introduced around WotLK and are commonly populated only from ~4.x;
# no 0.5.x client carries them (measured 0% across the entire 0_5_3_3368 corpus, 2026-07-18).
_ERA_RESTRICTED: dict[str, tuple[str, str]] = {
    "mccv_rgb": ("3.0", "MCCV vertex colors introduced WotLK+ (commonly 4.x); absent from pre-3.0 clients"),
}


@dataclass(frozen=True)
class CatalogSignal:
    name: str
    dtype: str
    shape: tuple[int, ...]
    policy: str  # "copy-if-verified" | "fresh-only" | "blacklisted"
    required: bool
    notes: str
    era_min_build: str | None = None  # None => available for all eras
    era_reason: str = ""

    def available_for_build(self, build_id: str) -> bool:
        """True when this signal can exist for ``build_id`` (e.g. ``0_5_3_3368`` / ``3_3_5_12340``)."""
        if self.era_min_build is None:
            return True
        return _build_sort_key(build_id) >= _build_sort_key(self.era_min_build)


def _build_sort_key(build_id: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", build_id)
    if not parts:
        raise ValueError(f"cannot parse build identifier {build_id!r}")
    return tuple(int(p) for p in parts[:3])


class CatalogParseError(ValueError):
    """Raised when the frozen catalog table cannot be parsed as declared."""


def parse_catalog_table(doc_path: Path) -> list[CatalogSignal]:
    """Read the frozen catalog table from the architecture doc.

    Fails closed on a missing heading, a changed column order, or an unparseable row — a silently
    partial parse is exactly the kind of drift this module exists to prevent."""
    text = doc_path.read_text(encoding="utf-8")
    heading_index = text.find(CATALOG_HEADING)
    if heading_index < 0:
        raise CatalogParseError(f"{doc_path} has no '{CATALOG_HEADING}' heading")

    lines = text[heading_index:].splitlines()
    rows: list[list[str]] = []
    header_seen = False
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            if header_seen and rows:
                break  # table ended
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not header_seen:
            if [c.strip("*") for c in cells] != list(_HEADER_COLUMNS):
                raise CatalogParseError(
                    f"catalog table header changed: expected {_HEADER_COLUMNS}, got {tuple(cells)}"
                )
            header_seen = True
            continue
        if set("".join(cells)) <= {"-", " "}:
            continue  # separator row
        if len(cells) != len(_HEADER_COLUMNS):
            raise CatalogParseError(f"catalog row has {len(cells)} cells, expected {len(_HEADER_COLUMNS)}: {stripped!r}")
        rows.append(cells)

    if not rows:
        raise CatalogParseError(f"{doc_path} catalog table has no signal rows")

    signals: list[CatalogSignal] = []
    for cells in rows:
        name = cells[0].strip("`")
        shape_match = re.fullmatch(r"\((\d+(?:\s*,\s*\d+)*)\)", cells[2])
        if not shape_match:
            raise CatalogParseError(f"unparseable Shape cell for {name!r}: {cells[2]!r}")
        shape = tuple(int(part) for part in shape_match.group(1).split(","))
        policy = cells[3].strip("*")
        if policy not in ("copy-if-verified", "fresh-only", "blacklisted"):
            raise CatalogParseError(f"unknown policy {policy!r} for {name!r}")
        required = cells[4].strip("*").lower() == "yes"
        era = _ERA_RESTRICTED.get(name)
        signals.append(
            CatalogSignal(
                name=name,
                dtype=cells[1],
                shape=shape,
                policy=policy,
                required=required,
                notes=cells[5],
                era_min_build=era[0] if era else None,
                era_reason=era[1] if era else "",
            )
        )
    return signals
