"""Generate the v50 manifest template FROM the frozen signal catalog (Spec 112 T008).

The template JSON that gates every v50 store build was previously hand-authored and drifted from
the frozen catalog — it declared four signals the catalog had explicitly dropped (zero-filled
corpus-wide) and `mccv_rgb`, which does not exist in 0.5.3-era client data at all. This module
makes that drift structurally impossible: the generator is the ONLY writer of the template, its
input is the catalog table (via ``signal_catalog.parse_catalog_table`` — the single reader), and a
signal can reach the template only by being in the catalog, non-blacklisted, and era-available for
the target build. Era-unavailable signals become explicit ``UnavailableSignal`` records with an
``era_unavailable:`` reason instead of declared-then-zero-filled arrays (research.md Decision 3/4).

The output is produced by round-tripping a real ``DatasetStoreManifest`` through ``to_dict()``, so
it is guaranteed loadable by ``v50_build_dataset.py``'s ``_load_manifest`` — not a hand-shaped dict
that merely resembles one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harvester.v50.contracts import (
    REASON_ERA_UNAVAILABLE,
    DatasetSignal,
    DatasetStoreManifest,
    FinalizationState,
    MigrationPolicy,
    UnavailableSignal,
    validate_release,
)
from harvester.v50.signal_catalog import CatalogSignal, parse_catalog_table

_PLACEHOLDER_HASH = "sha256:" + "0" * 64

# The only two signals produced by minimap synthesis rather than the harvest stream.
_SYNTHESIS_SIGNALS = frozenset({"minimap_rgb", "minimap_rgb_1024"})


def build_manifest_template(
    catalog: list[CatalogSignal],
    *,
    build_id: str,
    release: str,
) -> DatasetStoreManifest:
    release = validate_release(release)
    declared: list[DatasetSignal] = []
    unavailable: list[UnavailableSignal] = []

    for entry in catalog:
        if entry.policy == "blacklisted":
            continue  # never declared in a store, matching the existing template's holes_16 handling
        if not entry.available_for_build(build_id):
            unavailable.append(
                UnavailableSignal(
                    name=entry.name,
                    reason=f"{REASON_ERA_UNAVAILABLE} {entry.era_reason} (build {build_id})",
                )
            )
            continue
        declared.append(
            DatasetSignal(
                name=entry.name,
                dtype=entry.dtype,
                row_shape=entry.shape,
                required=entry.required,
                authoritative_source="synthetic-minimap" if entry.name in _SYNTHESIS_SIGNALS else "harvest-stream",
                content_identity=_PLACEHOLDER_HASH,
                coverage_count=0,
                migration_policy=MigrationPolicy(entry.policy),
            )
        )

    return DatasetStoreManifest(
        release=release,
        store_id=_PLACEHOLDER_HASH,
        build_id=build_id,
        producer_identity=_PLACEHOLDER_HASH,
        client_build_evidence_id=_PLACEHOLDER_HASH,
        index_identity=_PLACEHOLDER_HASH,
        row_count=0,
        signals=tuple(declared),
        row_lineage_identity=_PLACEHOLDER_HASH,
        finalization_state=FinalizationState.INCOMPLETE,
        unavailable_signals=tuple(unavailable),
    )


def build_signals_config(catalog: list[CatalogSignal]) -> dict:
    """The companion signals-config JSON (``_load_signal_specs`` shape): every catalog signal with
    its blacklist status carried through, so the audit path and the template share one source."""
    return {
        "signals": [
            {
                "name": entry.name,
                "blacklisted": entry.policy == "blacklisted",
                "blacklist_reason": entry.notes if entry.policy == "blacklisted" else "",
                "has_flag_name": None,
            }
            for entry in catalog
        ]
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Derive the v50 manifest template from the frozen signal catalog")
    ap.add_argument("--catalog-doc", required=True, type=Path,
                    help="architecture doc containing the frozen signal catalog table")
    ap.add_argument("--build-id", required=True, help="e.g. 0_5_3_3368")
    ap.add_argument("--release", default="v50.1", type=validate_release)
    ap.add_argument("--output", required=True, type=Path, help="manifest template JSON path")
    ap.add_argument("--signals-output", type=Path, default=None,
                    help="also regenerate the signals-config JSON here")
    args = ap.parse_args()

    catalog = parse_catalog_table(args.catalog_doc)
    manifest = build_manifest_template(catalog, build_id=args.build_id, release=args.release)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    print(
        f"[{args.release}] template: {len(manifest.signals)} declared, "
        f"{len(manifest.unavailable_signals)} era-unavailable -> {args.output}"
    )
    for entry in manifest.unavailable_signals:
        print(f"  unavailable: {entry.name}: {entry.reason}")

    if args.signals_output is not None:
        args.signals_output.write_text(json.dumps(build_signals_config(catalog), indent=2), encoding="utf-8")
        print(f"signals config: {len(catalog)} entries -> {args.signals_output}")
    return 0
