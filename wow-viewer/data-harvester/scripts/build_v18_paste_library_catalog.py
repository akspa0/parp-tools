from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build V18 paste-library catalog with deterministic naming metadata.")
    parser.add_argument("--deduped-candidates", type=Path, required=True, help="Path to candidates_deduped.jsonl/.json or directory containing it.")
    parser.add_argument("--composition-graph", type=Path, default=None, help="Optional composition graph directory/file for family/AreaID enrichment.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _resolve_deduped_path(path: Path) -> Path:
    if path.is_file():
        return path
    cand = path / "candidates_deduped.jsonl"
    if cand.exists():
        return cand
    cand_json = path / "candidates_deduped.json"
    if cand_json.exists():
        return cand_json
    raise FileNotFoundError(f"No deduped candidates file under: {path}")


def _resolve_comp_path(path: Path) -> tuple[Path | None, Path | None]:
    if path.is_file():
        if "famil" in path.stem.lower():
            return path, None
        return None, path
    fam_jsonl = path / "composition_families.jsonl"
    fam_json = path / "composition_families.json"
    cand_jsonl = path / "composition_candidates.jsonl"
    cand_json = path / "composition_candidates.json"
    families = fam_jsonl if fam_jsonl.exists() else (fam_json if fam_json.exists() else None)
    candidates = cand_jsonl if cand_jsonl.exists() else (cand_json if cand_json.exists() else None)
    return families, candidates


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return list(payload.get("rows", []))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _selection_hash(rows: list[dict[str, Any]], keys: list[str]) -> str:
    h = hashlib.sha256()
    for row in rows:
        h.update("|".join(str(row.get(k, "")) for k in keys).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _infer_role_shape(row: dict[str, Any]) -> tuple[list[str], list[str], float]:
    w, h = [int(v) for v in row.get("canvas_bbox_wh", [0, 0])]
    transition = float(row.get("transition_mean", 0.0))
    hard = float(row.get("hard_mean", 0.0))
    multi = bool(row.get("multi_tile", False))
    aspect = float(w / max(1, h))

    role_tags: list[str] = []
    shape_tags: list[str] = []
    confidence = 0.35

    if multi:
        role_tags.append("connector")
        confidence += 0.10
    if aspect >= 1.55:
        shape_tags.append("horizontal")
        role_tags.extend(["left", "right"])
        confidence += 0.20
    elif aspect <= 0.65:
        shape_tags.append("vertical")
        role_tags.extend(["start", "end"])
        confidence += 0.20
    else:
        shape_tags.append("compact")
        role_tags.append("fill")
        confidence += 0.08

    if transition >= 1.35:
        role_tags.append("transition")
        confidence += 0.12
    if hard >= 1.40 and 0.75 <= aspect <= 1.25:
        role_tags.append("corner")
        shape_tags.append("angular")
        confidence += 0.08

    # deterministic unique/stable ordering
    role_tags = sorted(set(role_tags))
    shape_tags = sorted(set(shape_tags))
    confidence = float(max(0.05, min(0.99, confidence)))
    return role_tags, shape_tags, confidence


def _family_name(
    *,
    composition_family_id: str,
    dominant_area_id: str,
    role_tags: list[str],
    shape_tags: list[str],
    alpha_sig: str,
    confidence: float,
) -> tuple[str, list[str], float]:
    role = role_tags[0] if role_tags else "fill"
    shape = shape_tags[0] if shape_tags else "compact"
    area = dominant_area_id.replace(" ", "_").replace("/", "_")
    area = "".join(ch for ch in area if ch.isalnum() or ch in ("_", "-")) or "unknown"
    alpha_short = alpha_sig[-6:] if alpha_sig else "nosig"
    family_short = composition_family_id.split("_")[-1][:6] if composition_family_id else "cf0000"
    canonical_name = f"pf_{area}_{role}_{shape}_{family_short}_{alpha_short}".lower()
    aliases = [
        f"{area}_{role}",
        f"{shape}_{role}",
        f"{role}_family",
    ]
    aliases = [a.lower() for a in aliases]
    # Slight confidence boost if name has both area and role resolved.
    if area != "unknown" and role != "fill":
        confidence = float(min(0.99, confidence + 0.05))
    return canonical_name, aliases, confidence


def _load_composition_maps(path: Path | None) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    if path is None:
        return {}, {}
    families_path, candidates_path = _resolve_comp_path(path)
    cluster_to_family: dict[str, str] = {}
    family_meta: dict[str, dict[str, Any]] = {}

    if families_path is not None and families_path.exists():
        for row in _load_rows(families_path):
            family_id = str(row.get("composition_family_id", "")).strip()
            if not family_id:
                continue
            family_meta[family_id] = {
                "dominant_area_ids": list(row.get("dominant_area_ids", [])),
                "area_id_distribution": dict(row.get("area_id_distribution", {})),
                "cluster_count": int(row.get("cluster_count", 0)),
            }
            for cluster_id in row.get("cluster_ids", []):
                cid = str(cluster_id).strip()
                if cid:
                    cluster_to_family[cid] = family_id

    if candidates_path is not None and candidates_path.exists():
        for row in _load_rows(candidates_path):
            cid = str(row.get("cluster_id", "")).strip()
            family_id = str(row.get("composition_family_id", "")).strip()
            if cid and family_id and cid not in cluster_to_family:
                cluster_to_family[cid] = family_id
    return cluster_to_family, family_meta


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    deduped_path = _resolve_deduped_path(args.deduped_candidates)
    rows = _load_rows(deduped_path)
    if not rows:
        raise RuntimeError(f"No rows loaded from {deduped_path}")

    cluster_to_family, family_meta = _load_composition_maps(args.composition_graph)

    by_cluster: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        cluster_id = str(row.get("cluster_id", "")).strip()
        if not cluster_id:
            continue
        by_cluster.setdefault(cluster_id, []).append(row)

    catalog_rows: list[dict[str, Any]] = []
    for cluster_id, members in sorted(by_cluster.items(), key=lambda item: item[0]):
        members.sort(key=lambda r: int(r.get("variant_rank", 999999)))
        canonical = members[0]
        family_id = str(cluster_to_family.get(cluster_id, "compfam_00000_unassigned"))
        fam = family_meta.get(family_id, {})
        dominant_area_ids = list(fam.get("dominant_area_ids", []))
        dominant_area = str(dominant_area_ids[0] if dominant_area_ids else "unknown")

        role_tags, shape_tags, name_conf = _infer_role_shape(canonical)
        canonical_name, aliases, name_conf = _family_name(
            composition_family_id=family_id,
            dominant_area_id=dominant_area,
            role_tags=role_tags,
            shape_tags=shape_tags,
            alpha_sig=str(canonical.get("alpha_layer_signature", "")),
            confidence=name_conf,
        )
        review_state = "auto"
        review_required = bool(name_conf < 0.55)

        builds = sorted({str(m.get("build", "")) for m in members})
        maps = sorted({str(m.get("map", "")) for m in members})
        alpha_profiles = [m.get("alpha_layer_signature", "") for m in members]
        alpha_profile_counts: dict[str, int] = {}
        for sig in alpha_profiles:
            s = str(sig).strip()
            if not s:
                continue
            alpha_profile_counts[s] = int(alpha_profile_counts.get(s, 0) + 1)
        normal_relief_profile = {
            "hard_mean_min": float(min(float(m.get("hard_mean", 0.0)) for m in members)),
            "hard_mean_max": float(max(float(m.get("hard_mean", 0.0)) for m in members)),
            "transition_mean_min": float(min(float(m.get("transition_mean", 0.0)) for m in members)),
            "transition_mean_max": float(max(float(m.get("transition_mean", 0.0)) for m in members)),
        }

        stable_name_payload = "|".join(
            [
                cluster_id,
                family_id,
                canonical_name,
                ",".join(role_tags),
                ",".join(shape_tags),
                str(canonical.get("alpha_layer_signature", "")),
            ]
        )
        paste_id = f"paste_{hashlib.sha256(stable_name_payload.encode('utf-8')).hexdigest()[:14]}"

        catalog_rows.append(
            {
                "paste_id": paste_id,
                "cluster_id": cluster_id,
                "composition_family_id": family_id,
                "canonical_id": int(canonical.get("canonical_id", canonical.get("candidate_id", -1))),
                "canonical_candidate_id": int(canonical.get("candidate_id", -1)),
                "canonical_name": canonical_name,
                "aliases": aliases,
                "name_confidence": float(name_conf),
                "review_state": review_state,
                "review_required": review_required,
                "role_tags": role_tags,
                "shape_tags": shape_tags,
                "build_span": {"first": builds[0] if builds else "", "last": builds[-1] if builds else "", "count": len(builds)},
                "builds": builds,
                "maps": maps,
                "variant_count": int(len(members)),
                "variant_candidate_ids": [int(m.get("candidate_id", -1)) for m in members],
                "alpha_layer_profiles": dict(sorted(alpha_profile_counts.items(), key=lambda item: (item[1], item[0]), reverse=True)),
                "normal_relief_profile": normal_relief_profile,
                "area_id_distribution": fam.get("area_id_distribution", {"unknown": 1.0}),
                "dominant_area_ids": dominant_area_ids if dominant_area_ids else ["unknown"],
                "sampling_metadata": {
                    "cluster_balance_weight": float(1.0 / np.sqrt(max(1, int(len(members))))),
                    "family_cluster_count": int(fam.get("cluster_count", 1) or 1),
                    "family_balance_weight": float(1.0 / np.sqrt(max(1, int(fam.get("cluster_count", 1) or 1)))),
                },
            }
        )

    catalog_rows.sort(key=lambda r: (r["canonical_name"], r["paste_id"]))
    (out_dir / "paste_library_catalog.json").write_text(json.dumps(catalog_rows, indent=2), encoding="utf-8")
    _write_jsonl(out_dir / "paste_library_catalog.jsonl", catalog_rows)

    summary = {
        "families": int(len(catalog_rows)),
        "auto_review_required": int(sum(1 for row in catalog_rows if bool(row.get("review_required", False)))),
        "stable_name_hash": _selection_hash(catalog_rows, ["paste_id", "canonical_name", "cluster_id", "composition_family_id"]),
        "composition_graph_path": str(args.composition_graph) if args.composition_graph else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "config.snapshot.json").write_text(json.dumps(vars(args), indent=2, default=str, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
