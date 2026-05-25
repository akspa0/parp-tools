from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build V18 composition graph from deduped canvas candidates.")
    parser.add_argument("--deduped-candidates", type=Path, required=True, help="Path to candidates_deduped.jsonl/.json or a directory containing it.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adjacency-margin-px", type=int, default=64, help="Max bbox gap (pixels) for adjacency edges.")
    parser.add_argument("--cooccur-edge-min", type=int, default=2, help="Min co-occur count to keep non-adjacent co-occur edges.")
    parser.add_argument("--area-id-map", type=Path, default=None, help="Optional tile-level AreaID mapping (.jsonl/.json/.parquet-like json rows).")
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
        key = "|".join(str(row.get(k, "")) for k in keys)
        h.update(key.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _parse_area_id_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return list(payload.get("rows", []))


def _build_area_id_index(path: Path | None) -> dict[tuple[str, str, int, int], list[tuple[str, float]]]:
    if path is None:
        return {}
    rows = _parse_area_id_rows(path)
    out: dict[tuple[str, str, int, int], list[tuple[str, float]]] = {}
    for row in rows:
        build = str(row.get("build", ""))
        map_name = str(row.get("map", ""))
        tile_x = int(row.get("tile_x", -1))
        tile_y = int(row.get("tile_y", -1))
        if not build or not map_name or tile_x < 0 or tile_y < 0:
            continue
        area_id = str(row.get("area_id", "unknown")).strip() or "unknown"
        weight = float(row.get("weight", row.get("fraction", 1.0)) or 1.0)
        key = (build, map_name, tile_x, tile_y)
        out.setdefault(key, []).append((area_id, weight))
    return out


def _bbox_distance_and_intersection(a: list[int], b: list[int]) -> tuple[float, int]:
    ax0, ay0, ax1, ay1 = [int(v) for v in a]
    bx0, by0, bx1, by1 = [int(v) for v in b]
    dx = max(0, max(ax0 - bx1 - 1, bx0 - ax1 - 1))
    dy = max(0, max(ay0 - by1 - 1, by0 - ay1 - 1))
    distance = float(math.sqrt((dx * dx) + (dy * dy)))
    ix = max(0, min(ax1, bx1) - max(ax0, bx0) + 1)
    iy = max(0, min(ay1, by1) - max(ay0, by0) + 1)
    inter = int(ix * iy)
    return distance, inter


def _candidate_area_coverage(row: dict[str, Any], area_idx: dict[tuple[str, str, int, int], list[tuple[str, float]]]) -> tuple[dict[str, float], list[str]]:
    build = str(row.get("build", ""))
    map_name = str(row.get("map", ""))
    coverage = list(row.get("tile_coverage", []))
    area_weights: dict[str, float] = {}
    if not coverage:
        return {"unknown": 1.0}, ["unknown"]

    tile_weight = 1.0 / float(len(coverage))
    for cov in coverage:
        tile_x = int(cov.get("tile_x", -1))
        tile_y = int(cov.get("tile_y", -1))
        key = (build, map_name, tile_x, tile_y)
        area_entries = area_idx.get(key)
        if not area_entries:
            area_weights["unknown"] = float(area_weights.get("unknown", 0.0) + tile_weight)
            continue
        total = float(sum(max(0.0, float(w)) for _a, w in area_entries))
        if total <= 0.0:
            area_weights["unknown"] = float(area_weights.get("unknown", 0.0) + tile_weight)
            continue
        for area_id, w in area_entries:
            frac = float(max(0.0, w) / total)
            area_weights[area_id] = float(area_weights.get(area_id, 0.0) + (tile_weight * frac))

    if not area_weights:
        area_weights["unknown"] = 1.0
    dominant = [k for k, _v in sorted(area_weights.items(), key=lambda item: (item[1], item[0]), reverse=True)]
    return area_weights, dominant[:3]


def _candidate_key(row: dict[str, Any]) -> str:
    return f"{row.get('build','')}|{row.get('map','')}|{row.get('candidate_id','')}"


def _cluster_area_distribution(rows: list[dict[str, Any]]) -> dict[str, float]:
    acc: dict[str, float] = {}
    for row in rows:
        cov = row.get("area_id_coverage", {})
        if not isinstance(cov, dict):
            continue
        for area_id, value in cov.items():
            acc[str(area_id)] = float(acc.get(str(area_id), 0.0) + float(value))
    denom = float(len(rows)) if rows else 1.0
    for k in list(acc.keys()):
        acc[k] = float(acc[k] / denom)
    return dict(sorted(acc.items(), key=lambda item: (item[1], item[0]), reverse=True))


def _build_family_ids(cluster_ids: list[str], edges: list[dict[str, Any]]) -> dict[str, str]:
    adj: dict[str, set[str]] = {cluster_id: set() for cluster_id in cluster_ids}
    for edge in edges:
        a = str(edge.get("cluster_a", ""))
        b = str(edge.get("cluster_b", ""))
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)

    seen: set[str] = set()
    family_by_cluster: dict[str, str] = {}
    components: list[list[str]] = []

    for start in sorted(cluster_ids):
        if start in seen:
            continue
        q: deque[str] = deque([start])
        seen.add(start)
        comp: list[str] = []
        while q:
            node = q.popleft()
            comp.append(node)
            for nxt in sorted(adj.get(node, set())):
                if nxt in seen:
                    continue
                seen.add(nxt)
                q.append(nxt)
        components.append(sorted(comp))

    components.sort(key=lambda c: (len(c), c[0]), reverse=True)
    for idx, comp in enumerate(components, start=1):
        payload = "|".join(comp)
        fam_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        family_id = f"compfam_{idx:05d}_{fam_hash}"
        for cluster_id in comp:
            family_by_cluster[cluster_id] = family_id
    return family_by_cluster


def main() -> None:
    args = _parse_args()
    deduped_path = _resolve_deduped_path(args.deduped_candidates)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(deduped_path)
    if not rows:
        raise RuntimeError(f"No rows loaded from {deduped_path}")

    area_idx = _build_area_id_index(args.area_id_map)

    # Candidate-level area coverage (AreaID soft labels with unknown fallback).
    candidate_rows: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        area_cov, dominant = _candidate_area_coverage(out, area_idx)
        out["area_id_coverage"] = area_cov
        out["dominant_area_ids"] = dominant
        candidate_rows.append(out)
    candidate_rows.sort(key=lambda r: (str(r.get("build", "")), str(r.get("map", "")), int(r.get("candidate_id", -1))))

    # Build edges over candidate instances on the same canvas.
    by_canvas: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        canvas_id = str(row.get("canvas_id", f"{row.get('build','')}:{row.get('map','')}"))
        by_canvas.setdefault(canvas_id, []).append(row)
    for rows_ in by_canvas.values():
        rows_.sort(key=lambda r: int(r.get("candidate_id", -1)))

    edge_acc: dict[tuple[str, str], dict[str, Any]] = {}
    for canvas_id, canvas_rows in sorted(by_canvas.items()):
        for i in range(len(canvas_rows)):
            a = canvas_rows[i]
            for j in range(i + 1, len(canvas_rows)):
                b = canvas_rows[j]
                cluster_a = str(a.get("cluster_id", ""))
                cluster_b = str(b.get("cluster_id", ""))
                if not cluster_a or not cluster_b or cluster_a == cluster_b:
                    continue
                first, second = sorted([cluster_a, cluster_b])
                distance, inter = _bbox_distance_and_intersection(list(a.get("canvas_bbox", [0, 0, 0, 0])), list(b.get("canvas_bbox", [0, 0, 0, 0])))
                is_adjacent = bool(distance <= float(args.adjacency_margin_px))
                key = (first, second)
                edge = edge_acc.get(key)
                if edge is None:
                    edge = {
                        "cluster_a": first,
                        "cluster_b": second,
                        "cooccur_count": 0,
                        "adjacent_count": 0,
                        "overlap_count": 0,
                        "min_distance_px": float("inf"),
                        "canvases": set(),
                        "area_pair_counts": {},
                    }
                    edge_acc[key] = edge
                edge["cooccur_count"] = int(edge["cooccur_count"] + 1)
                if is_adjacent:
                    edge["adjacent_count"] = int(edge["adjacent_count"] + 1)
                if inter > 0:
                    edge["overlap_count"] = int(edge["overlap_count"] + 1)
                edge["min_distance_px"] = float(min(float(edge["min_distance_px"]), distance))
                edge["canvases"].add(canvas_id)

                dom_a = str((a.get("dominant_area_ids") or ["unknown"])[0])
                dom_b = str((b.get("dominant_area_ids") or ["unknown"])[0])
                ap = "|".join(sorted([dom_a, dom_b]))
                area_pair_counts = edge["area_pair_counts"]
                area_pair_counts[ap] = int(area_pair_counts.get(ap, 0) + 1)

    edges: list[dict[str, Any]] = []
    for (_a, _b), edge in sorted(edge_acc.items(), key=lambda item: (item[0][0], item[0][1])):
        keep = bool(edge["adjacent_count"] > 0 or edge["cooccur_count"] >= int(args.cooccur_edge_min))
        if not keep:
            continue
        edge_out = {
            "cluster_a": str(edge["cluster_a"]),
            "cluster_b": str(edge["cluster_b"]),
            "cooccur_count": int(edge["cooccur_count"]),
            "adjacent_count": int(edge["adjacent_count"]),
            "overlap_count": int(edge["overlap_count"]),
            "min_distance_px": float(edge["min_distance_px"]) if np.isfinite(float(edge["min_distance_px"])) else None,
            "canvas_count": int(len(edge["canvases"])),
            "area_pair_counts": dict(sorted(edge["area_pair_counts"].items(), key=lambda item: (item[1], item[0]), reverse=True)),
        }
        edges.append(edge_out)

    # Cluster nodes.
    by_cluster: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        cluster_id = str(row.get("cluster_id", ""))
        if not cluster_id:
            continue
        by_cluster.setdefault(cluster_id, []).append(row)

    cluster_nodes: list[dict[str, Any]] = []
    for cluster_id, members in sorted(by_cluster.items(), key=lambda item: item[0]):
        members_sorted = sorted(members, key=lambda r: int(r.get("variant_rank", 999999)))
        canonical = members_sorted[0]
        area_dist = _cluster_area_distribution(members_sorted)
        builds = sorted({str(m.get("build", "")) for m in members_sorted})
        maps = sorted({str(m.get("map", "")) for m in members_sorted})
        cluster_nodes.append(
            {
                "cluster_id": cluster_id,
                "canonical_id": int(canonical.get("candidate_id", -1)),
                "member_count": int(len(members_sorted)),
                "builds": builds,
                "maps": maps,
                "dominant_area_ids": [k for k, _v in list(area_dist.items())[:3]] if area_dist else ["unknown"],
                "area_id_distribution": area_dist if area_dist else {"unknown": 1.0},
                "cluster_key": str(canonical.get("cluster_key", "")),
                "alpha_layer_signature": str(canonical.get("alpha_layer_signature", "")),
            }
        )

    # Composition families from cluster graph connected components.
    cluster_ids = [n["cluster_id"] for n in cluster_nodes]
    family_by_cluster = _build_family_ids(cluster_ids, edges)

    family_rows_acc: dict[str, dict[str, Any]] = {}
    for node in cluster_nodes:
        cluster_id = str(node.get("cluster_id", ""))
        family_id = str(family_by_cluster.get(cluster_id, "compfam_00000_unassigned"))
        node["composition_family_id"] = family_id
        fam = family_rows_acc.get(family_id)
        if fam is None:
            fam = {
                "composition_family_id": family_id,
                "cluster_ids": [],
                "cluster_count": 0,
                "builds": set(),
                "maps": set(),
                "area_id_distribution": {},
            }
            family_rows_acc[family_id] = fam
        fam["cluster_ids"].append(cluster_id)
        fam["cluster_count"] = int(fam["cluster_count"] + 1)
        fam["builds"].update(node.get("builds", []))
        fam["maps"].update(node.get("maps", []))
        area_dist = node.get("area_id_distribution", {})
        if isinstance(area_dist, dict):
            for area_id, value in area_dist.items():
                fam["area_id_distribution"][str(area_id)] = float(fam["area_id_distribution"].get(str(area_id), 0.0) + float(value))

    family_rows: list[dict[str, Any]] = []
    for family_id, fam in sorted(family_rows_acc.items(), key=lambda item: item[0]):
        area_dist = dict(sorted(fam["area_id_distribution"].items(), key=lambda item: (item[1], item[0]), reverse=True))
        denom = float(max(1, int(fam["cluster_count"])))
        norm_area_dist = {k: float(v / denom) for k, v in area_dist.items()} if area_dist else {"unknown": 1.0}
        family_rows.append(
            {
                "composition_family_id": family_id,
                "cluster_ids": sorted(fam["cluster_ids"]),
                "cluster_count": int(fam["cluster_count"]),
                "builds": sorted(fam["builds"]),
                "maps": sorted(fam["maps"]),
                "dominant_area_ids": [k for k, _v in list(norm_area_dist.items())[:3]] if norm_area_dist else ["unknown"],
                "area_id_distribution": norm_area_dist,
            }
        )
    family_rows.sort(key=lambda r: (int(r["cluster_count"]), str(r["composition_family_id"])), reverse=True)

    # Candidate rows with composition-family assignment.
    candidate_augmented: list[dict[str, Any]] = []
    for row in candidate_rows:
        out = dict(row)
        cluster_id = str(out.get("cluster_id", ""))
        out["composition_family_id"] = str(family_by_cluster.get(cluster_id, "compfam_00000_unassigned"))
        candidate_augmented.append(out)
    candidate_augmented.sort(key=lambda r: (str(r.get("build", "")), str(r.get("map", "")), int(r.get("candidate_id", -1))))

    # Write artifacts.
    (output_dir / "composition_candidates.json").write_text(json.dumps(candidate_augmented, indent=2), encoding="utf-8")
    _write_jsonl(output_dir / "composition_candidates.jsonl", candidate_augmented)
    (output_dir / "composition_nodes.json").write_text(json.dumps(cluster_nodes, indent=2), encoding="utf-8")
    _write_jsonl(output_dir / "composition_nodes.jsonl", cluster_nodes)
    (output_dir / "composition_edges.json").write_text(json.dumps(edges, indent=2), encoding="utf-8")
    _write_jsonl(output_dir / "composition_edges.jsonl", edges)
    (output_dir / "composition_families.json").write_text(json.dumps(family_rows, indent=2), encoding="utf-8")
    _write_jsonl(output_dir / "composition_families.jsonl", family_rows)

    summary = {
        "input_candidates": int(len(rows)),
        "composition_candidates": int(len(candidate_augmented)),
        "clusters": int(len(cluster_nodes)),
        "edges": int(len(edges)),
        "composition_families": int(len(family_rows)),
        "adjacency_margin_px": int(args.adjacency_margin_px),
        "cooccur_edge_min": int(args.cooccur_edge_min),
        "area_id_map_path": str(args.area_id_map) if args.area_id_map is not None else None,
        "unknown_area_candidate_count": int(
            sum(1 for row in candidate_augmented if str((row.get("dominant_area_ids") or ["unknown"])[0]) == "unknown")
        ),
        "graph_hash": _selection_hash(
            edges,
            keys=["cluster_a", "cluster_b", "cooccur_count", "adjacent_count", "overlap_count", "min_distance_px", "canvas_count"],
        ),
        "family_hash": _selection_hash(family_rows, keys=["composition_family_id", "cluster_count"]),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "config.snapshot.json").write_text(json.dumps(vars(args), indent=2, default=str, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
