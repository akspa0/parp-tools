"""Find tiles whose signals disagree with each other — the tell that something is off-pattern.

Every analysis so far started from a hypothesis ("terrain is squeezed", "holes hide geometry") and
then went looking. This inverts that: it makes no assumption about WHICH signals matter, measures
how every pair of signals normally co-occurs across the corpus, and then reports the tiles that
break the pattern.

The logic is deliberately blunt. If signal A is present on 900 tiles and signal B is present on
898 of those, then A essentially implies B — and the 2 tiles where A is present and B is not are
anomalies worth a human look. A rule that holds 99% of the time and fails twice is far more
interesting than one that holds 60% of the time, because the 60% rule was never a rule.

This finds nothing on its own. It produces a ranked list of tiles that do not look like their
neighbours in signal-space, which is where to point the next hypothesis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

MISMATCH_SCHEMA = "v50-tile-mismatch-v1"

# An implication is only worth reporting if the rule is nearly universal AND the exceptions are
# rare. A "rule" with hundreds of violations is just two loosely-related signals.
MIN_SUPPORT = 30          # the antecedent must be present on at least this many tiles
MIN_CONFIDENCE = 0.95     # A -> B must hold at least this often
MAX_VIOLATION_FRACTION = 0.10  # and break on at most this share of the antecedent's tiles


def presence_matrix(
    group: Any, signal_names: list[str], row_count: int
) -> tuple[np.ndarray, list[str]]:
    """Boolean (rows x signals) presence table: is this signal non-empty on this tile?

    Presence, not content: a signal that exists but is entirely zero is treated as absent, because
    that is exactly the state every dropped/never-populated signal lands in.
    """
    usable = [name for name in signal_names if name in group]
    table = np.zeros((row_count, len(usable)), dtype=bool)
    for column, name in enumerate(usable):
        array = group[name]
        for row in range(row_count):
            table[row, column] = bool(np.asarray(array[row]).any())
    return table, usable


def implication_rules(table: np.ndarray, names: list[str]) -> list[dict[str, Any]]:
    """Every near-universal ``A -> B`` rule, with the rows that violate it.

    Returns rules sorted by how sharp they are: high confidence with few violations first, since a
    rule broken twice out of a thousand points at two specific tiles.
    """
    rules: list[dict[str, Any]] = []
    for a in range(len(names)):
        support = int(table[:, a].sum())
        if support < MIN_SUPPORT:
            continue
        for b in range(len(names)):
            if a == b:
                continue
            both = int((table[:, a] & table[:, b]).sum())
            confidence = both / support
            violations = support - both
            if confidence < MIN_CONFIDENCE or violations == 0:
                continue
            if violations / support > MAX_VIOLATION_FRACTION:
                continue
            rules.append({
                "antecedent": names[a],
                "consequent": names[b],
                "support": support,
                "confidence": confidence,
                "violations": violations,
                "violating_rows": np.flatnonzero(table[:, a] & ~table[:, b]).tolist(),
            })
    rules.sort(key=lambda r: (r["violations"], -r["confidence"]))
    return rules


def score_tiles(rules: list[dict[str, Any]], row_count: int) -> np.ndarray:
    """Per-tile anomaly score: how many near-universal rules this tile breaks, weighted.

    A tile breaking a 99.9%-confident rule is stranger than one breaking a 95% rule, so each
    violation contributes its rule's confidence rather than a flat 1.
    """
    score = np.zeros(row_count, dtype=np.float64)
    for rule in rules:
        for row in rule["violating_rows"]:
            score[row] += float(rule["confidence"])
    return score


def analyze_store(
    group: Any, index_rows: list[dict[str, Any]], signal_names: list[str]
) -> dict[str, Any]:
    """Presence table -> rules -> ranked anomalous tiles for one map."""
    row_count = len(index_rows)
    table, names = presence_matrix(group, signal_names, row_count)
    rules = implication_rules(table, names)
    score = score_tiles(rules, row_count)

    ranked = []
    for row in np.argsort(-score):
        if score[row] <= 0:
            break
        meta = index_rows[int(row)]
        broken = [
            f"{r['antecedent']} -> {r['consequent']}"
            for r in rules if int(row) in r["violating_rows"]
        ]
        ranked.append({
            "tile_key": f"{meta.get('map')}_{int(meta['tile_x']):02d}_{int(meta['tile_y']):02d}",
            "map": str(meta.get("map")),
            "tile_x": int(meta["tile_x"]),
            "tile_y": int(meta["tile_y"]),
            "row_id": int(row),
            "anomaly_score": float(score[row]),
            "broken_rules": broken,
            "present_signals": [names[c] for c in np.flatnonzero(table[int(row)])],
        })
    return {
        "signals": names,
        "signal_coverage": {n: int(table[:, i].sum()) for i, n in enumerate(names)},
        "rules": [{k: v for k, v in r.items() if k != "violating_rows"} for r in rules],
        "anomalous_tiles": ranked,
    }


def main() -> int:
    import argparse

    import pyarrow.parquet as pq
    import zarr

    parser = argparse.ArgumentParser(
        description="Find tiles whose signals contradict the corpus-wide co-occurrence pattern"
    )
    parser.add_argument("--store", required=True, type=Path, action="append", dest="stores",
                        metavar="STORE", help="a per-map v50 store; repeatable")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top", type=int, default=40, help="anomalous tiles to print per map")
    args = parser.parse_args()

    everything: dict[str, Any] = {"schema": MISMATCH_SCHEMA, "maps": {}}
    for store in args.stores:
        group = zarr.open_group(str(store), mode="r")
        index_rows = pq.read_table(store / "index.parquet").to_pylist()
        map_name = str(index_rows[0].get("map", store.stem))
        # Every 2D/3D per-tile array in the store is a candidate; no hand-picked list, because the
        # point is to find relationships nobody thought to look for.
        candidates = [
            name for name in sorted(group.array_keys())
            if group[name].ndim >= 2 and group[name].shape[0] == len(index_rows)
        ]
        result = analyze_store(group, index_rows, candidates)
        everything["maps"][map_name] = result
        print(f"{map_name:20s} {len(index_rows):>5} tiles | {len(result['signals'])} signals | "
              f"{len(result['rules'])} near-universal rules | "
              f"{len(result['anomalous_tiles'])} anomalous tiles", flush=True)
        for rule in result["rules"][:6]:
            print(f"     {rule['antecedent']} -> {rule['consequent']}  "
                  f"conf={rule['confidence']:.3f} support={rule['support']} "
                  f"BREAKS on {rule['violations']}", flush=True)
        for tile in result["anomalous_tiles"][: args.top][:6]:
            print(f"     ANOMALY {tile['tile_key']}  score={tile['anomaly_score']:.2f}  "
                  f"{'; '.join(tile['broken_rules'][:3])}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(everything, indent=2), encoding="utf-8")
    print(f"\n[DONE] -> {args.output}", flush=True)
    return 0
