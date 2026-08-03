"""A rule broken twice in a thousand points at two tiles; a rule broken constantly is not a rule."""

from __future__ import annotations

import numpy as np

from harvester.v50.tile_mismatch import (
    MIN_SUPPORT,
    analyze_store,
    implication_rules,
    presence_matrix,
    score_tiles,
)


def test_presence_treats_an_all_zero_signal_as_absent() -> None:
    """A declared-but-never-populated signal is exactly the state every dropped signal lands in."""
    group = {
        "height_257": np.array([[[1.0]], [[0.0]], [[2.0]]]),
        "never_filled": np.zeros((3, 1, 1)),
    }
    table, names = presence_matrix(group, ["height_257", "never_filled", "not_in_store"], 3)
    assert names == ["height_257", "never_filled"]
    assert table[:, 0].tolist() == [True, False, True]
    assert not table[:, 1].any()


def test_only_near_universal_rules_with_rare_exceptions_are_reported() -> None:
    n = 200
    table = np.zeros((n, 4), dtype=bool)
    table[:, 0] = True                       # A everywhere
    table[:, 1] = True; table[7, 1] = False  # B everywhere but one tile -> sharp rule
    table[:, 2] = False; table[:100, 2] = True   # C on half -> A->C is only 50% confident
    table[:, 3] = True; table[:60, 3] = False    # D breaks on 60 tiles -> too many violations

    rules = implication_rules(table, ["A", "B", "C", "D"])
    pairs = {(r["antecedent"], r["consequent"]): r for r in rules}

    assert ("A", "B") in pairs                       # 199/200: reported
    assert pairs[("A", "B")]["violations"] == 1
    assert pairs[("A", "B")]["violating_rows"] == [7]
    assert ("A", "C") not in pairs                   # 50% confidence: not a rule
    assert ("A", "D") not in pairs                   # 30% of tiles violate: too noisy

    # Sharpest rules first.
    assert rules[0]["violations"] <= rules[-1]["violations"]


def test_support_floor_rejects_a_rule_from_too_few_tiles() -> None:
    small = np.zeros((MIN_SUPPORT - 1, 2), dtype=bool)
    small[:, 0] = True; small[:, 1] = True; small[0, 1] = False
    assert implication_rules(small, ["A", "B"]) == []


def test_scores_weight_by_rule_confidence() -> None:
    rules = [
        {"antecedent": "A", "consequent": "B", "confidence": 0.99, "violating_rows": [3]},
        {"antecedent": "A", "consequent": "C", "confidence": 0.96, "violating_rows": [3, 5]},
    ]
    score = score_tiles(rules, 6)
    assert score[3] == 0.99 + 0.96   # breaks both
    assert score[5] == 0.96          # breaks one
    assert score[0] == 0.0


def test_analyze_store_ranks_the_odd_tile_out() -> None:
    n = 120
    group = {
        "height_257": np.ones((n, 2, 2)),
        "normal_xyz": np.ones((n, 2, 2)),
        "minimap_rgb": np.ones((n, 2, 2)),
    }
    # One tile has terrain but no normals -- the mismatch the whole tool exists to surface.
    group["normal_xyz"][11] = 0
    index_rows = [{"map": "Kalimdor", "tile_x": i % 10, "tile_y": i // 10} for i in range(n)]

    result = analyze_store(group, index_rows, list(group))
    assert result["signal_coverage"]["height_257"] == n
    assert result["signal_coverage"]["normal_xyz"] == n - 1
    assert result["anomalous_tiles"], "the odd tile out must be reported"
    top = result["anomalous_tiles"][0]
    assert top["row_id"] == 11
    assert any("normal_xyz" in r for r in top["broken_rules"])
    assert "normal_xyz" not in top["present_signals"]
