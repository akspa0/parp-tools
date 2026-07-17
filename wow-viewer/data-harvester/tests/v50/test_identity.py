"""Spec 109 T006 (identity half): deterministic file/metadata-tree/Parquet/manifest identities."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from harvester.v50.identity import hash_file, hash_manifest, hash_metadata_tree, hash_parquet_table


def test_hash_file_is_deterministic_and_content_sensitive(tmp_path: Path):
    file_a = tmp_path / "a.bin"
    file_b = tmp_path / "b.bin"
    file_a.write_bytes(b"same content")
    file_b.write_bytes(b"same content")
    file_c = tmp_path / "c.bin"
    file_c.write_bytes(b"different content")

    assert hash_file(file_a) == hash_file(file_b)
    assert hash_file(file_a) != hash_file(file_c)
    assert hash_file(file_a).startswith("sha256:")


def test_hash_metadata_tree_is_sensitive_to_structure_and_content(tmp_path: Path):
    tree_a = tmp_path / "tree_a"
    (tree_a / "sub").mkdir(parents=True)
    (tree_a / "sub" / "file.txt").write_text("hello")
    (tree_a / "root.txt").write_text("world")

    tree_b = tmp_path / "tree_b"
    (tree_b / "sub").mkdir(parents=True)
    (tree_b / "sub" / "file.txt").write_text("hello")
    (tree_b / "root.txt").write_text("world")

    assert hash_metadata_tree(tree_a) == hash_metadata_tree(tree_b)

    (tree_b / "root.txt").write_text("CHANGED")
    assert hash_metadata_tree(tree_a) != hash_metadata_tree(tree_b)


def test_hash_metadata_tree_requires_a_directory(tmp_path: Path):
    lone_file = tmp_path / "lone.txt"
    lone_file.write_text("x")
    with pytest.raises(NotADirectoryError):
        hash_metadata_tree(lone_file)


def test_hash_parquet_table_is_invariant_to_physical_layout_but_sensitive_to_content(tmp_path: Path):
    table = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    path_snappy = tmp_path / "snappy.parquet"
    path_uncompressed = tmp_path / "none.parquet"
    pq.write_table(table, path_snappy, compression="snappy")
    pq.write_table(table, path_uncompressed, compression="none")

    # Same logical content, different physical compression -- must be the same identity, or a
    # routine resave/recompress would spuriously invalidate row lineage (data-model.md).
    assert hash_parquet_table(path_snappy) == hash_parquet_table(path_uncompressed)

    different_table = pa.table({"a": [1, 2, 4], "b": ["x", "y", "z"]})
    path_different = tmp_path / "different.parquet"
    pq.write_table(different_table, path_different)
    assert hash_parquet_table(path_snappy) != hash_parquet_table(path_different)


def test_hash_manifest_is_invariant_to_key_order_but_sensitive_to_value(tmp_path: Path):
    manifest_a = {"release": "v50.1", "row_count": 10, "signals": ["a", "b"]}
    manifest_b = {"signals": ["a", "b"], "row_count": 10, "release": "v50.1"}
    assert hash_manifest(manifest_a) == hash_manifest(manifest_b)

    manifest_c = {"release": "v50.1", "row_count": 11, "signals": ["a", "b"]}
    assert hash_manifest(manifest_a) != hash_manifest(manifest_c)
