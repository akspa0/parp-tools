"""Spec 109 T007: resolved-path and protected-root tests (FR-020/FR-021). Symlink-escape tests are
skipped when the host cannot create symlinks without elevated privileges (plain Windows accounts
without Developer Mode/Administrator) -- the non-symlink cases below still fully exercise approved-
vs-protected precedence and out-of-root rejection."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from harvester.v50.path_policy import PathPolicy, PathPolicyError


def _probe_symlink_support() -> bool:
    probe_dir = Path(tempfile.mkdtemp(prefix="v50_symlink_probe_"))
    try:
        target = probe_dir / "target"
        link = probe_dir / "link"
        target.mkdir()
        os.symlink(target, link, target_is_directory=True)
        return True
    except OSError:
        return False
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)


_SYMLINKS_SUPPORTED = _probe_symlink_support()
_SKIP_NO_SYMLINKS = pytest.mark.skipif(
    not _SYMLINKS_SUPPORTED, reason="host cannot create symlinks without elevated privileges"
)


def test_resolves_a_path_cleanly_inside_a_single_approved_root(tmp_path: Path):
    approved = tmp_path / "approved"
    approved.mkdir()
    target = approved / "old_dataset"
    target.mkdir()

    policy = PathPolicy(approved_roots=[approved], protected_roots=[])

    assert policy.resolve_within_approved_root(target) == target.resolve()


def test_rejects_a_path_entirely_outside_every_approved_root(tmp_path: Path):
    approved = tmp_path / "approved"
    approved.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    policy = PathPolicy(approved_roots=[approved], protected_roots=[])

    with pytest.raises(PathPolicyError, match="does not resolve inside any approved root"):
        policy.resolve_within_approved_root(outside)


def test_protected_root_wins_even_when_nested_inside_an_approved_root(tmp_path: Path):
    approved = tmp_path / "approved"
    protected = approved / "specs"  # e.g. specs/ nested under an otherwise-generated output root
    protected.mkdir(parents=True)

    policy = PathPolicy(approved_roots=[approved], protected_roots=[protected])

    with pytest.raises(PathPolicyError, match="protected root"):
        policy.resolve_within_approved_root(protected)

    # A sibling of the protected directory, still under the approved root, remains fine.
    sibling = approved / "old_output"
    sibling.mkdir()
    assert policy.resolve_within_approved_root(sibling) == sibling.resolve()


def test_rejects_a_nonexistent_path_instead_of_guessing(tmp_path: Path):
    approved = tmp_path / "approved"
    approved.mkdir()
    policy = PathPolicy(approved_roots=[approved], protected_roots=[])

    with pytest.raises(PathPolicyError, match="does not exist"):
        policy.resolve_within_approved_root(approved / "never_created")


def test_is_approved_and_is_protected_agree_with_resolve(tmp_path: Path):
    approved = tmp_path / "approved"
    protected = tmp_path / "protected"
    approved.mkdir()
    protected.mkdir()
    inside_approved = approved / "x"
    inside_approved.mkdir()

    policy = PathPolicy(approved_roots=[approved], protected_roots=[protected])

    assert policy.is_approved(inside_approved) is True
    assert policy.is_protected(inside_approved) is False
    assert policy.is_protected(protected) is True
    assert policy.is_approved(protected) is False


@_SKIP_NO_SYMLINKS
def test_rejects_a_symlink_that_escapes_every_approved_root(tmp_path: Path):
    approved = tmp_path / "approved"
    approved.mkdir()
    outside = tmp_path / "outside_target"
    outside.mkdir()
    (outside / "secret.txt").write_text("do not touch")

    escape_link = approved / "looks_safe"
    os.symlink(outside, escape_link, target_is_directory=True)

    policy = PathPolicy(approved_roots=[approved], protected_roots=[])

    with pytest.raises(PathPolicyError, match="does not resolve inside any approved root"):
        policy.resolve_within_approved_root(escape_link)


@_SKIP_NO_SYMLINKS
def test_rejects_a_symlink_that_redirects_into_a_protected_root(tmp_path: Path):
    approved = tmp_path / "approved"
    protected = tmp_path / "protected_specs"
    approved.mkdir()
    protected.mkdir()

    redirect_link = approved / "sneaky"
    os.symlink(protected, redirect_link, target_is_directory=True)

    policy = PathPolicy(approved_roots=[approved], protected_roots=[protected])

    with pytest.raises(PathPolicyError, match="protected root"):
        policy.resolve_within_approved_root(redirect_link)
