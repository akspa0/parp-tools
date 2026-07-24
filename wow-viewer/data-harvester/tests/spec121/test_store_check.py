"""Spec 121 T004: store prerequisite check tests."""

from __future__ import annotations

import numpy as np

from harvester.spec121.store_check import (
    LATTICE_ARRAYS,
    OBJECT_MASK_ARRAYS,
    REQUIRED_ARRAYS,
    check_store_arrays,
    missing_required,
    object_mask_present,
    report,
)


class FakeGroup:
    """Minimal zarr-group stand-in: array_keys() + membership + shaped arrays."""

    def __init__(self, names) -> None:
        self._names = list(names)

    def array_keys(self):
        return list(self._names)

    def __contains__(self, name) -> bool:
        return name in self._names

    def __getitem__(self, name):
        if name not in self._names:
            raise KeyError(name)
        return np.zeros((2, 4, 4), dtype=np.float32)


FULL = list(REQUIRED_ARRAYS) + list(OBJECT_MASK_ARRAYS)


def test_report_marks_full_store_usable():
    result = report(FakeGroup(FULL))
    assert result["schema"] == "v121-store-check-v1"
    assert result["usable"] is True
    assert result["missing_required"] == []
    assert result["object_mask_signal_present"] is True


def test_missing_lattice_arrays_make_store_unusable():
    group = FakeGroup(["minimap_rgb"])
    assert set(missing_required(group)) == set(LATTICE_ARRAYS)
    assert report(group)["usable"] is False


def test_object_mask_absence_is_not_a_required_failure():
    group = FakeGroup(list(REQUIRED_ARRAYS))
    assert object_mask_present(group) is False
    assert report(group)["usable"] is True
    assert report(group)["object_mask_signal_present"] is False


def test_check_store_arrays_covers_every_known_name():
    presence = check_store_arrays(FakeGroup(FULL))
    assert set(presence) == set(REQUIRED_ARRAYS) | set(OBJECT_MASK_ARRAYS)
    assert all(presence.values())
