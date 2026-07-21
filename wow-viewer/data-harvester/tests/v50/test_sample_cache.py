"""The in-RAM sample cache must be an optimisation only — never a change in what is trained on."""

from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")

from harvester.v50 import direct_geometry_train  # noqa: E402


class _CountingDataset:
    """Mirrors RowDataset's caching contract with a build step we can count."""

    def __init__(self, rows: list[int], *, cache: bool = False) -> None:
        self.rows = rows
        self._cache: dict[int, tuple] | None = {} if cache else None
        self.builds = 0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        if self._cache is not None:
            cached = self._cache.get(i)
            if cached is not None:
                return cached
        self.builds += 1
        generator = torch.Generator().manual_seed(self.rows[i])
        built = (torch.rand(3, 4, 4, generator=generator), torch.tensor(float(self.rows[i])))
        if self._cache is not None:
            self._cache[i] = built
        return built


class TestCacheSemantics:
    def test_cached_values_match_uncached_exactly(self) -> None:
        """Two passes with the cache on must equal the uncached values bit-for-bit, or a cached
        run is not comparable to an uncached one."""
        plain = _CountingDataset([7, 8, 9])
        cached = _CountingDataset([7, 8, 9], cache=True)
        for i in range(3):
            cached[i]  # populate
        for i in range(3):
            expected_x, expected_y = plain[i]
            got_x, got_y = cached[i]
            assert torch.equal(got_x, expected_x)
            assert torch.equal(got_y, expected_y)

    def test_second_epoch_does_no_rebuild(self) -> None:
        cached = _CountingDataset([1, 2, 3], cache=True)
        for _ in range(4):
            for i in range(3):
                cached[i]
        assert cached.builds == 3

    def test_disabled_cache_rebuilds_every_time(self) -> None:
        plain = _CountingDataset([1, 2, 3])
        for _ in range(3):
            for i in range(3):
                plain[i]
        assert plain.builds == 9

    def test_repeated_reads_are_stable(self) -> None:
        """A cached entry is handed back by reference; confirm callers observe a stable value."""
        cached = _CountingDataset([5], cache=True)
        first = cached[0][0].clone()
        for _ in range(3):
            assert torch.equal(cached[0][0], first)


class TestTrainerWiring:
    def test_cache_is_opt_in(self) -> None:
        """Default must stay off so existing runs keep their exact prior behaviour."""
        source = inspect.getsource(direct_geometry_train.main)
        assert '"--cache-samples", action="store_true"' in source

    def test_both_loaders_honour_the_flag(self) -> None:
        source = inspect.getsource(direct_geometry_train.main)
        assert source.count("cache=args.cache_samples") == 2

    def test_ram_estimate_precedes_dry_run_exit(self) -> None:
        """Regression: the estimate is only useful if the dry run prints it, and the dry run
        returns before the use_* flags are resolved."""
        source = inspect.getsource(direct_geometry_train.main)
        assert source.index("if args.cache_samples:") < source.index("if not args.confirm_run:")

    def test_estimate_avoids_flags_defined_after_it(self) -> None:
        """Regression: an earlier draft read use_liquid_mask/use_brush_mask here, which are defined
        below the dry-run exit — an UnboundLocalError the moment --cache-samples was passed."""
        source = inspect.getsource(direct_geometry_train.main)
        block = source[source.index("if args.cache_samples:"):source.index("if not args.confirm_run:")]
        assert "use_liquid_mask" not in block
        assert "use_brush_mask" not in block
