"""v60 unified dataset package (Spec 134).

The v60 line is a distinct dataset family from v50. It consolidates all harvested
builds into a single unified Zarr store with the decomposed terrain signals
(terrain_shadow_256, signal_class, surviving_height_levels). Code that is
v60-specific lives here, not in ``harvester.v50``.
"""
