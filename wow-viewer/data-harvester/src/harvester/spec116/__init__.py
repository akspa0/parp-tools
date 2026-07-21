"""Spec 116: relational terrain layer reconstruction.

Reframes terrain reconstruction as a relational schema: layer entries are ordered rows, a layer's
texture reference is a foreign key into that tile's own local MTEX table, and the corpus is
assembled from a discrete alphabet of reused pieces.

This package consumes the existing v50 Zarr curriculum store (no new harvest) and reuses the
Spec 115 surface-family taxonomy (``harvester.v50.terrain_feature_labels``, revision ``v115.1``).
All training and heavy rebuilds are user-run; CLIs are dry-run-first.
"""
