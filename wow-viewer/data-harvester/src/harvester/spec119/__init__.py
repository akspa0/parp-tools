"""Spec 119: object-library segmentation & classifier package.

Two small, from-scratch, independently checkpointed specialists trained on the object-library
zarr itself (Spec 118 capture pipeline output), plus a quality lens over the library. Distinct
from Spec 118 US3's ``ObjectSegmentNet`` (which segments minimap tiles); Spec 119 segments and
classifies the captured object crops, supervised by the library's own masks/labels.
"""
