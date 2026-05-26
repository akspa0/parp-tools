"""V18 dataset: reads V16 Zarr stores with V18 refinement awareness.

Built on the V16 Zarr corpus contract. V18-specific extras:
- Manifest-balanced (paste-deduped, family-balanced)
- Uses V161Dataset under the hood

Usage:
    from harvester.v18_dataset import V18Dataset
    ds = V18Dataset("output/datasets/v16", builds=["3_3_5_12340"])
"""

from harvester.v16_1_dataset import V161Dataset as V18Dataset

__all__ = ["V18Dataset"]
