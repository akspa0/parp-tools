"""Spec 077 height-only terrain augmentation.

Geometrically-exact flip/rotate helpers for height-only training samples.
Terrain height is a scalar field, so the dihedral group D4 (identity, hflip,
vflip, hflip+vflip, rot90/180/270) is an exact symmetry of the height target.
That is only true for orientation-free inputs. Baked minimap RGB is not
orientation-free: terrain lighting and shadows have a fixed world direction,
so production minimap-to-height runs should use the shadow-safe identity-only
policy unless they intentionally ablate with ``--augment-policy d4``.

For transforms that are explicitly requested, the minimap prior, weight mask,
and normal mask transform as plain images. The only non-trivial part is the
``normal_xyz`` target, whose x/y gradient channels must be negated and/or
swapped to stay consistent with the transformed height field.

Normal convention (see ``harvester.height_to_normal.analytic_normals_from_height``):
``n = normalize([-dh/dx, -dh/dy, +1])``. Under a spatial transform of the
height field, the gradient operators transform as vectors, so the normal's
x/y channels transform with the same rotation/reflection matrix as the
image plane, while the z channel is unchanged.

This module is deterministic given a chosen transform id and a per-sample
random state, so train runs are reproducible and val runs can opt out
entirely (``augment=False``).

Spec 077 FR-013/FR-014/FR-023 are respected: augmentation only changes the
geometry of a single height sample; it does not add heads, share weights,
or mix signals across models.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

# Transform ids for the D4 dihedral group. ``rot90`` is counter-clockwise in
# numpy/image coordinates (rows = y, cols = x), matching ``np.rot90``.
TransformId = Literal[
    "identity",
    "hflip",
    "vflip",
    "hflip_vflip",
    "rot90",
    "rot180",
    "rot270",
    "transpose",
]

ALL_TRANSFORMS: tuple[TransformId, ...] = (
    "identity",
    "hflip",
    "vflip",
    "hflip_vflip",
    "rot90",
    "rot180",
    "rot270",
    "transpose",
)

SHADOW_SAFE_TRANSFORMS: tuple[TransformId, ...] = ("identity",)


def sample_transform(
    rng: np.random.Generator,
    transforms: tuple[TransformId, ...] = ALL_TRANSFORMS,
) -> TransformId:
    """Pick a uniformly random transform id from an allowed transform set."""
    if not transforms:
        raise ValueError("At least one augmentation transform is required.")
    idx = int(rng.integers(0, len(transforms)))
    return transforms[idx]


def _spatial_flip(arr: np.ndarray, hflip: bool, vflip: bool) -> np.ndarray:
    """Flip the last two (H, W) axes of a (..., H, W) array."""
    if hflip:
        arr = np.flip(arr, axis=-1)
    if vflip:
        arr = np.flip(arr, axis=-2)
    return np.ascontiguousarray(arr)


def _spatial_rot90(arr: np.ndarray, k: int) -> np.ndarray:
    """Rotate the last two (H, W) axes of a (..., H, W) array by k*90 CCW."""
    return np.ascontiguousarray(np.rot90(arr, k=k, axes=(-2, -1)))


def _transform_image(arr: np.ndarray, transform: TransformId) -> np.ndarray:
    """Apply a D4 transform to a plain image-like array (last two axes = H, W).

    Used for the minimap prior, raw minimap, teacher mask/confidence, height,
    weight, and normal mask. These are all scalar-per-pixel fields, so they
    transform as plain images with no channel sign changes.
    """
    if transform == "identity":
        return np.ascontiguousarray(arr)
    if transform == "hflip":
        return _spatial_flip(arr, hflip=True, vflip=False)
    if transform == "vflip":
        return _spatial_flip(arr, hflip=False, vflip=True)
    if transform == "hflip_vflip":
        return _spatial_flip(arr, hflip=True, vflip=True)
    if transform == "rot90":
        return _spatial_rot90(arr, k=1)
    if transform == "rot180":
        return _spatial_rot90(arr, k=2)
    if transform == "rot270":
        return _spatial_rot90(arr, k=3)
    if transform == "transpose":
        return np.ascontiguousarray(np.swapaxes(arr, -2, -1))
    raise ValueError(f"Unknown transform id: {transform}")


def _transform_normals(normals: np.ndarray, transform: TransformId) -> np.ndarray:
    """Apply a D4 transform to a normal field with channel-first layout.

    Accepts ``(3, H, W)`` (single tile, as stored by the dataset) or
    ``(B, 3, H, W)`` (batched). Channel 0 = x, 1 = y, 2 = z, following the
    ``[-dh/dx, -dh/dy, +1]`` convention. The spatial part is transformed
    like an image, and the x/y channels are negated/swapped so the normal
    stays consistent with the transformed height field.

    Derivation (image coords: x = col, y = row, rot90 is CCW):
      * hflip (x -> -x): spatial flip on W axis; normal_x negated.
      * vflip (y -> -y): spatial flip on H axis; normal_y negated.
      * rot90 CCW (x -> y, y -> -x): spatial rot90; channels (x, y) -> (-y, x).
      * rot180: both negations; spatial rot180.
      * rot270 CCW (x -> -y, y -> x): spatial rot270; channels (x, y) -> (y, -x).
      * transpose (x <-> y): spatial transpose; channels (x, y) -> (y, x).
    """
    batched = normals.ndim == 4 and normals.shape[1] == 3
    if batched:
        # Recurse per-batch to reuse the (3, H, W) path.
        return np.stack([_transform_normals(normals[b], transform) for b in range(normals.shape[0])], axis=0)
    if normals.ndim != 3 or normals.shape[0] != 3:
        raise ValueError(f"Expected normals of shape (3, H, W) or (B, 3, H, W); got {normals.shape}")
    x = normals[0:1]
    y = normals[1:2]
    z = normals[2:3]
    if transform == "identity":
        return np.ascontiguousarray(normals)
    if transform == "hflip":
        return np.ascontiguousarray(np.concatenate([_transform_image(x, transform), _transform_image(y, transform), _transform_image(z, transform)], axis=0) * np.array([[-1.0], [1.0], [1.0]], dtype=normals.dtype)[:, :, None])
    if transform == "vflip":
        return np.ascontiguousarray(np.concatenate([_transform_image(x, transform), _transform_image(y, transform), _transform_image(z, transform)], axis=0) * np.array([[1.0], [-1.0], [1.0]], dtype=normals.dtype)[:, :, None])
    if transform == "hflip_vflip":
        return np.ascontiguousarray(np.concatenate([_transform_image(x, transform), _transform_image(y, transform), _transform_image(z, transform)], axis=0) * np.array([[-1.0], [-1.0], [1.0]], dtype=normals.dtype)[:, :, None])
    if transform == "rot90":
        # rot90 CCW: (n_x, n_y) -> (n_y, -n_x)
        new_x = _transform_image(y, transform)
        new_y = -_transform_image(x, transform)
        return np.ascontiguousarray(np.concatenate([new_x, new_y, _transform_image(z, transform)], axis=0))
    if transform == "rot180":
        new_x = -_transform_image(x, transform)
        new_y = -_transform_image(y, transform)
        return np.ascontiguousarray(np.concatenate([new_x, new_y, _transform_image(z, transform)], axis=0))
    if transform == "rot270":
        # rot270 CCW: (n_x, n_y) -> (-n_y, n_x)
        new_x = -_transform_image(y, transform)
        new_y = _transform_image(x, transform)
        return np.ascontiguousarray(np.concatenate([new_x, new_y, _transform_image(z, transform)], axis=0))
    if transform == "transpose":
        new_x = _transform_image(y, transform)
        new_y = _transform_image(x, transform)
        return np.ascontiguousarray(np.concatenate([new_x, new_y, _transform_image(z, transform)], axis=0))
    raise ValueError(f"Unknown transform id: {transform}")


def augment_sample(
    sample: dict,
    transform: TransformId,
) -> dict:
    """Apply a D4 transform to a height-only training sample dict.

    Transforms every spatial array in the sample consistently:
      * ``input_prior`` (C, 256, 256) — plain image
      * ``raw_minimap_rgb`` (3, 256, 256) — plain image
      * ``teacher_object_mask`` / ``teacher_object_confidence`` (1, 256, 256) — plain image
      * ``height_257`` (1, 257, 257) — plain image (scalar field)
      * ``weight_257`` (1, 257, 257) — plain image (scalar mask)
      * ``normal_mask`` (1, 257, 257) — plain image (scalar mask)
      * ``normal_xyz`` (3, 257, 257) — normal field (channel sign/swap aware)

    Non-scalar metadata (``meta_*``) is passed through unchanged. The input
    dict is not mutated; a new dict is returned.
    """
    out: dict = {}
    for key, value in sample.items():
        if key.startswith("meta_"):
            out[key] = value
            continue
        if isinstance(value, np.ndarray) and value.ndim >= 3:
            if key == "normal_xyz":
                out[key] = _transform_normals(value, transform)
            else:
                out[key] = _transform_image(value, transform)
        else:
            out[key] = value
    return out
