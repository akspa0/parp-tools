# M2 Regression Scene Catalog

This file records concrete live-viewer scenes that should be reused for before/after M2 rendering checks.

## development_grizzlyhills_tree05_cluster

- Map: `development`
- Build: `3.3.5.12340`
- Scene bookmark: `Scene: map=development build=3.3.5.12340 WoW=(16731.0, 12516.0, 374.0) Facing=170.0° S Local=(4550.7, 335.7, 374.0) Yaw=80.0 Pitch=-1.0 FOV=60.0`
- Shot-point template entry: `development_grizzlyhills_tree05_cluster`

Selected object captured in the scene:

- Asset: `GRIZZLYHILLS_TREE05.M2`
- Virtual path: `WORLD\EXPANSION02\DOODADS\GRIZZLYHILLS\TREES\GRIZZLYHILLS_TREE05.M2`
- UniqueId: `14147796`
- Object local position: `(4541.0, 567.1, 266.8)`
- Object WoW position: `(16198.6, 12494.9, 266.8)`
- Rotation: `(0.0, 0.0, 53.7)`
- Scale: `1.804`
- Bounds: `(4518.1, 506.9, 259.6) - (4642.1, 630.6, 528.8)`

Observed issue from the captured scene:

- tree canopy cards are visibly distorted and folded into the trunk cluster instead of reading as coherent foliage silhouettes
- this should be treated as one representative world-scene regression, not the only affected tree family

Use this scene together with future continent flythrough bookmarks to build a broader M2 regression set.