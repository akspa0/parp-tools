using System.Numerics;

namespace WoWViewer.Terrain;

// Moved from WorldScene (Spec 255 W0): formerly a private nested record; body unchanged.
internal readonly record struct SelectedSceneObjectKey(
    ObjectType ObjectType,
    int UniqueId,
    int PlacementEntryIndex,
    int TileX,
    int TileY,
    bool HasTileCoordinate,
    string ModelKey,
    Vector3 PlacementPosition);
