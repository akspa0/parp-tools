using System.Numerics;

namespace WoWViewer.Terrain;

public readonly record struct SceneObjectPickHit(
    ObjectType ObjectType,
    int ObjectIndex,
    float Distance,
    string ModelName,
    string ModelPath,
    int UniqueId,
    Vector3 PlacementPosition,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    Vector3 SelectionPoint,
    float SelectionPointDistanceSq,
    bool SharesClickedChunk,
    int ChunkGridDistance,
    int ParentWmoIndex = -1)
{
    public string KindLabel => ObjectType switch
    {
        ObjectType.Wmo => "WMO",
        ObjectType.WmoDoodad => "WMO Doodad",
        _ => "MDX"
    };
}
