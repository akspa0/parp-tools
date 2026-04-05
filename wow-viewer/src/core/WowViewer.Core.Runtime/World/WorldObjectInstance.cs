using System.Numerics;

namespace WowViewer.Core.Runtime.World;

public struct WorldObjectInstance
{
    public string ModelKey;
    public Matrix4x4 Transform;
    public Vector3 BoundsMin;
    public Vector3 BoundsMax;
    public Vector3 LocalBoundsMin;
    public Vector3 LocalBoundsMax;
    public string ModelName;
    public Vector3 PlacementPosition;
    public Vector3 PlacementRotation;
    public float PlacementScale;
    public string ModelPath;
    public int UniqueId;
    public int PlacementEntryIndex;
    public int TileX;
    public int TileY;
    public bool HasTileCoordinate;
    public bool BoundsResolved;
}