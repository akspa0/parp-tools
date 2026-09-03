using System.Numerics;

namespace WowViewer.Core.Runtime.World;

public struct WorldObjectInstance
{
    public string ModelKey;
    public string AssetKind;
    public Matrix4x4 Transform;
    public Vector3 BoundsMin;
    public Vector3 BoundsMax;
    public Vector3 LocalBoundsMin;
    public Vector3 LocalBoundsMax;

    /// <summary>
    /// Tight, geometry-derived model-space bounds used for the selection highlight and for ray
    /// picking. Kept separate from <see cref="LocalBoundsMin"/>/<see cref="LocalBoundsMax"/> because
    /// those feed frustum and distance culling and must stay conservative: an M2's declared extent is
    /// an animation/collision volume and is routinely much larger than the mesh, which is what made
    /// selection boxes fail to describe the object they select.
    /// </summary>
    public Vector3 SelectionLocalBoundsMin;
    public Vector3 SelectionLocalBoundsMax;
    public bool SelectionBoundsResolved;
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
    public bool HasOpaqueRenderContent;
    public bool HasTransparentRenderContent;
    public uint? WmoVersion;
    public int WmoGroupCount;
    public int WmoPortalCount;
    public int WmoDoodadSetCount;
    public int WmoDoodadMdxCount;
    public int WmoDoodadM2Count;
    public int WmoDoodadUnknownCount;
}