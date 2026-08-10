namespace WowViewer.Core.Runtime.World.SceneGraph;

public enum WorldSceneNodeKind
{
    Map,
    Tile,
    Chunk,
    WmoPlacement,
    WmoGroup,
    M2Placement,
    M2Attachment,
    Pm4Structure,
    Overlay,
    SyntheticProxy
}

[Flags]
public enum WorldSceneRenderPass
{
    None = 0,
    Opaque = 1 << 0,
    AlphaTested = 1 << 1,
    Transparent = 1 << 2,
    Liquid = 1 << 3,
    Overlay = 1 << 4
}
