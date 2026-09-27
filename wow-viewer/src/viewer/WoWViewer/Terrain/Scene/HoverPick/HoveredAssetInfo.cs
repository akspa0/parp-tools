using System.Numerics;

namespace WoWViewer.Terrain;

    public readonly struct HoveredAssetInfo
{
    public HoveredAssetInfo(
        string assetKind,
        string displayName,
        string sourcePath,
        string detailLine,
        Vector3 worldPosition,
        int additionalHitCount,
        (int tileX, int tileY, uint ck24, int objectPart)? pm4ObjectKey,
        ObjectType sceneObjectType = ObjectType.None,
        int sceneObjectIndex = -1,
        string? wlBodyKey = null,
        bool isPreciseRayHit = false,
        int parentWmoIndex = -1,
        string? parentSourcePath = null)
    {
        AssetKind = assetKind ?? string.Empty;
        DisplayName = displayName ?? string.Empty;
        SourcePath = sourcePath ?? string.Empty;
        DetailLine = detailLine ?? string.Empty;
        WorldPosition = worldPosition;
        AdditionalHitCount = Math.Max(0, additionalHitCount);
        Pm4ObjectKey = pm4ObjectKey;
        SceneObjectType = sceneObjectType;
        SceneObjectIndex = sceneObjectIndex;
        WlBodyKey = wlBodyKey ?? string.Empty;
        IsPreciseRayHit = isPreciseRayHit;
        ParentWmoIndex = parentWmoIndex;
        ParentSourcePath = parentSourcePath ?? string.Empty;
    }

    public string AssetKind { get; }
    public string DisplayName { get; }
    public string SourcePath { get; }
    public string DetailLine { get; }
    public Vector3 WorldPosition { get; }
    public int AdditionalHitCount { get; }
    public (int tileX, int tileY, uint ck24, int objectPart)? Pm4ObjectKey { get; }
    public ObjectType SceneObjectType { get; }
    public int SceneObjectIndex { get; }
    public string WlBodyKey { get; }
    public bool IsPreciseRayHit { get; }
    public int ParentWmoIndex { get; }
    public string ParentSourcePath { get; }
    public bool HasSceneObject => SceneObjectType is ObjectType.Mdx or ObjectType.Wmo or ObjectType.WmoDoodad && SceneObjectIndex >= 0;

    public HoveredAssetInfo WithPreciseRayHit() => new(
        AssetKind, DisplayName, SourcePath, DetailLine, WorldPosition, AdditionalHitCount, Pm4ObjectKey,
        SceneObjectType, SceneObjectIndex, WlBodyKey, isPreciseRayHit: true, ParentWmoIndex, ParentSourcePath);
}
