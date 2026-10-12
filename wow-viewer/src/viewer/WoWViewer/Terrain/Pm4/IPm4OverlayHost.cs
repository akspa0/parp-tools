using System.Numerics;
using WoWViewer.DataSources;
using WoWViewer.Rendering;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;

namespace WoWViewer.Terrain;

/// <summary>
/// The world-scene state <see cref="Pm4OverlayScene"/> reads. Implemented explicitly by
/// <see cref="WorldScene"/>; the overlay never reaches into the scene any other way (AGENTS.md §10).
/// </summary>
internal interface IPm4OverlayHost
{
    IDataSource? DataSource { get; }
    TerrainManager TerrainManager { get; }
    WorldAssetManager Assets { get; }
    bool InstancesDirty { get; }
    void RebuildInstanceLists();
    Dictionary<(int, int), List<ObjectInstance>> TileWmoInstances { get; }
    Dictionary<(int, int), List<ObjectInstance>> TileMdxInstances { get; }
    List<ObjectInstance> WmoInstances { get; }
    bool HasLastRenderedCameraPosition { get; }
    Vector3 LastRenderedCameraPosition { get; }
    bool IsHoverPickDistanceAllowed(float distance);
    bool IsHoverPickPositionAllowed(Vector3 worldPosition);
    FrustumCuller FrustumCuller { get; }
}
