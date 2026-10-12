using System.Numerics;
using Silk.NET.OpenGL;
using WoWViewer.DataSources;
using WoWViewer.Rendering;
using WoWViewer.Terrain;

namespace WoWViewer;

internal sealed partial class GroundEffectSceneService
{
    private ref IDataSource? _dataSource => ref _host.DataSource;
    private ref TerrainManager? _terrainManager => ref _host.TerrainManager;
    private ref WorldScene? _worldScene => ref _host.WorldScene;
    private ref Camera _camera => ref _host.Camera;
}
