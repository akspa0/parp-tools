using WoWViewer.DataSources;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WowViewer.Core.IO.Casc;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps.AdtAhdr;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// WorldLoaderService: members moved from ViewerApp_CascAhdr.cs; this file keeps that file's using directives so every
// name in the moved code resolves exactly as it did there.
internal sealed partial class WorldLoaderService
{

    /// <summary>Shared world load for adapters that are not WDT-driven (mirrors the Rosetta datastore path).</summary>
    internal void LoadTerrainFromAdapter(ITerrainAdapter adapter, string mapName, string modelInfoHeader)
    {
        _terrainWeakSignalRestore.ResetTerrainWeakSignalRestoreSessionState(preserveToggle: true);
        InvalidatePm4DerivedReports();
        _worldScene?.Dispose();
        _worldScene = null;
        _terrainManager?.Dispose();
        _terrainManager = null;
        _vlmTerrainManager?.Dispose();
        _vlmTerrainManager = null;
        _sqlSpawnStreaming.ResetSqlSpawnStreamingState(clearSceneSpawns: false);

        _loadingScreen?.Enable(_dataSource);
        PresentLoadingFrame();

        try
        {
            int loadStep = 0;
            void OnLoadStatus(string status)
            {
                _statusMessage = status;
                loadStep++;
                _loadingScreen?.UpdateProgress(loadStep, 20);
                PresentLoadingFrame();
            }

            var tm = new TerrainManager(_gl, adapter, mapName, _dataSource);
            _worldScene = new WorldScene(_gl, tm, _dataSource, _texResolver, _dbcBuild, _minimapRenderer, onStatus: OnLoadStatus);
            _terrainManager = _worldScene.Terrain;
            _terrainManager.DetailedTileCountOverride = _savedDetailedAdtTileCountOverride;
            ApplyGlobalFogDefaults(_terrainManager.Lighting);
            _renderer = _worldScene;
            _worldScene.EnableLitFallback($"{mapName} loaded directly; LIT/analytical lighting enabled.");

            var startPos = _terrainManager.GetInitialCameraPosition();
            _camera.Position = startPos;
            _camera.Yaw = 180f;
            _camera.Pitch = -20f;
            _terrainManager.UpdateAOI(startPos, _camera.Forward);

            _modelInfo = modelInfoHeader + $"\nCamera: ({startPos.X:F0}, {startPos.Y:F0}, {startPos.Z:F0})\n";
            _statusMessage = $"Loaded {mapName} ({adapter.ExistingTiles.Count} tiles)";
            _loadingScreen?.SetWorldLoaded();
            PresentLoadingFrame();
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[ViewerApp] {mapName} load failed: {ex}");
            _statusMessage = $"Load failed: {ex.Message}";
            _modelInfo = $"{mapName} load error:\n{ex.Message}";
            _worldScene?.Dispose();
            _worldScene = null;
            _terrainManager = null;
            _loadingScreen?.Disable();
        }
    }
}
