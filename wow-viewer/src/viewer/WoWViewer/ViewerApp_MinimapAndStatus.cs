using System.Numerics;
using ImGuiNET;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WoWViewer.UI;
using WowViewer.Core.Maps;
using WowViewer.Core.Runtime.World.Minimap;

namespace WoWViewer;

/// <summary>
/// Partial class containing status and minimap windows.
/// </summary>
public partial class ViewerApp
{
    bool IViewerAppHost.TryGetActiveMinimapState(out List<(int tx, int ty)>? existingTiles, out Func<int, int, bool>? isTileLoaded, out int loadedTileCount, out string? mapName) => _minimapAndStatus.TryGetActiveMinimapState(out existingTiles, out isTileLoaded, out loadedTileCount, out mapName);
    void IViewerAppHost.FocusCameraOnLitLight(int lightIndex, bool closeFullscreenAfterFocus) => _minimapAndStatus.FocusCameraOnLitLight(lightIndex, closeFullscreenAfterFocus);
    void IViewerAppHost.DrawInteractiveMinimapSurface(string interactionId, Vector2 cursorPos, float mapSize, List<(int tx, int ty)> existingTiles, Func<int, int, bool> isTileLoaded, string? mapName, MinimapAndStatusService.MinimapTeleportMode teleportMode, out float viewMinTx, out float viewMinTy, out float cellSize) => _minimapAndStatus.DrawInteractiveMinimapSurface(interactionId, cursorPos, mapSize, existingTiles, isTileLoaded, mapName, teleportMode, out viewMinTx, out viewMinTy, out cellSize);
    void IViewerAppHost.ClampMinimapPanOffset() => _minimapAndStatus.ClampMinimapPanOffset();
    void IViewerAppHost.ToggleFullscreenMinimap() => _minimapAndStatus.ToggleFullscreenMinimap();
    void IViewerAppHost.DrawMinimapContent(int loadedTileCount, string? mapName, List<(int tx, int ty)> existingTiles, Func<int, int, bool> isTileLoaded) => _minimapAndStatus.DrawMinimapContent(loadedTileCount, mapName, existingTiles, isTileLoaded);
}
