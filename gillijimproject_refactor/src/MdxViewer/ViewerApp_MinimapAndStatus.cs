using System.Numerics;
using ImGuiNET;
using MdxViewer.Rendering;
using MdxViewer.Terrain;

namespace MdxViewer;

/// <summary>
/// Partial class containing status and minimap windows.
/// </summary>
public partial class ViewerApp
{
    private const float MinimapTileCount = 64f;
    private const float MinimapWorldTileSize = WoWConstants.ChunkSize;

    private void HandleMinimapInteraction(string interactionId, Vector2 cursorPos, float mapSize, float viewMinTx, float viewMinTy, float cellSize)
    {
        ImGui.SetCursorScreenPos(cursorPos);
        ImGui.InvisibleButton(interactionId, new Vector2(mapSize, mapSize));
        bool isHovered = ImGui.IsItemHovered();
        bool isActive = ImGui.IsItemActive();
        Vector2 mousePos = ImGui.GetMousePos();

        if (isHovered || isActive)
        {
            if (ImGui.IsMouseClicked(ImGuiMouseButton.Left))
            {
                _minimapDragging = true;
                _minimapDragStart = mousePos;
                _minimapDragOrigin = mousePos;
            }
            else if (ImGui.IsMouseDown(ImGuiMouseButton.Left) && _minimapDragging)
            {
                Vector2 delta = mousePos - _minimapDragStart;
                if (delta.LengthSquared() > 0.01f)
                {
                    _minimapPanOffset -= new Vector2(delta.Y / cellSize, delta.X / cellSize);
                    _minimapDragStart = mousePos;
                }
            }
            else if (ImGui.IsMouseReleased(ImGuiMouseButton.Left) && _minimapDragging)
            {
                Vector2 totalDelta = mousePos - _minimapDragOrigin;
                if (totalDelta.Length() <= MinimapClickMovementThresholdPixels
                    && TryGetMinimapClickTarget(mousePos, cursorPos, cellSize, viewMinTx, viewMinTy, out float clickTileX, out float clickTileY))
                {
                    RegisterMinimapTeleportClick(clickTileX, clickTileY);
                }

                _minimapDragging = false;
            }
        }
        else if (_minimapDragging)
        {
            _minimapDragging = false;
        }
    }

    private static bool TryGetMinimapClickTarget(Vector2 mousePos, Vector2 cursorPos, float cellSize, float viewMinTx, float viewMinTy, out float clickTileX, out float clickTileY)
    {
        clickTileY = (mousePos.X - cursorPos.X) / cellSize + viewMinTy;
        clickTileX = (mousePos.Y - cursorPos.Y) / cellSize + viewMinTx;
        return clickTileX >= 0f && clickTileX < MinimapTileCount && clickTileY >= 0f && clickTileY < MinimapTileCount;
    }

    private void PrepareFullscreenMinimapState()
    {
        _minimapDragging = false;
        _minimapDragStart = Vector2.Zero;
        _minimapDragOrigin = Vector2.Zero;
        ClampMinimapPanOffset();
    }

    private void ClampMinimapPanOffset()
    {
        float camTileX = (WoWConstants.MapOrigin - _camera.Position.X) / MinimapWorldTileSize;
        float camTileY = (WoWConstants.MapOrigin - _camera.Position.Y) / MinimapWorldTileSize;
        float viewSpan = Math.Clamp(_minimapZoom * 2f, 2f, MinimapTileCount);
        float viewRadius = viewSpan * 0.5f;
        float maxViewMin = MathF.Max(0f, MinimapTileCount - viewSpan);
        float baseViewMinTx = camTileX - viewRadius;
        float baseViewMinTy = camTileY - viewRadius;
        float minPanX = -baseViewMinTx;
        float maxPanX = maxViewMin - baseViewMinTx;
        float minPanY = -baseViewMinTy;
        float maxPanY = maxViewMin - baseViewMinTy;

        _minimapPanOffset = new Vector2(
            Math.Clamp(_minimapPanOffset.X, minPanX, maxPanX),
            Math.Clamp(_minimapPanOffset.Y, minPanY, maxPanY));
    }

    private void RegisterMinimapTeleportClick(float clickTileX, float clickTileY)
    {
        int tileX = (int)MathF.Floor(clickTileX);
        int tileY = (int)MathF.Floor(clickTileY);
        DateTime now = DateTime.UtcNow;

        if (!_pendingMinimapTeleportTile.HasValue
            || _pendingMinimapTeleportTile.Value.tileX != tileX
            || _pendingMinimapTeleportTile.Value.tileY != tileY
            || now - _pendingMinimapTeleportLastClickUtc > MinimapTeleportConfirmWindow)
        {
            _pendingMinimapTeleportTile = (tileX, tileY);
            _pendingMinimapTeleportClickCount = 1;
            _pendingMinimapTeleportLastClickUtc = now;
            _statusMessage = $"Minimap teleport armed for tile ({tileX},{tileY}) 1/{MinimapTeleportConfirmClicks}. Click the same tile {MinimapTeleportConfirmClicks - 1} more times to teleport.";
            return;
        }

        _pendingMinimapTeleportClickCount++;
        _pendingMinimapTeleportLastClickUtc = now;

        if (_pendingMinimapTeleportClickCount < MinimapTeleportConfirmClicks)
        {
            int remainingClicks = MinimapTeleportConfirmClicks - _pendingMinimapTeleportClickCount;
            _statusMessage = $"Minimap teleport still armed for tile ({tileX},{tileY}) {_pendingMinimapTeleportClickCount}/{MinimapTeleportConfirmClicks}. Click {remainingClicks} more time{(remainingClicks == 1 ? string.Empty : "s")} to teleport.";
            return;
        }

        float worldX = WoWConstants.MapOrigin - clickTileX * MinimapWorldTileSize;
        float worldY = WoWConstants.MapOrigin - clickTileY * MinimapWorldTileSize;
        _camera.Position = new Vector3(worldX, worldY, _camera.Position.Z);
        _statusMessage = $"Minimap teleported camera to tile ({tileX},{tileY}).";
        ClearPendingMinimapTeleport();
    }

    private void ClearPendingMinimapTeleport()
    {
        _pendingMinimapTeleportTile = null;
        _pendingMinimapTeleportClickCount = 0;
        _pendingMinimapTeleportLastClickUtc = DateTime.MinValue;
    }

    private void DrawStatusBar()
    {
        var io = ImGui.GetIO();
        var windowHeight = io.DisplaySize.Y;
        ImGui.SetNextWindowPos(new Vector2(0, windowHeight - 24));
        ImGui.SetNextWindowSize(new Vector2(io.DisplaySize.X, 24));
        ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, new Vector2(8, 4));
        if (ImGui.Begin("##statusbar", ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.NoResize |
            ImGuiWindowFlags.NoMove | ImGuiWindowFlags.NoScrollbar | ImGuiWindowFlags.NoSavedSettings))
        {
            ImGui.Text(_statusMessage);
            if (!string.IsNullOrEmpty(_currentAreaName))
            {
                ImGui.SameLine();
                ImGui.TextColored(new Vector4(1f, 0.9f, 0.5f, 1f), $"  Area: {_currentAreaName}");
            }

            if (_terrainManager != null || _vlmTerrainManager != null)
            {
                var pos = _camera.Position;
                float wowX = WoWConstants.MapOrigin - pos.Y;
                float wowY = WoWConstants.MapOrigin - pos.X;
                float wowZ = pos.Z;
                string coordText = $"Local: ({pos.X:F0}, {pos.Y:F0}, {pos.Z:F0})  WoW: ({wowX:F0}, {wowY:F0}, {wowZ:F0})";
                float coordWidth = ImGui.CalcTextSize(coordText).X;
                float centerX = (io.DisplaySize.X - coordWidth) * 0.5f;
                ImGui.SameLine(centerX);
                ImGui.TextColored(new Vector4(0.7f, 0.85f, 1f, 1f), coordText);
            }

            string fpsText = $"{_currentFps:F0} FPS  {_frameTimeMs:F1} ms";
            float textWidth = ImGui.CalcTextSize(fpsText).X;
            ImGui.SameLine(io.DisplaySize.X - textWidth - 16);
            var fpsColor = _currentFps >= 30 ? new Vector4(0.4f, 1f, 0.4f, 1f)
                         : _currentFps >= 15 ? new Vector4(1f, 1f, 0.4f, 1f)
                         : new Vector4(1f, 0.4f, 0.4f, 1f);
            ImGui.TextColored(fpsColor, fpsText);
        }
        ImGui.End();
        ImGui.PopStyleVar();
    }

    private void DrawMinimapWindow()
    {
        List<(int tx, int ty)>? existingTiles = null;
        Func<int, int, bool>? isTileLoaded = null;
        int loadedTileCount = 0;
        string? mapName = null;
        bool hasWorldLoaded = false;

        if (_terrainManager != null)
        {
            var adapter = _terrainManager.Adapter;
            existingTiles = adapter.ExistingTiles.Select(idx => (idx / 64, idx % 64)).ToList();
            isTileLoaded = _terrainManager.IsTileLoaded;
            loadedTileCount = _terrainManager.LoadedTileCount;
            mapName = _terrainManager.MapName;
            hasWorldLoaded = true;
        }
        else if (_vlmTerrainManager != null)
        {
            existingTiles = _vlmTerrainManager.Loader.TileCoords.ToList();
            isTileLoaded = _vlmTerrainManager.IsTileLoaded;
            loadedTileCount = _vlmTerrainManager.LoadedTileCount;
            mapName = _vlmTerrainManager.MapName;
            hasWorldLoaded = true;
        }

        var io = ImGui.GetIO();

        if (!_useDockspaceUi)
        {
            float rightOffset = _showRightSidebar ? SidebarWidth + 20 : 20;
            ImGui.SetNextWindowSize(new Vector2(360, 360), ImGuiCond.FirstUseEver);
            ImGui.SetNextWindowSizeConstraints(new Vector2(300, 300), new Vector2(520, 520));
            ImGui.SetNextWindowPos(new Vector2(io.DisplaySize.X - 360 - rightOffset, MenuBarHeight + ToolbarHeight + 20), ImGuiCond.FirstUseEver);
        }

        if (!ImGui.Begin("Minimap", ref _showMinimapWindow,
            ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoScrollbar | ImGuiWindowFlags.NoScrollWithMouse))
        {
            ImGui.End();
            return;
        }

        if (_useDockspaceUi)
            CaptureDockPanelState(ref _minimapDockState);

        if (!hasWorldLoaded)
        {
            ImGui.TextWrapped("Load a world map or VLM project to activate the minimap.");
            ImGui.Spacing();
            ImGui.TextDisabled("Once a world is loaded, the minimap will show loaded tiles, support zoom and pan, and allow triple-click teleport.");
            ImGui.Spacing();

            if (ImGui.Button("Open Game Folder..."))
            {
                _showFolderInput = true;
                _folderInputBuf = string.IsNullOrWhiteSpace(_lastGameFolderPath) ? "" : _lastGameFolderPath;
            }

            ImGui.SameLine();
            if (ImGui.Button("Open File..."))
                _wantOpenFile = true;

            if (_dataSource != null)
            {
                ImGui.Separator();
                ImGui.Text($"Source: {_dataSource.Name}");
                if (_discoveredMaps.Count > 0)
                    ImGui.TextDisabled($"Discovered maps: {_discoveredMaps.Count}");
            }

            ImGui.End();
            return;
        }

        float camTileX = (WoWConstants.MapOrigin - _camera.Position.X) / MinimapWorldTileSize;
        float camTileY = (WoWConstants.MapOrigin - _camera.Position.Y) / MinimapWorldTileSize;
        ClampMinimapPanOffset();
        int ctX = (int)MathF.Floor(camTileX);
        int ctY = (int)MathF.Floor(camTileY);

        ImGui.Text($"Tile: ({ctX},{ctY})");
        ImGui.SameLine();
        if (ImGui.SmallButton("-##minimapZoomOut"))
            _minimapZoom = Math.Clamp(_minimapZoom + 0.5f, 1f, 32f);
        ImGui.SameLine();
        if (ImGui.SmallButton("+##minimapZoomIn"))
            _minimapZoom = Math.Clamp(_minimapZoom - 0.5f, 1f, 32f);
        ImGui.SameLine();
        ImGui.TextDisabled($"Zoom {_minimapZoom:F1}x");
        ImGui.SameLine();
        ImGui.TextDisabled($"Loaded {loadedTileCount}");

        if (_minimapRenderer != null && (_minimapRenderer.IsBusy || _minimapRenderer.UploadedTileCount > 0 || _minimapRenderer.FailedTileCount > 0))
        {
            float progress = _minimapRenderer.LoadingProgress;
            string overlay = _minimapRenderer.IsBusy
                ? $"Minimap {progress * 100f:F0}%  {_minimapRenderer.PendingTileCount} queued"
                : $"Minimap ready  {_minimapRenderer.UploadedTileCount} tiles";
            ImGui.ProgressBar(progress, new Vector2(MathF.Min(220f, ImGui.GetContentRegionAvail().X), 0f), overlay);
            if (_minimapRenderer.FailedTileCount > 0)
                ImGui.TextDisabled($"Missing or failed tiles: {_minimapRenderer.FailedTileCount}");
        }

        float controlsHeight = ImGui.GetCursorPosY() + 8f;
        float mapAvailableWidth = ImGui.GetContentRegionAvail().X;
        float mapAvailableHeight = ImGui.GetContentRegionAvail().Y - 4f;
        float mapSize = MathF.Max(64f, MathF.Min(mapAvailableWidth, mapAvailableHeight));

        var cursorPos = ImGui.GetCursorScreenPos();

        if (ImGui.IsWindowHovered() && ImGui.IsMouseHoveringRect(cursorPos, cursorPos + new Vector2(mapSize, mapSize)))
        {
            float wheel = io.MouseWheel;
            if (wheel != 0)
                _minimapZoom = Math.Clamp(_minimapZoom - wheel * 0.5f, 1f, 32f);
        }

        MinimapHelpers.RenderMinimapContent(
            cursorPos, mapSize, existingTiles, isTileLoaded, _minimapRenderer, mapName,
            camTileX, camTileY, _minimapZoom, _minimapPanOffset, _camera, _worldScene,
            out float viewMinTx, out float viewMinTy, out float cellSize);

        HandleMinimapInteraction("##minimapInteraction", cursorPos, mapSize, viewMinTx, viewMinTy, cellSize);

        ImGui.SetCursorPosY(controlsHeight + mapSize + 2f);

        ImGui.End();
    }

    private void DrawFullscreenMinimap()
    {
        List<(int tx, int ty)>? existingTiles = null;
        Func<int, int, bool>? isTileLoaded = null;
        int loadedTileCount = 0;
        string? mapName = null;

        if (_terrainManager != null)
        {
            var adapter = _terrainManager.Adapter;
            existingTiles = adapter.ExistingTiles.Select(idx => (idx / 64, idx % 64)).ToList();
            isTileLoaded = _terrainManager.IsTileLoaded;
            loadedTileCount = _terrainManager.LoadedTileCount;
            mapName = _terrainManager.MapName;
        }
        else if (_vlmTerrainManager != null)
        {
            existingTiles = _vlmTerrainManager.Loader.TileCoords.ToList();
            isTileLoaded = _vlmTerrainManager.IsTileLoaded;
            loadedTileCount = _vlmTerrainManager.LoadedTileCount;
            mapName = _vlmTerrainManager.MapName;
        }
        else return;

        var io = ImGui.GetIO();
        float mapSize = MathF.Min(io.DisplaySize.X * 0.8f, io.DisplaySize.Y * 0.8f);
        float padding = (io.DisplaySize.X - mapSize) * 0.5f;
        float topPadding = (io.DisplaySize.Y - mapSize) * 0.5f;

        ImGui.SetNextWindowPos(Vector2.Zero);
        ImGui.SetNextWindowSize(io.DisplaySize);
        ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, Vector2.Zero);
        ImGui.PushStyleColor(ImGuiCol.WindowBg, new Vector4(0, 0, 0, 0.85f));

        if (ImGui.Begin("##FullscreenMinimap", ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.NoResize |
            ImGuiWindowFlags.NoMove | ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoSavedSettings |
            ImGuiWindowFlags.NoScrollbar))
        {
            ImGui.SetCursorPos(new Vector2(padding, topPadding));
            var cursorPos = ImGui.GetCursorScreenPos();

            if (ImGui.IsWindowHovered())
            {
                float wheel = io.MouseWheel;
                if (wheel != 0)
                    _minimapZoom = Math.Clamp(_minimapZoom - wheel * 0.5f, 1f, 32f);
            }

            float camTileX = (WoWConstants.MapOrigin - _camera.Position.X) / MinimapWorldTileSize;
            float camTileY = (WoWConstants.MapOrigin - _camera.Position.Y) / MinimapWorldTileSize;
            ClampMinimapPanOffset();

            MinimapHelpers.RenderMinimapContent(
                cursorPos, mapSize, existingTiles, isTileLoaded, _minimapRenderer, mapName,
                camTileX, camTileY, _minimapZoom, _minimapPanOffset, _camera, _worldScene,
                out float viewMinTx, out float viewMinTy, out float cellSize);

            HandleMinimapInteraction("##fullscreenMinimapInteraction", cursorPos, mapSize, viewMinTx, viewMinTy, cellSize);

            ImGui.SetCursorPos(new Vector2(padding, topPadding + mapSize + 10));
            int ctX = (int)MathF.Floor(camTileX);
            int ctY = (int)MathF.Floor(camTileY);
            ImGui.TextColored(new Vector4(1, 1, 1, 1), $"Tile: ({ctX},{ctY})  Zoom: {_minimapZoom:F1}x  Loaded: {loadedTileCount}");
            if (_minimapRenderer != null && (_minimapRenderer.IsBusy || _minimapRenderer.UploadedTileCount > 0))
            {
                float progress = _minimapRenderer.LoadingProgress;
                ImGui.ProgressBar(progress, new Vector2(MathF.Min(260f, mapSize * 0.45f), 0f),
                    _minimapRenderer.IsBusy
                        ? $"Minimap {progress * 100f:F0}%"
                        : "Minimap ready");
            }
            ImGui.SameLine();
            ImGui.TextColored(new Vector4(0.7f, 0.7f, 0.7f, 1), $"  |  Press M to close  |  Scroll to zoom  |  Drag to pan  |  Triple-click same tile to teleport");

            if (_minimapPanOffset != Vector2.Zero)
            {
                ImGui.SameLine();
                if (ImGui.SmallButton("Reset Pan"))
                    _minimapPanOffset = Vector2.Zero;
            }
        }
        ImGui.End();
        ImGui.PopStyleColor();
        ImGui.PopStyleVar();
    }
}