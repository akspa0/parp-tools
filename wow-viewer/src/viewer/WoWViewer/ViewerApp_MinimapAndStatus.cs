using System.Numerics;
using ImGuiNET;
using WoWViewer.Rendering;
using WoWViewer.Terrain;

namespace WoWViewer;

/// <summary>
/// Partial class containing status and minimap windows.
/// </summary>
public partial class ViewerApp
{
    private const float MinimapTileCount = 64f;
    private const float MinimapWorldTileSize = WoWConstants.ChunkSize;

    private enum MinimapTeleportMode
    {
        Armed,
        Immediate,
    }

    private bool TryGetActiveMinimapState(
        out List<(int tx, int ty)>? existingTiles,
        out Func<int, int, bool>? isTileLoaded,
        out int loadedTileCount,
        out string? mapName)
    {
        existingTiles = null;
        isTileLoaded = null;
        loadedTileCount = 0;
        mapName = null;

        if (_terrainManager != null)
        {
            var adapter = _terrainManager.Adapter;
            existingTiles = adapter.ExistingTiles.Select(idx => (idx / 64, idx % 64)).ToList();
            isTileLoaded = _terrainManager.IsTileLoaded;
            loadedTileCount = _terrainManager.LoadedTileCount;
            mapName = _terrainManager.MapName;
            return true;
        }

        if (_vlmTerrainManager != null)
        {
            existingTiles = _vlmTerrainManager.Loader.TileCoords.ToList();
            isTileLoaded = _vlmTerrainManager.IsTileLoaded;
            loadedTileCount = _vlmTerrainManager.LoadedTileCount;
            mapName = _vlmTerrainManager.MapName;
            return true;
        }

        return false;
    }

    private void HandleMinimapInteraction(string interactionId, Vector2 cursorPos, float mapSize, float viewMinTx, float viewMinTy, float cellSize, MinimapTeleportMode teleportMode)
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
                    && MinimapHelpers.TryGetLitMarkerAtPoint(_worldScene, mousePos, cursorPos, mapSize, viewMinTx, viewMinTy, cellSize, out int litLightIndex))
                {
                    _worldScene!.SelectedLitLightIndex = litLightIndex;
                    if (ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left))
                        FocusCameraOnLitLight(litLightIndex, closeFullscreenAfterFocus: true);
                    else
                        _statusMessage = $"Selected LIT entry [{litLightIndex}] from minimap.";
                }
                else if (totalDelta.Length() <= MinimapClickMovementThresholdPixels
                    && TryGetMinimapClickTarget(mousePos, cursorPos, cellSize, viewMinTx, viewMinTy, out float clickTileX, out float clickTileY))
                {
                    if (teleportMode == MinimapTeleportMode.Immediate)
                        TeleportCameraToMinimapTile(clickTileX, clickTileY, closeFullscreenAfterTeleport: true);
                    else
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

    private void TeleportCameraToMinimapTile(float clickTileX, float clickTileY, bool closeFullscreenAfterTeleport)
    {
        int tileX = (int)MathF.Floor(clickTileX);
        int tileY = (int)MathF.Floor(clickTileY);
        float worldX = WoWConstants.MapOrigin - clickTileX * MinimapWorldTileSize;
        float worldY = WoWConstants.MapOrigin - clickTileY * MinimapWorldTileSize;
        _camera.Position = new Vector3(worldX, worldY, _camera.Position.Z);
        _statusMessage = $"Minimap teleported camera to tile ({tileX},{tileY}).";

        if (closeFullscreenAfterTeleport && _fullscreenMinimap)
        {
            _fullscreenMinimap = false;
            _minimapDragging = false;
        }

        ClearPendingMinimapTeleport();
    }

    private void FocusCameraOnLitLight(int lightIndex, bool closeFullscreenAfterFocus)
    {
        if (_worldScene?.LitLoader is not { HasData: true } lit
            || lightIndex < 0
            || lightIndex >= lit.Lights.Count)
        {
            _statusMessage = "Selected LIT entry is unavailable.";
            return;
        }

        LitLoader.LitLight light = lit.Lights[lightIndex];
        if (!light.IsNavigable)
        {
            _statusMessage = $"LIT entry [{lightIndex}] has no navigable position.";
            return;
        }

        float distance = MathF.Max(80f, MathF.Min(MathF.Max(light.Radius, light.Dropoff), 250f));
        float height = MathF.Max(50f, distance * 0.58f);
        _worldScene.SelectedLitLightIndex = lightIndex;
        _camera.Position = light.Position + new Vector3(distance, 0f, height);
        _camera.Yaw = 180f;
        _camera.Pitch = -30f;
        _statusMessage = $"Focused LIT entry [{lightIndex}] {light.DisplayName}.";

        if (closeFullscreenAfterFocus && _fullscreenMinimap)
        {
            _fullscreenMinimap = false;
            _minimapDragging = false;
        }

        ClearPendingMinimapTeleport();
    }

    private static float ComputeMinimapSquareSize(float availableWidth, float availableHeight, float minimumSize)
    {
        return MathF.Max(minimumSize, MathF.Min(availableWidth, availableHeight));
    }

    private void DrawInteractiveMinimapSurface(
        string interactionId,
        Vector2 cursorPos,
        float mapSize,
        List<(int tx, int ty)> existingTiles,
        Func<int, int, bool> isTileLoaded,
        string? mapName,
        MinimapTeleportMode teleportMode,
        out float viewMinTx,
        out float viewMinTy,
        out float cellSize)
    {
        var io = ImGui.GetIO();
        if (ImGui.IsMouseHoveringRect(cursorPos, cursorPos + new Vector2(mapSize, mapSize)))
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
            out viewMinTx, out viewMinTy, out cellSize);

        HandleMinimapInteraction(interactionId, cursorPos, mapSize, viewMinTx, viewMinTy, cellSize, teleportMode);
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

        TeleportCameraToMinimapTile(clickTileX, clickTileY, closeFullscreenAfterTeleport: _fullscreenMinimap);
    }

    private void ToggleFullscreenMinimap()
    {
        _fullscreenMinimap = !_fullscreenMinimap;
        if (_fullscreenMinimap)
            PrepareFullscreenMinimapState();
        else
            _minimapDragging = false;
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
            string rightStatusText = string.IsNullOrWhiteSpace(_statusMessage)
                ? "Ready"
                : _statusMessage.Replace(Environment.NewLine, " ").Trim();

            string leftText = string.Empty;
            if (_terrainManager != null || _vlmTerrainManager != null)
            {
                var pos = _camera.Position;
                float wowX = WoWConstants.MapOrigin - pos.Y;
                float wowY = WoWConstants.MapOrigin - pos.X;
                float wowZ = pos.Z;
                float facingDegrees = GetWorldFacingDegrees(_camera.Yaw);
                string facingLabel = GetWorldFacingLabel(facingDegrees);
                leftText = $"Local: ({pos.X:F0}, {pos.Y:F0}, {pos.Z:F0})  WoW: ({wowX:F0}, {wowY:F0}, {wowZ:F0})  Facing: {facingDegrees:F1}° {facingLabel ?? string.Empty}";
                if (!string.IsNullOrWhiteSpace(_currentAreaName))
                    leftText = $"{leftText}  |  Area: {_currentAreaName}";
            }
            else if (!string.IsNullOrWhiteSpace(_currentAreaName))
            {
                leftText = $"Area: {_currentAreaName}";
            }

            float leftWidth = string.IsNullOrEmpty(leftText) ? 0f : GetImGuiTextWidth(leftText) + 8f;
            float rightWidth = GetImGuiTextWidth(rightStatusText) + 8f;

            if (!string.IsNullOrEmpty(leftText))
                ImGui.TextUnformatted(leftText);

            // Push right text to the right edge
            float pad = io.DisplaySize.X - rightWidth - 16f;
            if (pad > 0f)
            {
                ImGui.SameLine();
                ImGui.SetCursorPosX(pad);
            }
            ImGui.TextUnformatted(rightStatusText);
        }
        ImGui.End();
        ImGui.PopStyleVar();
    }
    private static float GetImGuiTextWidth(string? text)
    {
        return string.IsNullOrEmpty(text) ? 0f : ImGui.CalcTextSize(text).X;
    }

    private void DrawMinimapWindow()
    {
        bool hasWorldLoaded = TryGetActiveMinimapState(out var existingTiles, out var isTileLoaded, out int loadedTileCount, out string? mapName);

        var io = ImGui.GetIO();
        var panel = GetShellPanelDefinition(ShellPanelId.Minimap);

        if (!_useDockspaceUi)
        {
            float rightOffset = IsShellPanelActive(ShellPanelId.Inspector) ? _rightSidebarWidth + 20 : 20;
            ImGui.SetNextWindowSize(new Vector2(panel.DefaultWidth, panel.DefaultWidth), ImGuiCond.FirstUseEver);
            ImGui.SetNextWindowSizeConstraints(new Vector2(panel.MinWidth, panel.MinWidth), new Vector2(panel.MaxWidth, panel.MaxWidth));
            ImGui.SetNextWindowPos(new Vector2(io.DisplaySize.X - panel.DefaultWidth - rightOffset, MenuBarHeight + ToolbarHeight + 20), ImGuiCond.FirstUseEver);
        }
        else
        {
            PrepareDockableShellPanelWindow(
                ShellPanelId.Minimap,
                new Vector2(panel.DefaultWidth, panel.DefaultWidth),
                new Vector2(panel.CompactMinWidth, panel.CompactMinWidth),
                new Vector2(panel.MaxWidth, panel.MaxWidth));
        }

        if (!ImGui.Begin("Minimap", ref _showMinimapWindow,
            ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoScrollbar | ImGuiWindowFlags.NoScrollWithMouse))
        {
            ImGui.End();
            return;
        }

        if (_useDockspaceUi)
            CaptureDockPanelState(ShellPanelId.Minimap);

        DrawMinimapContent(loadedTileCount, mapName, existingTiles, isTileLoaded);
        ImGui.End();
    }

    /// <summary>
    /// Headless minimap body (no Begin/End). Call from any container that already
    /// has an ImGui context active: docked panel, floating window, tab content
    /// region, fullscreen overlay. Caller manages the surrounding ImGui window.
    /// </summary>
    private void DrawMinimapContent(int loadedTileCount, string? mapName, List<(int tx, int ty)> existingTiles, Func<int, int, bool> isTileLoaded)
    {
        if (existingTiles == null || isTileLoaded == null)
        {
            ImGui.TextWrapped("Load a world map or MK dataset project to activate the minimap.");
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
                ? $"Minimap {progress * 100f:F0}%  {_minimapRenderer.PendingTileCount} pending"
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
        DrawInteractiveMinimapSurface(
            "##minimapInteraction",
            cursorPos,
            mapSize,
            existingTiles,
            isTileLoaded,
            mapName,
            MinimapTeleportMode.Armed,
            out _,
            out _,
            out _);

        ImGui.SetCursorPosY(controlsHeight + mapSize + 2f);
    }

    private void DrawFullscreenMinimap()
    {
        if (!TryGetActiveMinimapState(out var existingTiles, out var isTileLoaded, out int loadedTileCount, out string? mapName)) return;

        var io = ImGui.GetIO();
        const float horizontalMargin = 24f;
        const float verticalMargin = 24f;
        const float footerHeight = 60f;
        float mapSize = ComputeMinimapSquareSize(
            io.DisplaySize.X - horizontalMargin * 2f,
            io.DisplaySize.Y - verticalMargin * 2f - footerHeight,
            minimumSize: 128f);
        float padding = MathF.Max(horizontalMargin, (io.DisplaySize.X - mapSize) * 0.5f);
        float topPadding = MathF.Max(verticalMargin, (io.DisplaySize.Y - footerHeight - mapSize) * 0.5f);

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

            float camTileX = (WoWConstants.MapOrigin - _camera.Position.X) / MinimapWorldTileSize;
            float camTileY = (WoWConstants.MapOrigin - _camera.Position.Y) / MinimapWorldTileSize;

            DrawInteractiveMinimapSurface(
                "##fullscreenMinimapInteraction",
                cursorPos,
                mapSize,
                existingTiles,
                isTileLoaded,
                mapName,
                MinimapTeleportMode.Immediate,
                out _,
                out _,
                out _);

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
            ImGui.TextColored(new Vector4(0.7f, 0.7f, 0.7f, 1), "  |  Press M to close  |  Scroll to zoom  |  Drag to pan  |  Click tile to teleport");

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
