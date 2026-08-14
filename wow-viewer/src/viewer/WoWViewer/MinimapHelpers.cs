using System.Numerics;
using ImGuiNET;
using WoWViewer.Rendering;
using WoWViewer.Terrain;

namespace WoWViewer;

/// <summary>
/// Helper methods for minimap rendering shared between sidebar and fullscreen views
/// </summary>
internal static class MinimapHelpers
{
    private const float MapTileCount = 64f;
    private const float MinimapWorldTileSize = WoWConstants.ChunkSize;

    public static void RenderMinimapContent(
        Vector2 cursorPos,
        float mapSize,
        List<(int tx, int ty)> existingTiles,
        Func<int, int, bool> isTileLoaded,
        MinimapRenderer? minimapRenderer,
        string? mapName,
        float camTileX,
        float camTileY,
        float minimapZoom,
        Vector2 panOffset,
        Camera camera,
        WorldScene? worldScene,
        out float viewMinTx,
        out float viewMinTy,
        out float cellSize)
    {
        var drawList = ImGui.GetWindowDrawList();

        // View window: minimapZoom tiles in each direction from camera + pan offset
        float viewRadius = minimapZoom;
        float viewSpan = viewRadius * 2f;
        float maxViewMin = MathF.Max(0f, MapTileCount - viewSpan);
        viewMinTx = Math.Clamp(camTileX - viewRadius + panOffset.X, 0f, maxViewMin);
        float viewMaxTx = MathF.Min(MapTileCount, viewMinTx + viewSpan);
        viewMinTy = Math.Clamp(camTileY - viewRadius + panOffset.Y, 0f, maxViewMin);
        float viewMaxTy = MathF.Min(MapTileCount, viewMinTy + viewSpan);
        cellSize = mapSize / viewSpan;

        // Background
        drawList.AddRectFilled(cursorPos, cursorPos + new Vector2(mapSize, mapSize), 0xFF1A1A1A);

        // Clip to minimap area
        drawList.PushClipRect(cursorPos, cursorPos + new Vector2(mapSize, mapSize), true);

        // Draw existing tiles
        foreach (var (tx, ty) in existingTiles)
        {
            if (tx + 1 < viewMinTx || tx > viewMaxTx || ty + 1 < viewMinTy || ty > viewMaxTy)
                continue;

            float x = cursorPos.X + (ty - viewMinTy) * cellSize;
            float y = cursorPos.Y + (tx - viewMinTx) * cellSize;

            bool drewTexture = false;
            if (minimapRenderer != null && !string.IsNullOrEmpty(mapName))
            {
                string? overlayMap = worldScene?.SecondaryOverlayMap;
                uint tileTex = minimapRenderer.GetTileTexture(mapName, ty, tx, overlayMap);
                if (tileTex != 0)
                {
                    var texId = (IntPtr)tileTex;
                    var p1 = new Vector2(x, y);
                    var p2 = new Vector2(x + cellSize, y);
                    var p3 = new Vector2(x + cellSize, y + cellSize);
                    var p4 = new Vector2(x, y + cellSize);
                    drawList.AddImageQuad(texId, p1, p2, p3, p4,
                        new Vector2(0, 0), new Vector2(1, 0),
                        new Vector2(1, 1), new Vector2(0, 1),
                        0xFFFFFFFF);
                    drewTexture = true;
                }
            }

            if (!drewTexture)
            {
                bool loaded = isTileLoaded(tx, ty);
                uint color = loaded ? 0xFF00AA00 : 0xFF004400;
                drawList.AddRectFilled(new Vector2(x, y), new Vector2(x + cellSize, y + cellSize), color);
            }
        }

        // Camera position (centered, adjusted for pan)
        float clampedCamTileX = Math.Clamp(camTileX, 0f, MapTileCount);
        float clampedCamTileY = Math.Clamp(camTileY, 0f, MapTileCount);
        float camOffsetX = (clampedCamTileY - viewMinTy) * cellSize;
        float camOffsetY = (clampedCamTileX - viewMinTx) * cellSize;
        float camScreenX = cursorPos.X + camOffsetX;
        float camScreenY = cursorPos.Y + camOffsetY;

        // Camera direction indicator
        float yawRad = camera.Yaw * MathF.PI / 180f;
        float dirLen = mapSize * 0.08f;
        float dotRadius = mapSize * 0.02f;
        float dirX = camScreenX - MathF.Sin(yawRad) * dirLen;
        float dirY = camScreenY - MathF.Cos(yawRad) * dirLen;
        drawList.AddLine(new Vector2(camScreenX, camScreenY), new Vector2(dirX, dirY), 0xFFFFFF00, MathF.Max(2f, mapSize * 0.012f));
        drawList.AddCircleFilled(new Vector2(camScreenX, camScreenY), MathF.Max(3f, dotRadius), 0xFFFFFFFF);

        // POI markers
        if (worldScene?.PoiLoader != null && worldScene.ShowPoi)
        {
            foreach (var poi in worldScene.PoiLoader.Entries)
            {
                float poiTileX = (WoWConstants.MapOrigin - poi.Position.X) / MinimapWorldTileSize;
                float poiTileY = (WoWConstants.MapOrigin - poi.Position.Y) / MinimapWorldTileSize;
                float px = cursorPos.X + (poiTileY - viewMinTy) * cellSize;
                float py = cursorPos.Y + (poiTileX - viewMinTx) * cellSize;
                if (px >= cursorPos.X && px <= cursorPos.X + mapSize && py >= cursorPos.Y && py <= cursorPos.Y + mapSize)
                    drawList.AddCircleFilled(new Vector2(px, py), MathF.Max(2.5f, cellSize * 0.15f), 0xFFFF00FF);
            }
        }

        // LIT markers are a diagnostic layer only. They deliberately share the same loaded source
        // and selected index as the Lighting panel and never change fog or lighting selection.
        if (worldScene?.ShowLitMinimapMarkers == true && worldScene.LitLoader is { HasData: true } lit)
        {
            float timeOfDay = worldScene.LastLitSample?.TimeOfDay ?? 0.5f;
            for (int lightIndex = 0; lightIndex < lit.Lights.Count; lightIndex++)
            {
                LitLoader.LitLight light = lit.Lights[lightIndex];
                if (!TryGetLitMarkerPosition(light, cursorPos, mapSize, viewMinTx, viewMinTy, cellSize, out Vector2 markerPos))
                    continue;

                uint color = ToImGuiColor(lit.EvaluateOverlayColor(light, timeOfDay));
                bool selected = lightIndex == worldScene.SelectedLitLightIndex;
                float coverageRadius = Math.Clamp(
                    MathF.Max(3f, MathF.Max(light.Radius, light.Dropoff) / MinimapWorldTileSize * cellSize),
                    3f,
                    MathF.Max(3f, mapSize * 0.4f));
                float markerRadius = selected ? MathF.Max(5f, cellSize * 0.12f) : MathF.Max(3.5f, cellSize * 0.08f);

                // The ring alone disappears against authored minimap art. A low-alpha fog-color
                // fill makes the LIT radius readable without turning the diagnostic into a map tint.
                drawList.AddCircleFilled(markerPos, coverageRadius, WithAlpha(color, selected ? 0x38u : 0x24u), 32);
                drawList.AddCircle(markerPos, coverageRadius, color, 24, selected ? 2.5f : 1.25f);
                drawList.AddCircleFilled(markerPos, markerRadius, selected ? 0xFFFFFFFF : color);
                if (selected)
                    drawList.AddCircle(markerPos, markerRadius + 3f, color, 16, 1.75f);
            }
        }

        // Taxi paths
        if (worldScene?.TaxiLoader != null && worldScene.ShowTaxi)
        {
            foreach (var route in worldScene.TaxiLoader.Routes)
            {
                if (!worldScene.IsTaxiRouteVisible(route)) continue;
                for (int i = 0; i < route.Waypoints.Count - 1; i++)
                {
                    var a = route.Waypoints[i];
                    var b = route.Waypoints[i + 1];
                    float ax = cursorPos.X + ((WoWConstants.MapOrigin - a.Y) / MinimapWorldTileSize - viewMinTy) * cellSize;
                    float ay = cursorPos.Y + ((WoWConstants.MapOrigin - a.X) / MinimapWorldTileSize - viewMinTx) * cellSize;
                    float bx = cursorPos.X + ((WoWConstants.MapOrigin - b.Y) / MinimapWorldTileSize - viewMinTy) * cellSize;
                    float by = cursorPos.Y + ((WoWConstants.MapOrigin - b.X) / MinimapWorldTileSize - viewMinTx) * cellSize;
                    drawList.AddLine(new Vector2(ax, ay), new Vector2(bx, by), 0xFFFFFF00, 1.5f);
                }
            }
            foreach (var node in worldScene.TaxiLoader.Nodes)
            {
                if (!worldScene.IsTaxiNodeVisible(node)) continue;
                float nx = cursorPos.X + ((WoWConstants.MapOrigin - node.Position.Y) / MinimapWorldTileSize - viewMinTy) * cellSize;
                float ny = cursorPos.Y + ((WoWConstants.MapOrigin - node.Position.X) / MinimapWorldTileSize - viewMinTx) * cellSize;
                if (nx >= cursorPos.X && nx <= cursorPos.X + mapSize && ny >= cursorPos.Y && ny <= cursorPos.Y + mapSize)
                    drawList.AddCircleFilled(new Vector2(nx, ny), MathF.Max(3f, cellSize * 0.2f), 0xFF00FFFF);
            }
        }

        drawList.PopClipRect();

        // Border
        drawList.AddRect(cursorPos, cursorPos + new Vector2(mapSize, mapSize), 0xFF666666);
    }

    /// <summary>Returns the nearest visible positional LIT marker under a minimap point.</summary>
    public static bool TryGetLitMarkerAtPoint(
        WorldScene? worldScene,
        Vector2 mousePos,
        Vector2 cursorPos,
        float mapSize,
        float viewMinTx,
        float viewMinTy,
        float cellSize,
        out int lightIndex)
    {
        lightIndex = -1;
        if (worldScene?.ShowLitMinimapMarkers != true || worldScene.LitLoader is not { HasData: true } lit)
            return false;

        float bestDistanceSquared = float.MaxValue;
        float hitRadius = MathF.Max(8f, cellSize * 0.16f);
        float hitRadiusSquared = hitRadius * hitRadius;
        for (int index = 0; index < lit.Lights.Count; index++)
        {
            if (!TryGetLitMarkerPosition(lit.Lights[index], cursorPos, mapSize, viewMinTx, viewMinTy, cellSize, out Vector2 markerPos))
                continue;

            float distanceSquared = Vector2.DistanceSquared(mousePos, markerPos);
            if (distanceSquared <= hitRadiusSquared && distanceSquared < bestDistanceSquared)
            {
                lightIndex = index;
                bestDistanceSquared = distanceSquared;
            }
        }

        return lightIndex >= 0;
    }

    public static void HandleMinimapClick(
        Vector2 mousePos,
        Vector2 cursorPos,
        float mapSize,
        float viewMinTx,
        float viewMinTy,
        float cellSize,
        Camera camera,
        bool isDrag,
        ref Vector2 panOffset,
        ref Vector2 dragStart,
        ref bool dragging)
    {
        if (isDrag)
        {
            if (!dragging)
            {
                dragging = true;
                dragStart = mousePos;
            }
            else
            {
                Vector2 delta = mousePos - dragStart;
                panOffset -= new Vector2(delta.Y / cellSize, delta.X / cellSize);
                dragStart = mousePos;
            }
        }
        else
        {
            // Single click or double-click to teleport
            float clickTileY = (mousePos.X - cursorPos.X) / cellSize + viewMinTy;
            float clickTileX = (mousePos.Y - cursorPos.Y) / cellSize + viewMinTx;
            if (clickTileX >= 0 && clickTileX < 64 && clickTileY >= 0 && clickTileY < 64)
            {
                float worldX = WoWConstants.MapOrigin - clickTileX * MinimapWorldTileSize;
                float worldY = WoWConstants.MapOrigin - clickTileY * MinimapWorldTileSize;
                camera.Position = new Vector3(worldX, worldY, camera.Position.Z);
            }
        }
    }

    private static bool TryGetLitMarkerPosition(
        LitLoader.LitLight light,
        Vector2 cursorPos,
        float mapSize,
        float viewMinTx,
        float viewMinTy,
        float cellSize,
        out Vector2 markerPos)
    {
        markerPos = default;
        if (!light.IsNavigable)
            return false;

        float tileX = (WoWConstants.MapOrigin - light.Position.X) / MinimapWorldTileSize;
        float tileY = (WoWConstants.MapOrigin - light.Position.Y) / MinimapWorldTileSize;
        float x = cursorPos.X + (tileY - viewMinTy) * cellSize;
        float y = cursorPos.Y + (tileX - viewMinTx) * cellSize;
        if (x < cursorPos.X || x > cursorPos.X + mapSize || y < cursorPos.Y || y > cursorPos.Y + mapSize)
            return false;

        markerPos = new Vector2(x, y);
        return true;
    }

    private static uint ToImGuiColor(Vector3 color)
    {
        uint red = (uint)Math.Clamp((int)MathF.Round(color.X * 255f), 0, 255);
        uint green = (uint)Math.Clamp((int)MathF.Round(color.Y * 255f), 0, 255);
        uint blue = (uint)Math.Clamp((int)MathF.Round(color.Z * 255f), 0, 255);
        return 0xFF000000u | blue << 16 | green << 8 | red;
    }

    private static uint WithAlpha(uint color, uint alpha)
        => (color & 0x00FFFFFFu) | ((alpha & 0xFFu) << 24);
}
