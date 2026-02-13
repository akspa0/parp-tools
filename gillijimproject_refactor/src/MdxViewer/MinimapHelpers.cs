using System.Numerics;
using ImGuiNET;
using MdxViewer.Rendering;
using MdxViewer.Terrain;

namespace MdxViewer;

/// <summary>
/// Helper methods for minimap rendering shared between sidebar and fullscreen views
/// </summary>
internal static class MinimapHelpers
{
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
        viewMinTx = camTileX - viewRadius + panOffset.X;
        float viewMaxTx = camTileX + viewRadius + panOffset.X;
        viewMinTy = camTileY - viewRadius + panOffset.Y;
        float viewMaxTy = camTileY + viewRadius + panOffset.Y;
        float viewSpan = viewRadius * 2f;
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
                uint tileTex = minimapRenderer.GetTileTexture(mapName, ty, tx);
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
        float camOffsetX = (camTileX - viewMinTx) * cellSize;
        float camOffsetY = (camTileY - viewMinTy) * cellSize;
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
                float poiTileX = (WoWConstants.MapOrigin - poi.Position.X) / WoWConstants.ChunkSize;
                float poiTileY = (WoWConstants.MapOrigin - poi.Position.Y) / WoWConstants.ChunkSize;
                float px = cursorPos.X + (poiTileY - viewMinTy) * cellSize;
                float py = cursorPos.Y + (poiTileX - viewMinTx) * cellSize;
                if (px >= cursorPos.X && px <= cursorPos.X + mapSize && py >= cursorPos.Y && py <= cursorPos.Y + mapSize)
                    drawList.AddCircleFilled(new Vector2(px, py), MathF.Max(2.5f, cellSize * 0.15f), 0xFFFF00FF);
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
                    float ax = cursorPos.X + ((WoWConstants.MapOrigin - a.Y) / WoWConstants.ChunkSize - viewMinTy) * cellSize;
                    float ay = cursorPos.Y + ((WoWConstants.MapOrigin - a.X) / WoWConstants.ChunkSize - viewMinTx) * cellSize;
                    float bx = cursorPos.X + ((WoWConstants.MapOrigin - b.Y) / WoWConstants.ChunkSize - viewMinTy) * cellSize;
                    float by = cursorPos.Y + ((WoWConstants.MapOrigin - b.X) / WoWConstants.ChunkSize - viewMinTx) * cellSize;
                    drawList.AddLine(new Vector2(ax, ay), new Vector2(bx, by), 0xFFFFFF00, 1.5f);
                }
            }
            foreach (var node in worldScene.TaxiLoader.Nodes)
            {
                if (!worldScene.IsTaxiNodeVisible(node)) continue;
                float nx = cursorPos.X + ((WoWConstants.MapOrigin - node.Position.Y) / WoWConstants.ChunkSize - viewMinTy) * cellSize;
                float ny = cursorPos.Y + ((WoWConstants.MapOrigin - node.Position.X) / WoWConstants.ChunkSize - viewMinTx) * cellSize;
                if (nx >= cursorPos.X && nx <= cursorPos.X + mapSize && ny >= cursorPos.Y && ny <= cursorPos.Y + mapSize)
                    drawList.AddCircleFilled(new Vector2(nx, ny), MathF.Max(3f, cellSize * 0.2f), 0xFF00FFFF);
            }
        }

        drawList.PopClipRect();

        // Border
        drawList.AddRect(cursorPos, cursorPos + new Vector2(mapSize, mapSize), 0xFF666666);
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
                float worldX = WoWConstants.MapOrigin - clickTileX * WoWConstants.ChunkSize;
                float worldY = WoWConstants.MapOrigin - clickTileY * WoWConstants.ChunkSize;
                camera.Position = new Vector3(worldX, worldY, camera.Position.Z);
            }
        }
    }
}
