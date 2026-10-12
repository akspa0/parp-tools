using System.Numerics;
using ImGuiNET;
using WoWViewer.Terrain;
using WowViewer.Core.Maps;

namespace WoWViewer.UI;

/// <summary>
/// Operational mode for the fullscreen minimap surface.
/// </summary>
public enum FullscreenMinimapMode
{
    Navigate,
    DonorTileTool
}

/// <summary>
/// Spec 236: Owned service class providing interactive donor tile selection and placement
/// directly from the large/fullscreen minimap surface (Spec 228 god-class freeze compliant).
/// </summary>
public class MinimapDonorToolService
{
    public FullscreenMinimapMode Mode { get; set; } = FullscreenMinimapMode.Navigate;
    public (int tx, int ty)? SelectedDonorSource { get; set; }
    public (int tx, int ty)? HoveredTile { get; set; }
    public string StatusNotification { get; set; } = "";
    private double _notificationExpiryTime = 0.0;

    public void ToggleMode()
    {
        Mode = Mode == FullscreenMinimapMode.Navigate
            ? FullscreenMinimapMode.DonorTileTool
            : FullscreenMinimapMode.Navigate;

        if (Mode == FullscreenMinimapMode.DonorTileTool)
        {
            SetNotification("Donor Tile Tool active: Left-click tile to select as Donor Source, then Left-click target slot to place.");
        }
        else
        {
            SetNotification("Navigation Mode active: Drag to pan, triple-click to teleport.");
        }
    }

    public void SetNotification(string message, double durationSeconds = 4.5)
    {
        StatusNotification = message;
        _notificationExpiryTime = ImGui.GetTime() + durationSeconds;
    }

    public void Update()
    {
        if (!string.IsNullOrEmpty(StatusNotification) && ImGui.GetTime() > _notificationExpiryTime)
        {
            StatusNotification = "";
        }
    }

    /// <summary>
    /// Renders the top toolbar in the fullscreen minimap.
    /// </summary>
    public void DrawToolbar(WorldScene worldScene, TerrainManager? terrainManager)
    {
        Update();

        // 1. Mode switcher button
        bool isToolActive = Mode == FullscreenMinimapMode.DonorTileTool;
        if (isToolActive)
        {
            ImGui.PushStyleColor(ImGuiCol.Button, new Vector4(0.12f, 0.45f, 0.22f, 1f));
            ImGui.PushStyleColor(ImGuiCol.ButtonHovered, new Vector4(0.18f, 0.58f, 0.28f, 1f));
            ImGui.PushStyleColor(ImGuiCol.ButtonActive, new Vector4(0.10f, 0.38f, 0.18f, 1f));
        }
        else
        {
            ImGui.PushStyleColor(ImGuiCol.Button, new Vector4(0.24f, 0.24f, 0.28f, 1f));
            ImGui.PushStyleColor(ImGuiCol.ButtonHovered, new Vector4(0.32f, 0.32f, 0.38f, 1f));
            ImGui.PushStyleColor(ImGuiCol.ButtonActive, new Vector4(0.20f, 0.20f, 0.24f, 1f));
        }

        string modeLabel = isToolActive
            ? "TOOL: DONOR TILE PLACER [ACTIVE] (Press T to switch)"
            : "MODE: NAVIGATE (Press T to switch to Donor Placer)";

        if (ImGui.Button(modeLabel))
        {
            ToggleMode();
        }
        ImGui.PopStyleColor(3);

        if (!isToolActive)
            return;

        ImGui.SameLine();
        ImGui.TextDisabled("|");
        ImGui.SameLine();

        // 2. Active Phase Layer selection
        IReadOnlyList<PhaseLayerSettings> layers = worldScene.PhaseLayers;
        if (layers.Count == 0)
        {
            ImGui.TextColored(new Vector4(1f, 0.4f, 0.4f, 1f), "No Phase Layers Loaded");
            return;
        }

        int selectedIndex = Math.Clamp(worldScene.SelectedPhaseLayerIndex, 0, layers.Count - 1);
        PhaseLayerSettings activeLayer = layers[selectedIndex];

        ImGui.SetNextItemWidth(220f);
        if (ImGui.BeginCombo("##activePhaseLayerCombo", $"Layer: {activeLayer.MapName}"))
        {
            for (int i = 0; i < layers.Count; i++)
            {
                bool isSelected = i == worldScene.SelectedPhaseLayerIndex;
                if (ImGui.Selectable($"#{i}: {layers[i].MapName}##layerSelect{i}", isSelected))
                {
                    worldScene.SelectedPhaseLayerIndex = i;
                }
                if (isSelected)
                    ImGui.SetItemDefaultFocus();
            }
            ImGui.EndCombo();
        }

        ImGui.SameLine();
        ImGui.TextDisabled("|");
        ImGui.SameLine();

        // 3. Selected Donor Source status and clear button
        if (SelectedDonorSource.HasValue)
        {
            ImGui.TextColored(new Vector4(0.2f, 1f, 0.3f, 1f), $"Source: ({SelectedDonorSource.Value.tx},{SelectedDonorSource.Value.ty})");
            ImGui.SameLine();
            if (ImGui.SmallButton("Clear Source##clearSrc"))
            {
                SelectedDonorSource = null;
                SetNotification("Cleared donor source.");
            }
        }
        else
        {
            ImGui.TextColored(new Vector4(1f, 0.85f, 0.2f, 1f), "Click any tile to select Donor Source");
        }

        ImGui.SameLine();
        ImGui.TextDisabled("|");
        ImGui.SameLine();

        // 4. Placed-Only toggle and Clear All button
        ImGui.TextDisabled($"Placements: {activeLayer.TilePlacements.Count}");
        ImGui.SameLine();
        bool placedOnly = activeLayer.UsePlacedTilesOnly;
        if (ImGui.Checkbox("Placed-Only##placedOnlyToggle", ref placedOnly))
        {
            activeLayer.UsePlacedTilesOnly = placedOnly;
            terrainManager?.RefreshPhaseLayers();
            SetNotification($"Layer '{activeLayer.MapName}' Placed-Only mode: {placedOnly}");
        }

        if (activeLayer.TilePlacements.Count > 0)
        {
            ImGui.SameLine();
            if (ImGui.SmallButton("Clear Placements##clearPlacements"))
            {
                activeLayer.TilePlacements.Clear();
                activeLayer.UsePlacedTilesOnly = false;
                terrainManager?.RefreshPhaseLayers();
                SetNotification($"Cleared all placed tiles for layer '{activeLayer.MapName}'.");
            }
        }
    }

    /// <summary>
    /// Handles minimap mouse clicks in DonorTileTool mode.
    /// Returns true if the click was consumed by this tool.
    /// </summary>
    public bool HandleClick(
        WorldScene worldScene,
        TerrainManager? terrainManager,
        float clickTx,
        float clickTy,
        bool isRightClick)
    {
        if (Mode != FullscreenMinimapMode.DonorTileTool)
            return false;

        int tx = (int)MathF.Floor(clickTx);
        int ty = (int)MathF.Floor(clickTy);
        if (tx < 0 || tx > 63 || ty < 0 || ty > 63)
            return false;

        IReadOnlyList<PhaseLayerSettings> layers = worldScene.PhaseLayers;
        if (layers.Count == 0)
        {
            SetNotification("No phase layer exists to receive placements.");
            return true;
        }

        int selectedIndex = Math.Clamp(worldScene.SelectedPhaseLayerIndex, 0, layers.Count - 1);
        PhaseLayerSettings layer = layers[selectedIndex];

        if (isRightClick)
        {
            if (SelectedDonorSource.HasValue)
            {
                SelectedDonorSource = null;
                SetNotification("Cancelled donor source selection.");
            }
            else
            {
                int removed = 0;
                for (int i = layer.TilePlacements.Count - 1; i >= 0; i--)
                {
                    if (layer.TilePlacements[i].TargetTileX == tx && layer.TilePlacements[i].TargetTileY == ty)
                    {
                        layer.TilePlacements.RemoveAt(i);
                        removed++;
                    }
                }
                if (removed > 0)
                {
                    terrainManager?.RefreshPhaseLayers();
                    SetNotification($"Removed placement at target ({tx}, {ty}).");
                }
            }
            return true;
        }

        // Left-click
        if (!SelectedDonorSource.HasValue)
        {
            SelectedDonorSource = (tx, ty);
            SetNotification($"Selected donor source tile ({tx}, {ty}). Now click target slot to place.");
            return true;
        }

        // Place tile
        int sx = SelectedDonorSource.Value.tx;
        int sy = SelectedDonorSource.Value.ty;

        for (int i = layer.TilePlacements.Count - 1; i >= 0; i--)
        {
            if (layer.TilePlacements[i].TargetTileX == tx && layer.TilePlacements[i].TargetTileY == ty)
            {
                layer.TilePlacements.RemoveAt(i);
            }
        }
        layer.TilePlacements.Add(new PhaseTilePlacement(sx, sy, tx, ty));
        layer.UsePlacedTilesOnly = true;
        layer.Enabled = true;

        terrainManager?.RefreshPhaseLayers();
        SetNotification($"Placed donor ({sx}, {sy}) -> target ({tx}, {ty})! Layer switched to Placed-Only.");
        return true;
    }

    /// <summary>
    /// Renders overlays for donor source, target placements, preview outlines, and status alerts.
    /// </summary>
    public void RenderOverlays(
        ImDrawListPtr drawList,
        Vector2 cursorPos,
        float viewMinTx,
        float viewMinTy,
        float cellSize,
        float mapSize,
        WorldScene worldScene)
    {
        if (worldScene == null)
            return;

        IReadOnlyList<PhaseLayerSettings> layers = worldScene.PhaseLayers;
        if (layers.Count == 0)
            return;

        int selectedIndex = Math.Clamp(worldScene.SelectedPhaseLayerIndex, 0, layers.Count - 1);
        PhaseLayerSettings layer = layers[selectedIndex];

        // 1. Draw existing placed tiles for the active layer
        foreach (PhaseTilePlacement placement in layer.TilePlacements)
        {
            int tgtTx = placement.TargetTileX;
            int tgtTy = placement.TargetTileY;
            float tgtX = cursorPos.X + (tgtTy - viewMinTy) * cellSize;
            float tgtY = cursorPos.Y + (tgtTx - viewMinTx) * cellSize;

            if (tgtTx + 1 >= viewMinTx && tgtTx <= viewMinTx + mapSize / cellSize &&
                tgtTy + 1 >= viewMinTy && tgtTy <= viewMinTy + mapSize / cellSize)
            {
                var pMin = new Vector2(tgtX, tgtY);
                var pMax = new Vector2(tgtX + cellSize, tgtY + cellSize);

                drawList.AddRectFilled(pMin, pMax, 0x33FFD700);
                drawList.AddRect(pMin, pMax, 0xFFFFD700, 0f, ImDrawFlags.None, 2.5f);

                string badge = $"P:({placement.DonorTileX},{placement.DonorTileY})";
                drawList.AddRectFilled(pMin, pMin + new Vector2(MathF.Min(cellSize, 58f), 14f), 0xCCB8860B);
                drawList.AddText(pMin + new Vector2(2f, 0f), 0xFFFFFFFF, badge);
            }

            int srcTx = placement.DonorTileX;
            int srcTy = placement.DonorTileY;
            float srcX = cursorPos.X + (srcTy - viewMinTy) * cellSize;
            float srcY = cursorPos.Y + (srcTx - viewMinTx) * cellSize;

            if (srcTx + 1 >= viewMinTx && srcTx <= viewMinTx + mapSize / cellSize &&
                srcTy + 1 >= viewMinTy && srcTy <= viewMinTy + mapSize / cellSize)
            {
                var sMin = new Vector2(srcX, srcY);
                var sMax = new Vector2(srcX + cellSize, srcY + cellSize);
                drawList.AddRect(sMin, sMax, 0xFF00FF7F, 0f, ImDrawFlags.None, 1.5f);

                // Line connecting donor center to target center
                Vector2 sCenter = sMin + new Vector2(cellSize * 0.5f, cellSize * 0.5f);
                Vector2 tCenter = new Vector2(tgtX + cellSize * 0.5f, tgtY + cellSize * 0.5f);
                drawList.AddLine(sCenter, tCenter, 0x77FFD700, 1.5f);
                drawList.AddCircleFilled(tCenter, 3.5f, 0xFFFFD700);
            }
        }

        // 2. Selected Donor Source indicator
        if (SelectedDonorSource.HasValue)
        {
            int stx = SelectedDonorSource.Value.tx;
            int sty = SelectedDonorSource.Value.ty;
            float sx = cursorPos.X + (sty - viewMinTy) * cellSize;
            float sy = cursorPos.Y + (stx - viewMinTx) * cellSize;

            if (stx + 1 >= viewMinTx && stx <= viewMinTx + mapSize / cellSize &&
                sty + 1 >= viewMinTy && sty <= viewMinTy + mapSize / cellSize)
            {
                var sMin = new Vector2(sx, sy);
                var sMax = new Vector2(sx + cellSize, sy + cellSize);

                float pulse = (MathF.Sin((float)ImGui.GetTime() * 6f) + 1f) * 0.5f;
                uint pulseColor = FootprintColor(0.2f + 0.8f * pulse, 1.0f, 0.2f, 0xFF);

                drawList.AddRect(sMin, sMax, pulseColor, 0f, ImDrawFlags.None, 3.5f);
                drawList.AddRectFilled(sMin, sMin + new Vector2(MathF.Min(cellSize, 34f), 15f), 0xFF008800);
                drawList.AddText(sMin + new Vector2(2f, 0f), 0xFFFFFFFF, "SRC");

                // If hovered over a target, draw target preview and connector line
                if (Mode == FullscreenMinimapMode.DonorTileTool && HoveredTile.HasValue)
                {
                    int htx = HoveredTile.Value.tx;
                    int hty = HoveredTile.Value.ty;
                    float hx = cursorPos.X + (hty - viewMinTy) * cellSize;
                    float hy = cursorPos.Y + (htx - viewMinTx) * cellSize;

                    var hMin = new Vector2(hx, hy);
                    var hMax = new Vector2(hx + cellSize, hy + cellSize);

                    drawList.AddRectFilled(hMin, hMax, 0x4400FFFF);
                    drawList.AddRect(hMin, hMax, 0xFF00FFFF, 0f, ImDrawFlags.None, 2.5f);

                    Vector2 sCenter = sMin + new Vector2(cellSize * 0.5f, cellSize * 0.5f);
                    Vector2 hCenter = hMin + new Vector2(cellSize * 0.5f, cellSize * 0.5f);
                    drawList.AddLine(sCenter, hCenter, 0xFF00FFFF, 2.0f);
                    drawList.AddCircleFilled(hCenter, 4f, 0xFF00FFFF);

                    drawList.AddRectFilled(hMin, hMin + new Vector2(MathF.Min(cellSize, 52f), 15f), 0xCC008888);
                    drawList.AddText(hMin + new Vector2(2f, 0f), 0xFFFFFFFF, "TARGET");
                }
            }
        }

        // 3. Status Notification Toast
        if (!string.IsNullOrEmpty(StatusNotification))
        {
            Vector2 bannerSize = new Vector2(mapSize - 32f, 28f);
            Vector2 bannerPos = new Vector2(cursorPos.X + 16f, cursorPos.Y + mapSize - 38f);
            drawList.AddRectFilled(bannerPos, bannerPos + bannerSize, 0xDD181818, 4f);
            drawList.AddRect(bannerPos, bannerPos + bannerSize, 0xFF00FF7F, 4f, ImDrawFlags.None, 1.5f);
            drawList.AddText(bannerPos + new Vector2(10f, 6f), 0xFFFFFFFF, StatusNotification);
        }
    }

    private static uint FootprintColor(float r, float g, float b, uint alpha)
    {
        uint ri = (uint)(Math.Clamp(r, 0f, 1f) * 255f);
        uint gi = (uint)(Math.Clamp(g, 0f, 1f) * 255f);
        uint bi = (uint)(Math.Clamp(b, 0f, 1f) * 255f);
        return (alpha << 24) | (bi << 16) | (gi << 8) | ri;
    }
}
