using System.Numerics;
using ImGuiNET;

namespace MdxViewer;

/// <summary>
/// Partial class containing the WDL preview dialog for map spawn point selection.
/// </summary>
public partial class ViewerApp
{
    private void DrawWdlPreviewDialog()
    {
        if (_selectedMapForPreview == null || _wdlPreviewRenderer == null || !_wdlPreviewRenderer.HasPreview)
        {
            _showWdlPreview = false;
            return;
        }

        ImGui.SetNextWindowSize(new Vector2(600, 700), ImGuiCond.FirstUseEver);
        ImGui.SetNextWindowPos(new Vector2(
            ImGui.GetIO().DisplaySize.X / 2 - 300,
            ImGui.GetIO().DisplaySize.Y / 2 - 350), ImGuiCond.FirstUseEver);

        if (ImGui.Begin($"Map Preview - {_selectedMapForPreview.Name}", ref _showWdlPreview, ImGuiWindowFlags.NoCollapse))
        {
            ImGui.TextWrapped("Click on the map preview to select a spawn point, then click 'Load Map' to start at that location.");
            ImGui.Separator();

            // Preview image
            float previewSize = 512f;
            var cursorPos = ImGui.GetCursorScreenPos();
            
            // Draw the WDL preview texture
            ImGui.Image((nint)_wdlPreviewRenderer.TextureId, new Vector2(previewSize, previewSize));

            // Handle mouse clicks on the preview
            if (ImGui.IsItemHovered() && ImGui.IsMouseClicked(ImGuiMouseButton.Left))
            {
                var mousePos = ImGui.GetMousePos();
                var relativePos = mousePos - cursorPos;
                
                // Scale to texture coordinates
                float scaleX = _wdlPreviewRenderer.Width / previewSize;
                float scaleY = _wdlPreviewRenderer.Height / previewSize;
                var texturePos = new Vector2(relativePos.X * scaleX, relativePos.Y * scaleY);

                // Convert to tile coordinates
                var tile = _wdlPreviewRenderer.PixelToTile(texturePos);
                if (tile.HasValue)
                {
                    _selectedSpawnTile = new Vector2(tile.Value.tileX, tile.Value.tileY);
                }
            }

            // Draw selected spawn point marker
            if (_selectedSpawnTile.HasValue)
            {
                var drawList = ImGui.GetWindowDrawList();
                float tileSize = previewSize / 64f;
                float markerX = cursorPos.X + (_selectedSpawnTile.Value.X + 0.5f) * tileSize;
                float markerY = cursorPos.Y + (_selectedSpawnTile.Value.Y + 0.5f) * tileSize;
                
                // Draw crosshair
                uint color = ImGui.ColorConvertFloat4ToU32(new Vector4(1f, 0f, 0f, 1f));
                float crossSize = 10f;
                drawList.AddLine(new Vector2(markerX - crossSize, markerY), new Vector2(markerX + crossSize, markerY), color, 2f);
                drawList.AddLine(new Vector2(markerX, markerY - crossSize), new Vector2(markerX, markerY + crossSize), color, 2f);
                
                // Draw circle around it
                drawList.AddCircle(new Vector2(markerX, markerY), tileSize * 0.5f, color, 16, 2f);
            }

            ImGui.Separator();

            // Spawn point info
            if (_selectedSpawnTile.HasValue)
            {
                ImGui.Text($"Selected Tile: ({_selectedSpawnTile.Value.X:F0}, {_selectedSpawnTile.Value.Y:F0})");
                
                var worldPos = _wdlPreviewRenderer.TileToWorldPosition((int)_selectedSpawnTile.Value.X, (int)_selectedSpawnTile.Value.Y);
                ImGui.Text($"World Position: ({worldPos.X:F1}, {worldPos.Y:F1}, {worldPos.Z:F1})");
            }
            else
            {
                ImGui.TextColored(new Vector4(0.7f, 0.7f, 0.7f, 1f), "No spawn point selected. Click on the map to choose.");
            }

            ImGui.Separator();

            // Action buttons
            bool canLoad = _selectedMapForPreview.HasWdt && _selectedSpawnTile.HasValue;
            if (!canLoad) ImGui.BeginDisabled();
            
            if (ImGui.Button("Load Map at Selected Point", new Vector2(-1, 0)))
            {
                // Load the map and teleport to selected spawn point
                string wdtPath = $"World\\Maps\\{_selectedMapForPreview.Directory}\\{_selectedMapForPreview.Directory}.wdt";
                LoadFileFromDataSource(wdtPath);
                
                // Teleport camera to selected spawn point after a short delay (let map load first)
                if (_selectedSpawnTile.HasValue && _wdlPreviewRenderer != null)
                {
                    var spawnPos = _wdlPreviewRenderer.TileToWorldPosition(
                        (int)_selectedSpawnTile.Value.X, 
                        (int)_selectedSpawnTile.Value.Y);
                    
                    // Set camera position
                    _camera.Position = spawnPos;
                }
                
                _showWdlPreview = false;
            }
            
            if (!canLoad) ImGui.EndDisabled();

            if (ImGui.IsItemHovered(ImGuiHoveredFlags.AllowWhenDisabled) && !canLoad)
            {
                ImGui.SetTooltip(_selectedSpawnTile.HasValue 
                    ? "WDT file not found for this map" 
                    : "Select a spawn point on the map first");
            }

            ImGui.SameLine();
            if (ImGui.Button("Cancel", new Vector2(100, 0)))
            {
                _showWdlPreview = false;
            }

            // Legend
            ImGui.Separator();
            ImGui.Text("Color Legend:");
            ImGui.SameLine();
            ImGui.TextColored(new Vector4(0f, 0.5f, 1f, 1f), "■");
            ImGui.SameLine();
            ImGui.Text("Low");
            ImGui.SameLine();
            ImGui.TextColored(new Vector4(0f, 1f, 0.5f, 1f), "■");
            ImGui.SameLine();
            ImGui.Text("Mid");
            ImGui.SameLine();
            ImGui.TextColored(new Vector4(0.7f, 0.5f, 0.2f, 1f), "■");
            ImGui.SameLine();
            ImGui.Text("High");
            ImGui.SameLine();
            ImGui.Spacing();
            ImGui.SameLine();
            ImGui.TextColored(new Vector4(0.1f, 0.1f, 0.1f, 1f), "■");
            ImGui.SameLine();
            ImGui.Text("No Data");
        }
        ImGui.End();
    }
}
