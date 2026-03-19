using System.Numerics;
using ImGuiNET;
using MdxViewer.Terrain;

namespace MdxViewer;

/// <summary>
/// Partial class containing the WDL preview dialog for map spawn point selection.
/// </summary>
public partial class ViewerApp
{
    private void DrawWdlPreviewDialog()
    {
        if (_selectedMapForPreview == null || _wdlPreviewRenderer == null)
        {
            _showWdlPreview = false;
            return;
        }

        if (!_wdlPreviewRenderer.HasPreview)
        {
            TryLoadSelectedWdlPreviewFromCache(_selectedMapForPreview.Directory);
        }

        var previewState = GetSelectedWdlPreviewState();

        if (!_wdlPreviewRenderer.HasPreview)
        {
            if (previewState == WdlPreviewWarmState.Failed)
            {
                var failedMap = _selectedMapForPreview;
                _showWdlPreview = false;
                if (failedMap != null)
                    LoadMapAtDefaultSpawn(failedMap);
                return;
            }

            ImGui.SetNextWindowSize(new Vector2(400, 150), ImGuiCond.FirstUseEver);
            ImGui.SetNextWindowPos(new Vector2(
                ImGui.GetIO().DisplaySize.X / 2 - 200,
                ImGui.GetIO().DisplaySize.Y / 2 - 75), ImGuiCond.FirstUseEver);

            string title = previewState is WdlPreviewWarmState.Loading or WdlPreviewWarmState.NotQueued
                ? $"Preparing Map Preview - {_selectedMapForPreview.Name}"
                : $"WDL Preview Error - {_selectedMapForPreview.Name}";

            if (ImGui.Begin(title, ref _showWdlPreview))
            {
                if (previewState is WdlPreviewWarmState.Loading or WdlPreviewWarmState.NotQueued)
                {
                    ImGui.TextWrapped("Preparing the WDL heightmap preview for this map.");
                    if (!string.IsNullOrEmpty(_wdlPreviewWarmupStatus))
                    {
                        ImGui.Spacing();
                        ImGui.TextWrapped(_wdlPreviewWarmupStatus);
                    }
                }
                else
                {
                    ImGui.TextColored(new Vector4(1f, 0.3f, 0.3f, 1f), "Failed to load WDL preview.");
                    var error = GetSelectedWdlPreviewError();
                    if (!string.IsNullOrEmpty(error))
                        ImGui.TextWrapped(error);
                }

                ImGui.Separator();
                if (ImGui.Button("Close"))
                    _showWdlPreview = false;
            }
            ImGui.End();
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

            float previewSize = 512f;
            var cursorPos = ImGui.GetCursorScreenPos();

            ImGui.Image((nint)_wdlPreviewRenderer.TextureId, new Vector2(previewSize, previewSize));

            if (ImGui.IsItemHovered() && ImGui.IsMouseClicked(ImGuiMouseButton.Left))
            {
                var mousePos = ImGui.GetMousePos();
                var relativePos = mousePos - cursorPos;

                float scaleX = _wdlPreviewRenderer.Width / previewSize;
                float scaleY = _wdlPreviewRenderer.Height / previewSize;
                var texturePos = new Vector2(relativePos.X * scaleX, relativePos.Y * scaleY);

                var tile = _wdlPreviewRenderer.PixelToTile(texturePos);
                if (tile.HasValue)
                    _selectedSpawnTile = new Vector2(tile.Value.tileX, tile.Value.tileY);
            }

            if (_selectedSpawnTile.HasValue)
            {
                var drawList = ImGui.GetWindowDrawList();
                float tileSize = previewSize / 64f;
                float markerX = cursorPos.X + (_selectedSpawnTile.Value.X + 0.5f) * tileSize;
                float markerY = cursorPos.Y + (_selectedSpawnTile.Value.Y + 0.5f) * tileSize;

                uint color = ImGui.ColorConvertFloat4ToU32(new Vector4(1f, 0f, 0f, 1f));
                float crossSize = 10f;
                drawList.AddLine(new Vector2(markerX - crossSize, markerY), new Vector2(markerX + crossSize, markerY), color, 2f);
                drawList.AddLine(new Vector2(markerX, markerY - crossSize), new Vector2(markerX, markerY + crossSize), color, 2f);
                drawList.AddCircle(new Vector2(markerX, markerY), tileSize * 0.5f, color, 16, 2f);
            }

            ImGui.Separator();

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

            bool canLoad = _selectedMapForPreview.HasWdt && _selectedSpawnTile.HasValue;
            if (!canLoad) ImGui.BeginDisabled();

            if (ImGui.Button("Load Map at Selected Point", new Vector2(-1, 0)))
                LoadSelectedPreviewMapAtSpawn();

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
