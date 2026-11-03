using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using ImGuiNET;
using Silk.NET.OpenGL;
using WoWViewer.DataSources;
using WoWViewer.Terrain;
using WowViewer.Core.Maps;

namespace WoWViewer;

/// <summary>
/// Interactive full-map donor tile picker and placement service.
/// Allows visual selection of one or more donor tiles from a donor map overview/WDL preview,
/// and allows choosing target placement on the currently loaded map.
/// </summary>
internal sealed class DonorTilePickerService : IDisposable
{
    private readonly IViewerAppHost _host;
    private WdlPreviewRenderer? _donorRenderer;
    private WdlPreviewRenderer? _targetRenderer;

    private bool _isOpen;
    private string _donorMap = string.Empty;
    private string _targetMap = string.Empty;

    private readonly HashSet<(int row, int col)> _selectedDonorTiles = new();
    private (int row, int col)? _targetAnchor;
    private bool _placeAtMatchingCoords = true;

    private Vector2? _dragStart;
    private Vector2? _dragCurrent;
    private bool _isDraggingBox;

    public bool IsOpen => _isOpen;

    public DonorTilePickerService(IViewerAppHost host)
    {
        _host = host;
    }

    public void Open(string donorMap, string? targetMap = null)
    {
        _isOpen = true;
        _donorMap = string.IsNullOrWhiteSpace(donorMap)
            ? (_host.DiscoveredMaps.FirstOrDefault()?.Directory ?? string.Empty)
            : donorMap;

        _targetMap = targetMap
            ?? _host.TerrainManager?.MapName
            ?? _host.VlmTerrainManager?.MapName
            ?? (_host.DiscoveredMaps.FirstOrDefault()?.Directory ?? string.Empty);

        _selectedDonorTiles.Clear();
        _targetAnchor = null;
        _placeAtMatchingCoords = true;

        EnsureDonorRendererLoaded();
        EnsureTargetRendererLoaded();
    }

    public void Close()
    {
        _isOpen = false;
        _selectedDonorTiles.Clear();
        _targetAnchor = null;
    }

    private void EnsureDonorRendererLoaded()
    {
        if (string.IsNullOrWhiteSpace(_donorMap) || _host.DataSource == null)
            return;

        if (_donorRenderer == null)
            _donorRenderer = new WdlPreviewRenderer(_host.Gl);

        if (_donorRenderer.MapDirectory != _donorMap)
        {
            if (_host.WdlPreviewCacheService != null && _host.WdlPreviewCacheService.TryGetPreview(_donorMap, out var cached))
                _donorRenderer.LoadPreview(cached);
            else
                _donorRenderer.LoadWdl(_host.DataSource, _donorMap);
        }
    }

    private void EnsureTargetRendererLoaded()
    {
        if (string.IsNullOrWhiteSpace(_targetMap) || _host.DataSource == null)
            return;

        if (_targetRenderer == null)
            _targetRenderer = new WdlPreviewRenderer(_host.Gl);

        if (_targetRenderer.MapDirectory != _targetMap)
        {
            if (_host.WdlPreviewCacheService != null && _host.WdlPreviewCacheService.TryGetPreview(_targetMap, out var cached))
                _targetRenderer.LoadPreview(cached);
            else
                _targetRenderer.LoadWdl(_host.DataSource, _targetMap);
        }
    }

    public void DrawDialog()
    {
        if (!_isOpen)
            return;

        Vector2 display = ImGui.GetIO().DisplaySize;
        Vector2 minSize = new(780f, 820f);
        Vector2 targetSize = Vector2.Min(minSize, display * 0.92f);
        ImGui.SetNextWindowSize(targetSize, ImGuiCond.Appearing);
        ImGui.SetNextWindowPos(display * 0.5f, ImGuiCond.Appearing, new Vector2(0.5f, 0.5f));

        if (!ImGui.Begin("Map Donor Tile Picker###DonorTilePickerModal", ref _isOpen, ImGuiWindowFlags.NoCollapse))
        {
            ImGui.End();
            return;
        }

        // Header controls
        ImGui.TextWrapped("Select one or more donor tiles from the visual map overview. Click tiles to toggle selection or drag to box-select.");
        ImGui.Separator();

        // Donor map selector combo
        var maps = _host.DiscoveredMaps;
        if (maps.Count > 0 && ImGui.BeginCombo("Donor Map##pickerDonorMap", string.IsNullOrWhiteSpace(_donorMap) ? "Select Map..." : _donorMap))
        {
            foreach (var m in maps)
            {
                bool isSelected = string.Equals(m.Directory, _donorMap, StringComparison.OrdinalIgnoreCase);
                string label = m.HasDbcEntry ? $"[{m.Id:D3}] {m.Name} ({m.Directory})" : $"[custom] {m.Name} ({m.Directory})";
                if (ImGui.Selectable(label, isSelected))
                {
                    _donorMap = m.Directory;
                    _selectedDonorTiles.Clear();
                    EnsureDonorRendererLoaded();
                }
                if (isSelected) ImGui.SetItemDefaultFocus();
            }
            ImGui.EndCombo();
        }

        // Occupied tiles for donor map
        var occupiedTiles = GetOccupiedTilesForMap(_donorMap);
        var occupiedSet = new HashSet<(int row, int col)>(occupiedTiles.Select(t => (t.TileX, t.TileY)));

        // Toolbar buttons
        if (ImGui.Button("Select All Occupied Tiles"))
        {
            foreach (var t in occupiedTiles)
                _selectedDonorTiles.Add((t.TileX, t.TileY));
        }
        ImGui.SameLine();
        if (ImGui.Button("Clear Selection"))
        {
            _selectedDonorTiles.Clear();
        }
        ImGui.SameLine();
        ImGui.TextDisabled($"Selected: {_selectedDonorTiles.Count} tile(s) | Available: {occupiedTiles.Count} tile(s)");

        ImGui.Separator();

        // Placement mode selection
        ImGui.RadioButton("Place at matching coordinates (same xx_yy)##modeMatching", _placeAtMatchingCoords);
        if (ImGui.IsItemClicked()) _placeAtMatchingCoords = true;
        ImGui.SameLine();
        ImGui.RadioButton("Place at target anchor on current map##modeAnchor", !_placeAtMatchingCoords);
        if (ImGui.IsItemClicked())
        {
            _placeAtMatchingCoords = false;
            EnsureTargetRendererLoaded();
        }

        if (!_placeAtMatchingCoords)
        {
            string anchorText = _targetAnchor.HasValue
                ? $"Anchor Target: {_targetAnchor.Value.col:D2}_{_targetAnchor.Value.row:D2} on {_targetMap}"
                : "Click on the map surface below while holding Ctrl, or click target slot.";
            ImGui.TextColored(new Vector4(0.3f, 0.9f, 0.4f, 1f), anchorText);
        }

        ImGui.Separator();

        // 64x64 Map Surface
        float availableY = ImGui.GetContentRegionAvail().Y - ImGui.GetFrameHeightWithSpacing() * 2.5f;
        float mapSize = MathF.Max(256f, MathF.Min(ImGui.GetContentRegionAvail().X, availableY));
        var cursorPos = ImGui.GetCursorScreenPos();
        var drawList = ImGui.GetWindowDrawList();

        // Background
        drawList.AddRectFilled(cursorPos, cursorPos + new Vector2(mapSize, mapSize), 0xFF141416);

        // Preview texture
        if (_donorRenderer != null && _donorRenderer.HasPreview)
        {
            ImGui.Image((nint)_donorRenderer.TextureId, new Vector2(mapSize, mapSize));
        }
        else
        {
            ImGui.Dummy(new Vector2(mapSize, mapSize));
        }

        float cellSize = mapSize / 64f;
        Vector2 mousePos = ImGui.GetIO().MousePos;
        bool isHovered = mousePos.X >= cursorPos.X && mousePos.X < cursorPos.X + mapSize
            && mousePos.Y >= cursorPos.Y && mousePos.Y < cursorPos.Y + mapSize;

        int hoveredCol = isHovered ? Math.Clamp((int)((mousePos.X - cursorPos.X) / cellSize), 0, 63) : -1;
        int hoveredRow = isHovered ? Math.Clamp((int)((mousePos.Y - cursorPos.Y) / cellSize), 0, 63) : -1;

        // Draw grid overlay for occupied and selected tiles
        foreach ((int occRow, int occCol) in occupiedSet)
        {
            Vector2 pMin = cursorPos + new Vector2(occCol * cellSize, occRow * cellSize);
            Vector2 pMax = pMin + new Vector2(cellSize, cellSize);
            drawList.AddRect(pMin, pMax, 0x44FFFFFF, 0f, ImDrawFlags.None, 1f);
        }

        // Draw selected tiles
        foreach ((int selRow, int selCol) in _selectedDonorTiles)
        {
            Vector2 pMin = cursorPos + new Vector2(selCol * cellSize, selRow * cellSize);
            Vector2 pMax = pMin + new Vector2(cellSize, cellSize);
            drawList.AddRectFilled(pMin, pMax, 0x6600E5FF);
            drawList.AddRect(pMin, pMax, 0xFF00E5FF, 0f, ImDrawFlags.None, 1.5f);
        }

        // Draw hovered tile
        if (hoveredCol >= 0 && hoveredRow >= 0)
        {
            Vector2 pMin = cursorPos + new Vector2(hoveredCol * cellSize, hoveredRow * cellSize);
            Vector2 pMax = pMin + new Vector2(cellSize, cellSize);
            drawList.AddRect(pMin, pMax, 0xFFFFAA00, 0f, ImDrawFlags.None, 2f);

            bool occ = occupiedSet.Contains((hoveredRow, hoveredCol));
            bool sel = _selectedDonorTiles.Contains((hoveredRow, hoveredCol));
            ImGui.SetTooltip($"Tile: {hoveredCol:D2}_{hoveredRow:D2} (col={hoveredCol}, row={hoveredRow}) | {(occ ? "Occupied" : "Empty")} | {(sel ? "[SELECTED]" : "Click to select")}");
        }

        // Interaction over map surface
        ImGui.SetCursorScreenPos(cursorPos);
        ImGui.InvisibleButton("##donorMapTileSurface", new Vector2(mapSize, mapSize));

        if (ImGui.IsItemHovered())
        {
            if (ImGui.IsMouseClicked(ImGuiMouseButton.Left))
            {
                if (ImGui.GetIO().KeyCtrl && !_placeAtMatchingCoords)
                {
                    _targetAnchor = (hoveredRow, hoveredCol);
                }
                else
                {
                    _dragStart = mousePos;
                    _dragCurrent = mousePos;
                    _isDraggingBox = false;
                }
            }

            if (ImGui.IsMouseDown(ImGuiMouseButton.Left) && _dragStart.HasValue)
            {
                _dragCurrent = mousePos;
                if (Vector2.Distance(_dragStart.Value, mousePos) > 6f)
                    _isDraggingBox = true;
            }

            if (ImGui.IsMouseReleased(ImGuiMouseButton.Left) && _dragStart.HasValue)
            {
                if (_isDraggingBox && _dragCurrent.HasValue)
                {
                    float minX = MathF.Min(_dragStart.Value.X, _dragCurrent.Value.X) - cursorPos.X;
                    float maxX = MathF.Max(_dragStart.Value.X, _dragCurrent.Value.X) - cursorPos.X;
                    float minY = MathF.Min(_dragStart.Value.Y, _dragCurrent.Value.Y) - cursorPos.Y;
                    float maxY = MathF.Max(_dragStart.Value.Y, _dragCurrent.Value.Y) - cursorPos.Y;

                    int startCol = Math.Clamp((int)(minX / cellSize), 0, 63);
                    int endCol = Math.Clamp((int)(maxX / cellSize), 0, 63);
                    int startRow = Math.Clamp((int)(minY / cellSize), 0, 63);
                    int endRow = Math.Clamp((int)(maxY / cellSize), 0, 63);

                    for (int r = startRow; r <= endRow; r++)
                    {
                        for (int c = startCol; c <= endCol; c++)
                        {
                            if (occupiedSet.Contains((r, c)))
                                _selectedDonorTiles.Add((r, c));
                        }
                    }
                }
                else if (hoveredCol >= 0 && hoveredRow >= 0)
                {
                    // Single click toggle
                    if (_selectedDonorTiles.Contains((hoveredRow, hoveredCol)))
                        _selectedDonorTiles.Remove((hoveredRow, hoveredCol));
                    else
                        _selectedDonorTiles.Add((hoveredRow, hoveredCol));
                }

                _dragStart = null;
                _dragCurrent = null;
                _isDraggingBox = false;
            }
        }

        // Draw active drag box
        if (_isDraggingBox && _dragStart.HasValue && _dragCurrent.HasValue)
        {
            Vector2 boxMin = new(MathF.Min(_dragStart.Value.X, _dragCurrent.Value.X), MathF.Min(_dragStart.Value.Y, _dragCurrent.Value.Y));
            Vector2 boxMax = new(MathF.Max(_dragStart.Value.X, _dragCurrent.Value.X), MathF.Max(_dragStart.Value.Y, _dragCurrent.Value.Y));
            drawList.AddRectFilled(boxMin, boxMax, 0x33FFAA00);
            drawList.AddRect(boxMin, boxMax, 0xFFFFAA00, 0f, ImDrawFlags.None, 1.5f);
        }

        ImGui.Separator();

        // Footer buttons
        bool canAdd = !string.IsNullOrWhiteSpace(_donorMap) && _selectedDonorTiles.Count > 0;
        if (!canAdd) ImGui.BeginDisabled();

        string addButtonText = _selectedDonorTiles.Count == 1
            ? $"Add Single Donor Tile as Layer ({_selectedDonorTiles.First().col:D2}_{_selectedDonorTiles.First().row:D2})"
            : $"Add {_selectedDonorTiles.Count} Donor Tiles as Layer";

        if (ImGui.Button(addButtonText, new Vector2(-120f, 0f)))
        {
            ApplySelectedDonorTiles();
            _isOpen = false;
        }

        if (!canAdd) ImGui.EndDisabled();

        ImGui.SameLine();
        if (ImGui.Button("Cancel", new Vector2(100f, 0f)))
        {
            _isOpen = false;
        }

        ImGui.End();
    }

    private void ApplySelectedDonorTiles()
    {
        if (_selectedDonorTiles.Count == 0 || string.IsNullOrWhiteSpace(_donorMap))
            return;

        var terrain = _host.TerrainManager;
        if (terrain == null)
            return;

        int minRow = _selectedDonorTiles.Min(t => t.row);
        int minCol = _selectedDonorTiles.Min(t => t.col);

        int deltaRow = 0;
        int deltaCol = 0;
        if (!_placeAtMatchingCoords && _targetAnchor.HasValue)
        {
            deltaRow = _targetAnchor.Value.row - minRow;
            deltaCol = _targetAnchor.Value.col - minCol;
        }

        var layer = new PhaseLayerSettings
        {
            MapName = _donorMap,
            TileOffsetX = 0,
            TileOffsetY = 0,
            UsePlacedTilesOnly = true,
            Channels = PhaseDataChannel.All,
            OnlyTakeWhatThePhaseCarries = true,
        };

        foreach ((int donorRow, int donorCol) in _selectedDonorTiles)
        {
            int targetRow = Math.Clamp(donorRow + deltaRow, 0, 63);
            int targetCol = Math.Clamp(donorCol + deltaCol, 0, 63);

            layer.TilePlacements.Add(new PhaseTilePlacement(
                DonorTileX: donorRow,
                DonorTileY: donorCol,
                TargetTileX: targetRow,
                TargetTileY: targetCol));
        }

        terrain.PhaseLayers.Add(layer);
        terrain.RefreshPhaseLayers();
    }

    private IReadOnlyList<(int TileX, int TileY)> GetOccupiedTilesForMap(string mapName)
    {
        if (string.IsNullOrWhiteSpace(mapName) || _host.TerrainManager == null)
            return Array.Empty<(int, int)>();

        var layer = new PhaseLayerSettings { MapName = mapName };
        return _host.TerrainManager.GetLayerFootprint(layer);
    }

    public void Dispose()
    {
        _donorRenderer?.Dispose();
        _donorRenderer = null;
        _targetRenderer?.Dispose();
        _targetRenderer = null;
    }
}

