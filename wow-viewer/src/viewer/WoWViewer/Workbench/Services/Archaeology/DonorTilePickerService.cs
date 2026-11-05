using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using ImGuiNET;
using Silk.NET.OpenGL;
using WoWViewer.DataSources;
using WoWViewer.Rendering;
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

    // Zoom/pan state for the 64x64 map surface. _pan is the normalized (0..1) top-left
    // corner of the visible window; _zoom is the magnification factor.
    private const float MinZoom = 1f;
    private const float MaxZoom = 16f;
    private float _zoom = MinZoom;
    private Vector2 _pan = Vector2.Zero;
    private Vector2? _panDragStart;
    private Vector2 _panAtDragStart;

    // Per-tile settings applied to every tile added from the current selection.
    private float _selectionZOffset;
    private float _selectionZScale = 1f;

    // Per-donor-tile chunk inclusion masks. Missing entries mean "all chunks".
    private readonly Dictionary<(int row, int col), PhaseChunkMask> _chunkMasks = new();
    private (int row, int col)? _chunkEditTile;

    // Whole-map placement offset (tile units) for the "add entire map" action.
    private int _wholeMapOffsetX;
    private int _wholeMapOffsetY;

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
        _zoom = MinZoom;
        _pan = Vector2.Zero;
        _panDragStart = null;
        _selectionZOffset = 0f;
        _selectionZScale = 1f;
        _chunkMasks.Clear();
        _chunkEditTile = null;
        _wholeMapOffsetX = 0;
        _wholeMapOffsetY = 0;

        EnsureDonorRendererLoaded();
        EnsureTargetRendererLoaded();
    }

    public void Close()
    {
        _isOpen = false;
        _selectedDonorTiles.Clear();
        _targetAnchor = null;
        _chunkEditTile = null;
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

        // Zoom / pan controls
        ImGui.TextUnformatted("Zoom");
        ImGui.SameLine();
        if (ImGui.SmallButton("-##pickerZoomOut"))
            SetZoom(_zoom / 1.5f, null);
        ImGui.SameLine();
        if (ImGui.SmallButton("+##pickerZoomIn"))
            SetZoom(_zoom * 1.5f, null);
        ImGui.SameLine();
        if (ImGui.SmallButton("Fit##pickerZoomFit"))
        {
            _zoom = MinZoom;
            _pan = Vector2.Zero;
        }
        ImGui.SameLine();
        ImGui.TextDisabled($"{_zoom:0.0}x  |  scroll to zoom, middle/right-drag to pan");
        ImGui.SameLine();
        ImGui.TextDisabled($"Visible: {VisibleTileRangeText()}");

        ImGui.Separator();

        // 64x64 Map Surface (zoomable)
        float availableY = ImGui.GetContentRegionAvail().Y - ImGui.GetFrameHeightWithSpacing() * 2.5f;
        float mapSize = MathF.Max(256f, MathF.Min(ImGui.GetContentRegionAvail().X, availableY));
        var cursorPos = ImGui.GetCursorScreenPos();
        var drawList = ImGui.GetWindowDrawList();
        Vector2 mousePos = ImGui.GetIO().MousePos;

        bool isHovered = mousePos.X >= cursorPos.X && mousePos.X < cursorPos.X + mapSize
            && mousePos.Y >= cursorPos.Y && mousePos.Y < cursorPos.Y + mapSize;

        // Wheel zoom, anchored on the tile under the cursor.
        if (isHovered)
        {
            float wheel = ImGui.GetIO().MouseWheel;
            if (MathF.Abs(wheel) > 0.001f)
            {
                Vector2 local = (mousePos - cursorPos) / mapSize;
                Vector2 uvBefore = _pan + local * (1f / _zoom);
                float newZoom = Math.Clamp(_zoom * MathF.Pow(1.2f, wheel), MinZoom, MaxZoom);
                _zoom = newZoom;
                _pan = uvBefore - local * (1f / _zoom);
                ClampPan();
            }
        }

        // Pan with middle or right mouse drag.
        if (isHovered && (ImGui.IsMouseClicked(ImGuiMouseButton.Middle) || ImGui.IsMouseClicked(ImGuiMouseButton.Right)))
        {
            _panDragStart = mousePos;
            _panAtDragStart = _pan;
        }
        if (_panDragStart.HasValue && (ImGui.IsMouseDown(ImGuiMouseButton.Middle) || ImGui.IsMouseDown(ImGuiMouseButton.Right)))
        {
            Vector2 delta = (mousePos - _panDragStart.Value) / mapSize * (1f / _zoom);
            _pan = _panAtDragStart - delta;
            ClampPan();
        }
        if (ImGui.IsMouseReleased(ImGuiMouseButton.Middle) || ImGui.IsMouseReleased(ImGuiMouseButton.Right))
            _panDragStart = null;

        // Recompute hovered tile after zoom/pan.
        float viewSize = 1f / _zoom;
        Vector2 localMouse = (mousePos - cursorPos) / mapSize;
        Vector2 uvMouse = _pan + localMouse * viewSize;
        int hoveredCol = isHovered ? Math.Clamp((int)(uvMouse.X * 64f), 0, 63) : -1;
        int hoveredRow = isHovered ? Math.Clamp((int)(uvMouse.Y * 64f), 0, 63) : -1;

        // Visible tile range for grid drawing.
        int firstCol = Math.Clamp((int)MathF.Floor(_pan.X * 64f), 0, 63);
        int lastCol = Math.Clamp((int)MathF.Ceiling((_pan.X + viewSize) * 64f), 0, 63);
        int firstRow = Math.Clamp((int)MathF.Floor(_pan.Y * 64f), 0, 63);
        int lastRow = Math.Clamp((int)MathF.Ceiling((_pan.Y + viewSize) * 64f), 0, 63);

        Vector2 TileToScreen(int col, int row)
        {
            Vector2 uv = new((col / 64f - _pan.X) / viewSize, (row / 64f - _pan.Y) / viewSize);
            return cursorPos + uv * mapSize;
        }

        // Background
        drawList.AddRectFilled(cursorPos, cursorPos + new Vector2(mapSize, mapSize), 0xFF141416);

        // Donor tiles are drawn from the client's minimap art, not the WDL heightmap preview:
        // the operator needs to recognize the terrain they are picking, and a heightmap does not
        // show roads, buildings, or biome. Each visible tile is its own minimap texture.
        MinimapRenderer? minimapRenderer = _host.MinimapRenderer;
        for (int row = firstRow; row <= lastRow; row++)
        {
            for (int col = firstCol; col <= lastCol; col++)
            {
                Vector2 pMin = TileToScreen(col, row);
                Vector2 pMax = TileToScreen(col + 1, row + 1);
                uint texture = minimapRenderer?.GetTileTexture(_donorMap, col, row) ?? 0;
                if (texture != 0)
                    drawList.AddImage((nint)texture, pMin, pMax);
                else
                    drawList.AddRectFilled(pMin, pMax, 0xFF1E1E22);
            }
        }

        // Draw grid overlay for occupied tiles in the visible window.
        foreach ((int occRow, int occCol) in occupiedSet)
        {
            if (occCol < firstCol || occCol > lastCol || occRow < firstRow || occRow > lastRow)
                continue;
            Vector2 pMin = TileToScreen(occCol, occRow);
            Vector2 pMax = TileToScreen(occCol + 1, occRow + 1);
            drawList.AddRect(pMin, pMax, 0x44FFFFFF, 0f, ImDrawFlags.None, 1f);
        }

        // Draw selected tiles in the visible window.
        foreach ((int selRow, int selCol) in _selectedDonorTiles)
        {
            if (selCol < firstCol || selCol > lastCol || selRow < firstRow || selRow > lastRow)
                continue;
            Vector2 pMin = TileToScreen(selCol, selRow);
            Vector2 pMax = TileToScreen(selCol + 1, selRow + 1);
            drawList.AddRectFilled(pMin, pMax, 0x6600E5FF);
            drawList.AddRect(pMin, pMax, 0xFF00E5FF, 0f, ImDrawFlags.None, 1.5f);
        }

        // Draw hovered tile.
        if (hoveredCol >= 0 && hoveredRow >= 0)
        {
            Vector2 pMin = TileToScreen(hoveredCol, hoveredRow);
            Vector2 pMax = TileToScreen(hoveredCol + 1, hoveredRow + 1);
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
                    Vector2 uvA = _pan + (Vector2.Min(_dragStart.Value, _dragCurrent.Value) - cursorPos) / mapSize * viewSize;
                    Vector2 uvB = _pan + (Vector2.Max(_dragStart.Value, _dragCurrent.Value) - cursorPos) / mapSize * viewSize;

                    int startCol = Math.Clamp((int)MathF.Floor(uvA.X * 64f), 0, 63);
                    int endCol = Math.Clamp((int)MathF.Floor(uvB.X * 64f), 0, 63);
                    int startRow = Math.Clamp((int)MathF.Floor(uvA.Y * 64f), 0, 63);
                    int endRow = Math.Clamp((int)MathF.Floor(uvB.Y * 64f), 0, 63);

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

        // Per-tile settings applied to every tile added from this selection.
        if (ImGui.TreeNode("Selection settings (applied to added tiles)##pickerSelectionSettings"))
        {
            WoWViewer.UI.SharedUiWidgets.SetNextItemWidth(120f);
            ImGui.InputFloat("Z offset##pickerSelZOffset", ref _selectionZOffset, 0.1f, 1f, "%.2f");
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Vertical offset applied to every tile added from this selection.");
            WoWViewer.UI.SharedUiWidgets.SetNextItemWidth(120f);
            ImGui.InputFloat("Z scale##pickerSelZScale", ref _selectionZScale, 0.01f, 0.1f, "%.3f");
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Vertical scale applied to every tile added from this selection.");
            if (ImGui.SmallButton("Reset##pickerSelZReset"))
            {
                _selectionZOffset = 0f;
                _selectionZScale = 1f;
            }
            ImGui.TreePop();
        }

        // Chunk-level inclusion for the selected donor tile. The development map canvas is not
        // aligned to tile boundaries, so a donor tile often needs only part of its 16x16 chunks.
        if (ImGui.TreeNode("Chunk mask (16x16)##pickerChunkMask"))
        {
            if (_selectedDonorTiles.Count == 0)
            {
                ImGui.TextDisabled("Select a donor tile first.");
            }
            else
            {
                if (_chunkEditTile is not { } editTile || !_selectedDonorTiles.Contains(editTile))
                    _chunkEditTile = _selectedDonorTiles.First();

                if (_selectedDonorTiles.Count > 1)
                {
                    string current = $"{_chunkEditTile.Value.col:D2}_{_chunkEditTile.Value.row:D2}";
                    if (ImGui.BeginCombo("Edit tile##pickerChunkTile", current))
                    {
                        foreach ((int row, int col) in _selectedDonorTiles.OrderBy(t => t.row).ThenBy(t => t.col))
                        {
                            string label = $"{col:D2}_{row:D2}";
                            if (ImGui.Selectable(label, _chunkEditTile == (row, col)))
                                _chunkEditTile = (row, col);
                        }
                        ImGui.EndCombo();
                    }
                }

                (int editRow, int editCol) = _chunkEditTile.Value;
                PhaseChunkMask mask = _chunkMasks.TryGetValue((editRow, editCol), out PhaseChunkMask existing)
                    ? existing
                    : PhaseChunkMask.All;

                if (ImGui.SmallButton("All##pickerChunkAll"))
                    mask = PhaseChunkMask.All;
                ImGui.SameLine();
                if (ImGui.SmallButton("None##pickerChunkNone"))
                    mask = PhaseChunkMask.None;
                ImGui.SameLine();
                if (ImGui.SmallButton("Invert##pickerChunkInvert"))
                {
                    PhaseChunkMask inverted = PhaseChunkMask.None;
                    for (int y = 0; y < 16; y++)
                        for (int x = 0; x < 16; x++)
                            inverted = inverted.With(x, y, !mask.IsIncluded(x, y));
                    mask = inverted;
                }
                ImGui.SameLine();
                ImGui.TextDisabled($"{mask.IncludedCount}/256 chunks included");

                const float cell = 14f;
                for (int y = 0; y < 16; y++)
                {
                    for (int x = 0; x < 16; x++)
                    {
                        bool included = mask.IsIncluded(x, y);
                        ImGui.PushStyleColor(ImGuiCol.Button, included
                            ? new Vector4(0.20f, 0.55f, 0.30f, 1f)
                            : new Vector4(0.18f, 0.18f, 0.20f, 1f));
                        if (ImGui.Button($"##pickerChunk{x}_{y}", new Vector2(cell, cell)))
                            mask = mask.With(x, y, !included);
                        ImGui.PopStyleColor();
                        if (ImGui.IsItemHovered())
                            ImGui.SetTooltip($"Chunk {x:D2}_{y:D2}: {(included ? "included" : "excluded")} (click to toggle)");
                        if (x < 15) ImGui.SameLine(0f, 1f);
                    }
                }

                _chunkMasks[(editRow, editCol)] = mask;
            }
            ImGui.TreePop();
        }

        // Whole-map placement: drop the donor map's entire footprint at an arbitrary tile offset.
        if (ImGui.TreeNode("Place entire map##pickerWholeMap"))
        {
            WoWViewer.UI.SharedUiWidgets.SetNextItemWidth(100f);
            ImGui.InputInt("Offset X##pickerWholeMapX", ref _wholeMapOffsetX);
            ImGui.SameLine();
            WoWViewer.UI.SharedUiWidgets.SetNextItemWidth(100f);
            ImGui.InputInt("Offset Y##pickerWholeMapY", ref _wholeMapOffsetY);
            _wholeMapOffsetX = Math.Clamp(_wholeMapOffsetX, -63, 63);
            _wholeMapOffsetY = Math.Clamp(_wholeMapOffsetY, -63, 63);
            ImGui.TextDisabled("Positive X moves the donor map east; positive Y moves it south.");
            if (ImGui.Button("Add Whole Map at Offset##pickerAddWholeMap"))
            {
                AddWholeMapLayer(_wholeMapOffsetX, _wholeMapOffsetY);
                _isOpen = false;
            }
            ImGui.SameLine();
            if (ImGui.Button("Add Whole Map at Matching Coordinates##pickerAddWholeMapZero"))
            {
                AddWholeMapLayer(0, 0);
                _isOpen = false;
            }
            ImGui.TreePop();
        }

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

            PhaseChunkMask? chunkMask = _chunkMasks.TryGetValue((donorRow, donorCol), out PhaseChunkMask mask)
                && !mask.IsAllIncluded
                    ? mask
                    : null;

            layer.TilePlacements.Add(new PhaseTilePlacement(
                DonorTileX: donorRow,
                DonorTileY: donorCol,
                TargetTileX: targetRow,
                TargetTileY: targetCol,
                ZOffset: _selectionZOffset,
                ZScale: _selectionZScale,
                ChunkMask: chunkMask));
        }

        terrain.PhaseLayers.Add(layer);
        terrain.RefreshPhaseLayers();
    }

    private void AddWholeMapLayer(int offsetX, int offsetY)
    {
        var terrain = _host.TerrainManager;
        if (terrain == null || string.IsNullOrWhiteSpace(_donorMap))
            return;

        var layer = new PhaseLayerSettings
        {
            MapName = _donorMap,
            TileOffsetX = offsetX,
            TileOffsetY = offsetY,
            UsePlacedTilesOnly = false,
            Channels = PhaseDataChannel.All,
            OnlyTakeWhatThePhaseCarries = true,
        };
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

    private void SetZoom(float zoom, Vector2? anchorLocal)
    {
        float newZoom = Math.Clamp(zoom, MinZoom, MaxZoom);
        if (anchorLocal.HasValue)
        {
            Vector2 local = anchorLocal.Value;
            Vector2 uvBefore = _pan + local * (1f / _zoom);
            _zoom = newZoom;
            _pan = uvBefore - local * (1f / _zoom);
        }
        else
        {
            _zoom = newZoom;
        }
        ClampPan();
    }

    private void ClampPan()
    {
        float viewSize = 1f / _zoom;
        _pan = new Vector2(
            Math.Clamp(_pan.X, 0f, MathF.Max(0f, 1f - viewSize)),
            Math.Clamp(_pan.Y, 0f, MathF.Max(0f, 1f - viewSize)));
    }

    private string VisibleTileRangeText()
    {
        float viewSize = 1f / _zoom;
        int firstCol = Math.Clamp((int)MathF.Floor(_pan.X * 64f), 0, 63);
        int lastCol = Math.Clamp((int)MathF.Ceiling((_pan.X + viewSize) * 64f) - 1, 0, 63);
        int firstRow = Math.Clamp((int)MathF.Floor(_pan.Y * 64f), 0, 63);
        int lastRow = Math.Clamp((int)MathF.Ceiling((_pan.Y + viewSize) * 64f) - 1, 0, 63);
        return $"{firstCol:D2}_{firstRow:D2} .. {lastCol:D2}_{lastRow:D2}";
    }

    public void Dispose()
    {
        _donorRenderer?.Dispose();
        _donorRenderer = null;
        _targetRenderer?.Dispose();
        _targetRenderer = null;
    }
}

