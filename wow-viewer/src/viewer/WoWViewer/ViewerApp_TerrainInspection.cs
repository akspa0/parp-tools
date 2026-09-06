using System;
using System.Collections.Generic;
using System.Numerics;
using WowViewer.Core.Runtime.World.Inspection;
using WowViewer.Core.World;
using WoWViewer.Terrain;
using WoWViewer.Rendering;

namespace WoWViewer;

public partial class ViewerApp
{
    /// <summary>
    /// The Inspector's terrain target is deliberately independent from the legacy investigation
    /// target. A click pins a chunk until its owning world/map changes; ordinary hover and camera
    /// inspection remain live fallbacks when no pin exists.
    /// </summary>
    private TerrainRenderer.TerrainChunkInfo? _selectedTerrainChunk;
    private object? _selectedTerrainChunkOwner;
    private string? _selectedTerrainChunkMapName;
    private int _selectedTerrainChunkMapId = int.MinValue;

    private enum TerrainInspectorTargetSource
    {
        Pinned,
        Hovered,
        Camera,
    }

    private bool AppendPinnedTerrainChunkInspection(InspectorContentBuilder builder, bool primary = false)
    {
        if (!TryGetPinnedTerrainChunkInspectionTarget(
                out TerrainRenderer.TerrainChunkInfo chunkInfo,
                out TerrainInspectorTargetSource targetSource))
        {
            return false;
        }

        if (!TryResolvePinnedTerrainChunkInspectionData(
                chunkInfo,
                out TerrainChunkData? chunkData,
                out IReadOnlyList<string>? tileTextures,
                out TileLoadResult? tileResult)
            || chunkData == null)
        {
            return false;
        }

        if (primary)
        {
            builder.ObjectType = "ADT";
            builder.Headline = $"ADT Chunk ({chunkInfo.TileY}, {chunkInfo.TileX}) MCNK ({chunkInfo.ChunkX}, {chunkInfo.ChunkY})";
        }

        var meta = builder.AddSection(primary ? "Chunk Metadata" : "ADT Terrain Context");
        meta.Row("Target Source", DescribeTerrainInspectorTargetSource(targetSource), isImportant: true);
        meta.Row("Tile (X, Y)", $"({chunkInfo.TileX}, {chunkInfo.TileY})");
        meta.Row("MCNK (X, Y)", $"({chunkInfo.ChunkX}, {chunkInfo.ChunkY})");
        meta.Row("Area ID", $"{chunkData.AreaId}", isImportant: true);
        meta.Row("Area Name", ResolveTerrainAreaName(chunkData.AreaId), isImportant: true);
        meta.Row("MCNK Flags", $"0x{unchecked((uint)chunkData.McnkFlags):X8} ({DescribeMcnkFlags(chunkData.McnkFlags)})");
        meta.Row("Alpha Source Flags", $"0x{unchecked((uint)chunkData.AlphaSourceFlags):X8}");
        meta.Row("Holes Mask", $"0x{chunkData.HoleMask:X4} ({CountTerrainHoleCells(chunkData.HoleMask)}/16 2x2 groups)");
        meta.Row("World Position", FormatVector(chunkData.WorldPosition));
        meta.Row("Elevation", DescribeTerrainElevation(chunkData));
        meta.Row("Terrain Bounds", $"Min {FormatVector(chunkInfo.BoundsMin)}  Max {FormatVector(chunkInfo.BoundsMax)}");
        meta.Row("Vertices / Normals", $"{chunkData.Heights.Length} / {chunkData.Normals.Length}");
        meta.Row("Layers", $"{chunkData.Layers.Length}");
        meta.Row("Alpha Maps", $"{chunkData.AlphaMaps.Count}");
        meta.Row("Shadow Map", DescribeTerrainBytePayload(chunkData.ShadowMap, "64x64"));
        meta.Row("MCCV Vertex Colors", DescribeTerrainBytePayload(chunkData.MccvColors, "145 BGRA verts"));

        if (chunkData.Liquid is LiquidChunkData liquid)
        {
            meta.Row(
                "Liquid",
                $"{liquid.Type}  height {liquid.MinHeight:F2}..{liquid.MaxHeight:F2} yd  "
                + $"vertices={liquid.Heights.Length} flags={(liquid.TileFlags?.Length ?? 0)}",
                isImportant: true);
        }
        else
        {
            meta.Row("Liquid", "None");
        }

        meta.Row("Chunk Placement Refs", $"MDX/MCRD={chunkData.McrdReferences.Length}  WMO/MCRW={chunkData.McrwReferences.Length}");
        if (tileResult != null)
        {
            meta.Row("Tile Placements", $"MDDF/MDX={tileResult.MddfPlacements.Count}  MODF/WMO={tileResult.ModfPlacements.Count}");
        }

        if (chunkData.Layers.Length > 0)
        {
            InspectorSectionBuilder layersSection = builder.AddSection(primary ? "Texture Layers (Stratigraphy)" : "Terrain Texture Layers");
            for (int i = 0; i < chunkData.Layers.Length; i++)
            {
                TerrainLayer layer = chunkData.Layers[i];
                string textureName = ResolvePinnedTerrainTextureName(tileTextures, layer.TextureIndex);
                bool hasAlpha = i > 0 && chunkData.AlphaMaps.ContainsKey(i);
                layersSection.Row(
                    $"Layer {i}",
                    $"{textureName} (tex#{layer.TextureIndex}, flags=0x{layer.Flags:X8}, effect={layer.EffectId}, alpha={(hasAlpha ? "yes" : "no")})");
            }
        }

        var context = builder.AddSection("Chunk Context");
        AppendTerrainCameraContext(context, chunkData, chunkInfo);
        if (TryBuildTerrainWeakSignalTextureGuidance(chunkData, out TerrainWeakSignalTextureGuidance? guidance)
            && guidance != null)
        {
            context.Row(
                "Weak Sub-cells",
                $"{guidance.SelectedCellCount} ({guidance.BorderSelectedCellCount} border) "
                + $"range {guidance.ObservedMinHeight:F1}..{guidance.ObservedMaxHeight:F1} yd");
        }
        else
        {
            context.Row("Weak Sub-cells", "None detected for the current source Z band and alpha grouping");
        }

        builder.AddSection(primary ? "Chunk Actions" : "Terrain Actions")
            .Action("frame_terrain_chunk", "Frame Chunk")
            .Action("copy_chunk_texture_summary", "Copy Texture Summary")
            .Action("copy_terrain_coordinates", "Copy Coordinates")
            .Action("clear_terrain_chunk_selection", "Clear Chunk Selection");

        return true;
    }

    private bool TryGetPinnedTerrainChunkInspectionTarget(
        out TerrainRenderer.TerrainChunkInfo info,
        out TerrainInspectorTargetSource source)
    {
        info = default;
        source = default;

        TerrainRenderer? renderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
        if (renderer == null)
        {
            ClearSelectedTerrainChunk();
            return false;
        }

        InvalidateSelectedTerrainChunkIfWorldChanged();
        if (_selectedTerrainChunk is TerrainRenderer.TerrainChunkInfo pinned)
        {
            // A pin is tied to the resident renderer generation. AOI eviction must not leave the
            // Inspector showing stale bounds; clear it and continue through the live fallbacks.
            if (renderer.TryGetChunkInfo(pinned.TileX, pinned.TileY, pinned.ChunkX, pinned.ChunkY, out TerrainRenderer.TerrainChunkInfo residentPinned))
            {
                info = residentPinned;
                source = TerrainInspectorTargetSource.Pinned;
                return true;
            }

            ClearSelectedTerrainChunk();
        }

        if (TryPickTerrainChunkUnderMouse(renderer, out info))
        {
            source = TerrainInspectorTargetSource.Hovered;
            return true;
        }

        TerrainRenderer.TerrainChunkInfo? cameraChunk = renderer.GetChunkInfoAt(_camera.Position.X, _camera.Position.Y);
        if (cameraChunk.HasValue)
        {
            info = cameraChunk.Value;
            source = TerrainInspectorTargetSource.Camera;
            return true;
        }

        return false;
    }

    private bool TryResolvePinnedTerrainChunkInspectionData(
        TerrainRenderer.TerrainChunkInfo chunkInfo,
        out TerrainChunkData? chunkData,
        out IReadOnlyList<string>? tileTextures,
        out TileLoadResult? tileResult)
    {
        chunkData = null;
        tileTextures = null;
        tileResult = null;

        if (_terrainManager != null)
        {
            if (!_terrainManager.TryGetTileLoadResult(chunkInfo.TileX, chunkInfo.TileY, out TileLoadResult result))
                return false;

            tileResult = result;
            chunkData = FindPinnedTerrainChunk(result, chunkInfo.ChunkX, chunkInfo.ChunkY);
            _terrainManager.Adapter.TileTextures.TryGetValue((chunkInfo.TileX, chunkInfo.TileY), out List<string>? textures);
            tileTextures = textures;
            return chunkData != null;
        }

        if (_vlmTerrainManager != null
            && _vlmTerrainManager.TryGetTileLoadResult(chunkInfo.TileX, chunkInfo.TileY, out TileLoadResult vlmResult))
        {
            tileResult = vlmResult;
            chunkData = FindPinnedTerrainChunk(vlmResult, chunkInfo.ChunkX, chunkInfo.ChunkY);
            _vlmTerrainManager.Loader.TileTextures.TryGetValue((chunkInfo.TileX, chunkInfo.TileY), out List<string>? textures);
            tileTextures = textures;
            return chunkData != null;
        }

        return false;
    }

    private static TerrainChunkData? FindPinnedTerrainChunk(TileLoadResult result, int chunkX, int chunkY)
    {
        for (int i = 0; i < result.Chunks.Count; i++)
        {
            TerrainChunkData chunk = result.Chunks[i];
            if (chunk.ChunkX == chunkX && chunk.ChunkY == chunkY)
                return chunk;
        }

        return null;
    }

    private void SetSelectedTerrainChunk(TerrainRenderer.TerrainChunkInfo info)
    {
        object? owner = GetActiveTerrainInspectionOwner();
        if (owner == null)
        {
            ClearSelectedTerrainChunk();
            return;
        }

        _selectedTerrainChunk = info;
        _selectedTerrainChunkOwner = owner;
        _selectedTerrainChunkMapName = GetActiveTerrainInspectionMapName();
        _selectedTerrainChunkMapId = _currentMapId;
        _statusMessage = $"Pinned ADT chunk tile({info.TileX},{info.TileY}) MCNK({info.ChunkX},{info.ChunkY}) for Inspector.";
    }

    private void SelectTerrainChunkFromClick(TerrainRenderer.TerrainChunkInfo info)
    {
        // A terrain pin is the active selection context only when the click did not resolve to a
        // higher-priority scene object. Clear the old object context so a previous MDX/WMO cannot
        // keep the Inspector on an unrelated detail surface after the operator clicks the ground.
        _worldScene?.ClearSelection();
        _worldScene?.ClearTaxiSelection();
        _worldScene?.ClearPm4ObjectSelection();
        ClearSelectedWlLiquidBody(clearListIsolation: true);
        ClearSelectedAreaPoiInfo();
        _selectedObjectIndex = -1;
        _selectedObjectType = string.Empty;
        _selectedObjectInfo = string.Empty;
        SetSelectedTerrainChunk(info);
    }

    private void ClearSelectedTerrainChunk()
    {
        bool hadSelection = _selectedTerrainChunk.HasValue;
        _selectedTerrainChunk = null;
        _selectedTerrainChunkOwner = null;
        _selectedTerrainChunkMapName = null;
        _selectedTerrainChunkMapId = int.MinValue;
        if (hadSelection)
            _statusMessage = "Cleared pinned terrain chunk selection.";
    }

    private void InvalidateSelectedTerrainChunkIfWorldChanged()
    {
        if (!_selectedTerrainChunk.HasValue)
            return;

        object? activeOwner = GetActiveTerrainInspectionOwner();
        string? activeMapName = GetActiveTerrainInspectionMapName();
        bool ownerChanged = activeOwner == null || !ReferenceEquals(_selectedTerrainChunkOwner, activeOwner);
        bool mapChanged = !string.Equals(_selectedTerrainChunkMapName, activeMapName, StringComparison.OrdinalIgnoreCase);
        bool mapIdChanged = _selectedTerrainChunkMapId >= 0 && _currentMapId >= 0 && _selectedTerrainChunkMapId != _currentMapId;
        if (ownerChanged || mapChanged || mapIdChanged)
            ClearSelectedTerrainChunk();
    }

    private object? GetActiveTerrainInspectionOwner()
        => _terrainManager is not null ? _terrainManager : _vlmTerrainManager;

    private string? GetActiveTerrainInspectionMapName()
        => _terrainManager?.MapName ?? _vlmTerrainManager?.MapName;

    private static string DescribeTerrainInspectorTargetSource(TerrainInspectorTargetSource source)
        => source switch
        {
            TerrainInspectorTargetSource.Pinned => "Pinned Click Target",
            TerrainInspectorTargetSource.Hovered => "Hovered Cursor Chunk",
            _ => "Camera Chunk",
        };

    private string ResolveTerrainAreaName(int areaId)
    {
        if (_areaTableService == null)
            return "Unavailable (AreaTable not loaded)";

        AreaLookupResult lookup = _areaTableService.ResolveArea(areaId, _currentMapId);
        if (lookup.PrimaryText == null)
            return $"Unknown ({areaId}; {lookup.Reason})";

        string display = lookup.ZoneText == lookup.SubzoneText || lookup.ZoneText == null
            ? lookup.SubzoneText ?? lookup.PrimaryText
            : $"{lookup.ZoneText} > {lookup.SubzoneText}";
        return lookup.IsResolved ? display : $"{display} ({lookup.Reason})";
    }

    private static int CountTerrainHoleCells(int holeMask)
    {
        uint value = unchecked((uint)holeMask) & 0xFFFFu;
        int count = 0;
        while (value != 0)
        {
            count += (int)(value & 1u);
            value >>= 1;
        }

        return count;
    }

    private static string DescribeTerrainElevation(TerrainChunkData chunk)
    {
        if (chunk.Heights.Length == 0)
            return "Unavailable";

        float min = float.MaxValue;
        float max = float.MinValue;
        for (int i = 0; i < chunk.Heights.Length; i++)
        {
            float height = chunk.Heights[i];
            if (!float.IsFinite(height))
                continue;
            min = MathF.Min(min, height);
            max = MathF.Max(max, height);
        }

        return min == float.MaxValue ? "Unavailable" : $"{min:F2} .. {max:F2} yd (Δ {(max - min):F2})";
    }

    private static string DescribeTerrainBytePayload(byte[]? bytes, string shape)
        => bytes is { Length: > 0 } ? $"Present ({shape}, {bytes.Length} bytes)" : "None";

    private void AppendTerrainCameraContext(
        InspectorSectionBuilder section,
        TerrainChunkData chunk,
        TerrainRenderer.TerrainChunkInfo chunkInfo)
    {
        float localX = _camera.Position.X - chunk.WorldPosition.X;
        float localY = _camera.Position.Y - chunk.WorldPosition.Y;
        float spanX = MathF.Abs(chunkInfo.BoundsMax.X - chunkInfo.BoundsMin.X);
        float spanY = MathF.Abs(chunkInfo.BoundsMax.Y - chunkInfo.BoundsMin.Y);
        float cellSizeX = spanX / 16f;
        float cellSizeY = spanY / 16f;
        bool inside = localX >= 0f && localX <= spanX && localY >= 0f && localY <= spanY
            && cellSizeX > 0f && cellSizeY > 0f;
        if (inside)
        {
            int cellX = Math.Clamp((int)MathF.Floor(localX / cellSizeX), 0, 15);
            int cellY = Math.Clamp((int)MathF.Floor(localY / cellSizeY), 0, 15);
            section.Row("Camera Cell", $"({cellX}, {cellY})");
        }
        else
        {
            section.Row("Camera Cell", "Outside chunk");
        }

        section.Row("Camera Local Offset", $"({localX:F2}, {localY:F2}, {_camera.Position.Z - chunk.WorldPosition.Z:F2}) yd");
    }

    private static string ResolvePinnedTerrainTextureName(IReadOnlyList<string>? tileTextures, int textureIndex)
    {
        if (tileTextures == null || textureIndex < 0 || textureIndex >= tileTextures.Count)
            return "<missing>";

        string name = tileTextures[textureIndex];
        return string.IsNullOrWhiteSpace(name) ? "<empty>" : name;
    }

    private static string BuildTerrainChunkCoordinates(TerrainRenderer.TerrainChunkInfo info)
    {
        float minWowX = WoWConstants.MapOrigin - info.BoundsMax.Y;
        float maxWowX = WoWConstants.MapOrigin - info.BoundsMin.Y;
        float minWowY = WoWConstants.MapOrigin - info.BoundsMax.X;
        float maxWowY = WoWConstants.MapOrigin - info.BoundsMin.X;
        return $"Tile ({info.TileY}, {info.TileX}) MCNK ({info.ChunkX}, {info.ChunkY})\n"
            + $"Renderer bounds: min {FormatVector(info.BoundsMin)} max {FormatVector(info.BoundsMax)}\n"
            + $"WoW bounds: X {minWowX:F2}..{maxWowX:F2}, Y {minWowY:F2}..{maxWowY:F2}, Z {info.BoundsMin.Z:F2}..{info.BoundsMax.Z:F2}";
    }

    private static string FormatVector(Vector3 value)
        => $"({value.X:F2}, {value.Y:F2}, {value.Z:F2})";

    private bool BuildWorldOverviewInspector(InspectorContentBuilder builder)
    {
        if (_worldScene == null && _terrainManager == null && _vlmTerrainManager == null)
            return false;

        string mapName = _terrainManager?.MapName
            ?? _vlmTerrainManager?.MapName
            ?? "World";
        builder.ObjectType = "World";
        builder.Headline = $"World Overview: {mapName}";

        var overview = builder.AddSection("World Overview");
        overview.Row("Map", mapName, isImportant: true);
        overview.Row("Camera Position", FormatVector(_camera.Position));
        int tileX = (int)MathF.Floor((WoWConstants.MapOrigin - _camera.Position.X) / WoWConstants.ChunkSize);
        int tileY = (int)MathF.Floor((WoWConstants.MapOrigin - _camera.Position.Y) / WoWConstants.ChunkSize);
        overview.Row("Camera Tile (X, Y)", $"({tileX}, {tileY})");
        overview.Row("Area", string.IsNullOrWhiteSpace(_currentAreaName) ? "Unknown" : _currentAreaName);

        if (_terrainManager != null)
        {
            overview.Row("Loaded Tiles", $"{_terrainManager.LoadedTileCount}");
            overview.Row("Loaded Chunks", $"{_terrainManager.LoadedChunkCount}");
            overview.Row("Streaming", _terrainManager.IsStreaming ? $"pending={_terrainManager.PendingTerrainLoadCount}" : "settled");
        }
        else if (_vlmTerrainManager != null)
        {
            overview.Row("Loaded Tiles", $"{_vlmTerrainManager.LoadedTileCount}");
            overview.Row("Loaded Chunks", $"{_vlmTerrainManager.LoadedChunkCount}");
        }

        if (_worldScene != null)
        {
            overview.Row("World Objects", $"MDX={_worldScene.MdxInstanceCount}  WMO={_worldScene.WmoInstanceCount}");
            overview.Row("Pending Object Loads", $"{_worldScene.PendingWorldObjectLoadCount}");
        }

        return true;
    }
}
