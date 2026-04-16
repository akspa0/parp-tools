using System.Numerics;
using System.Text.Json;
using ImGuiNET;
using MdxViewer.Export;
using Silk.NET.OpenGL;
using Image = SixLabors.ImageSharp.Image;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace MdxViewer;

public partial class ViewerApp
{
    private void DrawTerrainAnalysisWindow()
    {
        if (_terrainManager == null && _vlmTerrainManager == null)
        {
            _showTerrainAnalysisWindow = false;
            return;
        }

        EnsureTerrainAnalysisTextures();

        var cameraTile = GetCameraTile();
        if (_terrainAnalysisFollowCameraTile && (!_terrainAnalysisPreviewTile.HasValue || _terrainAnalysisPreviewTile.Value != cameraTile))
            RefreshTerrainAnalysisCurrentTile(cameraTile);

        ImGui.SetNextWindowSize(new Vector2(980f, 860f), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin("Terrain Analysis", ref _showTerrainAnalysisWindow, ImGuiWindowFlags.NoCollapse))
        {
            ImGui.End();
            return;
        }

        ImGui.Text($"Camera Tile: ({cameraTile.tileX}, {cameraTile.tileY})");
        if (_terrainAnalysisPreviewTile.HasValue)
        {
            var previewTile = _terrainAnalysisPreviewTile.Value;
            ImGui.SameLine();
            ImGui.TextDisabled($"Preview Tile: ({previewTile.tileX}, {previewTile.tileY})");
        }

        ImGui.Checkbox("Follow camera tile", ref _terrainAnalysisFollowCameraTile);
        ImGui.SameLine();
        if (ImGui.Button("Refresh Current Tile"))
            RefreshTerrainAnalysisCurrentTile(_terrainAnalysisPreviewTile ?? cameraTile);

        ImGui.SameLine();
        if (ImGui.Button("Save Preview Set"))
            SaveTerrainAnalysisPreviewSet(_terrainAnalysisPreviewTile ?? cameraTile);

        int scopeIndex = _terrainAnalysisGlobalScope == TerrainTileScope.WholeMap ? 1 : 0;
        string[] scopeLabels = { "Loaded tiles", "Whole map" };
        if (ImGui.Combo("Global Bounds Source", ref scopeIndex, scopeLabels, scopeLabels.Length))
            _terrainAnalysisGlobalScope = scopeIndex == 1 ? TerrainTileScope.WholeMap : TerrainTileScope.LoadedTiles;

        ImGui.SameLine();
        if (ImGui.Button("Recompute Global Bounds"))
            RefreshTerrainAnalysisGlobalBounds();

        if (_terrainAnalysisHasGlobalBounds)
            ImGui.Text($"Global Range: {_terrainAnalysisGlobalMin:F3} to {_terrainAnalysisGlobalMax:F3} across {_terrainAnalysisGlobalTileCount} tile(s)");
        else
            ImGui.TextDisabled("Global bounds not computed yet.");

        if (_terrainAnalysisPreviewTile.HasValue)
            ImGui.Text($"Current Tile Range: {_terrainAnalysisPreviewTileMin:F3} to {_terrainAnalysisPreviewTileMax:F3}");

        if (_terrainAnalysisHasGlobalBounds && _terrainAnalysisPreviewTile.HasValue)
        {
            ImGui.Text($"Current Tile Relief: {_terrainAnalysisPreviewTileMax - _terrainAnalysisPreviewTileMin:F3} ({_terrainAnalysisPreviewVisibilityRatio:P2} of global range, x{_terrainAnalysisPreviewAmplification:F1} local amplification)");
            if (_terrainAnalysisPreviewCompareTile.HasValue && _terrainAnalysisPreviewSimilarity.HasValue)
            {
                var compareTile = _terrainAnalysisPreviewCompareTile.Value;
                ImGui.Text($"Offset Match ({_terrainAnalysisHiddenCompareOffsetX:+#;-#;0}, {_terrainAnalysisHiddenCompareOffsetY:+#;-#;0}): tile ({compareTile.tileX}, {compareTile.tileY}) similarity {_terrainAnalysisPreviewSimilarity.Value:P1}");
            }
        }

        if (!string.IsNullOrWhiteSpace(_terrainAnalysisStatus))
            ImGui.TextWrapped(_terrainAnalysisStatus);

        ImGui.Separator();

        if (ImGui.BeginTable("##terrainAnalysisPreviews", 2, ImGuiTableFlags.SizingStretchSame))
        {
            ImGui.TableNextColumn();
            DrawTerrainAnalysisPreviewPane(
                "Current Tile Heightmap",
                _terrainAnalysisLocalTexture,
                "Per-tile normalization. This stretches the current tile across its own min/max range.");

            ImGui.TableNextColumn();
            DrawTerrainAnalysisPreviewPane(
                "Map-Normalized Tile Heightmap",
                _terrainAnalysisGlobalTexture,
                _terrainAnalysisHasGlobalBounds
                    ? "Current tile remapped against the selected loaded-tile or whole-map min/max bounds."
                    : "Compute global bounds to expose terrain that is visually compressed inside the local tile range.");

            ImGui.EndTable();
        }

        ImGui.Separator();
        if (ImGui.CollapsingHeader("Packed Alpha/Shadow Atlas", ImGuiTreeNodeFlags.DefaultOpen))
        {
            ImGui.TextWrapped("RGB encodes terrain alpha layers 1-3, and the alpha channel encodes the shadow map. This matches the packed atlas now emitted by the ML dataset exporter.");
            DrawTerrainAnalysisPreviewPane("Alpha/Shadow Atlas", _terrainAnalysisAlphaTexture, null);
        }

        ImGui.Separator();
        DrawHiddenTerrainCandidatesSection();

        ImGui.End();
    }

    private void DrawHiddenTerrainCandidatesSection()
    {
        if (_terrainManager == null && _vlmTerrainManager == null)
        {
            ImGui.TextDisabled("Hidden-terrain candidate scanning requires an active terrain source.");
            return;
        }

        if (!ImGui.CollapsingHeader("Hidden Terrain Candidates", ImGuiTreeNodeFlags.DefaultOpen))
            return;

        ImGui.TextWrapped("Compares each tile's locally normalized relief against an offset neighbor using the active viewer terrain source. This works against normal map loading too, including odd split-ADT or placeholder cases already handled by the terrain adapters.");

        int hiddenScopeIndex = _terrainAnalysisHiddenScope switch
        {
            TerrainTileScope.WholeMap => 1,
            TerrainTileScope.CurrentTile => 2,
            _ => 0,
        };
        string[] hiddenScopeLabels = { "Loaded tiles", "Whole map", "Current tile" };
        ImGui.SetNextItemWidth(180f);
        if (ImGui.Combo("Candidate Source", ref hiddenScopeIndex, hiddenScopeLabels, hiddenScopeLabels.Length))
        {
            _terrainAnalysisHiddenScope = hiddenScopeIndex switch
            {
                1 => TerrainTileScope.WholeMap,
                2 => TerrainTileScope.CurrentTile,
                _ => TerrainTileScope.LoadedTiles,
            };
        }

        ImGui.SetNextItemWidth(120f);
        ImGui.InputInt("Offset X", ref _terrainAnalysisHiddenCompareOffsetX);
        ImGui.SameLine();
        ImGui.SetNextItemWidth(120f);
        ImGui.InputInt("Offset Y", ref _terrainAnalysisHiddenCompareOffsetY);
        ImGui.SameLine();
        if (ImGui.SmallButton("Use 2 Tiles South"))
        {
            _terrainAnalysisHiddenCompareOffsetX = 0;
            _terrainAnalysisHiddenCompareOffsetY = 2;
        }

        ImGui.SetNextItemWidth(220f);
        ImGui.SliderFloat("Min Similarity", ref _terrainAnalysisHiddenMinSimilarity, 0.50f, 0.999f, "%.3f");
        ImGui.SetNextItemWidth(220f);
        ImGui.SliderFloat("Max Visibility Ratio", ref _terrainAnalysisHiddenMaxVisibilityRatio, 0.005f, 0.250f, "%.3f");
        ImGui.SetNextItemWidth(220f);
        ImGui.SliderInt("Max Results", ref _terrainAnalysisHiddenMaxResults, 1, 100);

        if (ImGui.Button("Scan Tiles"))
            RefreshHiddenTerrainCandidates();

        ImGui.SameLine();
        if (ImGui.Button("Clear Results"))
        {
            _terrainAnalysisHiddenCandidates.Clear();
            _terrainAnalysisHiddenSelectedIndex = -1;
            _terrainAnalysisHiddenStatus = "Hidden-terrain results cleared.";
        }

        if (!string.IsNullOrWhiteSpace(_terrainAnalysisHiddenStatus))
            ImGui.TextWrapped(_terrainAnalysisHiddenStatus);

        if (_terrainAnalysisHiddenCandidates.Count == 0)
        {
            ImGui.TextDisabled("No candidate tiles loaded yet.");
            return;
        }

        if (!ImGui.BeginTable("##hiddenTerrainCandidates", 6, ImGuiTableFlags.Borders | ImGuiTableFlags.RowBg | ImGuiTableFlags.ScrollY | ImGuiTableFlags.SizingStretchProp, new Vector2(0f, 280f)))
            return;

        ImGui.TableSetupColumn("Tile");
        ImGui.TableSetupColumn("Offset Match");
        ImGui.TableSetupColumn("Similarity");
        ImGui.TableSetupColumn("Visibility");
        ImGui.TableSetupColumn("Relief");
        ImGui.TableSetupColumn("Preview", ImGuiTableColumnFlags.WidthFixed, 90f);
        ImGui.TableHeadersRow();

        for (int index = 0; index < _terrainAnalysisHiddenCandidates.Count; index++)
        {
            TerrainHiddenTileCandidate candidate = _terrainAnalysisHiddenCandidates[index];
            ImGui.TableNextRow();

            ImGui.TableNextColumn();
            bool isSelected = _terrainAnalysisHiddenSelectedIndex == index;
            if (ImGui.Selectable($"({candidate.Tile.tileX}, {candidate.Tile.tileY})##hiddenCandidate{index}", isSelected, ImGuiSelectableFlags.SpanAllColumns))
            {
                _terrainAnalysisHiddenSelectedIndex = index;
                PreviewTerrainAnalysisTile(candidate.Tile);
            }

            ImGui.TableNextColumn();
            ImGui.Text($"({candidate.CompareTile.tileX}, {candidate.CompareTile.tileY})");

            ImGui.TableNextColumn();
            ImGui.Text($"{candidate.Similarity:P1}");

            ImGui.TableNextColumn();
            ImGui.Text($"{candidate.VisibilityRatio:P2}");

            ImGui.TableNextColumn();
            ImGui.Text($"{candidate.ReliefRange:F3}");

            ImGui.TableNextColumn();
            if (ImGui.SmallButton($"Preview##hiddenPreview{index}"))
            {
                _terrainAnalysisHiddenSelectedIndex = index;
                PreviewTerrainAnalysisTile(candidate.Tile);
            }
        }

        ImGui.EndTable();
    }

    private void DrawTerrainAnalysisPreviewPane(string title, TerrainAnalysisPreviewTexture? texture, string? description)
    {
        ImGui.Text(title);
        if (!string.IsNullOrWhiteSpace(description))
            ImGui.TextWrapped(description);

        if (texture == null || !texture.HasTexture)
        {
            ImGui.TextDisabled("Preview unavailable.");
            return;
        }

        float maxWidth = MathF.Max(1f, ImGui.GetContentRegionAvail().X);
        float previewWidth = MathF.Min(maxWidth, 460f);
        float previewHeight = previewWidth * texture.Height / MathF.Max(1f, texture.Width);
        ImGui.Image((nint)texture.TextureId, new Vector2(previewWidth, previewHeight));
    }

    private void RefreshTerrainAnalysisCurrentTile((int tileX, int tileY) tile)
    {
        var chunks = LoadTileChunksForExport(tile.tileX, tile.tileY);
        if (chunks == null || chunks.Count == 0)
        {
            _terrainAnalysisStatus = $"No terrain data available for tile ({tile.tileX}, {tile.tileY}).";
            ClearTerrainAnalysisTextures(clearGlobal: false);
            _terrainAnalysisPreviewTile = tile;
            return;
        }

        var tileHeightmap = TerrainHeightmapIo.BuildTileHeightmap257(chunks);
        _terrainAnalysisPreviewTile = tile;
        _terrainAnalysisPreviewTileMin = tileHeightmap.MinHeight;
        _terrainAnalysisPreviewTileMax = tileHeightmap.MaxHeight;
        float reliefRange = Math.Max(tileHeightmap.MaxHeight - tileHeightmap.MinHeight, 0f);

        var localPixels = BuildHeightPreviewPixels(
            tileHeightmap.Heights,
            tileHeightmap.MinHeight,
            tileHeightmap.MaxHeight,
            TerrainHeightmapIo.TileHeightmapSize,
            TerrainHeightmapIo.TileHeightmapSize);
        _terrainAnalysisLocalTexture?.Update(localPixels, TerrainHeightmapIo.TileHeightmapSize, TerrainHeightmapIo.TileHeightmapSize);

        if (_terrainAnalysisHasGlobalBounds)
        {
            float globalRange = Math.Max(_terrainAnalysisGlobalMax - _terrainAnalysisGlobalMin, 1e-6f);
            _terrainAnalysisPreviewVisibilityRatio = reliefRange / globalRange;
            _terrainAnalysisPreviewAmplification = reliefRange > 1e-6f
                ? globalRange / reliefRange
                : 0f;

            var globalPixels = BuildHeightPreviewPixels(
                tileHeightmap.Heights,
                _terrainAnalysisGlobalMin,
                _terrainAnalysisGlobalMax,
                TerrainHeightmapIo.TileHeightmapSize,
                TerrainHeightmapIo.TileHeightmapSize);
            _terrainAnalysisGlobalTexture?.Update(globalPixels, TerrainHeightmapIo.TileHeightmapSize, TerrainHeightmapIo.TileHeightmapSize);
        }
        else
        {
            _terrainAnalysisPreviewVisibilityRatio = 0f;
            _terrainAnalysisPreviewAmplification = 1f;
            _terrainAnalysisGlobalTexture?.Dispose();
            _terrainAnalysisGlobalTexture = new TerrainAnalysisPreviewTexture(_gl);
        }

        UpdateTerrainAnalysisPreviewSimilarity(tile, tileHeightmap);

        using (var atlas = TerrainImageIo.BuildAlphaAtlasFromChunks(chunks))
        {
            var alphaPixels = new byte[atlas.Width * atlas.Height * 4];
            atlas.CopyPixelDataTo(alphaPixels);
            _terrainAnalysisAlphaTexture?.Update(alphaPixels, atlas.Width, atlas.Height);
        }

        _terrainAnalysisStatus = $"Terrain analysis refreshed for tile ({tile.tileX}, {tile.tileY}).";
    }

    private void RefreshTerrainAnalysisGlobalBounds()
    {
        var scope = _terrainAnalysisGlobalScope == TerrainTileScope.WholeMap
            ? TerrainTileScope.WholeMap
            : TerrainTileScope.LoadedTiles;
        var tiles = GetTileScopeList(scope);
        if (tiles.Count == 0)
        {
            _terrainAnalysisHasGlobalBounds = false;
            _terrainAnalysisGlobalTileCount = 0;
            _terrainAnalysisStatus = "No tiles available for global terrain analysis.";
            return;
        }

        float minHeight = float.MaxValue;
        float maxHeight = float.MinValue;
        int validTileCount = 0;

        foreach (var tile in tiles)
        {
            var chunks = LoadTileChunksForExport(tile.tileX, tile.tileY);
            if (!TryGetChunkHeightRange(chunks, out float tileMin, out float tileMax))
                continue;

            validTileCount++;
            if (tileMin < minHeight) minHeight = tileMin;
            if (tileMax > maxHeight) maxHeight = tileMax;
        }

        if (validTileCount == 0)
        {
            _terrainAnalysisHasGlobalBounds = false;
            _terrainAnalysisGlobalTileCount = 0;
            _terrainAnalysisStatus = "No valid heights were found while computing global bounds.";
            return;
        }

        _terrainAnalysisHasGlobalBounds = true;
        _terrainAnalysisGlobalMin = minHeight;
        _terrainAnalysisGlobalMax = maxHeight;
        _terrainAnalysisGlobalTileCount = validTileCount;
        _terrainAnalysisStatus = _terrainAnalysisGlobalScope == TerrainTileScope.WholeMap
            ? $"Computed whole-map terrain bounds across {validTileCount} tile(s)."
            : $"Computed loaded-tile terrain bounds across {validTileCount} tile(s).";

        RefreshTerrainAnalysisCurrentTile(_terrainAnalysisPreviewTile ?? GetCameraTile());
    }

    private static bool TryGetChunkHeightRange(IReadOnlyList<Terrain.TerrainChunkData>? chunks, out float minHeight, out float maxHeight)
    {
        minHeight = float.MaxValue;
        maxHeight = float.MinValue;

        if (chunks == null)
            return false;

        foreach (var chunk in chunks)
        {
            if (chunk.Heights == null)
                continue;

            foreach (float height in chunk.Heights)
            {
                if (float.IsNaN(height) || float.IsInfinity(height))
                    continue;

                if (height < minHeight) minHeight = height;
                if (height > maxHeight) maxHeight = height;
            }
        }

        return minHeight != float.MaxValue && maxHeight != float.MinValue;
    }

    private void EnsureTerrainAnalysisTextures()
    {
        _terrainAnalysisLocalTexture ??= new TerrainAnalysisPreviewTexture(_gl);
        _terrainAnalysisGlobalTexture ??= new TerrainAnalysisPreviewTexture(_gl);
        _terrainAnalysisAlphaTexture ??= new TerrainAnalysisPreviewTexture(_gl);
    }

    private void ClearTerrainAnalysisTextures(bool clearGlobal)
    {
        _terrainAnalysisLocalTexture?.Dispose();
        _terrainAnalysisLocalTexture = new TerrainAnalysisPreviewTexture(_gl);
        _terrainAnalysisAlphaTexture?.Dispose();
        _terrainAnalysisAlphaTexture = new TerrainAnalysisPreviewTexture(_gl);

        if (clearGlobal)
        {
            _terrainAnalysisGlobalTexture?.Dispose();
            _terrainAnalysisGlobalTexture = new TerrainAnalysisPreviewTexture(_gl);
        }

        _terrainAnalysisPreviewCompareTile = null;
        _terrainAnalysisPreviewSimilarity = null;
        _terrainAnalysisPreviewVisibilityRatio = 0f;
        _terrainAnalysisPreviewAmplification = 1f;
    }

    private void PreviewTerrainAnalysisTile((int tileX, int tileY) tile)
    {
        _terrainAnalysisFollowCameraTile = false;
        RefreshTerrainAnalysisCurrentTile(tile);
    }

    private string CreateTerrainAnalysisOutputDirectory((int tileX, int tileY) tile)
    {
        string root = Path.Combine(EnsureEditorProjectOutputDirectory(), "terrain-analysis", $"tile_{tile.tileX}_{tile.tileY}");
        Directory.CreateDirectory(root);

        string timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
        string candidate = Path.Combine(root, timestamp);
        int suffix = 1;
        while (Directory.Exists(candidate))
        {
            candidate = Path.Combine(root, $"{timestamp}_{suffix:D2}");
            suffix++;
        }

        Directory.CreateDirectory(candidate);
        return candidate;
    }

    private void SaveTerrainAnalysisPreviewSet((int tileX, int tileY) tile)
    {
        var chunks = LoadTileChunksForExport(tile.tileX, tile.tileY);
        if (chunks == null || chunks.Count == 0)
        {
            _terrainAnalysisStatus = $"No terrain data available to save for tile ({tile.tileX}, {tile.tileY}).";
            return;
        }

        TerrainHeightmapIo.TileHeightmap257 tileHeightmap = TerrainHeightmapIo.BuildTileHeightmap257(chunks);
        string outputDir = CreateTerrainAnalysisOutputDirectory(tile);
        string tileStem = $"tile_{tile.tileX}_{tile.tileY}";

        byte[] localPixels = BuildHeightPreviewPixels(
            tileHeightmap.Heights,
            tileHeightmap.MinHeight,
            tileHeightmap.MaxHeight,
            TerrainHeightmapIo.TileHeightmapSize,
            TerrainHeightmapIo.TileHeightmapSize);
        string localPath = Path.Combine(outputDir, $"{tileStem}_preview_local.png");
        using (Image<Rgba32> localImage = Image.LoadPixelData<Rgba32>(localPixels, TerrainHeightmapIo.TileHeightmapSize, TerrainHeightmapIo.TileHeightmapSize))
            localImage.SaveAsPng(localPath);

        string? globalPath = null;
        if (_terrainAnalysisHasGlobalBounds)
        {
            byte[] globalPixels = BuildHeightPreviewPixels(
                tileHeightmap.Heights,
                _terrainAnalysisGlobalMin,
                _terrainAnalysisGlobalMax,
                TerrainHeightmapIo.TileHeightmapSize,
                TerrainHeightmapIo.TileHeightmapSize);
            globalPath = Path.Combine(outputDir, $"{tileStem}_preview_global.png");
            using (Image<Rgba32> globalImage = Image.LoadPixelData<Rgba32>(globalPixels, TerrainHeightmapIo.TileHeightmapSize, TerrainHeightmapIo.TileHeightmapSize))
                globalImage.SaveAsPng(globalPath);
        }

        string atlasPath = Path.Combine(outputDir, $"{tileStem}_alpha_shadow_atlas.png");
        using (Image<Rgba32> atlasImage = TerrainImageIo.BuildAlphaAtlasFromChunks(chunks))
            atlasImage.SaveAsPng(atlasPath);

        string metadataPath = Path.Combine(outputDir, $"{tileStem}_preview_metadata.json");
        var metadata = new
        {
            tile_x = tile.tileX,
            tile_y = tile.tileY,
            saved_at_utc = DateTime.UtcNow.ToString("O"),
            local_min_height = tileHeightmap.MinHeight,
            local_max_height = tileHeightmap.MaxHeight,
            global_min_height = _terrainAnalysisHasGlobalBounds ? _terrainAnalysisGlobalMin : (float?)null,
            global_max_height = _terrainAnalysisHasGlobalBounds ? _terrainAnalysisGlobalMax : (float?)null,
            local_preview = Path.GetFileName(localPath),
            global_preview = globalPath == null ? null : Path.GetFileName(globalPath),
            alpha_shadow_atlas = Path.GetFileName(atlasPath),
            global_bounds_source = _terrainAnalysisHasGlobalBounds ? _terrainAnalysisGlobalScope.ToString() : null,
        };
        File.WriteAllText(metadataPath, JsonSerializer.Serialize(metadata, new JsonSerializerOptions { WriteIndented = true }));

        _terrainAnalysisStatus = $"Saved terrain analysis preview set for tile ({tile.tileX}, {tile.tileY}) to {outputDir}.";
    }

    private void RefreshHiddenTerrainCandidates()
    {
        _terrainAnalysisHiddenCandidates.Clear();
        _terrainAnalysisHiddenSelectedIndex = -1;

        if (_terrainManager == null && _vlmTerrainManager == null)
        {
            _terrainAnalysisHiddenStatus = "Hidden-terrain scanning requires an active terrain source.";
            return;
        }

        IReadOnlyList<(int tileX, int tileY)> tiles = GetTileScopeList(_terrainAnalysisHiddenScope);
        if (tiles.Count == 0)
        {
            _terrainAnalysisHiddenStatus = "No tiles are available for hidden-terrain scanning.";
            return;
        }

        var summaries = new Dictionary<(int tileX, int tileY), TerrainHiddenTileSummary>(tiles.Count);
        float globalMin = float.MaxValue;
        float globalMax = float.MinValue;

        foreach (var tile in tiles)
        {
            if (!TryBuildTerrainHiddenTileSummary(tile, out TerrainHiddenTileSummary summary))
                continue;

            summaries[tile] = summary;
            if (summary.MinHeight < globalMin) globalMin = summary.MinHeight;
            if (summary.MaxHeight > globalMax) globalMax = summary.MaxHeight;
        }

        if (summaries.Count == 0)
        {
            _terrainAnalysisHiddenStatus = "No valid tile heightmaps were available for hidden-terrain scanning.";
            return;
        }

        float globalRange = Math.Max(globalMax - globalMin, 1e-6f);
        foreach (var summary in summaries.Values)
        {
            var compareTile = (
                summary.Tile.tileX + _terrainAnalysisHiddenCompareOffsetX,
                summary.Tile.tileY + _terrainAnalysisHiddenCompareOffsetY);

            if (!summaries.TryGetValue(compareTile, out TerrainHiddenTileSummary? compareSummary))
                continue;

            float visibilityRatio = summary.ReliefRange / globalRange;
            if (visibilityRatio > _terrainAnalysisHiddenMaxVisibilityRatio)
                continue;

            float similarity = ComputeTerrainHiddenSimilarity(summary.Feature, compareSummary.Feature);
            if (similarity < _terrainAnalysisHiddenMinSimilarity)
                continue;

            _terrainAnalysisHiddenCandidates.Add(new TerrainHiddenTileCandidate
            {
                Tile = summary.Tile,
                CompareTile = compareTile,
                ReliefRange = summary.ReliefRange,
                VisibilityRatio = visibilityRatio,
                Similarity = similarity
            });
        }

        _terrainAnalysisHiddenCandidates.Sort(static (left, right) =>
        {
            int similarityCompare = right.Similarity.CompareTo(left.Similarity);
            if (similarityCompare != 0)
                return similarityCompare;

            return left.VisibilityRatio.CompareTo(right.VisibilityRatio);
        });

        int maxResults = Math.Clamp(_terrainAnalysisHiddenMaxResults, 1, 100);
        if (_terrainAnalysisHiddenCandidates.Count > maxResults)
            _terrainAnalysisHiddenCandidates.RemoveRange(maxResults, _terrainAnalysisHiddenCandidates.Count - maxResults);

        _terrainAnalysisHiddenStatus = _terrainAnalysisHiddenCandidates.Count == 0
            ? $"No candidates matched similarity >= {_terrainAnalysisHiddenMinSimilarity:F3} with visibility <= {_terrainAnalysisHiddenMaxVisibilityRatio:F3}."
            : $"Found {_terrainAnalysisHiddenCandidates.Count} candidate tile(s) using offset ({_terrainAnalysisHiddenCompareOffsetX:+#;-#;0}, {_terrainAnalysisHiddenCompareOffsetY:+#;-#;0}) across {summaries.Count} scanned tile(s) from {_terrainAnalysisHiddenScope}.";
    }

    private void UpdateTerrainAnalysisPreviewSimilarity((int tileX, int tileY) tile, TerrainHeightmapIo.TileHeightmap257 tileHeightmap)
    {
        _terrainAnalysisPreviewCompareTile = null;
        _terrainAnalysisPreviewSimilarity = null;

        var compareTile = (
            tile.tileX + _terrainAnalysisHiddenCompareOffsetX,
            tile.tileY + _terrainAnalysisHiddenCompareOffsetY);

        if (!TryBuildTerrainHiddenTileSummary(compareTile, out TerrainHiddenTileSummary compareSummary))
            return;

        float[] currentFeature = BuildTerrainHiddenFeature(tileHeightmap.Heights, tileHeightmap.MinHeight, tileHeightmap.MaxHeight);
        _terrainAnalysisPreviewCompareTile = compareTile;
        _terrainAnalysisPreviewSimilarity = ComputeTerrainHiddenSimilarity(currentFeature, compareSummary.Feature);
    }

    private bool TryBuildTerrainHiddenTileSummary((int tileX, int tileY) tile, out TerrainHiddenTileSummary summary)
    {
        summary = new TerrainHiddenTileSummary();

        IReadOnlyList<Terrain.TerrainChunkData>? chunks = LoadTileChunksForExport(tile.tileX, tile.tileY);
        if (chunks == null || chunks.Count == 0)
            return false;

        TerrainHeightmapIo.TileHeightmap257 tileHeightmap = TerrainHeightmapIo.BuildTileHeightmap257(chunks);
        summary = new TerrainHiddenTileSummary
        {
            Tile = tile,
            MinHeight = tileHeightmap.MinHeight,
            MaxHeight = tileHeightmap.MaxHeight,
            ReliefRange = Math.Max(tileHeightmap.MaxHeight - tileHeightmap.MinHeight, 0f),
            Feature = BuildTerrainHiddenFeature(tileHeightmap.Heights, tileHeightmap.MinHeight, tileHeightmap.MaxHeight)
        };
        return true;
    }

    private static float[] BuildTerrainHiddenFeature(float[] heights, float minHeight, float maxHeight)
    {
        const int featureSize = 33;
        const int sourceSize = TerrainHeightmapIo.TileHeightmapSize;

        var feature = new float[featureSize * featureSize];
        float range = maxHeight - minHeight;
        if (range <= 1e-6f)
            range = 1f;

        float step = (sourceSize - 1f) / (featureSize - 1f);
        for (int y = 0; y < featureSize; y++)
        {
            int sampleY = Math.Min(sourceSize - 1, (int)MathF.Round(y * step));
            for (int x = 0; x < featureSize; x++)
            {
                int sampleX = Math.Min(sourceSize - 1, (int)MathF.Round(x * step));
                float height = heights[sampleY * sourceSize + sampleX];
                feature[y * featureSize + x] = NormalizeHeight(height, minHeight, range);
            }
        }

        return feature;
    }

    private static float ComputeTerrainHiddenSimilarity(float[] left, float[] right)
    {
        int count = Math.Min(left.Length, right.Length);
        if (count == 0)
            return 0f;

        float diffSum = 0f;
        for (int index = 0; index < count; index++)
            diffSum += MathF.Abs(left[index] - right[index]);

        float averageDifference = diffSum / count;
        return Math.Clamp(1f - averageDifference, 0f, 1f);
    }

    private static byte[] BuildHeightPreviewPixels(float[] heights, float minHeight, float maxHeight, int width, int height)
    {
        var pixels = new byte[width * height * 4];
        float range = maxHeight - minHeight;
        if (range <= 1e-6f)
            range = 1f;

        Vector3 lightDir = Vector3.Normalize(new Vector3(0.35f, 1f, 0.45f));

        for (int y = 0; y < height; y++)
        {
            int rowOffset = y * width;
            int upY = Math.Max(y - 1, 0);
            int downY = Math.Min(y + 1, height - 1);

            for (int x = 0; x < width; x++)
            {
                int leftX = Math.Max(x - 1, 0);
                int rightX = Math.Min(x + 1, width - 1);

                float h = NormalizeHeight(heights[rowOffset + x], minHeight, range);
                float hLeft = NormalizeHeight(heights[rowOffset + leftX], minHeight, range);
                float hRight = NormalizeHeight(heights[rowOffset + rightX], minHeight, range);
                float hUp = NormalizeHeight(heights[upY * width + x], minHeight, range);
                float hDown = NormalizeHeight(heights[downY * width + x], minHeight, range);

                float dx = hRight - hLeft;
                float dy = hDown - hUp;
                Vector3 normal = Vector3.Normalize(new Vector3(-dx * 4f, 1f, -dy * 4f));
                float shade = Math.Clamp(Vector3.Dot(normal, lightDir), 0.2f, 1f);

                var (baseR, baseG, baseB) = GetTerrainAnalysisColor(h);
                float lit = 0.45f + shade * 0.55f;

                int pixelIndex = (rowOffset + x) * 4;
                pixels[pixelIndex + 0] = (byte)Math.Clamp((int)MathF.Round(baseR * lit), 0, 255);
                pixels[pixelIndex + 1] = (byte)Math.Clamp((int)MathF.Round(baseG * lit), 0, 255);
                pixels[pixelIndex + 2] = (byte)Math.Clamp((int)MathF.Round(baseB * lit), 0, 255);
                pixels[pixelIndex + 3] = 255;
            }
        }

        return pixels;
    }

    private static float NormalizeHeight(float height, float minHeight, float range)
    {
        float normalized = (height - minHeight) / range;
        return Math.Clamp(normalized, 0f, 1f);
    }

    private static (byte r, byte g, byte b) GetTerrainAnalysisColor(float normalized)
    {
        normalized = Math.Clamp(normalized, 0f, 1f);

        float r;
        float g;
        float b;
        if (normalized < 0.33f)
        {
            float t = normalized / 0.33f;
            r = 0f;
            g = t * 0.5f;
            b = 0.5f + t * 0.5f;
        }
        else if (normalized < 0.66f)
        {
            float t = (normalized - 0.33f) / 0.33f;
            r = t * 0.3f;
            g = 0.5f + t * 0.5f;
            b = 1f - t;
        }
        else
        {
            float t = (normalized - 0.66f) / 0.34f;
            r = 0.3f + t * 0.4f;
            g = 1f - t * 0.5f;
            b = t * 0.2f;
        }

        return ((byte)(r * 255f), (byte)(g * 255f), (byte)(b * 255f));
    }
}

sealed class TerrainHiddenTileCandidate
{
    public (int tileX, int tileY) Tile { get; init; }
    public (int tileX, int tileY) CompareTile { get; init; }
    public float ReliefRange { get; init; }
    public float VisibilityRatio { get; init; }
    public float Similarity { get; init; }
}

sealed class TerrainHiddenTileSummary
{
    public (int tileX, int tileY) Tile { get; init; }
    public float MinHeight { get; init; }
    public float MaxHeight { get; init; }
    public float ReliefRange { get; init; }
    public float[] Feature { get; init; } = Array.Empty<float>();
}

sealed class TerrainAnalysisPreviewTexture : IDisposable
{
    private readonly GL _gl;

    public TerrainAnalysisPreviewTexture(GL gl)
    {
        _gl = gl;
    }

    public uint TextureId { get; private set; }
    public int Width { get; private set; }
    public int Height { get; private set; }
    public bool HasTexture => TextureId != 0 && Width > 0 && Height > 0;

    public unsafe void Update(byte[] rgbaPixels, int width, int height)
    {
        if (rgbaPixels.Length < width * height * 4)
            throw new ArgumentException("RGBA pixel buffer is smaller than the requested texture size.", nameof(rgbaPixels));

        if (TextureId == 0)
            TextureId = _gl.GenTexture();

        Width = width;
        Height = height;

        _gl.BindTexture(TextureTarget.Texture2D, TextureId);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);

        fixed (byte* ptr = rgbaPixels)
        {
            _gl.TexImage2D(
                TextureTarget.Texture2D,
                0,
                InternalFormat.Rgba,
                (uint)width,
                (uint)height,
                0,
                PixelFormat.Rgba,
                PixelType.UnsignedByte,
                ptr);
        }

        _gl.BindTexture(TextureTarget.Texture2D, 0);
    }

    public void Dispose()
    {
        if (TextureId != 0)
        {
            _gl.DeleteTexture(TextureId);
            TextureId = 0;
        }

        Width = 0;
        Height = 0;
    }
}