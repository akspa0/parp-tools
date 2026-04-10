using System.Numerics;
using ImGuiNET;
using MdxViewer.Export;
using Silk.NET.OpenGL;
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

        ImGui.End();
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

        var localPixels = BuildHeightPreviewPixels(
            tileHeightmap.Heights,
            tileHeightmap.MinHeight,
            tileHeightmap.MaxHeight,
            TerrainHeightmapIo.TileHeightmapSize,
            TerrainHeightmapIo.TileHeightmapSize);
        _terrainAnalysisLocalTexture?.Update(localPixels, TerrainHeightmapIo.TileHeightmapSize, TerrainHeightmapIo.TileHeightmapSize);

        if (_terrainAnalysisHasGlobalBounds)
        {
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
            _terrainAnalysisGlobalTexture?.Dispose();
            _terrainAnalysisGlobalTexture = new TerrainAnalysisPreviewTexture(_gl);
        }

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