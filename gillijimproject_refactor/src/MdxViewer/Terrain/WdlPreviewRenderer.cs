using System.Numerics;
using MdxViewer.DataSources;
using MdxViewer.Logging;
using MdxViewer.Rendering;
using Silk.NET.OpenGL;
using WoWMapConverter.Core.VLM;

namespace MdxViewer.Terrain;

/// <summary>
/// Renders a WDL file as a low-resolution heightmap preview texture.
/// Used for map selection UI to show terrain overview and allow spawn point selection.
/// </summary>
public class WdlPreviewRenderer : IDisposable
{
    private const int PreviewTileCount = 64;
    private const int DisplayPixelsPerTile = 8;

    private readonly GL _gl;
    private uint _textureId;
    private int _textureWidth;
    private int _textureHeight;
    private WdlPreviewData? _previewData;
    private string _mapDirectory = "";

    public bool HasPreview => _textureId != 0 && _previewData != null;
    public uint TextureId => _textureId;
    public int Width => _textureWidth;
    public int Height => _textureHeight;
    public string MapDirectory => _mapDirectory;

    public WdlPreviewRenderer(GL gl)
    {
        _gl = gl;
    }

    /// <summary>
    /// Load and render a WDL file as a preview texture.
    /// </summary>
    public string? LastError { get; internal set; }

    public bool LoadWdl(IDataSource dataSource, string mapDirectory)
    {
        if (!WdlPreviewDataBuilder.TryBuild(dataSource, mapDirectory, out var previewData, out var error))
        {
            LastError = error ?? $"Failed to build WDL preview for {mapDirectory}";
            ViewerLog.Error(ViewerLog.Category.Terrain, $"[WDL Preview] {LastError}");
            return false;
        }

        return LoadPreview(previewData!);
    }

    internal bool LoadPreview(WdlPreviewData previewData)
    {
        if (previewData.PreviewRgba == null || previewData.PreviewRgba.Length == 0)
        {
            LastError = $"Preview payload for {previewData.MapDirectory} is empty.";
            return false;
        }

        _previewData = previewData;
        _mapDirectory = previewData.MapDirectory;
        LastError = null;
        GeneratePreviewTexture(previewData);
        return true;
    }

    public void ClearPreview()
    {
        _previewData = null;
        _mapDirectory = "";
        LastError = null;

        if (_textureId != 0)
        {
            _gl.DeleteTexture(_textureId);
            _textureId = 0;
        }

        _textureWidth = 0;
        _textureHeight = 0;
    }

    private void GeneratePreviewTexture(WdlPreviewData previewData)
    {
        byte[] pixelData = BuildDisplayTexture(previewData, out _textureWidth, out _textureHeight);

        if (_textureId != 0)
            _gl.DeleteTexture(_textureId);

        _textureId = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2D, _textureId);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);

        unsafe
        {
            fixed (byte* ptr = pixelData)
            {
                _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba, (uint)_textureWidth, (uint)_textureHeight,
                    0, PixelFormat.Rgba, PixelType.UnsignedByte, ptr);
            }
        }

        _gl.BindTexture(TextureTarget.Texture2D, 0);
    }

    /// <summary>
    /// Convert pixel coordinates on the preview texture to WDL tile coordinates (0-63).
    /// </summary>
    public (int tileX, int tileY)? PixelToTile(Vector2 pixelPos)
    {
        if (!HasPreview) return null;

        float tileWidth = _textureWidth / (float)PreviewTileCount;
        float tileHeight = _textureHeight / (float)PreviewTileCount;
        int tileX = (int)(pixelPos.X / tileWidth);
        int tileY = (int)(pixelPos.Y / tileHeight);

        if (tileX < 0 || tileX >= PreviewTileCount || tileY < 0 || tileY >= PreviewTileCount)
            return null;

        return (tileX, tileY);
    }

    private static byte[] BuildDisplayTexture(WdlPreviewData previewData, out int width, out int height)
    {
        bool hasHeightField = previewData.TileCenterHeights.Length == PreviewTileCount * PreviewTileCount
            && previewData.TileDataMask.Length == PreviewTileCount * PreviewTileCount;

        width = Math.Max(previewData.Width, PreviewTileCount * DisplayPixelsPerTile);
        height = Math.Max(previewData.Height, PreviewTileCount * DisplayPixelsPerTile);

        if (!hasHeightField)
        {
            width = previewData.Width;
            height = previewData.Height;
            return previewData.PreviewRgba;
        }

        var pixels = new byte[width * height * 4];
        float minHeight = previewData.MinHeight;
        float maxHeight = previewData.MaxHeight;
        float heightRange = MathF.Max(1f, maxHeight - minHeight);

        for (int py = 0; py < height; py++)
        {
            float sampleTileY = ((py + 0.5f) / height) * PreviewTileCount - 0.5f;
            for (int px = 0; px < width; px++)
            {
                float sampleTileX = ((px + 0.5f) / width) * PreviewTileCount - 0.5f;
                int pixelIndex = (py * width + px) * 4;

                if (!TrySampleHeight(previewData, sampleTileX, sampleTileY, out float heightSample))
                {
                    pixels[pixelIndex + 0] = 25;
                    pixels[pixelIndex + 1] = 25;
                    pixels[pixelIndex + 2] = 25;
                    pixels[pixelIndex + 3] = 255;
                    continue;
                }

                float normalized = (heightSample - minHeight) / heightRange;
                var color = GetPreviewColor(normalized);
                pixels[pixelIndex + 0] = color.r;
                pixels[pixelIndex + 1] = color.g;
                pixels[pixelIndex + 2] = color.b;
                pixels[pixelIndex + 3] = 255;
            }
        }

        return pixels;
    }

    private static bool TrySampleHeight(WdlPreviewData previewData, float sampleTileX, float sampleTileY, out float height)
    {
        int x0 = Math.Clamp((int)MathF.Floor(sampleTileX), 0, PreviewTileCount - 1);
        int y0 = Math.Clamp((int)MathF.Floor(sampleTileY), 0, PreviewTileCount - 1);
        int x1 = Math.Clamp(x0 + 1, 0, PreviewTileCount - 1);
        int y1 = Math.Clamp(y0 + 1, 0, PreviewTileCount - 1);

        float fx = Math.Clamp(sampleTileX - x0, 0f, 1f);
        float fy = Math.Clamp(sampleTileY - y0, 0f, 1f);

        float totalWeight = 0f;
        float weightedHeight = 0f;

        AccumulateSample(previewData, x0, y0, (1f - fx) * (1f - fy), ref weightedHeight, ref totalWeight);
        AccumulateSample(previewData, x1, y0, fx * (1f - fy), ref weightedHeight, ref totalWeight);
        AccumulateSample(previewData, x0, y1, (1f - fx) * fy, ref weightedHeight, ref totalWeight);
        AccumulateSample(previewData, x1, y1, fx * fy, ref weightedHeight, ref totalWeight);

        if (totalWeight <= 0f)
        {
            height = 0f;
            return false;
        }

        height = weightedHeight / totalWeight;
        return true;
    }

    private static void AccumulateSample(WdlPreviewData previewData, int tileX, int tileY, float weight, ref float weightedHeight, ref float totalWeight)
    {
        if (weight <= 0f)
            return;

        int index = tileY * PreviewTileCount + tileX;
        if ((uint)index >= previewData.TileCenterHeights.Length ||
            (uint)index >= previewData.TileDataMask.Length ||
            previewData.TileDataMask[index] == 0)
        {
            return;
        }

        weightedHeight += previewData.TileCenterHeights[index] * weight;
        totalWeight += weight;
    }

    private static (byte r, byte g, byte b) GetPreviewColor(float normalized)
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

    /// <summary>
    /// Convert WDL tile coordinates to world coordinates for camera positioning.
    /// </summary>
    public Vector3 TileToWorldPosition(int tileX, int tileY)
    {
        var sourceTile = PreviewTileToSourceTile(tileX, tileY);

        float height = 0f;
        if (_previewData != null)
        {
            int tileIndex = tileY * 64 + tileX;
            if ((uint)tileIndex < _previewData.TileCenterHeights.Length &&
                (uint)tileIndex < _previewData.TileDataMask.Length &&
                _previewData.TileDataMask[tileIndex] != 0)
            {
                height = _previewData.TileCenterHeights[tileIndex];
            }
        }

        return GetTileSpawnPosition(sourceTile.tileX, sourceTile.tileY, height);
    }

    internal static (int tileX, int tileY) PreviewTileToSourceTile(int previewTileX, int previewTileY)
    {
        return (previewTileY, previewTileX);
    }

    internal static Vector3 GetTileSpawnPosition(int tileX, int tileY, float height)
    {
        // WDL preview selection is on the 64x64 world-map tile grid used by the terrain loader.
        // In this codebase that grid spacing is WoWConstants.ChunkSize (533.3333...), not TileSize.
        float rendererX = WoWConstants.MapOrigin - ((tileX + 0.5f) * WoWConstants.ChunkSize);
        float rendererY = WoWConstants.MapOrigin - ((tileY + 0.5f) * WoWConstants.ChunkSize);
        return new Vector3(rendererX, rendererY, height + 100f);
    }

    public void Dispose()
    {
        ClearPreview();
    }
}
