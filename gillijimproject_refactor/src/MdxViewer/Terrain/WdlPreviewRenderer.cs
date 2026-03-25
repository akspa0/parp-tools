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
        _textureWidth = previewData.Width;
        _textureHeight = previewData.Height;

        if (_textureId != 0)
            _gl.DeleteTexture(_textureId);

        _textureId = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2D, _textureId);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);

        unsafe
        {
            fixed (byte* ptr = previewData.PreviewRgba)
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

        int tileSize = _textureWidth / 64;
        int tileX = (int)(pixelPos.X / tileSize);
        int tileY = (int)(pixelPos.Y / tileSize);

        if (tileX < 0 || tileX >= 64 || tileY < 0 || tileY >= 64)
            return null;

        return (tileX, tileY);
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
        float rendererX = WoWConstants.MapOrigin - ((tileX + 0.5f) * WoWConstants.TileSize);
        float rendererY = WoWConstants.MapOrigin - ((tileY + 0.5f) * WoWConstants.TileSize);
        return new Vector3(rendererX, rendererY, height + 100f);
    }

    public void Dispose()
    {
        ClearPreview();
    }
}
