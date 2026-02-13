using System.Numerics;
using MdxViewer.DataSources;
using MdxViewer.Logging;
using Silk.NET.OpenGL;
using WoWRollback.Core.Services.Parsers;

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
    private WdlParser.WdlData? _wdlData;
    private string _mapDirectory = "";

    public bool HasPreview => _textureId != 0 && _wdlData != null;
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
    public bool LoadWdl(IDataSource dataSource, string mapDirectory)
    {
        _mapDirectory = mapDirectory;
        string wdlPath = $"World\\Maps\\{mapDirectory}\\{mapDirectory}.wdl";

        try
        {
            byte[]? wdlData = dataSource.ReadFile(wdlPath);
            if (wdlData == null || wdlData.Length == 0)
            {
                ViewerLog.Info(ViewerLog.Category.Terrain, $"[WDL Preview] No WDL data for {mapDirectory}");
                return false;
            }

            _wdlData = WdlParser.Parse(wdlData);
            if (_wdlData == null)
            {
                ViewerLog.Error(ViewerLog.Category.Terrain, $"[WDL Preview] Failed to parse WDL for {mapDirectory}");
                return false;
            }

            // Count tiles with data
            int tilesWithData = _wdlData.Tiles.Count(t => t?.HasData == true);
            ViewerLog.Info(ViewerLog.Category.Terrain, $"[WDL Preview] Loaded {mapDirectory}.wdl: {tilesWithData}/4096 tiles");

            // Generate preview texture
            GeneratePreviewTexture();
            return true;
        }
        catch (Exception ex)
        {
            ViewerLog.Error(ViewerLog.Category.Terrain, $"[WDL Preview] Error loading {mapDirectory}: {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// Generate a heightmap preview texture from WDL data.
    /// 64x64 tiles, each rendered as a single pixel (average height).
    /// Color-coded: blue=low, green=mid, brown=high.
    /// </summary>
    private void GeneratePreviewTexture()
    {
        if (_wdlData == null) return;

        const int tileSize = 8; // Each WDL tile rendered as 8x8 pixels for better visibility
        _textureWidth = 64 * tileSize;
        _textureHeight = 64 * tileSize;

        byte[] pixels = new byte[_textureWidth * _textureHeight * 4]; // RGBA

        // First pass: find min/max heights for normalization
        short minHeight = short.MaxValue;
        short maxHeight = short.MinValue;
        foreach (var tile in _wdlData.Tiles)
        {
            if (tile?.HasData != true) continue;
            for (int r = 0; r < 17; r++)
            {
                for (int c = 0; c < 17; c++)
                {
                    short h = tile.Height17[r, c];
                    if (h < minHeight) minHeight = h;
                    if (h > maxHeight) maxHeight = h;
                }
            }
        }

        float heightRange = maxHeight - minHeight;
        if (heightRange < 1f) heightRange = 1f;

        // Second pass: render tiles
        for (int tileY = 0; tileY < 64; tileY++)
        {
            for (int tileX = 0; tileX < 64; tileX++)
            {
                int tileIndex = tileY * 64 + tileX;
                var tile = _wdlData.Tiles[tileIndex];

                Vector3 color;
                if (tile?.HasData == true)
                {
                    // Calculate average height for this tile
                    float avgHeight = 0f;
                    int count = 0;
                    for (int r = 0; r < 17; r++)
                    {
                        for (int c = 0; c < 17; c++)
                        {
                            avgHeight += tile.Height17[r, c];
                            count++;
                        }
                    }
                    avgHeight /= count;

                    // Normalize to 0-1 range
                    float normalized = (avgHeight - minHeight) / heightRange;

                    // Color gradient: blue (low) -> green (mid) -> brown (high)
                    if (normalized < 0.33f)
                    {
                        // Blue to cyan
                        float t = normalized / 0.33f;
                        color = new Vector3(0f, t * 0.5f, 0.5f + t * 0.5f);
                    }
                    else if (normalized < 0.66f)
                    {
                        // Cyan to green
                        float t = (normalized - 0.33f) / 0.33f;
                        color = new Vector3(t * 0.3f, 0.5f + t * 0.5f, 1f - t);
                    }
                    else
                    {
                        // Green to brown
                        float t = (normalized - 0.66f) / 0.34f;
                        color = new Vector3(0.3f + t * 0.4f, 1f - t * 0.5f, t * 0.2f);
                    }
                }
                else
                {
                    // No data - dark gray
                    color = new Vector3(0.1f, 0.1f, 0.1f);
                }

                // Fill tileSize x tileSize block with this color
                for (int py = 0; py < tileSize; py++)
                {
                    for (int px = 0; px < tileSize; px++)
                    {
                        int pixelX = tileX * tileSize + px;
                        int pixelY = tileY * tileSize + py;
                        int pixelIndex = (pixelY * _textureWidth + pixelX) * 4;

                        pixels[pixelIndex + 0] = (byte)(color.X * 255);
                        pixels[pixelIndex + 1] = (byte)(color.Y * 255);
                        pixels[pixelIndex + 2] = (byte)(color.Z * 255);
                        pixels[pixelIndex + 3] = 255;
                    }
                }
            }
        }

        // Upload to GPU
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
            fixed (byte* ptr = pixels)
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
        const float MapOrigin = 17066.666f;
        const float TileSize = 533.33333f;

        // WDL tile (0,0) is top-left, corresponds to ADT (63,63) in world space
        // Transform: rendererX = MapOrigin - wowY, rendererY = MapOrigin - wowX
        float rendererX = MapOrigin - (tileY * TileSize);
        float rendererY = MapOrigin - (tileX * TileSize);

        // Get height at tile center if available
        float height = 0f;
        if (_wdlData != null)
        {
            int tileIndex = tileY * 64 + tileX;
            var tile = _wdlData.Tiles[tileIndex];
            if (tile?.HasData == true)
            {
                // Use center height (8,8 in 17x17 grid)
                height = tile.Height17[8, 8];
            }
        }

        return new Vector3(rendererX, rendererY, height + 100f); // +100 to be above terrain
    }

    public void Dispose()
    {
        if (_textureId != 0)
        {
            _gl.DeleteTexture(_textureId);
            _textureId = 0;
        }
    }
}
