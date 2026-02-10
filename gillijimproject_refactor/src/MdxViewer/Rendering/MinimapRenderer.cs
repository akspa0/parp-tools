using System.Numerics;
using MdxViewer.DataSources;
using SereniaBLPLib;
using Silk.NET.OpenGL;
using WoWMapConverter.Core.Services;

namespace MdxViewer.Rendering;

/// <summary>
/// Handles loading and caching of minimap tile textures for display in the UI.
/// Uses Md5TranslateResolver to handle hashed file paths in early WoW versions.
/// </summary>
public class MinimapRenderer : IDisposable
{
    private readonly GL _gl;
    private readonly IDataSource _dataSource;
    private readonly Md5TranslateIndex? _md5Index;
    private readonly Dictionary<string, uint> _textureCache = new(StringComparer.OrdinalIgnoreCase);

    public MinimapRenderer(GL gl, IDataSource dataSource, Md5TranslateIndex? md5Index)
    {
        _gl = gl;
        _dataSource = dataSource;
        _md5Index = md5Index;
    }

    /// <summary>
    /// Gets the GL texture handle for a specific minimap tile.
    /// Returns 0 if the tile is not found or failed to load.
    /// </summary>
    public uint GetTileTexture(string mapName, int tx, int ty)
    {
        string plainPath = MinimapService.GetMinimapTilePath(mapName, tx, ty);
        
        if (_textureCache.TryGetValue(plainPath, out uint cached))
            return cached;

        uint tex = LoadTile(plainPath);
        _textureCache[plainPath] = tex;
        return tex;
    }

    private unsafe uint LoadTile(string plainPath)
    {
        byte[]? data = null;

        // 1. Try MD5 translation if available
        if (_md5Index != null)
        {
            var normalized = _md5Index.Normalize(plainPath);
            if (_md5Index.PlainToHash.TryGetValue(normalized, out string? hashedPath))
            {
                data = _dataSource.ReadFile(hashedPath);
                if (data == null)
                {
                    // Try variants (textures/minimap/ vs textures/minimaps/ etc)
                    data = _dataSource.ReadFile(hashedPath.Replace('/', '\\'));
                }
            }
        }

        // 2. Fallback to direct path
        if (data == null)
        {
            data = _dataSource.ReadFile(plainPath);
            if (data == null)
                data = _dataSource.ReadFile(plainPath.Replace('/', '\\'));
        }

        if (data == null || data.Length == 0)
            return 0;

        try
        {
            using var ms = new MemoryStream(data);
            using var blp = new BlpFile(ms);
            var bmp = blp.GetBitmap(0);

            int w = bmp.Width, h = bmp.Height;
            var pixels = new byte[w * h * 4];
            var rect = new System.Drawing.Rectangle(0, 0, w, h);
            var bmpData = bmp.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadOnly,
                System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            
            try
            {
                var srcBytes = new byte[bmpData.Stride * h];
                System.Runtime.InteropServices.Marshal.Copy(bmpData.Scan0, srcBytes, 0, srcBytes.Length);

                // BGRA → RGBA
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        int srcIdx = y * bmpData.Stride + x * 4;
                        int dstIdx = (y * w + x) * 4;
                        pixels[dstIdx + 0] = srcBytes[srcIdx + 2]; // R
                        pixels[dstIdx + 1] = srcBytes[srcIdx + 1]; // G
                        pixels[dstIdx + 2] = srcBytes[srcIdx + 0]; // B
                        pixels[dstIdx + 3] = srcBytes[srcIdx + 3]; // A
                    }
                }
            }
            finally
            {
                bmp.UnlockBits(bmpData);
            }
            bmp.Dispose();

            uint tex = _gl.GenTexture();
            _gl.BindTexture(TextureTarget.Texture2D, tex);
            fixed (byte* ptr = pixels)
                _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba,
                    (uint)w, (uint)h, 0, PixelFormat.Rgba, PixelType.UnsignedByte, ptr);

            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);
            _gl.BindTexture(TextureTarget.Texture2D, 0);

            return tex;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[MinimapRenderer] Failed to load tile {plainPath}: {ex.Message}");
            return 0;
        }
    }

    public void Dispose()
    {
        foreach (var tex in _textureCache.Values)
        {
            if (tex != 0) _gl.DeleteTexture(tex);
        }
        _textureCache.Clear();
    }
}
