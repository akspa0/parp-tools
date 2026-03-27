using System.Numerics;
using System.Security.Cryptography;
using MdxViewer.DataSources;
using MdxViewer.Logging;
using SereniaBLPLib;
using Silk.NET.OpenGL;
using WowViewer.Core.IO.Files;

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
    private readonly string _cacheRoot;
    private readonly Dictionary<string, uint> _textureCache = new(StringComparer.OrdinalIgnoreCase);

    public MinimapRenderer(GL gl, IDataSource dataSource, Md5TranslateIndex? md5Index, string cacheRoot)
    {
        _gl = gl;
        _dataSource = dataSource;
        _md5Index = md5Index;
        _cacheRoot = cacheRoot;
        Directory.CreateDirectory(_cacheRoot);
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

        uint tex = LoadTile(mapName, tx, ty, plainPath);
        _textureCache[plainPath] = tex;
        return tex;
    }

    private unsafe uint LoadTile(string mapName, int tx, int ty, string cacheKey)
    {
        if (TryLoadCachedBitmap(cacheKey, out DecodedMinimapTile? cachedTile) && cachedTile != null)
            return UploadTexture(cachedTile);

        byte[]? data = null;
        foreach (string candidatePath in EnumerateTileCandidates(mapName, tx, ty))
        {
            data = TryReadTileData(candidatePath);
            if (data != null && data.Length > 0)
                break;
        }

        if (data == null || data.Length == 0)
            return 0;

        try
        {
            using var ms = new MemoryStream(data);
            using var blp = new BlpFile(ms);
            var bmp = blp.GetBitmap(0);
            DecodedMinimapTile decoded = ConvertBitmap(bmp);
            TrySaveCachedBitmap(cacheKey, bmp);
            bmp.Dispose();
            return UploadTexture(decoded);
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[MinimapRenderer] Failed to load tile {cacheKey}: {ex.Message}");
            return 0;
        }
    }

    private byte[]? TryReadTileData(string plainPath)
    {
        byte[]? data = null;

        if (_md5Index != null)
        {
            var normalized = _md5Index.Normalize(plainPath);
            if (_md5Index.PlainToHash.TryGetValue(normalized, out string? hashedPath))
            {
                data = _dataSource.ReadFile(hashedPath);
                if (data == null)
                    data = _dataSource.ReadFile(hashedPath.Replace('/', '\\'));
            }
        }

        if (data == null)
        {
            data = _dataSource.ReadFile(plainPath);
            if (data == null)
                data = _dataSource.ReadFile(plainPath.Replace('/', '\\'));
        }

        return data;
    }

    private static IEnumerable<string> EnumerateTileCandidates(string mapName, int x, int y)
    {
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var yieldReturnList = new List<string>();

        void AddCandidate(string candidate)
        {
            if (!string.IsNullOrWhiteSpace(candidate) && seen.Add(candidate))
                yieldReturnList.Add(candidate);
        }

        string normalizedMapName = mapName.ToLowerInvariant();
        string x2 = x.ToString("D2");
        string y2 = y.ToString("D2");
        string trsFormat = $"map{x}_{y2}.blp";

        AddCandidate($"{normalizedMapName}\\{trsFormat}");
        AddCandidate($"{normalizedMapName}/{trsFormat}");
        AddCandidate($"textures/minimap/{normalizedMapName}/{trsFormat}");

        AddCandidate($"textures/minimap/{normalizedMapName}/{normalizedMapName}_{x2}_{y2}.blp");
        AddCandidate($"textures/minimap/{normalizedMapName}/map{x2}_{y2}.blp");
        AddCandidate($"{normalizedMapName}/map{x2}_{y2}.blp");

        string mapNameSpace = InsertSpaceBeforeCapitals(mapName).ToLowerInvariant();
        if (!string.Equals(mapNameSpace, normalizedMapName, StringComparison.OrdinalIgnoreCase))
        {
            AddCandidate($"{mapNameSpace}\\{trsFormat}");
            AddCandidate($"textures/minimap/{mapNameSpace}/{trsFormat}");
            AddCandidate($"textures/minimap/{mapNameSpace}/{mapNameSpace}_{x2}_{y2}.blp");
            AddCandidate($"textures/minimap/{mapNameSpace}/map{x2}_{y2}.blp");
            AddCandidate($"{mapNameSpace}/map{x2}_{y2}.blp");
        }

        AddCandidate($"world/minimaps/{normalizedMapName}/map{x2}_{y2}.blp");
        AddCandidate($"world/minimaps/{normalizedMapName}/map{x}_{y}.blp");
        AddCandidate($"textures/minimap/{normalizedMapName}_{x2}_{y2}.blp");
        AddCandidate($"textures/minimap/{normalizedMapName}_{x}_{y}.blp");

        return yieldReturnList;
    }

    private static string InsertSpaceBeforeCapitals(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return value;

        var builder = new System.Text.StringBuilder(value.Length + 8);
        for (int index = 0; index < value.Length; index++)
        {
            char ch = value[index];
            if (index > 0 && char.IsUpper(ch) && !char.IsWhiteSpace(value[index - 1]))
                builder.Append(' ');

            builder.Append(ch);
        }

        return builder.ToString();
    }

    private sealed record DecodedMinimapTile(int Width, int Height, byte[] Pixels);

    private static DecodedMinimapTile ConvertBitmap(System.Drawing.Bitmap bmp)
    {
        int width = bmp.Width;
        int height = bmp.Height;
        var pixels = new byte[width * height * 4];
        var rect = new System.Drawing.Rectangle(0, 0, width, height);
        var bmpData = bmp.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadOnly,
            System.Drawing.Imaging.PixelFormat.Format32bppArgb);

        try
        {
            var srcBytes = new byte[bmpData.Stride * height];
            System.Runtime.InteropServices.Marshal.Copy(bmpData.Scan0, srcBytes, 0, srcBytes.Length);

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int srcIdx = y * bmpData.Stride + x * 4;
                    int dstIdx = (y * width + x) * 4;
                    pixels[dstIdx + 0] = srcBytes[srcIdx + 2];
                    pixels[dstIdx + 1] = srcBytes[srcIdx + 1];
                    pixels[dstIdx + 2] = srcBytes[srcIdx + 0];
                    pixels[dstIdx + 3] = srcBytes[srcIdx + 3];
                }
            }
        }
        finally
        {
            bmp.UnlockBits(bmpData);
        }

        return new DecodedMinimapTile(width, height, pixels);
    }

    private unsafe uint UploadTexture(DecodedMinimapTile tile)
    {
        uint tex = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2D, tex);
        fixed (byte* ptr = tile.Pixels)
            _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba,
                (uint)tile.Width, (uint)tile.Height, 0, PixelFormat.Rgba, PixelType.UnsignedByte, ptr);

        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);
        _gl.BindTexture(TextureTarget.Texture2D, 0);
        return tex;
    }

    private bool TryLoadCachedBitmap(string plainPath, out DecodedMinimapTile? tile)
    {
        tile = null;
        string cachePath = GetCachePath(plainPath);
        if (!File.Exists(cachePath))
            return false;

        try
        {
            using var bmp = new System.Drawing.Bitmap(cachePath);
            tile = ConvertBitmap(bmp);
            return true;
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[MinimapRenderer] Failed to load cached tile {cachePath}: {ex.Message}");
            return false;
        }
    }

    private void TrySaveCachedBitmap(string plainPath, System.Drawing.Bitmap bmp)
    {
        string cachePath = GetCachePath(plainPath);
        string? cacheDirectory = Path.GetDirectoryName(cachePath);
        if (!string.IsNullOrEmpty(cacheDirectory))
            Directory.CreateDirectory(cacheDirectory);

        string tempPath = cachePath + ".tmp";
        try
        {
            bmp.Save(tempPath, System.Drawing.Imaging.ImageFormat.Png);
            File.Move(tempPath, cachePath, overwrite: true);
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[MinimapRenderer] Failed to save cached tile {cachePath}: {ex.Message}");
            if (File.Exists(tempPath))
                File.Delete(tempPath);
        }
    }

    private string GetCachePath(string plainPath)
    {
        string normalized = plainPath.Replace('\\', '/').ToLowerInvariant();
        string hash = Convert.ToHexString(SHA1.HashData(System.Text.Encoding.UTF8.GetBytes(normalized))).ToLowerInvariant();
        return Path.Combine(_cacheRoot, hash + ".png");
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
