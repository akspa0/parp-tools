using Silk.NET.OpenGL;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WowViewer.Core.IO.Files;

namespace WowViewer.Core.Renderer.Texture;

public sealed class TextureCache : IDisposable
{
    private readonly GL _gl;
    private readonly IArchiveCatalog _archiveCatalog;
    private readonly Dictionary<string, uint> _textures = new(StringComparer.OrdinalIgnoreCase);
    private bool _disposed;

    public TextureCache(GL gl, IArchiveCatalog archiveCatalog)
    {
        _gl = gl ?? throw new ArgumentNullException(nameof(gl));
        _archiveCatalog = archiveCatalog ?? throw new ArgumentNullException(nameof(archiveCatalog));
    }

    public uint GetOrCreateTexture(string texturePath)
    {
        if (string.IsNullOrEmpty(texturePath))
            return 0;

        if (_textures.TryGetValue(texturePath, out uint existing))
            return existing;

        uint texture = LoadBlpTexture(texturePath);
        _textures[texturePath] = texture;
        return texture;
    }

    public unsafe uint CreateTextureArray(List<string> texturePaths, int targetSize = 256)
    {
        int layerCount = Math.Max(1, texturePaths.Count);

        uint textureArray = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2DArray, textureArray);
        _gl.TexImage3D(TextureTarget.Texture2DArray, 0, InternalFormat.Rgba8, (uint)targetSize, (uint)targetSize, (uint)layerCount, 0, PixelFormat.Rgba, PixelType.UnsignedByte, null);

        for (int layer = 0; layer < layerCount; layer++)
        {
            byte[] pixels;
            int width, height;

            if (layer < texturePaths.Count && TryLoadTexturePixels(texturePaths[layer], out width, out height, out pixels))
            {
                if (width != targetSize || height != targetSize)
                    pixels = ResampleNearest(pixels, width, height, targetSize, targetSize);
            }
            else
            {
                pixels = CreateSolidPixels(targetSize, targetSize, 255, 255, 255, 255);
            }

            fixed (byte* ptr = pixels)
            {
                _gl.TexSubImage3D(TextureTarget.Texture2DArray, 0, 0, 0, layer, (uint)targetSize, (uint)targetSize, 1, PixelFormat.Rgba, PixelType.UnsignedByte, ptr);
            }
        }

        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.LinearMipmapLinear);
        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureWrapS, (int)TextureWrapMode.Repeat);
        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureWrapT, (int)TextureWrapMode.Repeat);
        _gl.GenerateMipmap(TextureTarget.Texture2DArray);
        _gl.BindTexture(TextureTarget.Texture2DArray, 0);

        return textureArray;
    }

    public unsafe uint CreateAlphaShadowArray(byte[] alphaShadow)
    {
        uint texture = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2DArray, texture);

        fixed (byte* ptr = alphaShadow)
        {
            _gl.TexImage3D(TextureTarget.Texture2DArray, 0, InternalFormat.Rgba8, 64, 64, 256, 0, PixelFormat.Rgba, PixelType.UnsignedByte, ptr);
        }

        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);
        _gl.BindTexture(TextureTarget.Texture2DArray, 0);

        return texture;
    }

    private unsafe uint LoadBlpTexture(string texturePath)
    {
        byte[]? blpData = _archiveCatalog.ReadFile(texturePath)
                          ?? _archiveCatalog.ReadFile(texturePath.Replace('/', '\\'));

        if (blpData == null || blpData.Length == 0)
            return 0;

        try
        {
            using var stream = new MemoryStream(blpData);
            using var blp = new SereniaBLPLib.BlpFile(stream);
            using Image<Rgba32> image = blp.GetImage(0);
            int width = image.Width;
            int height = image.Height;
            // ImageSharp CopyPixelDataTo yields tightly-packed RGBA, top-down — the exact
            // layout the GL upload below expects (PixelFormat.Rgba), so no channel swap.
            var pixels = new byte[width * height * 4];
            image.CopyPixelDataTo(pixels);

            uint texture = _gl.GenTexture();
            _gl.BindTexture(TextureTarget.Texture2D, texture);
            fixed (byte* ptr = pixels)
            {
                _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba, (uint)width, (uint)height, 0, PixelFormat.Rgba, PixelType.UnsignedByte, ptr);
            }

            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.LinearMipmapLinear);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.Repeat);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.Repeat);
            _gl.GenerateMipmap(TextureTarget.Texture2D);
            _gl.BindTexture(TextureTarget.Texture2D, 0);

            return texture;
        }
        catch
        {
            return 0;
        }
    }

    private bool TryLoadTexturePixels(string texturePath, out int width, out int height, out byte[] pixels)
    {
        width = 0;
        height = 0;
        pixels = Array.Empty<byte>();

        byte[]? blpData = _archiveCatalog.ReadFile(texturePath)
                          ?? _archiveCatalog.ReadFile(texturePath.Replace('/', '\\'));

        if (blpData == null || blpData.Length == 0)
            return false;

        try
        {
            using var stream = new MemoryStream(blpData);
            using var blp = new SereniaBLPLib.BlpFile(stream);
            using Image<Rgba32> image = blp.GetImage(0);
            width = image.Width;
            height = image.Height;
            // Tightly-packed RGBA from ImageSharp; consumers (ResampleNearest, the Rgba GL
            // upload path) treat these bytes as RGBA already, so no channel swap is needed.
            pixels = new byte[width * height * 4];
            image.CopyPixelDataTo(pixels);

            return true;
        }
        catch
        {
            return false;
        }
    }

    private static byte[] ResampleNearest(byte[] source, int srcW, int srcH, int dstW, int dstH)
    {
        var dest = new byte[dstW * dstH * 4];
        for (int y = 0; y < dstH; y++)
        {
            int srcY = (int)((long)y * srcH / dstH);
            for (int x = 0; x < dstW; x++)
            {
                int srcX = (int)((long)x * srcW / dstW);
                int si = (srcY * srcW + srcX) * 4;
                int di = (y * dstW + x) * 4;
                dest[di + 0] = source[si + 0];
                dest[di + 1] = source[si + 1];
                dest[di + 2] = source[si + 2];
                dest[di + 3] = source[si + 3];
            }
        }
        return dest;
    }

    private static byte[] CreateSolidPixels(int w, int h, byte r, byte g, byte b, byte a)
    {
        var pixels = new byte[w * h * 4];
        for (int i = 0; i < pixels.Length; i += 4)
        {
            pixels[i + 0] = r;
            pixels[i + 1] = g;
            pixels[i + 2] = b;
            pixels[i + 3] = a;
        }
        return pixels;
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        foreach (uint tex in _textures.Values)
        {
            if (tex != 0)
                _gl.DeleteTexture(tex);
        }
        _textures.Clear();
    }
}
