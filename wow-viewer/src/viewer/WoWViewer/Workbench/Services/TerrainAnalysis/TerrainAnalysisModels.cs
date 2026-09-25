using System.Numerics;
using System.Text.Json;
using ImGuiNET;
using WoWViewer.Export;
using Silk.NET.OpenGL;
using Image = SixLabors.ImageSharp.Image;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace WoWViewer;

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
