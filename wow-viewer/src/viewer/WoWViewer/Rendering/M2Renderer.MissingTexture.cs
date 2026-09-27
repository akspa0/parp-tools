using WoWViewer.Logging;

namespace WoWViewer.Rendering;

// Spec 256 amendment 2026-09-27 (operator): "we should just point to the missing or error blp for missing
// textures, instead of searching for missing textures, that's what the real engine does." A section whose
// textures cannot be loaded binds one shared placeholder — the client's own error texture when the data
// source has it, else a generated checkerboard — instead of probing the data source for a substitute.
public sealed partial class M2Renderer
{
    /// <summary>The client's error texture, bound wherever a texture is missing.</summary>
    internal const string MissingTexturePath = @"Textures\ShaneCube.blp";

    private const string MissingTextureCacheKey = "missing|";
    private const int GeneratedMissingTextureSize = 8;

    private void ApplyMissingTexture(SectionBuffers section)
    {
        section.TexturePending = false;
        uint textureId = AcquireMissingTexture();
        if (textureId == 0)
            return;

        section.TextureId = textureId;
        section.HasTexture = true;
        section.UvSet = 0;
        section.GeneratedTexCoord = false;
    }

    // One GL texture per data source, shared and reference-counted through M2TextureCache like any other.
    private uint AcquireMissingTexture()
    {
        if (_gl == null)
            return 0;

        if (!M2TextureCache.TryAcquire(_dataSource, MissingTextureCacheKey, out uint textureId))
        {
            textureId = CreateMissingTexture();
            if (textureId == 0)
                return 0;

            M2TextureCache.Add(_dataSource, textureId, MissingTextureCacheKey);
        }

        KeepOneReference(textureId);
        return textureId;
    }

    private uint CreateMissingTexture()
    {
        byte[]? blpData = _dataSource?.ReadFile(MissingTexturePath);
        if (blpData is { Length: > 0 })
        {
            uint textureId = LoadTextureFromBlp(blpData, MissingTexturePath, clampS: false, clampT: false);
            if (textureId != 0)
                return textureId;
        }

        ViewerLog.Info(ViewerLog.Category.Mdx, $"[M2] {MissingTexturePath} not available; using a generated missing-texture checkerboard");
        const int size = GeneratedMissingTextureSize;
        byte[] pixels = new byte[size * size * 4];
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                int offset = ((y * size) + x) * 4;
                bool magenta = ((x / 2) + (y / 2)) % 2 == 0;
                pixels[offset] = magenta ? (byte)255 : (byte)0;
                pixels[offset + 1] = 0;
                pixels[offset + 2] = magenta ? (byte)255 : (byte)0;
                pixels[offset + 3] = 255;
            }
        }

        return UploadTexture(pixels, size, size, clampS: false, clampT: false);
    }
}
