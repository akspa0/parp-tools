using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace WoWMapConverter.Core.VLM;

/// <summary>
/// Utility helpers for producing terrain-oriented minimap variants from exported signals.
/// </summary>
public static class VlmMinimapCleanupService
{
    internal static Rgba32 DecodeStoredMccvColor(byte[] rawBytes, int baseIndex)
    {
        ArgumentNullException.ThrowIfNull(rawBytes);
        if (baseIndex < 0 || baseIndex + 3 >= rawBytes.Length)
            throw new ArgumentOutOfRangeException(nameof(baseIndex));

        // MCCV bytes are stored as BGRA in the ADT chunk payload.
        return new Rgba32(
            rawBytes[baseIndex + 2],
            rawBytes[baseIndex + 1],
            rawBytes[baseIndex],
            rawBytes[baseIndex + 3]);
    }

    internal static Rgba32 DecodeStoredMccvColor(Rgba32 rawPixel)
    {
        // MdxViewer's MCCV PNG export preserves raw file-order bytes in the PNG
        // channels for tooling compatibility. Decode that raw-view pixel back to
        // renderer-style RGBA tint before applying cleanup.
        return new Rgba32(rawPixel.B, rawPixel.G, rawPixel.R, rawPixel.A);
    }

    public static byte[] RemoveMccvTint(string sourceMinimapPath, string mccvMapPath)
    {
        using Image<Rgba32> source = Image.Load<Rgba32>(sourceMinimapPath);
        using Image<Rgba32> mccv = Image.Load<Rgba32>(mccvMapPath);
        return RemoveMccvTint(source, mccv);
    }

    public static byte[] RemoveMccvTint(Image<Rgba32> source, Image<Rgba32> mccv)
    {
        using Image<Rgba32> working = source.Clone();
        using Image<Rgba32> overlay = mccv.Clone();

        if (overlay.Width != working.Width || overlay.Height != working.Height)
            overlay.Mutate(ctx => ctx.Resize(working.Width, working.Height));

        for (int y = 0; y < working.Height; y++)
        {
            for (int x = 0; x < working.Width; x++)
            {
                Rgba32 src = working[x, y];
                Rgba32 tint = DecodeStoredMccvColor(overlay[x, y]);

                working[x, y] = new Rgba32(
                    RemoveTint(src.R, tint.R),
                    RemoveTint(src.G, tint.G),
                    RemoveTint(src.B, tint.B),
                    src.A);
            }
        }

        using MemoryStream ms = new();
        working.SaveAsPng(ms);
        return ms.ToArray();
    }

    private static byte RemoveTint(byte source, byte tintChannel)
    {
        // Match MdxViewer's terrain shader: vertexTint = clamp(vVertexColor.rgb * 2.0, 0.0, 2.0)
        float tintFactor = Math.Clamp((tintChannel / 255f) * 2f, 1f / 255f, 2f);
        return (byte)Math.Clamp((int)MathF.Round(source / tintFactor), 0, 255);
    }

}