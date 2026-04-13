using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace WoWMapConverter.Core.VLM;

/// <summary>
/// Utility helpers for producing terrain-oriented minimap variants from exported signals.
/// </summary>
public static class VlmMinimapCleanupService
{
    private const int NeutralMccvValue = 127;

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
            overlay.Mutate(ctx => ctx.Resize(working.Width, working.Height, KnownResamplers.Bilinear));

        working.ProcessPixelRows(sourceAccessor =>
        {
            overlay.ProcessPixelRows(overlayAccessor =>
            {
                for (int y = 0; y < sourceAccessor.Height; y++)
                {
                    Span<Rgba32> srcRow = sourceAccessor.GetRowSpan(y);
                    Span<Rgba32> overlayRow = overlayAccessor.GetRowSpan(y);
                    for (int x = 0; x < srcRow.Length; x++)
                    {
                        Rgba32 src = srcRow[x];
                        Rgba32 tint = overlayRow[x];

                        int deltaR = tint.R - NeutralMccvValue;
                        int deltaG = tint.G - NeutralMccvValue;
                        int deltaB = tint.B - NeutralMccvValue;

                        srcRow[x] = new Rgba32(
                            ClampToByte(src.R - deltaR),
                            ClampToByte(src.G - deltaG),
                            ClampToByte(src.B - deltaB),
                            src.A);
                    }
                }
            });
        });

        using MemoryStream ms = new();
        working.SaveAsPng(ms);
        return ms.ToArray();
    }

    private static byte ClampToByte(int value) => (byte)Math.Clamp(value, 0, 255);
}