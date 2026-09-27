using Silk.NET.OpenGL;
using WowViewer.Core.Blp;
using WowViewer.Core.IO.Blp;

namespace WoWViewer.Rendering;

/// <summary>
/// Spec 256 P2c: a DXT-compressed BLP2 prepared for a direct GPU upload — its own mip levels, no CPU decode.
/// Format choice matches SereniaBLPLib's decoder (alpha depth &gt; 1 → DXT5 when the pixel format says so,
/// else DXT3; otherwise RGBA-DXT1, whose transparent block mode the library also decodes as alpha 0).
/// </summary>
internal sealed class BlpCompressedTexture
{
    public required byte[] Data { get; init; }
    public required InternalFormat Format { get; init; }
    public required IReadOnlyList<(int Width, int Height, int Offset, int Length)> Levels { get; init; }

    /// <summary>Set once on the render thread; workers read it to decide whether to skip the CPU decode.</summary>
    public static volatile bool S3tcSupported;

    public static void DetectSupport(GL gl)
    {
        if (!S3tcSupported)
            S3tcSupported = gl.IsExtensionPresent("GL_EXT_texture_compression_s3tc");
    }

    /// <summary>
    /// Returns the compressed levels 0..n of a DXT BLP2, stopping at the first level that is missing, out of
    /// bounds or too small (the GL texture's max level is set to the last one kept). Null when the file is not
    /// a usable DXT BLP2 or S3TC is unavailable; callers then decode as before.
    /// </summary>
    public static BlpCompressedTexture? TryCreate(byte[] data, string sourceName)
    {
        if (!S3tcSupported || data.Length < 4)
            return null;

        BlpSummary summary;
        try
        {
            using var stream = new MemoryStream(data, writable: false);
            summary = BlpSummaryReader.Read(stream, sourceName);
        }
        catch (Exception)
        {
            return null;
        }

        if (summary.Format != BlpFormat.Blp2 || summary.Compression != BlpCompressionType.Dxtc)
            return null;

        (InternalFormat format, int blockBytes) = summary.AlphaDepthBits > 1
            ? (summary.PixelFormat == WowViewer.Core.Blp.BlpPixelFormat.Dxt5
                ? (InternalFormat.CompressedRgbaS3TCDxt5Ext, 16)
                : (InternalFormat.CompressedRgbaS3TCDxt3Ext, 16))
            : (InternalFormat.CompressedRgbaS3TCDxt1Ext, 8);

        var levels = new List<(int, int, int, int)>();
        for (int level = 0; level < 16; level++)
        {
            BlpMipMapEntry? mip = null;
            foreach (BlpMipMapEntry entry in summary.MipMaps)
            {
                if (entry.Level == level)
                {
                    mip = entry;
                    break;
                }
            }

            int width = Math.Max(1, summary.Width >> level);
            int height = Math.Max(1, summary.Height >> level);
            long expected = (long)Math.Max(1, (width + 3) / 4) * Math.Max(1, (height + 3) / 4) * blockBytes;
            if (mip is not { IsInBounds: true } || mip.SizeBytes < expected || mip.Offset + expected > data.Length)
                break;

            levels.Add((width, height, (int)mip.Offset, (int)expected));
            if (width == 1 && height == 1)
                break;
        }

        return levels.Count == 0 ? null : new BlpCompressedTexture { Data = data, Format = format, Levels = levels };
    }
}
