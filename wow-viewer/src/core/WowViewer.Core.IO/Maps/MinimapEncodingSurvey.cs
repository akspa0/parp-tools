using SereniaBLPLib;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Per-build/map distribution of tile encodings.
/// </summary>
/// <param name="Build">Build identity.</param>
/// <param name="Map">Map name.</param>
/// <param name="EncodingDistribution">Encoding label -> tile count.</param>
/// <param name="BuildRecognised">Whether the build was recognised by era-gating (FR-006).</param>
public sealed record EncodingSurveyResult(
    string Build,
    string Map,
    IReadOnlyDictionary<string, int> EncodingDistribution,
    bool BuildRecognised);

/// <summary>
/// Reports, per build and map, the distribution of encodings found, so the DXT1 assumption is
/// verified rather than inherited (FR-013).
///
/// WHY THIS EXISTS: only 0.5.3.3368 / Azeroth has been measured as DXT1. Every other build and map is
/// unverified. This survey detects each tile's actual encoding from the file (FR-001) and aggregates
/// it per build/map, so the DXT1 assumption is verified rather than spread.
/// </summary>
public static class MinimapEncodingSurvey
{
    /// <summary>
    /// Inspect authored tiles and report the distribution of encodings (format, compression mode,
    /// colour depth, dimensions, mip count) per file (FR-001, FR-013).
    /// </summary>
    public static EncodingSurveyResult Survey(
        string build,
        string map,
        IEnumerable<byte[]> authoredBlpFiles,
        bool buildRecognised)
    {
        ArgumentNullException.ThrowIfNull(authoredBlpFiles);

        var distribution = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (byte[] blp in authoredBlpFiles)
        {
            string label = DescribeEncoding(blp);
            distribution[label] = distribution.TryGetValue(label, out int count) ? count + 1 : 1;
        }

        return new EncodingSurveyResult(build, map, distribution, buildRecognised);
    }

    /// <summary>
    /// Describe a BLP file's encoding from its header. Returns a human-readable label such as
    /// "BLP2/DXTC/DXT1/256x256/1mip/noalpha" or "unknown".
    /// </summary>
    public static string DescribeEncoding(byte[] blp)
    {
        ArgumentNullException.ThrowIfNull(blp);

        if (blp.Length < 148)
            return "unknown";

        bool isBlp2 = blp[0] == 'B' && blp[1] == 'L' && blp[2] == 'P' && blp[3] == '2';
        if (!isBlp2)
            return "unknown";

        byte compression = blp[6];      // 2 = DXTC
        byte alphaDepth = blp[7];       // 0 = no alpha, 8 = alpha
        byte alphaEncoding = blp[8];    // 7 = DXT1, 8 = DXT3, 9 = DXT5
        byte hasMips = blp[9];
        int width = ReadInt32Le(blp, 12);
        int height = ReadInt32Le(blp, 16);

        string compressionName = compression switch
        {
            2 => "DXTC",
            _ => $"comp{compression}",
        };

        string alphaName = alphaEncoding switch
        {
            7 => "DXT1",
            8 => "DXT3",
            9 => "DXT5",
            _ => $"alpha{alphaEncoding}",
        };

        string alpha = alphaDepth == 0 ? "noalpha" : $"alpha{alphaDepth}";
        return $"BLP2/{compressionName}/{alphaName}/{width}x{height}/{hasMips}mip/{alpha}";
    }

    private static int ReadInt32Le(byte[] data, int offset) =>
        data[offset]
        | (data[offset + 1] << 8)
        | (data[offset + 2] << 16)
        | (data[offset + 3] << 24);
}
