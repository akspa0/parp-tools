using WowViewer.Core.IO.M2Chunked;
using WowViewer.Core.IO.Mdx;

namespace WowViewer.Core.IO.AssetReferences;

/// <summary>
/// Pulls the texture paths a model claims, across the model formats that read today.
/// </summary>
/// <remarks>
/// Model routes are not uniformly readable. Spec 154 measured that the MD20 routes between the Alpha
/// chunked format and 0x108 fail, so a caller sweeping such a build must record the route as blocked
/// rather than sweeping it and reporting no textures — an unread model and a model with no textures
/// are different facts.
/// </remarks>
public static class ModelReferenceExtractor
{
    /// <summary>
    /// Extracts every texture path named by a model. Empty texture slots are skipped: a model may
    /// declare a replaceable-texture slot with no path, which is a real authored state and not a
    /// reference to an empty path.
    /// </summary>
    public static IReadOnlyList<string> Extract(byte[] modelBytes, string sourcePath)
    {
        ArgumentNullException.ThrowIfNull(modelBytes);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        return IsMdx(modelBytes)
            ? ExtractFromMdx(modelBytes)
            : ExtractFromM2(modelBytes, sourcePath);
    }

    /// <summary>
    /// True when the bytes carry a model magic this extractor understands well enough to attempt.
    /// A caller can use this to separate "not a model" from "a model that failed".
    /// </summary>
    public static bool LooksLikeModel(byte[] modelBytes)
    {
        if (modelBytes.Length < 4)
            return false;

        return IsMdx(modelBytes) || IsMd20(modelBytes);
    }

    private static bool IsMdx(byte[] bytes)
        => bytes.Length >= 4 && bytes[0] == (byte)'M' && bytes[1] == (byte)'D' && bytes[2] == (byte)'L' && bytes[3] == (byte)'X';

    private static bool IsMd20(byte[] bytes)
        => bytes.Length >= 4 && bytes[0] == (byte)'M' && bytes[1] == (byte)'D' && bytes[2] == (byte)'2' && bytes[3] == (byte)'0';

    private static IReadOnlyList<string> ExtractFromMdx(byte[] modelBytes)
    {
        using MemoryStream stream = new(modelBytes, writable: false);
        MdxFile mdx = MdxFile.Load(stream);

        List<string> paths = [];
        HashSet<string> seen = new(StringComparer.OrdinalIgnoreCase);
        foreach (MdlTexture texture in mdx.Textures)
        {
            string path = texture.Path;
            if (string.IsNullOrWhiteSpace(path) || !seen.Add(path))
                continue;

            paths.Add(path);
        }

        return paths;
    }

    /// <summary>
    /// Texture paths for an MD20 model. The era readers expose their own inline geometry, which
    /// carries the texture table independently of the bone table — so an era whose bones do not parse
    /// can still yield its textures. Only when no era supplies inline geometry does this fall back to
    /// the classic geometry reader, whose failure is a genuine unreadable result.
    /// </summary>
    private static IReadOnlyList<string> ExtractFromM2(byte[] modelBytes, string sourcePath)
    {
        using MemoryStream stream = new(modelBytes, writable: false);
        Core.M2.M2ModelDocument document = M2ModelReaderDispatcher.Read(stream, sourcePath);

        List<string> paths = [];
        HashSet<string> seen = new(StringComparer.OrdinalIgnoreCase);

        foreach (string path in EnumerateInlineTexturePaths(document))
            AddPath(path);

        if (paths.Count > 0)
            return paths;

        stream.Position = 0;
        Core.M2.M2GeometryDocument geometry = M2.M2GeometryReader.Read(stream, sourcePath);
        foreach (Core.M2.M2GeometryTexture texture in geometry.Textures)
            AddPath(texture.Filename);

        return paths;

        void AddPath(string? path)
        {
            if (string.IsNullOrWhiteSpace(path) || !seen.Add(path))
                return;

            paths.Add(path);
        }
    }

    private static IEnumerable<string> EnumerateInlineTexturePaths(Core.M2.M2ModelDocument document)
    {
        if (document.InlineEra100Geometry is { } era100)
        {
            foreach (Core.M2.M2Era100Texture texture in era100.Textures)
                yield return texture.Filename;
        }

        if (document.InlineEra1121Geometry is { } era1121)
        {
            foreach (Core.M2.M2Era1121Texture texture in era1121.Textures)
                yield return texture.Filename;
        }
    }
}
