using System.Globalization;
using System.Security.Cryptography;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.M2Chunked;
using WowViewer.Core.IO.M2Era1121;
using WowViewer.Core.M2;

namespace WowViewer.Core.Anim;

public static class M2PoseSourceLoader
{
    public static M2AnimationPoseSource LoadFromFile(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        PathNormalizer.AssertNoStalePath(path);

        using FileStream stream = File.OpenRead(path);
        byte[] bytes = ReadAllBytes(stream);
        M2DispatchResult dispatch = M2ModelReaderDispatcher.ReadDetailed(new MemoryStream(bytes, writable: false), Path.GetFullPath(path));
        return new M2AnimationPoseSource(
            dispatch.Document,
            Path.GetFullPath(path),
            MapEraToSourceFormat(dispatch.Era),
            ComputeSha256Hex(bytes));
    }

    public static M2AnimationPoseSource LoadFromVirtualFile(string virtualPath, IEnumerable<string> archiveRoots)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(virtualPath);
        ArgumentNullException.ThrowIfNull(archiveRoots);

        PathNormalizer.AssertNoStalePath(virtualPath);
        string[] roots = archiveRoots.ToArray();

        byte[] bytes = ArchiveVirtualFileReader.ReadVirtualFile(virtualPath, roots, bootstrapOptions: null);
        M2DispatchResult dispatch = M2ModelReaderDispatcher.ReadDetailed(new MemoryStream(bytes, writable: false), virtualPath);
        return new M2AnimationPoseSource(
            dispatch.Document,
            virtualPath,
            MapEraToSourceFormat(dispatch.Era),
            ComputeSha256Hex(bytes));
    }

    private static string MapEraToSourceFormat(M2Era1121EraTag era)
    {
        return era switch
        {
            M2Era1121EraTag.Mdlx => "chunked",
            M2Era1121EraTag.Md20_1X_V100 or M2Era1121EraTag.Md20_1X_V101 => "era1121",
            M2Era1121EraTag.Md20_3X_V108 => "classic",
            _ => "unknown",
        };
    }

    private static string ComputeSha256Hex(byte[] bytes)
    {
        Span<byte> hash = stackalloc byte[32];
        SHA256.HashData(bytes, hash);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    private static byte[] ReadAllBytes(Stream stream)
    {
        if (!stream.CanSeek)
            throw new ArgumentException("M2 pose source loading requires a seekable stream.", nameof(stream));

        long previousPosition = stream.Position;
        try
        {
            stream.Position = 0;
            byte[] data = new byte[checked((int)stream.Length)];
            stream.ReadExactly(data);
            return data;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }
}
