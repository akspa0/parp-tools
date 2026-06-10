using System.Security.Cryptography;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Mdx;

namespace WowViewer.Core.Anim;

public static class MdxPoseSourceLoader
{
    public static MdxAnimationPoseSource LoadFromFile(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        PathNormalizer.AssertNoStalePath(path);

        byte[] bytes = File.ReadAllBytes(path);
        MdxFile mdx = MdxFile.Load(new MemoryStream(bytes, writable: false));
        return new MdxAnimationPoseSource(mdx, Path.GetFullPath(path), ComputeSha256Hex(bytes));
    }

    public static MdxAnimationPoseSource LoadFromVirtualFile(string virtualPath, IEnumerable<string> archiveRoots)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(virtualPath);
        ArgumentNullException.ThrowIfNull(archiveRoots);

        PathNormalizer.AssertNoStalePath(virtualPath);
        string[] roots = archiveRoots.ToArray();

        byte[] bytes = ArchiveVirtualFileReader.ReadVirtualFile(virtualPath, roots, bootstrapOptions: null);
        MdxFile mdx = MdxFile.Load(new MemoryStream(bytes, writable: false));
        return new MdxAnimationPoseSource(mdx, virtualPath, ComputeSha256Hex(bytes));
    }

    private static string ComputeSha256Hex(byte[] bytes)
    {
        Span<byte> hash = stackalloc byte[32];
        SHA256.HashData(bytes, hash);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }
}
