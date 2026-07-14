using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.M2Chunked;
using WowViewer.Core.IO.M2Era1121;
using WowViewer.Core.M2;

namespace WowViewer.Core.Tests;

public sealed class M2Era1121ModelReaderTests
{
    [Fact]
    public void M2Era1121ModelReader_Read_TestFixture_PopulatesInlineEra1121Geometry_WhenArchiveIsPresent()
    {
        if (!TryReadStagedVirtualFile(
                "1.X_Retail_Windows_enUS_1.12.1.5875",
                "creature\\bear\\bear.mdx",
                out byte[] bytes,
                out string sourcePath,
                out _))
        {
            return;
        }

        using MemoryStream stream = new(bytes, writable: false);
        M2ModelDocument document = M2Era1121ModelReader.Read(stream, sourcePath);

        Assert.NotNull(document.InlineEra1121Geometry);
        M2Era1121Geometry geometry = document.InlineEra1121Geometry!;
        Assert.True(geometry.Vertices.Count > 0, "Bear must have at least one vertex index.");
        Assert.True(geometry.Positions.Count > 0, "Bear must have at least one unique position.");
        Assert.True(geometry.Normals.Count > 0, "Bear must have at least one unique normal.");

        int validVertices = 0;
        foreach (M2Era1121VertexIndex v in geometry.Vertices)
        {
            if (v.PositionIndex < geometry.Positions.Count && v.NormalIndex < geometry.Normals.Count)
                validVertices++;
        }
        Assert.True(validVertices > 0, "At least one vertex must reference valid position and normal indices.");
        Assert.True(validVertices >= geometry.Vertices.Count / 2, "Majority of vertices must reference valid position and normal indices.");

        int finitePositions = 0;
        foreach (Vector3 p in geometry.Positions)
        {
            if (float.IsFinite(p.X) && float.IsFinite(p.Y) && float.IsFinite(p.Z))
                finitePositions++;
        }
        Assert.Equal(geometry.Positions.Count, finitePositions);

        int finiteNormals = 0;
        foreach (Vector3 n in geometry.Normals)
        {
            if (float.IsFinite(n.X) && float.IsFinite(n.Y) && float.IsFinite(n.Z))
                finiteNormals++;
        }
        Assert.Equal(geometry.Normals.Count, finiteNormals);
    }

    [Fact]
    public void Dispatcher_3X_Model_GoesTo3X_Reader_NotTo1121Reader()
    {
        byte[] md20 = CreateSyntheticMd20_3X("Synthetic3X");

        using MemoryStream stream = new(md20, writable: false);
        M2DispatchResult result = M2ModelReaderDispatcher.ReadDetailed(stream, "Creature\\Synthetic3X\\Synthetic3X.m2");

        Assert.Equal(M2Era1121EraTag.Md20_3X_V108, result.Era);
        Assert.Equal("MD20", result.Document.Signature);
        Assert.Equal(0x108u, result.Document.Version);
    }

    [Fact]
    public void M2Era1121ModelReader_FlagsPassThrough_And_HasNo3XOnlyCvarAccessors()
    {
        byte[] md20 = CreateSynthetic1121Md20("SyntheticFlags", flags: 0x80000000u);

        using MemoryStream stream = new(md20, writable: false);
        M2ModelDocument document = M2Era1121ModelReader.Read(stream, "Creature\\SyntheticFlags\\SyntheticFlags.mdx");

        Assert.Equal(0x80000000u, document.Flags);
        Assert.DoesNotContain("M2BatchParticles", GetAllPublicMemberNames(typeof(M2Era1121ModelReader)));
        Assert.DoesNotContain("M2ForceAdditiveParticleSort", GetAllPublicMemberNames(typeof(M2Era1121ModelReader)));
    }

    [Fact]
    public void Dispatcher_MDLX_GoesToChunkedReader_NotTo1121Reader()
    {
        byte[] mdlx = CreateSyntheticMdlx();

        using MemoryStream stream = new(mdlx, writable: false);
        M2DispatchResult result = M2ModelReaderDispatcher.ReadDetailed(stream, "World\\Generic\\SyntheticMdlx.mdx");

        Assert.Equal(M2Era1121EraTag.Mdlx, result.Era);
    }

    [Fact]
    public void Dispatcher_2X_Version_Throws_WithSpec049Message()
    {
        byte[] md20_2x = CreateSyntheticMd20_2X("Synthetic2X");

        using MemoryStream stream = new(md20_2x, writable: false);
        NotSupportedException ex = Assert.Throws<NotSupportedException>(() =>
            M2ModelReaderDispatcher.ReadDetailed(stream, "Creature\\Synthetic2X\\Synthetic2X.m2"));

        Assert.Contains("049", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Dispatcher_4X_Version_GoesToCataTag()
    {
        byte[] md20_4x = CreateSyntheticMd20_4X("Synthetic4X");

        using MemoryStream stream = new(md20_4x, writable: false);
        M2DispatchResult result = M2ModelReaderDispatcher.ReadDetailed(stream, "Creature\\Synthetic4X\\Synthetic4X.m2");

        Assert.Equal(M2Era1121EraTag.Md20_4X_V109, result.Era);
        Assert.Equal("MD20", result.Document.Signature);
        Assert.Equal(0x109u, result.Document.Version);
    }

    [Fact]
    public void Dispatcher_10A_Version_AlsoGoesToCataTag()
    {
        byte[] md20_10a = CreateSyntheticMd20_4X("Synthetic10A", version: 0x10Au);

        using MemoryStream stream = new(md20_10a, writable: false);
        M2DispatchResult result = M2ModelReaderDispatcher.ReadDetailed(stream, "Creature\\Synthetic10A\\Synthetic10A.m2");

        Assert.Equal(M2Era1121EraTag.Md20_4X_V109, result.Era);
    }

    [Fact]
    public void M2Era1121ModelReader_RejectsBadMagic()
    {
        byte[] bad = new byte[0x100];
        Encoding.ASCII.GetBytes("JUNK").CopyTo(bad, 0);

        using MemoryStream stream = new(bad, writable: false);
        Assert.Throws<InvalidDataException>(() => M2Era1121ModelReader.Read(stream, "Bad\\Magic.bin"));
    }

    [Fact]
    public void M2Era1121ModelReader_RejectsWrongVersion()
    {
        byte[] md20 = CreateSyntheticMd20_3X("SyntheticWrong");

        using MemoryStream stream = new(md20, writable: false);
        NotSupportedException ex = Assert.Throws<NotSupportedException>(() =>
            M2Era1121ModelReader.Read(stream, "Creature\\SyntheticWrong\\SyntheticWrong.mdx"));

        Assert.Contains("0x108", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    private static byte[] CreateSynthetic1121Md20(string modelName, uint flags)
    {
        byte[] nameBytes = Encoding.UTF8.GetBytes(modelName + "\0");
        int nameOffset = 0x150;
        byte[] data = new byte[nameOffset + nameBytes.Length];
        Encoding.ASCII.GetBytes("MD20").CopyTo(data, 0);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x04, 4), 0x100u);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x08, 4), (uint)nameBytes.Length);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x0C, 4), (uint)nameOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x10, 4), flags);
        nameBytes.CopyTo(data, nameOffset);
        return data;
    }

    private static byte[] CreateSyntheticMd20_3X(string modelName)
    {
        byte[] nameBytes = Encoding.UTF8.GetBytes(modelName + "\0");
        int nameOffset = 0x180;
        byte[] data = new byte[nameOffset + nameBytes.Length];
        Encoding.ASCII.GetBytes("MD20").CopyTo(data, 0);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x04, 4), 0x108u);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x08, 4), (uint)nameBytes.Length);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x0C, 4), (uint)nameOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x44, 4), 1u);
        nameBytes.CopyTo(data, nameOffset);
        return data;
    }

    private static byte[] CreateSyntheticMd20_2X(string modelName)
    {
        byte[] nameBytes = Encoding.UTF8.GetBytes(modelName + "\0");
        int nameOffset = 0x180;
        byte[] data = new byte[nameOffset + nameBytes.Length];
        Encoding.ASCII.GetBytes("MD20").CopyTo(data, 0);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x04, 4), 0x104u);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x08, 4), (uint)nameBytes.Length);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x0C, 4), (uint)nameOffset);
        nameBytes.CopyTo(data, nameOffset);
        return data;
    }

    private static byte[] CreateSyntheticMd20_4X(string modelName, uint version = 0x109u)
    {
        byte[] nameBytes = Encoding.UTF8.GetBytes(modelName + "\0");
        int nameOffset = 0x180;
        byte[] data = new byte[nameOffset + nameBytes.Length];
        Encoding.ASCII.GetBytes("MD20").CopyTo(data, 0);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x04, 4), version);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x08, 4), (uint)nameBytes.Length);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x0C, 4), (uint)nameOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x44, 4), 1u);
        nameBytes.CopyTo(data, nameOffset);
        return data;
    }

    private static byte[] CreateSyntheticMdlx()
    {
        byte[] data = new byte[16];
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0, 4), MdxMagic.Mdlx);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(4, 4), 0x56455253u);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(8, 4), 4u);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(12, 4), 0x01000000u);
        return data;
    }

    private static IEnumerable<string> GetAllPublicMemberNames(Type type)
    {
        foreach (System.Reflection.MemberInfo member in type.GetMembers(System.Reflection.BindingFlags.Public
            | System.Reflection.BindingFlags.Static
            | System.Reflection.BindingFlags.Instance
            | System.Reflection.BindingFlags.DeclaredOnly))
        {
            yield return member.Name;
        }
    }

    private static bool TryReadStagedVirtualFile(
        string stagedBuildDirectory,
        string virtualPath,
        out byte[] bytes,
        out string sourcePath,
        out Func<string, byte[]?> companionReader)
    {
        string root = GetWowViewerRoot();
        string clientDataPath = Path.Combine(root, "..", "output", "tmp", "wowarchive-clients", stagedBuildDirectory, "World of Warcraft", "Data");
        string listfilePath = Path.Combine(root, "libs", "wowdev", "wow-listfile", "listfile.txt");
        sourcePath = virtualPath.Replace('/', '\\');

        if (!Directory.Exists(clientDataPath) || !File.Exists(listfilePath))
        {
            bytes = [];
            companionReader = static _ => null;
            return false;
        }

        string ext = Path.GetExtension(sourcePath);
        string[] aliasExtensions = { ".mdx", ".m2", ".mdl" };
        string requestedPath = sourcePath;
        string requestedBase = Path.Combine(Path.GetDirectoryName(requestedPath) ?? string.Empty, Path.GetFileNameWithoutExtension(requestedPath));
        bool isModelExt = aliasExtensions.Any(a => a.Equals(ext, StringComparison.OrdinalIgnoreCase));
        List<string> candidateList = new() { requestedPath };
        if (isModelExt)
        {
            foreach (string a in aliasExtensions)
            {
                if (a.Equals(ext, StringComparison.OrdinalIgnoreCase)) continue;
                candidateList.Add(requestedBase + a);
            }
        }
        string[] candidatePaths = candidateList.ToArray();

        string resolvedPath = requestedPath;
        foreach (string candidate in candidatePaths)
        {
            try
            {
                bytes = ArchiveVirtualFileReader.ReadVirtualFile(candidate, [clientDataPath], listfilePath);
                resolvedPath = candidate;
                sourcePath = resolvedPath;
                string clientDataForLambda = clientDataPath;
                string listfileForLambda = listfilePath;
                companionReader = path =>
                {
                    try
                    {
                        return ArchiveVirtualFileReader.ReadVirtualFile(path.Replace('/', '\\'), [clientDataForLambda], listfileForLambda);
                    }
                    catch (FileNotFoundException)
                    {
                        return null;
                    }
                };
                return true;
            }
            catch (FileNotFoundException)
            {
            }
        }

        bytes = [];
        companionReader = static _ => null;
        return false;
    }

    private static string GetWowViewerRoot()
    {
        DirectoryInfo? current = new(AppContext.BaseDirectory);
        while (current is not null)
        {
            if (File.Exists(Path.Combine(current.FullName, "WowViewer.slnx")))
                return current.FullName;

            current = current.Parent;
        }

        throw new DirectoryNotFoundException("Could not locate the wow-viewer repository root from the test output directory.");
    }
}
