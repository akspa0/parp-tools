using System.Buffers.Binary;
using System.Text;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.M2Chunked;
using WowViewer.Core.M2;
using WowViewer.Core.Runtime.M2;

namespace WowViewer.Core.Tests;

public sealed class M2ChunkedReaderTests
{
    [Fact]
    public void ChunkWalker_WalksSyntheticChunks_InDeclaredOrder()
    {
        byte[] bytes = CreateChunkedFile(
            ("VERS", [1, 0, 0, 0]),
            ("MODL", Encoding.ASCII.GetBytes("test\0")),
            ("TEXS", Encoding.ASCII.GetBytes("texture.blp\0")));

        using MemoryStream stream = new(bytes, writable: false);
        using BinaryReader reader = new(stream, Encoding.ASCII, leaveOpen: true);
        IReadOnlyList<M2ChunkedChunkHeader> chunks = new M2ChunkedChunkWalker(reader).Walk();

        Assert.Equal(3, chunks.Count);
        Assert.Equal("VERS", chunks[0].FourCC);
        Assert.Equal("MODL", chunks[1].FourCC);
        Assert.Equal("TEXS", chunks[2].FourCC);
        Assert.DoesNotContain(chunks, static chunk => chunk.IsTruncated);
    }

    [Fact]
    public void ChunkWalker_TruncatesMalformedChunk_InsteadOfThrowing()
    {
        byte[] bytes = new byte[4 + 8 + 2];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0, 4), MdxMagic.Mdlx);
        Encoding.ASCII.GetBytes("VERS").CopyTo(bytes, 4);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(8, 4), 16u);
        bytes[12] = 0x34;
        bytes[13] = 0x12;

        using MemoryStream stream = new(bytes, writable: false);
        using BinaryReader reader = new(stream, Encoding.ASCII, leaveOpen: true);
        IReadOnlyList<M2ChunkedChunkHeader> chunks = new M2ChunkedChunkWalker(reader).Walk();

        Assert.Single(chunks);
        Assert.True(chunks[0].IsTruncated);
        Assert.Equal(2u, chunks[0].Size);
    }

    [Fact]
    public void Dispatcher_ReadsSyntheticStrictMd20_ThroughExistingPath()
    {
        byte[] md20 = CreateMinimalMd20Bytes("SyntheticDispatcher");

        using MemoryStream stream = new(md20, writable: false);
        var document = M2ModelReaderDispatcher.Read(stream, "Creature\\SyntheticDispatcher\\SyntheticDispatcher.m2");

        Assert.Equal("MD20", document.Signature);
        Assert.Equal("SyntheticDispatcher", document.ModelName);
    }

    [Fact]
    public void ChunkedReader_Reads_Staged1121BearMdx_WhenArchiveIsPresent()
    {
        if (!TryReadStagedVirtualFile(
                "1.X_Retail_Windows_enUS_1.12.1.5875",
                "creature\\bear\\bear.mdx",
                out byte[] bytes,
                out string sourcePath,
                out Func<string, byte[]?> companionReader))
        {
            return;
        }

        using MemoryStream stream = new(bytes, writable: false);
        M2ChunkedReadResult result = M2ChunkedModelReader.ReadDetailed(stream, sourcePath, companionReader);

        Assert.Equal("MD20", result.Model.Signature);
        Assert.True(result.Geometry.GeosetCount > 0);
        Assert.True(result.VertexCount > 0);
        Assert.True(result.Summary.MaterialCount > 0);
        Assert.NotEmpty(result.Chunks);
    }

    [Fact]
    public void ChunkedReader_Reads_Staged053HumanMaleMdx_WhenArchiveIsPresent()
    {
        if (!TryReadStagedVirtualFile(
                "0_5_3_3368",
                "character\\human\\male\\humanmale.mdx",
                out byte[] bytes,
                out string sourcePath,
                out Func<string, byte[]?> companionReader))
        {
            return;
        }

        using MemoryStream stream = new(bytes, writable: false);
        M2ChunkedReadResult result = M2ChunkedModelReader.ReadDetailed(stream, sourcePath, companionReader);

        Assert.Equal("MD20", result.Model.Signature);
        Assert.True(result.Geometry.GeosetCount > 0);
        Assert.True(result.VertexCount > 0);
        Assert.True(result.Summary.MaterialCount > 0);
        Assert.NotEmpty(result.Chunks);
    }

    [Fact]
    public void ChunkedReader_Produces_RuntimeReadyConvertedArtifacts_WhenArchiveIsPresent()
    {
        if (!TryReadStagedVirtualFile(
                "1.X_Retail_Windows_enUS_1.12.1.5875",
                "creature\\bear\\bear.mdx",
                out byte[] bytes,
                out string sourcePath,
                out Func<string, byte[]?> companionReader))
        {
            return;
        }

        using MemoryStream stream = new(bytes, writable: false);
        M2ChunkedReadResult result = M2ChunkedModelReader.ReadDetailed(stream, sourcePath, companionReader);

        using MemoryStream geometryStream = new(result.Conversion.ModelBytes, writable: false);
        M2GeometryDocument geometry = M2GeometryReader.Read(geometryStream, result.Conversion.ModelPath);

        using MemoryStream skinStream = new(result.Conversion.SkinBytes, writable: false);
        M2SkinDocument skin = M2SkinReader.Read(skinStream, result.Conversion.SkinPath);

        M2SkinProfileRuntimeState state = M2SkinProfileRuntime.Choose(result.Model, 0);
        state = M2SkinProfileRuntime.Load(state, skin);
        state = M2SkinProfileRuntime.Initialize(state);

        M2StaticRenderModel runtimeModel = M2StaticRenderModelBuilder.Build(geometry, state);

        Assert.NotEmpty(geometry.Vertices);
        Assert.True(skin.VertexLookupCount > 0);
        Assert.NotEmpty(runtimeModel.Sections);
    }

    private static byte[] CreateChunkedFile(params (string Tag, byte[] Payload)[] chunks)
    {
        using MemoryStream stream = new();
        using BinaryWriter writer = new(stream, Encoding.ASCII, leaveOpen: true);
        writer.Write(MdxMagic.Mdlx);
        foreach ((string tag, byte[] payload) in chunks)
        {
            writer.Write(Encoding.ASCII.GetBytes(tag));
            writer.Write((uint)payload.Length);
            writer.Write(payload);
        }

        writer.Flush();
        return stream.ToArray();
    }

    private static byte[] CreateMinimalMd20Bytes(string modelName)
    {
        byte[] nameBytes = Encoding.UTF8.GetBytes(modelName + "\0");
        int nameOffset = 0x120;
        byte[] data = new byte[nameOffset + nameBytes.Length];
        Encoding.ASCII.GetBytes("MD20").CopyTo(data, 0);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x04, 4), 0x108u);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x08, 4), (uint)nameBytes.Length);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x0C, 4), (uint)nameOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x44, 4), 1u);
        nameBytes.CopyTo(data, nameOffset);
        return data;
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

        try
        {
            bytes = ArchiveVirtualFileReader.ReadVirtualFile(sourcePath, [clientDataPath], listfilePath);
            companionReader = path =>
            {
                try
                {
                    return ArchiveVirtualFileReader.ReadVirtualFile(path.Replace('/', '\\'), [clientDataPath], listfilePath);
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
            bytes = [];
            companionReader = static _ => null;
            return false;
        }
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
