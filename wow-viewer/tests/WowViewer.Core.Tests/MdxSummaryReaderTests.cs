using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.Files;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.Tests;

public sealed class MdxSummaryReaderTests
{
    [Fact]
    public void Read_SyntheticMdxHeader_ProducesExpectedSummary()
    {
        byte[] bytes = CreateMdxBytes(
            version: 1300,
            modelName: "SyntheticChest",
            blendTime: 150,
            boundsMin: new Vector3(-1.0f, -2.0f, -3.0f),
            boundsMax: new Vector3(4.0f, 5.0f, 6.0f),
            textures:
            [
                (0u, "Textures\\SyntheticChestMain.blp", 3u),
                (1u, string.Empty, 0u),
            ],
            materials:
            [
                (0, [(0u, 0x10u, 0, -1, 0, 1.0f)]),
                (2, [(1u, 0x20u, 1, -1, 1, 0.5f)]),
            ],
            extraChunks:
            [
                CreateChunk("UNKN", [5, 6, 7, 8]),
            ]);

        using MemoryStream stream = new(bytes);
        MdxSummary summary = MdxSummaryReader.Read(stream, "synthetic.mdx");

        Assert.Equal("MDLX", summary.Signature);
        Assert.Equal(1300u, summary.Version);
        Assert.Equal("SyntheticChest", summary.ModelName);
        Assert.Equal(150u, summary.BlendTime);
        Assert.Equal(new Vector3(-1.0f, -2.0f, -3.0f), summary.BoundsMin);
        Assert.Equal(new Vector3(4.0f, 5.0f, 6.0f), summary.BoundsMax);
        Assert.Equal(5, summary.ChunkCount);
        Assert.Equal(4, summary.KnownChunkCount);
        Assert.Equal(1, summary.UnknownChunkCount);
        Assert.Equal(2, summary.TextureCount);
        Assert.Equal(1, summary.ReplaceableTextureCount);
        Assert.Equal("Textures\\SyntheticChestMain.blp", summary.Textures[0].Path);
        Assert.Equal(3u, summary.Textures[0].Flags);
        Assert.Equal(1u, summary.Textures[1].ReplaceableId);
        Assert.Null(summary.Textures[1].Path);
        Assert.Equal(2, summary.MaterialCount);
        Assert.Equal(2, summary.MaterialLayerCount);
        Assert.Equal(0, summary.Materials[0].PriorityPlane);
        Assert.Equal(1, summary.Materials[0].LayerCount);
        Assert.Equal(0u, summary.Materials[0].Layers[0].BlendMode);
        Assert.Equal(0x10u, summary.Materials[0].Layers[0].Flags);
        Assert.Equal(0, summary.Materials[0].Layers[0].TextureId);
        Assert.Equal(2, summary.Materials[1].PriorityPlane);
        Assert.Equal(1u, summary.Materials[1].Layers[0].BlendMode);
        Assert.Equal(0.5f, summary.Materials[1].Layers[0].StaticAlpha);
        Assert.Equal(MdxChunkIds.Vers, summary.Chunks[0].Id);
        Assert.Equal(MdxChunkIds.Modl, summary.Chunks[1].Id);
        Assert.Equal(MdxChunkIds.Texs, summary.Chunks[2].Id);
        Assert.Equal(MdxChunkIds.Mtls, summary.Chunks[3].Id);
        Assert.Equal("UNKN", summary.Chunks[4].Id.ToString());
        Assert.False(summary.Chunks[4].IsKnownChunk);
    }

    [Fact]
    public void Read_RealStandardArchiveMdx_ProducesExpectedSignals()
    {
        if (!Directory.Exists(MdxTestPaths.Standard060DataPath) || !File.Exists(MdxTestPaths.ListfilePath))
            return;

        using IArchiveCatalog catalog = new MpqArchiveCatalog();
        ArchiveCatalogBootstrapResult bootstrap = ArchiveCatalogBootstrapper.Bootstrap(catalog, [MdxTestPaths.Standard060DataPath], MdxTestPaths.ListfilePath);
        Assert.NotNull(bootstrap);

        if (!catalog.FileExists(MdxTestPaths.Standard060MdxVirtualPath))
            return;

        byte[]? bytes = catalog.ReadFile(MdxTestPaths.Standard060MdxVirtualPath);
        Assert.NotNull(bytes);

        using MemoryStream detectionStream = new(bytes);
        WowFileDetection detection = WowFileDetector.Detect(detectionStream, MdxTestPaths.Standard060MdxVirtualPath);
        Assert.Equal(WowFileKind.Mdx, detection.Kind);

        using MemoryStream summaryStream = new(bytes);
        MdxSummary summary = MdxSummaryReader.Read(summaryStream, MdxTestPaths.Standard060MdxVirtualPath);

        Assert.Equal("MDLX", summary.Signature);
        Assert.Equal(1300u, summary.Version);
        Assert.Equal("Chest01", summary.ModelName);
        Assert.True(summary.ChunkCount > 0);
        Assert.True(summary.KnownChunkCount > 0);
        Assert.Equal(2, summary.TextureCount);
        Assert.Equal(0, summary.ReplaceableTextureCount);
        Assert.Equal("WORLD\\GENERIC\\ACTIVEDOODADS\\CHEST01\\CHEST1SIDE.BLP", summary.Textures[0].Path);
        Assert.Equal("WORLD\\GENERIC\\ACTIVEDOODADS\\CHEST01\\CHEST1FRONT.BLP", summary.Textures[1].Path);
        Assert.Equal(2, summary.MaterialCount);
        Assert.Equal(2, summary.MaterialLayerCount);
        Assert.Equal(0, summary.Materials[0].PriorityPlane);
        Assert.Equal(0u, summary.Materials[0].Layers[0].BlendMode);
        Assert.Equal(0x10u, summary.Materials[0].Layers[0].Flags);
        Assert.Equal(0, summary.Materials[0].Layers[0].TextureId);
        Assert.Equal(1, summary.Materials[1].Layers[0].TextureId);
        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Vers);
        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Modl);
        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Mtls);
        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Texs);
    }

    private static byte[] CreateMdxBytes(uint version, string modelName, uint blendTime, Vector3 boundsMin, Vector3 boundsMax, IReadOnlyList<(uint ReplaceableId, string Path, uint Flags)> textures, IReadOnlyList<(int PriorityPlane, IReadOnlyList<(uint BlendMode, uint Flags, int TextureId, int TransformId, int CoordId, float StaticAlpha)> Layers)> materials, IReadOnlyList<byte[]> extraChunks)
    {
        List<byte> bytes =
        [
            (byte)'M', (byte)'D', (byte)'L', (byte)'X',
        ];

        bytes.AddRange(CreateChunk("VERS", CreateUInt32Payload(version)));
        bytes.AddRange(CreateChunk("MODL", CreateModlPayload(modelName, blendTime, boundsMin, boundsMax)));
        bytes.AddRange(CreateChunk("TEXS", CreateTexsPayload(textures)));
        bytes.AddRange(CreateChunk("MTLS", CreateMtlsPayload(materials)));
        foreach (byte[] chunk in extraChunks)
            bytes.AddRange(chunk);

        return [.. bytes];
    }

    private static byte[] CreateModlPayload(string modelName, uint blendTime, Vector3 boundsMin, Vector3 boundsMax)
    {
        byte[] payload = new byte[0x6C];
        WriteFixedAscii(payload, 0, 0x50, modelName);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(0x50, 4), boundsMin.X);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(0x54, 4), boundsMin.Y);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(0x58, 4), boundsMin.Z);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(0x5C, 4), boundsMax.X);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(0x60, 4), boundsMax.Y);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(0x64, 4), boundsMax.Z);
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(0x68, 4), blendTime);
        return payload;
    }

    private static byte[] CreateTexsPayload(IReadOnlyList<(uint ReplaceableId, string Path, uint Flags)> textures)
    {
        const int entrySize = 0x10C;
        const int pathSize = 0x104;

        byte[] payload = new byte[textures.Count * entrySize];
        for (int index = 0; index < textures.Count; index++)
        {
            int offset = index * entrySize;
            BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(offset, 4), textures[index].ReplaceableId);
            WriteFixedAscii(payload, offset + 4, pathSize, textures[index].Path);
            BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(offset + 4 + pathSize, 4), textures[index].Flags);
        }

        return payload;
    }

    private static byte[] CreateMtlsPayload(IReadOnlyList<(int PriorityPlane, IReadOnlyList<(uint BlendMode, uint Flags, int TextureId, int TransformId, int CoordId, float StaticAlpha)> Layers)> materials)
    {
        List<byte> payload = [];
        payload.AddRange(CreateUInt32Payload((uint)materials.Count));
        payload.AddRange(CreateUInt32Payload(0));

        foreach (var material in materials)
        {
            List<byte> materialBytes = [];
            materialBytes.AddRange(CreateInt32Payload(material.PriorityPlane));
            materialBytes.AddRange(CreateUInt32Payload((uint)material.Layers.Count));

            foreach (var layer in material.Layers)
            {
                List<byte> layerBytes = [];
                layerBytes.AddRange(CreateUInt32Payload(layer.BlendMode));
                layerBytes.AddRange(CreateUInt32Payload(layer.Flags));
                layerBytes.AddRange(CreateInt32Payload(layer.TextureId));
                layerBytes.AddRange(CreateInt32Payload(layer.TransformId));
                layerBytes.AddRange(CreateInt32Payload(layer.CoordId));
                layerBytes.AddRange(CreateSinglePayload(layer.StaticAlpha));

                materialBytes.AddRange(CreateSizedPayload(layerBytes));
            }

            payload.AddRange(CreateSizedPayload(materialBytes));
        }

        return [.. payload];
    }

    private static void WriteFixedAscii(byte[] buffer, int offset, int length, string value)
    {
        int count = Math.Min(length, value.Length);
        for (int index = 0; index < count; index++)
            buffer[offset + index] = (byte)value[index];
    }

    private static byte[] CreateChunk(string id, byte[] payload)
    {
        byte[] bytes = new byte[8 + payload.Length];
        for (int index = 0; index < 4; index++)
            bytes[index] = (byte)id[index];

        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4), (uint)payload.Length);
        Array.Copy(payload, 0, bytes, 8, payload.Length);
        return bytes;
    }

    private static byte[] CreateUInt32Payload(uint value)
    {
        return MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(value);
    }

    private static byte[] CreateInt32Payload(int value)
    {
        byte[] bytes = new byte[4];
        BinaryPrimitives.WriteInt32LittleEndian(bytes, value);
        return bytes;
    }

    private static byte[] CreateSinglePayload(float value)
    {
        byte[] bytes = new byte[4];
        BinaryPrimitives.WriteSingleLittleEndian(bytes, value);
        return bytes;
    }

    private static byte[] CreateSizedPayload(List<byte> payload)
    {
        byte[] bytes = new byte[4 + payload.Count];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0, 4), (uint)(4 + payload.Count));
        payload.CopyTo(bytes, 4);
        return bytes;
    }
}

internal static class MdxTestPaths
{
    public const string Standard060MdxVirtualPath = "world/generic/activedoodads/chest01/chest01.mdx";

    public static string Standard060DataPath => Path.Combine(GetWowViewerRoot(), "testdata", "0.6.0", "World of Warcraft", "Data");

    public static string ListfilePath => Path.Combine(GetWowViewerRoot(), "libs", "wowdev", "wow-listfile", "listfile.txt");

    private static string GetWowViewerRoot()
    {
        string? current = AppContext.BaseDirectory;
        while (!string.IsNullOrWhiteSpace(current))
        {
            if (File.Exists(Path.Combine(current, "WowViewer.slnx")))
                return current;

            current = Directory.GetParent(current)?.FullName;
        }

        throw new DirectoryNotFoundException("Could not locate wow-viewer workspace root from the current test context.");
    }
}