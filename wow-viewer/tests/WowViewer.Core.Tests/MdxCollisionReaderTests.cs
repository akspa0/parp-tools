using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.Files;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.Tests;

public sealed class MdxCollisionReaderTests
{
    [Fact]
    public void Read_SyntheticClassicClid_ProducesExpectedCollisionPayload()
    {
        byte[] bytes = CreateMdxBytes(
            version: 1300,
            modelName: "SyntheticCollisionPayload",
            boundsMin: new Vector3(-1.0f, -1.0f, -1.0f),
            boundsMax: new Vector3(1.0f, 1.0f, 1.0f),
            extraChunks:
            [
                CreateChunk("CLID", CreateCollisionPayload(
                    [
                        new Vector3(-1.0f, -2.0f, -3.0f),
                        new Vector3(4.0f, -2.0f, -3.0f),
                        new Vector3(4.0f, 5.0f, 6.0f),
                        new Vector3(-1.0f, 5.0f, 6.0f),
                    ],
                    [0, 1, 2, 0, 2, 3],
                    [
                        new Vector3(0.0f, 0.0f, 1.0f),
                        new Vector3(0.0f, 1.0f, 0.0f),
                    ])),
            ]);

        using MemoryStream stream = new(bytes);
        MdxCollisionFile collisionFile = MdxCollisionReader.Read(stream, "synthetic_clid_payload.mdx");

        Assert.Equal((uint)1300, collisionFile.Version);
        Assert.Equal("SyntheticCollisionPayload", collisionFile.ModelName);
        Assert.True(collisionFile.HasCollision);

        MdxCollisionMesh collision = Assert.IsType<MdxCollisionMesh>(collisionFile.Collision);
        Assert.Equal(4, collision.VertexCount);
        Assert.Equal(new Vector3(-1.0f, -2.0f, -3.0f), collision.Vertices[0]);
        Assert.Equal(new Vector3(-1.0f, 5.0f, 6.0f), collision.Vertices[3]);
        Assert.Equal(6, collision.TriangleIndexCount);
        Assert.Equal(2, collision.TriangleCount);
        Assert.Equal([0, 1, 2, 0, 2, 3], collision.TriangleIndices);
        Assert.Equal(2, collision.FacetNormalCount);
        Assert.Equal(new Vector3(0.0f, 1.0f, 0.0f), collision.FacetNormals[1]);
        Assert.Equal(3, collision.MaxTriangleIndex);
        Assert.Equal(new Vector3(-1.0f, -2.0f, -3.0f), collision.BoundsMin);
        Assert.Equal(new Vector3(4.0f, 5.0f, 6.0f), collision.BoundsMax);
    }

    [Fact]
    public void Read_RealAlpha053WispMdx_WithClid_ProducesPayloadSignals()
    {
        if (!File.Exists(MdxTestPaths.Alpha053WispMdxPath))
            return;

        MdxCollisionFile collisionFile = MdxCollisionReader.Read(MdxTestPaths.Alpha053WispMdxPath);

        Assert.True(collisionFile.HasCollision);
        MdxCollisionMesh collision = Assert.IsType<MdxCollisionMesh>(collisionFile.Collision);
        Assert.Equal(8, collision.VertexCount);
        Assert.Equal(36, collision.TriangleIndexCount);
        Assert.Equal(12, collision.TriangleCount);
        Assert.Equal(12, collision.FacetNormalCount);
        Assert.Equal(7, collision.MaxTriangleIndex);
        Assert.Equal(-0.3472222f, collision.BoundsMin!.Value.X, 6);
        Assert.Equal(-0.3472222f, collision.BoundsMin!.Value.Y, 6);
        Assert.Equal(0.0f, collision.BoundsMin!.Value.Z, 6);
        Assert.Equal(0.3472222f, collision.BoundsMax!.Value.X, 6);
        Assert.Equal(0.3472222f, collision.BoundsMax!.Value.Y, 6);
        Assert.Equal(2.083333f, collision.BoundsMax!.Value.Z, 6);
        Assert.Equal([0, 1, 2], collision.TriangleIndices.Take(3).ToArray());
    }

    [Fact]
    public void Read_RealStandardArchiveMdx_WithClid_ProducesPayloadSignals()
    {
        if (!Directory.Exists(MdxTestPaths.Standard060DataPath) || !File.Exists(MdxTestPaths.ListfilePath))
            return;

        using var catalog = new MpqArchiveCatalog();
        _ = ArchiveCatalogBootstrapper.Bootstrap(catalog, [MdxTestPaths.Standard060DataPath], MdxTestPaths.ListfilePath);

        string? virtualPath = MdxCollisionTestPaths.Standard060ClidCandidates.FirstOrDefault(catalog.FileExists);
        if (virtualPath is null)
            return;

        byte[]? bytes = catalog.ReadFile(virtualPath);
        Assert.NotNull(bytes);

        using MemoryStream detectionStream = new(bytes!);
        WowFileDetection detection = WowFileDetector.Detect(detectionStream, virtualPath);
        Assert.Equal(WowFileKind.Mdx, detection.Kind);

        using MemoryStream collisionStream = new(bytes!);
        MdxCollisionFile collisionFile = MdxCollisionReader.Read(collisionStream, virtualPath);

        Assert.True(collisionFile.HasCollision);
        MdxCollisionMesh collision = Assert.IsType<MdxCollisionMesh>(collisionFile.Collision);
        Assert.True(collision.VertexCount > 0);
        Assert.True(collision.TriangleIndexCount > 0);
        Assert.True(collision.TriangleCount > 0);
        Assert.True(collision.FacetNormalCount > 0);
        Assert.True(collision.MaxTriangleIndex >= 0);

        if (string.Equals(virtualPath, MdxCollisionTestPaths.Standard060DwarfFemaleMdxVirtualPath, StringComparison.OrdinalIgnoreCase))
        {
            Assert.Equal(8, collision.VertexCount);
            Assert.Equal(36, collision.TriangleIndexCount);
            Assert.Equal(12, collision.TriangleCount);
            Assert.Equal(12, collision.FacetNormalCount);
            Assert.Equal(7, collision.MaxTriangleIndex);
        }
    }

    private static byte[] CreateMdxBytes(uint version, string modelName, Vector3 boundsMin, Vector3 boundsMax, IReadOnlyList<byte[]> extraChunks)
    {
        List<byte> bytes = [];
        bytes.AddRange(Encoding.ASCII.GetBytes("MDLX"));
        bytes.AddRange(CreateChunk("VERS", CreateUInt32Payload(version)));
        bytes.AddRange(CreateChunk("MODL", CreateModlPayload(modelName, boundsMin, boundsMax)));
        foreach (byte[] chunk in extraChunks)
            bytes.AddRange(chunk);

        return [.. bytes];
    }

    private static byte[] CreateCollisionPayload(IReadOnlyList<Vector3> vertices, IReadOnlyList<ushort> triangleIndices, IReadOnlyList<Vector3> facetNormals)
    {
        List<byte> payload = [];

        WriteTagAndCount(payload, "VRTX", vertices.Count);
        foreach (Vector3 vertex in vertices)
        {
            payload.AddRange(CreateSinglePayload(vertex.X));
            payload.AddRange(CreateSinglePayload(vertex.Y));
            payload.AddRange(CreateSinglePayload(vertex.Z));
        }

        WriteTagAndCount(payload, "TRI ", triangleIndices.Count);
        foreach (ushort triangleIndex in triangleIndices)
        {
            byte[] indexBytes = new byte[2];
            BinaryPrimitives.WriteUInt16LittleEndian(indexBytes, triangleIndex);
            payload.AddRange(indexBytes);
        }

        WriteTagAndCount(payload, "NRMS", facetNormals.Count);
        foreach (Vector3 normal in facetNormals)
        {
            payload.AddRange(CreateSinglePayload(normal.X));
            payload.AddRange(CreateSinglePayload(normal.Y));
            payload.AddRange(CreateSinglePayload(normal.Z));
        }

        return [.. payload];
    }

    private static byte[] CreateModlPayload(string modelName, Vector3 boundsMin, Vector3 boundsMax)
    {
        byte[] payload = new byte[0x6C];
        WriteFixedAscii(payload, 0, 0x50, modelName);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(0x50, 4), boundsMin.X);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(0x54, 4), boundsMin.Y);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(0x58, 4), boundsMin.Z);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(0x5C, 4), boundsMax.X);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(0x60, 4), boundsMax.Y);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(0x64, 4), boundsMax.Z);
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(0x68, 4), 0u);
        return payload;
    }

    private static byte[] CreateChunk(string tag, byte[] payload)
    {
        byte[] bytes = new byte[8 + payload.Length];
        Encoding.ASCII.GetBytes(tag, bytes.AsSpan(0, 4));
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4, 4), (uint)payload.Length);
        payload.CopyTo(bytes.AsSpan(8));
        return bytes;
    }

    private static void WriteTagAndCount(List<byte> payload, string tag, int count)
    {
        payload.AddRange(Encoding.ASCII.GetBytes(tag));
        payload.AddRange(CreateInt32Payload(count));
    }

    private static byte[] CreateUInt32Payload(uint value)
    {
        byte[] bytes = new byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, value);
        return bytes;
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

    private static void WriteFixedAscii(byte[] destination, int offset, int length, string value)
    {
        Encoding.ASCII.GetBytes(value, destination.AsSpan(offset, Math.Min(length, value.Length)));
    }
}

internal static class MdxCollisionTestPaths
{
    public const string Standard060DwarfFemaleMdxVirtualPath = "character/dwarf/female/dwarffemale.mdx";
    public const string Standard060DwarfMaleMdxVirtualPath = "character/dwarf/male/dwarfmale.mdx";

    public static readonly string[] Standard060ClidCandidates =
    [
        Standard060DwarfFemaleMdxVirtualPath,
        Standard060DwarfMaleMdxVirtualPath,
    ];
}