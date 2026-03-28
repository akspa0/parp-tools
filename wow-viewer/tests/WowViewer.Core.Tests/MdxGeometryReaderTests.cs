using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.Files;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.Tests;

public sealed class MdxGeometryReaderTests
{
    [Fact]
    public void Read_SyntheticClassicGeos_ProducesExpectedGeosetPayload()
    {
        byte[] bytes = CreateMdxBytes(
            version: 1300,
            modelName: "SyntheticGeosPayload",
            boundsMin: new Vector3(-1.0f, -1.0f, -1.0f),
            boundsMax: new Vector3(1.0f, 1.0f, 1.0f),
            extraChunks:
            [
                CreateChunk("GEOS", CreateClassicGeosPayload(
                [
                    (3, 7, 2u, 0x10u, 5.5f, new Vector3(-1.0f, -2.0f, -3.0f), new Vector3(4.0f, 5.0f, 6.0f), 1),
                ])),
            ]);

        using MemoryStream stream = new(bytes);
        MdxGeometryFile geometry = MdxGeometryReader.Read(stream, "synthetic_geos_payload.mdx");

        Assert.Equal((uint)1300, geometry.Version);
        Assert.Equal("SyntheticGeosPayload", geometry.ModelName);
        Assert.Equal(1, geometry.GeosetCount);

        MdxGeosetGeometry geoset = geometry.Geosets[0];
        Assert.Equal(3, geoset.VertexCount);
        Assert.Equal(new Vector3(0.0f, 0.25f, 0.5f), geoset.Vertices[0]);
        Assert.Equal(new Vector3(2.0f, 2.25f, 2.5f), geoset.Vertices[2]);
        Assert.Equal(3, geoset.NormalCount);
        Assert.Equal(new Vector3(0.0f, 0.0f, 1.0f), geoset.Normals[1]);
        Assert.Equal(1, geoset.UvSetCount);
        Assert.Equal(3, geoset.PrimaryUvCount);
        Assert.Equal(new Vector2(0.1f, 0.05f), geoset.PrimaryUvSet[1]);
        Assert.Equal([4], geoset.PrimitiveTypes);
        Assert.Equal([3], geoset.FaceGroups);
        Assert.Equal([0, 1, 2], geoset.Indices);
        Assert.Equal([0, 1, 2], geoset.VertexGroups);
        Assert.Equal([3u], geoset.MatrixGroups);
        Assert.Equal([0u, 1u, 2u], geoset.MatrixIndices);
        Assert.Equal([0u, 1u, 2u], geoset.BoneIndices);
        Assert.Equal([255u, 255u, 255u], geoset.BoneWeights);
        Assert.Equal(7, geoset.MaterialId);
        Assert.Equal(2u, geoset.SelectionGroup);
        Assert.Equal(0x10u, geoset.Flags);
        Assert.Equal(5.5f, geoset.BoundsRadius);
        Assert.Equal(new Vector3(-1.0f, -2.0f, -3.0f), geoset.BoundsMin);
        Assert.Equal(new Vector3(4.0f, 5.0f, 6.0f), geoset.BoundsMax);
        Assert.Equal(1, geoset.AnimationExtentCount);
    }

    [Fact]
    public void Read_RealStandardArchiveMdx_WithGeos_ProducesPayloadSignals()
    {
        if (!Directory.Exists(MdxGeometryTestPaths.Standard060DataPath) || !File.Exists(MdxGeometryTestPaths.ListfilePath))
            return;

        using var catalog = new MpqArchiveCatalog();
        _ = ArchiveCatalogBootstrapper.Bootstrap(catalog, [MdxGeometryTestPaths.Standard060DataPath], MdxGeometryTestPaths.ListfilePath);

        string? virtualPath = MdxGeometryTestPaths.Standard060GeosCandidates.FirstOrDefault(catalog.FileExists);
        if (virtualPath is null)
            return;

        byte[]? bytes = catalog.ReadFile(virtualPath);
        Assert.NotNull(bytes);

        using MemoryStream detectionStream = new(bytes!);
        WowFileDetection detection = WowFileDetector.Detect(detectionStream, virtualPath);
        Assert.Equal(WowFileKind.Mdx, detection.Kind);

        using MemoryStream geometryStream = new(bytes);
        MdxGeometryFile geometry = MdxGeometryReader.Read(geometryStream, virtualPath);

        Assert.True(geometry.GeosetCount > 0);
        Assert.All(geometry.Geosets, static geoset =>
        {
            Assert.True(geoset.VertexCount > 0);
            Assert.True(geoset.IndexCount > 0);
            Assert.True(geoset.TriangleCount > 0);
            Assert.True(geoset.FaceGroupCount > 0);
            Assert.True(geoset.PrimitiveTypeCount > 0);
            Assert.All(geoset.PrimitiveTypes, static primitiveType => Assert.Equal(4, primitiveType));
            Assert.True(geoset.MaterialId >= 0 || geoset.MaterialId == -1);
        });

        if (string.Equals(virtualPath, MdxGeometryTestPaths.Standard060ChestMdxVirtualPath, StringComparison.OrdinalIgnoreCase))
        {
            Assert.Equal(2, geometry.GeosetCount);
            Assert.Equal(6, geometry.Geosets[0].IndexCount);
            Assert.Equal(102, geometry.Geosets[1].IndexCount);
            Assert.Equal(1, geometry.Geosets[0].MaterialId);
            Assert.Equal(0, geometry.Geosets[1].MaterialId);
        }
    }

    [Fact]
    public void Read_RealAlpha053Mdx_WithGeos_ProducesPayloadSignals()
    {
        string? inputPath = MdxGeometryTestPaths.Alpha053GeosCandidates.FirstOrDefault(File.Exists);
        if (inputPath is null)
            return;

        MdxGeometryFile geometry = MdxGeometryReader.Read(inputPath);

        Assert.True(geometry.GeosetCount > 0);
        Assert.All(geometry.Geosets, static geoset =>
        {
            Assert.True(geoset.VertexCount > 0);
            Assert.True(geoset.IndexCount > 0);
            Assert.True(geoset.MatrixIndexCount > 0);
            Assert.True(geoset.BoneIndexCount > 0);
            Assert.True(geoset.BoneWeightCount > 0);
            Assert.True(geoset.UvSetCount > 0);
        });
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

    private static byte[] CreateClassicGeosPayload(IReadOnlyList<(int VertexCount, int MaterialId, uint SelectionGroup, uint Flags, float BoundsRadius, Vector3 BoundsMin, Vector3 BoundsMax, int AnimationExtentCount)> geosets)
    {
        List<byte> payload = [];
        payload.AddRange(CreateInt32Payload(geosets.Count));

        foreach ((int vertexCount, int materialId, uint selectionGroup, uint flags, float boundsRadius, Vector3 boundsMin, Vector3 boundsMax, int animationExtentCount) in geosets)
        {
            int safeVertexCount = Math.Max(vertexCount, 3);
            int indexCount = safeVertexCount;

            List<byte> geosetPayload = [];
            WriteTagAndCount(geosetPayload, "VRTX", safeVertexCount);
            for (int index = 0; index < safeVertexCount; index++)
            {
                geosetPayload.AddRange(CreateSinglePayload(index));
                geosetPayload.AddRange(CreateSinglePayload(index + 0.25f));
                geosetPayload.AddRange(CreateSinglePayload(index + 0.5f));
            }

            WriteTagAndCount(geosetPayload, "NRMS", safeVertexCount);
            for (int index = 0; index < safeVertexCount; index++)
            {
                geosetPayload.AddRange(CreateSinglePayload(0.0f));
                geosetPayload.AddRange(CreateSinglePayload(0.0f));
                geosetPayload.AddRange(CreateSinglePayload(1.0f));
            }

            WriteTagAndCount(geosetPayload, "UVAS", 1);
            for (int index = 0; index < safeVertexCount; index++)
            {
                geosetPayload.AddRange(CreateSinglePayload(index / 10.0f));
                geosetPayload.AddRange(CreateSinglePayload(index / 20.0f));
            }

            WriteTagAndCount(geosetPayload, "PTYP", 1);
            geosetPayload.Add(4);

            WriteTagAndCount(geosetPayload, "PCNT", 1);
            geosetPayload.AddRange(CreateInt32Payload(indexCount));

            WriteTagAndCount(geosetPayload, "PVTX", indexCount);
            for (ushort index = 0; index < indexCount; index++)
            {
                byte[] indexBytes = new byte[2];
                BinaryPrimitives.WriteUInt16LittleEndian(indexBytes, index);
                geosetPayload.AddRange(indexBytes);
            }

            WriteTagAndCount(geosetPayload, "GNDX", safeVertexCount);
            for (int index = 0; index < safeVertexCount; index++)
                geosetPayload.Add((byte)index);

            WriteTagAndCount(geosetPayload, "MTGC", 1);
            geosetPayload.AddRange(CreateInt32Payload(safeVertexCount));

            WriteTagAndCount(geosetPayload, "MATS", safeVertexCount);
            for (int index = 0; index < safeVertexCount; index++)
                geosetPayload.AddRange(CreateInt32Payload(index));

            WriteTagAndCount(geosetPayload, "BIDX", safeVertexCount);
            for (int index = 0; index < safeVertexCount; index++)
                geosetPayload.AddRange(CreateUInt32Payload((uint)index));

            WriteTagAndCount(geosetPayload, "BWGT", safeVertexCount);
            for (int index = 0; index < safeVertexCount; index++)
                geosetPayload.AddRange(CreateUInt32Payload(255u));

            geosetPayload.AddRange(CreateInt32Payload(materialId));
            geosetPayload.AddRange(CreateInt32Payload(unchecked((int)selectionGroup)));
            geosetPayload.AddRange(CreateInt32Payload(unchecked((int)flags)));
            geosetPayload.AddRange(CreateSinglePayload(boundsRadius));
            geosetPayload.AddRange(CreateSinglePayload(boundsMin.X));
            geosetPayload.AddRange(CreateSinglePayload(boundsMin.Y));
            geosetPayload.AddRange(CreateSinglePayload(boundsMin.Z));
            geosetPayload.AddRange(CreateSinglePayload(boundsMax.X));
            geosetPayload.AddRange(CreateSinglePayload(boundsMax.Y));
            geosetPayload.AddRange(CreateSinglePayload(boundsMax.Z));
            geosetPayload.AddRange(CreateInt32Payload(animationExtentCount));

            for (int extentIndex = 0; extentIndex < animationExtentCount; extentIndex++)
            {
                geosetPayload.AddRange(CreateSinglePayload(boundsRadius + extentIndex));
                geosetPayload.AddRange(CreateSinglePayload(boundsMin.X));
                geosetPayload.AddRange(CreateSinglePayload(boundsMin.Y));
                geosetPayload.AddRange(CreateSinglePayload(boundsMin.Z));
                geosetPayload.AddRange(CreateSinglePayload(boundsMax.X));
                geosetPayload.AddRange(CreateSinglePayload(boundsMax.Y));
                geosetPayload.AddRange(CreateSinglePayload(boundsMax.Z));
            }

            payload.AddRange(CreateSizedPayload(geosetPayload));
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

    private static byte[] CreateSizedPayload(List<byte> payload)
    {
        byte[] bytes = new byte[4 + payload.Count];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0, 4), (uint)(4 + payload.Count));
        payload.CopyTo(bytes, 4);
        return bytes;
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

internal static class MdxGeometryTestPaths
{
    public const string Standard060ChestMdxVirtualPath = "world/generic/activedoodads/chest01/chest01.mdx";
    public const string Standard060AncientOfWarMdxVirtualPath = "Creature/AncientOfWar/AncientofWar.mdx";

    public static readonly string[] Standard060GeosCandidates =
    [
        Standard060AncientOfWarMdxVirtualPath,
        Standard060ChestMdxVirtualPath,
    ];

    public static readonly string[] Alpha053GeosCandidates =
    [
        Path.Combine(GetWowViewerRoot(), "testdata", "0.5.3", "tree", "Creature", "Wisp", "Wisp.mdx"),
        Path.Combine(GetWowViewerRoot(), "testdata", "0.5.3", "tree", "Creature", "WaterElemental", "WaterElemental.mdx"),
        Path.Combine(GetWowViewerRoot(), "testdata", "0.5.3", "tree", "Creature", "Banshee", "Banshee.mdx"),
    ];

    public static string Standard060DataPath => Path.Combine(GetWowViewerRoot(), "testdata", "0.6.0", "World of Warcraft", "Data");

    public static string ListfilePath => Path.Combine(GetWowViewerRoot(), "libs", "wowdev", "wow-listfile", "listfile.txt");

    private static string GetWowViewerRoot()
    {
        string? current = AppContext.BaseDirectory;
        while (!string.IsNullOrEmpty(current))
        {
            if (File.Exists(Path.Combine(current, "WowViewer.slnx")))
                return current;

            current = Directory.GetParent(current)?.FullName;
        }

        throw new DirectoryNotFoundException("Could not locate the wow-viewer repository root from the current test base directory.");
    }
}