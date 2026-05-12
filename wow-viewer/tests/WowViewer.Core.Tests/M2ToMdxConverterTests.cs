using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.M2;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.Tests;

public sealed class M2ToMdxConverterTests
{
    [Fact]
    public void Convert_SyntheticGeometryAndSkin_ProducesReadableClassicMdx()
    {
        M2ModelDocument model = M2ModelReader.Read(
            new MemoryStream(CreateMd20Bytes(
                version: 0x108u,
                modelName: "SyntheticCrate",
                boundsMin: new Vector3(-1.0f, -2.0f, -3.0f),
                boundsMax: new Vector3(4.0f, 5.0f, 6.0f),
                boundsRadius: 7.5f,
                embeddedSkinProfileCount: 0,
                embeddedSkinProfileOffset: 0,
                sequences:
                [
                    new SyntheticSequence(
                        AnimationId: 0,
                        VariationIndex: 0,
                        Duration: 1200u,
                        MoveSpeed: 1.25f,
                        Flags: 0u,
                        Frequency: 3,
                        ReplayMinimum: 0u,
                        ReplayMaximum: 1200u,
                        BlendTimeIn: 0,
                        BlendTimeOut: 0,
                        BoundsMin: new Vector3(-1.0f, -2.0f, -3.0f),
                        BoundsMax: new Vector3(4.0f, 5.0f, 6.0f),
                        BoundsRadius: 7.5f,
                        VariationNext: -1,
                        AliasNext: ushort.MaxValue),
                ]),
            writable: false),
            "Creature\\SyntheticCrate\\SyntheticCrate.m2");

        M2GeometryDocument geometry = new(
            model,
            vertices:
            [
                new M2GeometryVertex(new Vector3(0f, 0f, 0f), Vector3.UnitZ, new Vector2(0f, 0f), Vector2.Zero, Vector4.Zero, Vector4.Zero),
                new M2GeometryVertex(new Vector3(1f, 0f, 0f), Vector3.UnitZ, new Vector2(1f, 0f), Vector2.Zero, Vector4.Zero, Vector4.Zero),
                new M2GeometryVertex(new Vector3(0f, 1f, 0f), Vector3.UnitZ, new Vector2(0f, 1f), Vector2.Zero, Vector4.Zero, Vector4.Zero),
            ],
            textures:
            [
                new M2GeometryTexture("Textures\\SyntheticCrateMain.blp", 0, 0),
            ],
            renderFlags:
            [
                new M2GeometryRenderFlag(flags: 0x10, rawBlendMode: 2),
            ],
            textureLookup:
            [
                new M2GeometryTextureLookup(textureId: 0),
            ],
            textureUnitLookup:
            [
                new M2GeometryTextureUnitLookup(0),
            ],
            transparencyLookup: [],
            textureAnimationLookup: [],
            boneLookup: []);

        M2SkinDocument skin = new(
            sourcePath: "Creature\\SyntheticCrate\\SyntheticCrate00.skin",
            signature: "SKIN",
            vertexLookup: [0, 1, 2],
            vertexLookupOffset: 0,
            triangleIndices: [0, 1, 2],
            triangleIndexOffset: 0,
            boneEntries: [],
            boneEntryOffset: 0,
            submeshes: [new M2SkinSubmesh(1, 0, 0, 3, 0, 3)],
            submeshOffset: 0,
            batches: [new M2SkinBatch(0x2, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, ushort.MaxValue)],
            batchOffset: 0,
            globalVertexOffset: 0,
            shadowBatchCount: 0,
            shadowBatchOffset: 0);

        byte[] converted = M2ToMdxConverter.Convert(geometry, skin);

        using MemoryStream summaryStream = new(converted, writable: false);
        MdxSummary summary = MdxSummaryReader.Read(summaryStream, "synthetic_crate.mdx");

        using MemoryStream geometryStream = new(converted, writable: false);
        MdxGeometryFile geometryFile = MdxGeometryReader.Read(geometryStream, "synthetic_crate.mdx");

        Assert.Equal("MDLX", summary.Signature);
        Assert.Equal(1300u, summary.Version);
        Assert.Equal("SyntheticCrate", summary.ModelName);
        Assert.Equal(1, summary.SequenceCount);
        Assert.Equal("Stand", summary.Sequences[0].Name);
        Assert.Equal(1200, summary.Sequences[0].EndTime);
        Assert.Equal(1, summary.TextureCount);
        Assert.Equal("Textures\\SyntheticCrateMain.blp", summary.Textures[0].Path);
        Assert.Equal(1, summary.MaterialCount);
        Assert.Equal(2u, summary.Materials[0].Layers[0].BlendMode);
        Assert.Equal(0x10u, summary.Materials[0].Layers[0].Flags);
        Assert.Equal(0, summary.Materials[0].Layers[0].TextureId);
        Assert.Equal(1, summary.GeosetCount);
        Assert.Equal(3, summary.Geosets[0].VertexCount);
        Assert.Equal(3, summary.Geosets[0].IndexCount);
        Assert.Contains(summary.Chunks, static chunk => chunk.Id.ToString() == "GEOS");

        MdxGeosetGeometry geoset = Assert.Single(geometryFile.Geosets);
        Assert.Equal(3, geoset.VertexCount);
        Assert.Equal(3, geoset.IndexCount);
        Assert.Equal(1, geoset.TriangleCount);
        Assert.Equal(new Vector3(0f, 0f, 0f), geoset.Vertices[0]);
        Assert.Equal(new Vector2(1f, 0f), geoset.PrimaryUvSet[1]);
    }

    private readonly record struct SyntheticSequence(
        ushort AnimationId,
        ushort VariationIndex,
        uint Duration,
        float MoveSpeed,
        uint Flags,
        short Frequency,
        uint ReplayMinimum,
        uint ReplayMaximum,
        ushort BlendTimeIn,
        ushort BlendTimeOut,
        Vector3 BoundsMin,
        Vector3 BoundsMax,
        float BoundsRadius,
        short VariationNext,
        ushort AliasNext);

    private static byte[] CreateMd20Bytes(
        uint version,
        string modelName,
        Vector3 boundsMin,
        Vector3 boundsMax,
        float boundsRadius,
        uint embeddedSkinProfileCount,
        uint embeddedSkinProfileOffset,
        IReadOnlyList<SyntheticSequence>? sequences = null)
    {
        sequences ??= [];

        byte[] nameBytes = Encoding.UTF8.GetBytes(modelName + "\0");
        int nameOffset = 0x120;
        int cursor = nameOffset + nameBytes.Length;
        int sequenceOffset = Align(cursor, 0x10);
        cursor = sequenceOffset + (sequences.Count * 0x40);
        cursor = Math.Max(cursor, 0x120);

        byte[] data = new byte[cursor];
        Encoding.ASCII.GetBytes("MD20").CopyTo(data, 0);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x04, 4), version);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x08, 4), (uint)nameBytes.Length);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x0C, 4), (uint)nameOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x1C, 4), (uint)sequences.Count);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x20, 4), sequences.Count == 0 ? 0u : (uint)sequenceOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x44, 4), embeddedSkinProfileCount);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x48, 4), embeddedSkinProfileOffset);
        WriteVector3(data, 0xA0, boundsMin);
        WriteVector3(data, 0xAC, boundsMax);
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(0xB8, 4), BitConverter.SingleToInt32Bits(boundsRadius));
        nameBytes.CopyTo(data, nameOffset);

        for (int index = 0; index < sequences.Count; index++)
            WriteSequence(data, sequenceOffset + (index * 0x40), sequences[index]);

        return data;
    }

    private static int Align(int value, int alignment)
    {
        int remainder = value % alignment;
        return remainder == 0 ? value : value + (alignment - remainder);
    }

    private static void WriteSequence(byte[] data, int offset, SyntheticSequence sequence)
    {
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(offset + 0x00, 2), sequence.AnimationId);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(offset + 0x02, 2), sequence.VariationIndex);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset + 0x04, 4), sequence.Duration);
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(offset + 0x08, 4), BitConverter.SingleToInt32Bits(sequence.MoveSpeed));
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset + 0x0C, 4), sequence.Flags);
        BinaryPrimitives.WriteInt16LittleEndian(data.AsSpan(offset + 0x10, 2), sequence.Frequency);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset + 0x14, 4), sequence.ReplayMinimum);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset + 0x18, 4), sequence.ReplayMaximum);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(offset + 0x1C, 2), sequence.BlendTimeIn);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(offset + 0x1E, 2), sequence.BlendTimeOut);
        WriteVector3(data, offset + 0x20, sequence.BoundsMin);
        WriteVector3(data, offset + 0x2C, sequence.BoundsMax);
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(offset + 0x38, 4), BitConverter.SingleToInt32Bits(sequence.BoundsRadius));
        BinaryPrimitives.WriteInt16LittleEndian(data.AsSpan(offset + 0x3C, 2), sequence.VariationNext);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(offset + 0x3E, 2), sequence.AliasNext);
    }

    private static void WriteVector3(byte[] data, int offset, Vector3 value)
    {
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(offset + 0x00, 4), BitConverter.SingleToInt32Bits(value.X));
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(offset + 0x04, 4), BitConverter.SingleToInt32Bits(value.Y));
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(offset + 0x08, 4), BitConverter.SingleToInt32Bits(value.Z));
    }
}