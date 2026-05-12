using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.IO.M2;
using WowViewer.Core.M2;

namespace WowViewer.Core.Tests;

public sealed class MdxToM2ConverterTests
{
    [Fact]
    public void Convert_SyntheticClassicMdx_ProducesReadableStrictMd20AndSkin()
    {
        M2ModelDocument sourceModel = M2ModelReader.Read(
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

        M2GeometryDocument sourceGeometry = new(
            sourceModel,
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

        M2SkinDocument sourceSkin = new(
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

        byte[] mdxBytes = M2ToMdxConverter.Convert(sourceGeometry, sourceSkin);

        MdxToM2ConversionResult result = MdxToM2Converter.Convert(mdxBytes, "Creature\\SyntheticCrate\\SyntheticCrate.mdx");

        using MemoryStream modelStream = new(result.ModelBytes, writable: false);
        M2GeometryDocument convertedGeometry = M2GeometryReader.Read(modelStream, result.ModelPath);

        using MemoryStream skinStream = new(result.SkinBytes, writable: false);
        M2SkinDocument convertedSkin = M2SkinReader.Read(skinStream, result.SkinPath);

        Assert.Equal("Creature\\SyntheticCrate\\SyntheticCrate.m2", result.ModelPath);
        Assert.Equal("Creature\\SyntheticCrate\\SyntheticCrate00.skin", result.SkinPath);
        Assert.Equal("MD20", convertedGeometry.Model.Signature);
        Assert.Equal(0x108u, convertedGeometry.Model.Version);
        Assert.Equal("SyntheticCrate", convertedGeometry.Model.ModelName);
        Assert.Equal(1u, convertedGeometry.Model.ViewCount);
        Assert.Equal(1, convertedGeometry.Model.SequenceCount);
        Assert.Equal(3, convertedGeometry.Vertices.Count);
        Assert.Equal(new Vector3(1f, 0f, 0f), convertedGeometry.Vertices[1].Position);
        Assert.Single(convertedGeometry.Textures);
        Assert.Equal("Textures\\SyntheticCrateMain.blp", convertedGeometry.Textures[0].Filename);
        Assert.Single(convertedGeometry.RenderFlags);
        Assert.Equal((ushort)2, convertedGeometry.RenderFlags[0].RawBlendMode);
        Assert.Equal((ushort)0x10, convertedGeometry.RenderFlags[0].Flags);

        Assert.Equal("SKIN", convertedSkin.Signature);
        Assert.Equal([0, 1, 2], convertedSkin.VertexLookup);
        Assert.Equal([0, 1, 2], convertedSkin.TriangleIndices);
        Assert.Single(convertedSkin.Submeshes);
        Assert.Equal((ushort)3, convertedSkin.Submeshes[0].VertexCount);
        Assert.Equal((ushort)3, convertedSkin.Submeshes[0].IndexCount);
        Assert.Single(convertedSkin.Batches);
        Assert.Equal((ushort)0, convertedSkin.Batches[0].TextureComboIndex);
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