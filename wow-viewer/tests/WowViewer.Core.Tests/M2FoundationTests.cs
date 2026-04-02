using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.IO.M2;
using WowViewer.Core.M2;
using WowViewer.Core.Runtime.M2;

namespace WowViewer.Core.Tests;

public sealed class M2FoundationTests
{
    [Theory]
    [InlineData("Creature\\Wolf\\Wolf.m2", "Creature\\Wolf\\Wolf.m2")]
    [InlineData("Creature\\Wolf\\Wolf.mdx", "Creature\\Wolf\\Wolf.m2")]
    [InlineData("Creature\\Wolf\\Wolf.mdl", "Creature\\Wolf\\Wolf.m2")]
    public void FromPath_ModelIdentityCanonicalizesToM2(string requestedPath, string expectedCanonicalPath)
    {
        M2ModelIdentity identity = M2ModelIdentity.FromPath(requestedPath);

        Assert.Equal(requestedPath, identity.RequestedPath);
        Assert.Equal(expectedCanonicalPath, identity.CanonicalModelPath);
    }

    [Fact]
    public void Read_StrictMd20Model_ProducesExpectedDocument()
    {
        byte[] bytes = CreateMd20Bytes(
            version: 0x108u,
            modelName: "SyntheticRoot",
            boundsMin: new Vector3(-7.0f, -8.0f, -9.0f),
            boundsMax: new Vector3(10.0f, 11.0f, 12.0f),
            boundsRadius: 17.5f,
            embeddedSkinProfileCount: 2,
            embeddedSkinProfileOffset: 0x1C0);

        using MemoryStream stream = new(bytes);
        M2ModelDocument document = M2ModelReader.Read(stream, "Creature\\SyntheticRoot\\SyntheticRoot.mdx");

        Assert.Equal("Creature\\SyntheticRoot\\SyntheticRoot.m2", document.Identity.CanonicalModelPath);
        Assert.Equal("MD20", document.Signature);
        Assert.Equal(0x108u, document.Version);
        Assert.Equal("SyntheticRoot", document.ModelName);
        Assert.Equal(new Vector3(-7.0f, -8.0f, -9.0f), document.BoundsMin);
        Assert.Equal(new Vector3(10.0f, 11.0f, 12.0f), document.BoundsMax);
        Assert.Equal(17.5f, document.BoundsRadius);
        Assert.Equal(2u, document.EmbeddedSkinProfileCount);
        Assert.True(document.HasEmbeddedSkinProfiles);
    }

    [Fact]
    public void Read_Md21Root_ThrowsForStrictMd20Contract()
    {
        byte[] bytes = new byte[0xD0];
        Encoding.ASCII.GetBytes("MD21").CopyTo(bytes, 0);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0x04, 4), 0x108u);

        using MemoryStream stream = new(bytes);
        InvalidDataException ex = Assert.Throws<InvalidDataException>(() => M2ModelReader.Read(stream, "Creature\\Synthetic\\Synthetic.m2"));

        Assert.Contains("strict MD20 root", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Read_SkinDocument_ProducesExpectedTables()
    {
        byte[] bytes = CreateSkinBytes();

        using MemoryStream stream = new(bytes);
        M2SkinDocument document = M2SkinReader.Read(stream, "Creature\\SyntheticRoot\\SyntheticRoot00.skin");

        Assert.Equal("SKIN", document.Signature);
        Assert.Equal(4, document.VertexLookupCount);
        Assert.Equal([10, 11, 12, 13], document.VertexLookup);
        Assert.Equal(6, document.TriangleIndexCount);
        Assert.Equal([0, 1, 2, 2, 3, 0], document.TriangleIndices);
        Assert.Equal(4, document.BoneLookupCount);
        Assert.Equal([5, 6, 7, 8], document.BoneLookup);
        Assert.Equal(1, document.SubmeshCount);
        Assert.Equal((ushort)7, document.Submeshes[0].SkinSectionId);
        Assert.Equal((ushort)6, document.Submeshes[0].IndexCount);
        Assert.Equal(1, document.BatchCount);
        Assert.Equal((byte)0x2, document.Batches[0].Flags);
        Assert.Equal((byte)3, document.Batches[0].PriorityPlane);
        Assert.Equal((ushort)5, document.Batches[0].MaterialIndex);
        Assert.Equal((ushort)9, document.Batches[0].TextureComboIndex);
        Assert.Equal((ushort)2, document.Batches[0].TextureCoordComboIndex);
        Assert.Equal((ushort)4, document.Batches[0].TransparencyComboIndex);
        Assert.Equal((ushort)6, document.Batches[0].TextureAnimationLookupIndex);
        Assert.Equal(12u, document.GlobalVertexOffset);
        Assert.Equal(2u, document.ShadowBatchCount);
        Assert.True(document.HasShadowBatches);
    }

    [Fact]
    public void Runtime_ChooseLoadInitialize_PreservesExactSkinPath()
    {
        M2ModelDocument model = M2ModelReader.Read(new MemoryStream(CreateMd20Bytes(
            version: 0x108u,
            modelName: "SyntheticRoot",
            boundsMin: new Vector3(-1.0f, -1.0f, -1.0f),
            boundsMax: new Vector3(1.0f, 1.0f, 1.0f),
            boundsRadius: 2.5f,
            embeddedSkinProfileCount: 0,
            embeddedSkinProfileOffset: 0)), "Creature\\SyntheticRoot\\SyntheticRoot.m2");
        M2SkinDocument skin = M2SkinReader.Read(new MemoryStream(CreateSkinBytes()), "Creature\\SyntheticRoot\\SyntheticRoot00.skin");

        M2SkinProfileRuntimeState chosen = M2SkinProfileRuntime.Choose(model, 0);
        M2SkinProfileRuntimeState loaded = M2SkinProfileRuntime.Load(chosen, skin);
        M2SkinProfileRuntimeState initialized = M2SkinProfileRuntime.Initialize(loaded);

        Assert.Equal(M2SkinProfileStage.Chosen, chosen.Stage);
        Assert.Equal("Creature\\SyntheticRoot\\SyntheticRoot00.skin", chosen.Selection.CompanionPath);
        Assert.Equal(M2SkinProfileStage.Loaded, loaded.Stage);
        Assert.NotNull(loaded.LoadedSkin);
        Assert.Equal(M2SkinProfileStage.Initialized, initialized.Stage);
        Assert.NotNull(initialized.ActiveSkinProfile);
        Assert.Equal(1, initialized.ActiveSkinProfile!.ActiveSubmeshCount);
        Assert.Equal(1, initialized.ActiveSkinProfile.ActiveSectionCount);
        Assert.Equal(1, initialized.ActiveSkinProfile.SectionsWithBatchesCount);
        Assert.Equal(1, initialized.ActiveSkinProfile.ActiveBatchCount);
        Assert.Equal(0, initialized.ActiveSkinProfile.UnmatchedBatchCount);
        Assert.Equal((ushort)7, initialized.ActiveSkinProfile.ActiveSections[0].SkinSectionId);
        Assert.Equal(1, initialized.ActiveSkinProfile.ActiveSections[0].ActiveBatchCount);
        Assert.Equal(0, initialized.ActiveSkinProfile.ActiveSections[0].Batches[0].BatchIndex);
        Assert.Equal((ushort)5, initialized.ActiveSkinProfile.ActiveSections[0].Batches[0].MaterialIndex);
        Assert.Equal((ushort)6, initialized.ActiveSkinProfile.ActiveSections[0].Batches[0].TextureAnimationLookupIndex);
        Assert.False(initialized.ActiveSkinProfile.UsesCompatibilityFallback);
    }

    [Fact]
    public void Runtime_LoadRejectsNonExactSkinPath()
    {
        M2ModelDocument model = M2ModelReader.Read(new MemoryStream(CreateMd20Bytes(
            version: 0x108u,
            modelName: "SyntheticRoot",
            boundsMin: new Vector3(-1.0f, -1.0f, -1.0f),
            boundsMax: new Vector3(1.0f, 1.0f, 1.0f),
            boundsRadius: 2.5f,
            embeddedSkinProfileCount: 0,
            embeddedSkinProfileOffset: 0)), "Creature\\SyntheticRoot\\SyntheticRoot.m2");
        M2SkinDocument skin = M2SkinReader.Read(new MemoryStream(CreateSkinBytes()), "Creature\\SyntheticRoot\\SyntheticRoot01.skin");

        M2SkinProfileRuntimeState chosen = M2SkinProfileRuntime.Choose(model, 0);
        InvalidDataException ex = Assert.Throws<InvalidDataException>(() => M2SkinProfileRuntime.Load(chosen, skin));

        Assert.Contains("exact selected companion", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void StaticRenderModelBuilder_BuildsSectionGeometryFromActiveSkin()
    {
        M2ModelDocument model = M2ModelReader.Read(new MemoryStream(CreateMd20Bytes(
            version: 0x108u,
            modelName: "SyntheticRuntime",
            boundsMin: new Vector3(-2.0f, -2.0f, -2.0f),
            boundsMax: new Vector3(2.0f, 2.0f, 2.0f),
            boundsRadius: 4.0f,
            embeddedSkinProfileCount: 0,
            embeddedSkinProfileOffset: 0)), "Creature\\SyntheticRuntime\\SyntheticRuntime.m2");

        M2GeometryDocument geometry = new(
            model,
            [
                new M2GeometryVertex(new Vector3(0f, 0f, 0f), Vector3.UnitZ, new Vector2(0f, 0f), Vector2.Zero, Vector4.Zero, Vector4.Zero),
                new M2GeometryVertex(new Vector3(1f, 0f, 0f), Vector3.UnitZ, new Vector2(1f, 0f), Vector2.Zero, Vector4.Zero, Vector4.Zero),
                new M2GeometryVertex(new Vector3(1f, 1f, 0f), Vector3.UnitZ, new Vector2(1f, 1f), Vector2.Zero, Vector4.Zero, Vector4.Zero),
                new M2GeometryVertex(new Vector3(0f, 1f, 0f), Vector3.UnitZ, new Vector2(0f, 1f), Vector2.Zero, Vector4.Zero, Vector4.Zero),
            ],
            [new M2GeometryTexture("Creature\\SyntheticRuntime\\synthetic.blp", 0, 0)],
            [new M2GeometryRenderFlag(flags: 0x4, rawBlendMode: 2)],
            [new M2GeometryTextureLookup(textureId: 0)]);

        M2SkinDocument skin = new(
            sourcePath: "Creature\\SyntheticRuntime\\SyntheticRuntime00.skin",
            signature: "SKIN",
            vertexLookup: [0, 1, 2, 3],
            vertexLookupOffset: 0,
            triangleIndices: [0, 1, 2, 2, 3, 0],
            triangleIndexOffset: 0,
            boneLookup: [],
            boneLookupOffset: 0,
            submeshes: [new M2SkinSubmesh(skinSectionId: 7, level: 0, vertexStart: 0, vertexCount: 4, indexStart: 0, indexCount: 6)],
            submeshOffset: 0,
            batches: [new M2SkinBatch(flags: 0x2, priorityPlane: 3, skinSectionIndex: 0, colorIndex: -1, materialIndex: 0, textureComboIndex: 0, textureCoordComboIndex: 0, transparencyComboIndex: 0, textureAnimationLookupIndex: 0)],
            batchOffset: 0,
            globalVertexOffset: 0,
            shadowBatchCount: 0,
            shadowBatchOffset: 0);

        M2SkinProfileSelection selection = new(0, skin.SourcePath);
        M2SkinProfileRuntimeState chosen = new(model, selection, M2SkinProfileStage.Chosen, loadedSkin: null, activeSkinProfile: null);
        M2SkinProfileRuntimeState loaded = M2SkinProfileRuntime.Load(chosen, skin);
        M2SkinProfileRuntimeState initialized = M2SkinProfileRuntime.Initialize(loaded);

        M2StaticRenderModel runtimeModel = M2StaticRenderModelBuilder.Build(geometry, initialized);

        Assert.Single(runtimeModel.Sections);
        Assert.Equal((ushort)7, runtimeModel.Sections[0].SkinSectionId);
        Assert.Equal(4, runtimeModel.Sections[0].Vertices.Count);
        Assert.Equal(6, runtimeModel.Sections[0].Indices.Count);
        Assert.Equal(M2BlendMode.AlphaBlend, runtimeModel.Sections[0].Material.BlendMode);
        Assert.True(runtimeModel.Sections[0].Material.IsTransparent);
        Assert.True(runtimeModel.Sections[0].Material.IsTwoSided);
        Assert.Equal("Creature\\SyntheticRuntime\\synthetic.blp", runtimeModel.Sections[0].Material.TexturePath);
    }

    private static byte[] CreateMd20Bytes(
        uint version,
        string modelName,
        Vector3 boundsMin,
        Vector3 boundsMax,
        float boundsRadius,
        uint embeddedSkinProfileCount,
        uint embeddedSkinProfileOffset)
    {
        byte[] nameBytes = Encoding.UTF8.GetBytes(modelName + "\0");
        int nameOffset = 0xD0;
        byte[] data = new byte[nameOffset + nameBytes.Length];

        Encoding.ASCII.GetBytes("MD20").CopyTo(data, 0);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x04, 4), version);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x08, 4), (uint)nameBytes.Length);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x0C, 4), (uint)nameOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x4C, 4), embeddedSkinProfileCount);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x50, 4), embeddedSkinProfileOffset);
        WriteVector3(data, 0xB4, boundsMin);
        WriteVector3(data, 0xC0, boundsMax);
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(0xCC, 4), BitConverter.SingleToInt32Bits(boundsRadius));
        nameBytes.CopyTo(data, nameOffset);
        return data;
    }

    private static byte[] CreateSkinBytes()
    {
        ushort[] vertexLookup = [10, 11, 12, 13];
        ushort[] triangleIndices = [0, 1, 2, 2, 3, 0];
        ushort[] boneLookup = [5, 6, 7, 8];

        const int headerSize = 60;
        int vertexLookupOffset = headerSize;
        int triangleIndexOffset = vertexLookupOffset + (vertexLookup.Length * sizeof(ushort));
        int boneLookupOffset = triangleIndexOffset + (triangleIndices.Length * sizeof(ushort));
        int submeshOffset = boneLookupOffset + (boneLookup.Length * sizeof(ushort));
        int batchOffset = submeshOffset + 0x30;
        byte[] data = new byte[batchOffset + 0x18];

        Encoding.ASCII.GetBytes("SKIN").CopyTo(data, 0);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x04, 4), (uint)vertexLookup.Length);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x08, 4), (uint)vertexLookupOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x0C, 4), (uint)triangleIndices.Length);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x10, 4), (uint)triangleIndexOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x14, 4), (uint)boneLookup.Length);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x18, 4), (uint)boneLookupOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x1C, 4), 1u);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x20, 4), (uint)submeshOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x24, 4), 1u);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x28, 4), (uint)batchOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x2C, 4), 12u);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x30, 4), 2u);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x34, 4), 0x400u);

        for (int index = 0; index < vertexLookup.Length; index++)
            BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(vertexLookupOffset + (index * sizeof(ushort)), sizeof(ushort)), vertexLookup[index]);
        for (int index = 0; index < triangleIndices.Length; index++)
            BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(triangleIndexOffset + (index * sizeof(ushort)), sizeof(ushort)), triangleIndices[index]);
        for (int index = 0; index < boneLookup.Length; index++)
            BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(boneLookupOffset + (index * sizeof(ushort)), sizeof(ushort)), boneLookup[index]);

        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(submeshOffset + 0x00, 2), 7);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(submeshOffset + 0x02, 2), 1);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(submeshOffset + 0x04, 2), 0);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(submeshOffset + 0x06, 2), 4);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(submeshOffset + 0x08, 2), 0);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(submeshOffset + 0x0A, 2), 6);

        data[batchOffset + 0x00] = 0x2;
        data[batchOffset + 0x01] = 3;
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(batchOffset + 0x04, 2), 0);
        BinaryPrimitives.WriteInt16LittleEndian(data.AsSpan(batchOffset + 0x08, 2), -1);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(batchOffset + 0x0A, 2), 5);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(batchOffset + 0x10, 2), 9);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(batchOffset + 0x12, 2), 2);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(batchOffset + 0x14, 2), 4);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(batchOffset + 0x16, 2), 6);

        return data;
    }

    private static void WriteVector3(byte[] data, int offset, Vector3 value)
    {
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(offset + 0x00, 4), BitConverter.SingleToInt32Bits(value.X));
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(offset + 0x04, 4), BitConverter.SingleToInt32Bits(value.Y));
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(offset + 0x08, 4), BitConverter.SingleToInt32Bits(value.Z));
    }
}