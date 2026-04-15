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
    public void BuildAnimationPath_FormatsExpectedCompanionName()
    {
        M2ModelIdentity identity = M2ModelIdentity.FromPath("Creature\\Wolf\\Wolf.mdx");

        string path = identity.BuildAnimationPath(animationId: 5, variationIndex: 2);

        Assert.Equal("Creature\\Wolf\\Wolf0005-02.anim", path);
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
        Assert.Equal(2u, document.ViewCount);
        Assert.Equal(0u, document.EmbeddedSkinProfileCount);
        Assert.False(document.HasEmbeddedSkinProfiles);
        Assert.Empty(document.GlobalLoops);
        Assert.Empty(document.Sequences);
        Assert.Empty(document.SequenceLookup);
        Assert.Empty(document.Colors);
        Assert.Empty(document.TextureWeights);
        Assert.Empty(document.TextureTransforms);
        Assert.Empty(document.Lights);
    }

    [Fact]
    public void Read_ModelWithSequenceBlock_ParsesExternalAnimationMetadata()
    {
        byte[] bytes = CreateMd20Bytes(
            version: 0x109u,
            modelName: "SyntheticAnim",
            boundsMin: new Vector3(-1.0f, -1.0f, -1.0f),
            boundsMax: new Vector3(1.0f, 1.0f, 1.0f),
            boundsRadius: 3.0f,
            embeddedSkinProfileCount: 0,
            embeddedSkinProfileOffset: 0,
            globalLoops: [1234u],
            sequences:
            [
                new SyntheticSequence(AnimationId: 5, VariationIndex: 2, Duration: 1500u, MoveSpeed: 7.5f, Flags: 0u, Frequency: 3, ReplayMinimum: 1u, ReplayMaximum: 2u, BlendTimeIn: 50, BlendTimeOut: 100, BoundsMin: new Vector3(-2.0f, -3.0f, -4.0f), BoundsMax: new Vector3(2.0f, 3.0f, 4.0f), BoundsRadius: 6.5f, VariationNext: -1, AliasNext: ushort.MaxValue),
                new SyntheticSequence(AnimationId: 5, VariationIndex: 3, Duration: 900u, MoveSpeed: 0f, Flags: (uint)M2SequenceFlags.Alias, Frequency: 1, ReplayMinimum: 0u, ReplayMaximum: 0u, BlendTimeIn: 0, BlendTimeOut: 0, BoundsMin: Vector3.Zero, BoundsMax: Vector3.One, BoundsRadius: 1.0f, VariationNext: -1, AliasNext: 0),
            ],
            sequenceLookup: [(short)1, (short)-1]);

        using MemoryStream stream = new(bytes);
        M2ModelDocument document = M2ModelReader.Read(stream, "Creature\\SyntheticAnim\\SyntheticAnim.m2");

        Assert.Single(document.GlobalLoops);
        Assert.Equal((uint)1234, document.GlobalLoops[0]);
        Assert.Equal(2, document.Sequences.Count);
        Assert.Equal((ushort)5, document.Sequences[0].AnimationId);
        Assert.Equal((ushort)2, document.Sequences[0].VariationIndex);
        Assert.True(document.Sequences[0].UsesExternalAnimationFile);
        Assert.False(document.Sequences[0].UsesInlineAnimationData);
        Assert.True(document.Sequences[1].IsAlias);
        Assert.Equal((ushort)0, document.Sequences[1].AliasNext);
        Assert.Equal([(short)1, (short)-1], document.SequenceLookup);
    }

    [Fact]
    public void Read_ModelWithAnimatedBlockTables_ParsesDefinitions()
    {
        byte[] bytes = CreateMd20BytesWithAnimatedTables();

        using MemoryStream stream = new(bytes);
        M2ModelDocument document = M2ModelReader.Read(stream, "Creature\\SyntheticAnimated\\SyntheticAnimated.m2");

        Assert.Single(document.Colors);
        Assert.Single(document.TextureWeights);
        Assert.Single(document.TextureTransforms);
        Assert.Single(document.Lights);

        Assert.Equal(M2TrackInterpolation.Linear, document.Colors[0].ColorTrack.Interpolation);
        Assert.Equal((uint)1, document.Colors[0].ColorTrack.TimestampArray.Count);
        Assert.Equal(M2TrackInterpolation.Linear, document.TextureWeights[0].WeightTrack.Interpolation);
        Assert.Equal((uint)1, document.TextureTransforms[0].RotationTrack.ValueArray.Count);
        Assert.Equal((ushort)1, document.Lights[0].Type);
        Assert.Equal(new Vector3(1.0f, 2.0f, 3.0f), document.Lights[0].Position);
        Assert.Equal(M2TrackInterpolation.None, document.Lights[0].VisibilityTrack.Interpolation);
    }

    [Fact]
    public void Read_Md21Root_ThrowsForStrictMd20Contract()
    {
        byte[] bytes = new byte[0x110];
        Encoding.ASCII.GetBytes("MD21").CopyTo(bytes, 0);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0x04, 4), 0x108u);

        using MemoryStream stream = new(bytes);
        InvalidDataException ex = Assert.Throws<InvalidDataException>(() => M2ModelReader.Read(stream, "Creature\\Synthetic\\Synthetic.m2"));

        Assert.Contains("strict MD20 root", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnimatedRenderStateEvaluator_EvaluatesInlineMaterialAndLightTracks()
    {
        SyntheticAnimatedFixture fixture = CreateAnimatedEvaluationFixture(useExternalPayload: false);

        M2AnimatedRenderState state = M2AnimatedRenderStateEvaluator.Evaluate(fixture.Model, fixture.RenderModel, 0, 500);

        Assert.False(state.UsesExternalPayload);
        Assert.Single(state.Passes);
        Assert.Single(state.Lights);

        M2AnimatedRenderPassState pass = state.Passes[0];
        AssertVectorNear(new Vector3(0.5f, 0.5f, 0.0f), pass.Color, 0.01f);
        Assert.InRange(pass.ColorAlpha, 0.49f, 0.51f);
        Assert.InRange(pass.CombinedAlpha, 0.37f, 0.38f);
        Assert.Single(pass.TextureBindings);
        Assert.InRange(pass.TextureBindings[0].TransparencyAlpha, 0.74f, 0.76f);
        AssertVectorNear(new Vector3(0.5f, 1.0f, 0.0f), pass.TextureBindings[0].Translation, 0.01f);
        AssertVectorNear(new Vector3(1.5f, 1.5f, 1.5f), pass.TextureBindings[0].Scaling, 0.01f);

        M2AnimatedLightState light = state.Lights[0];
        AssertVectorNear(new Vector3(0.5f, 0.5f, 1.0f), light.AmbientColor, 0.01f);
        Assert.InRange(light.AmbientIntensity, 0.74f, 0.76f);
        AssertVectorNear(new Vector3(0.5f, 0.5f, 0.0f), light.DiffuseColor, 0.01f);
        Assert.InRange(light.DiffuseIntensity, 0.59f, 0.61f);
        Assert.InRange(light.AttenuationStart, 4.9f, 5.1f);
        Assert.InRange(light.AttenuationEnd, 14.9f, 15.1f);
        Assert.True(light.Visible);
    }

    [Fact]
    public void AnimatedRenderStateEvaluator_UsesExternalAnimPayloadWhenLoaded()
    {
        SyntheticAnimatedFixture fixture = CreateAnimatedEvaluationFixture(useExternalPayload: true);
        M2ExternalAnimationRuntimeState chosen = M2ExternalAnimationRuntime.Choose(fixture.Model, 0);
        M2ExternalAnimationDocument animation = M2AnimationReader.Read(new MemoryStream(fixture.ExternalPayload, writable: false), "Creature\\SyntheticAnimated\\SyntheticAnimated0007-00.anim");
        M2ExternalAnimationRuntimeState loaded = M2ExternalAnimationRuntime.Load(chosen, animation);

        M2AnimatedRenderState state = M2AnimatedRenderStateEvaluator.Evaluate(fixture.Model, fixture.RenderModel, 0, 250, loaded);

        Assert.True(state.UsesExternalPayload);
        Assert.Single(state.Passes);
        AssertVectorNear(new Vector3(0.75f, 0.25f, 0.0f), state.Passes[0].Color, 0.01f);
        Assert.InRange(state.Passes[0].ColorAlpha, 0.74f, 0.76f);
        Assert.InRange(state.Passes[0].TextureBindings[0].TransparencyAlpha, 0.87f, 0.88f);
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
    public void ExternalAnimationRuntime_ChooseResolvesAliasChainAndExactPath()
    {
        M2ModelDocument model = M2ModelReader.Read(new MemoryStream(CreateMd20Bytes(
            version: 0x109u,
            modelName: "SyntheticAnim",
            boundsMin: new Vector3(-1.0f, -1.0f, -1.0f),
            boundsMax: new Vector3(1.0f, 1.0f, 1.0f),
            boundsRadius: 3.0f,
            embeddedSkinProfileCount: 0,
            embeddedSkinProfileOffset: 0,
            sequences:
            [
                new SyntheticSequence(AnimationId: 12, VariationIndex: 1, Duration: 1000u, MoveSpeed: 0f, Flags: 0u, Frequency: 0, ReplayMinimum: 0u, ReplayMaximum: 0u, BlendTimeIn: 0, BlendTimeOut: 0, BoundsMin: Vector3.Zero, BoundsMax: Vector3.One, BoundsRadius: 1.0f, VariationNext: -1, AliasNext: ushort.MaxValue),
                new SyntheticSequence(AnimationId: 12, VariationIndex: 2, Duration: 1000u, MoveSpeed: 0f, Flags: (uint)M2SequenceFlags.Alias, Frequency: 0, ReplayMinimum: 0u, ReplayMaximum: 0u, BlendTimeIn: 0, BlendTimeOut: 0, BoundsMin: Vector3.Zero, BoundsMax: Vector3.One, BoundsRadius: 1.0f, VariationNext: -1, AliasNext: 0),
            ])), "Creature\\SyntheticAnim\\SyntheticAnim.m2");

        M2ExternalAnimationRuntimeState state = M2ExternalAnimationRuntime.Choose(model, 1);

        Assert.Equal(1, state.RequestedSequenceIndex);
        Assert.Equal(0, state.ResolvedSequenceIndex);
        Assert.Equal([1, 0], state.AliasChain);
        Assert.True(state.UsesExternalFile);
        Assert.Equal("Creature\\SyntheticAnim\\SyntheticAnim0012-01.anim", state.CompanionPath);
    }

    [Fact]
    public void ExternalAnimationRuntime_LoadMarksAliasChainReady()
    {
        M2ModelDocument model = M2ModelReader.Read(new MemoryStream(CreateMd20Bytes(
            version: 0x109u,
            modelName: "SyntheticAnim",
            boundsMin: new Vector3(-1.0f, -1.0f, -1.0f),
            boundsMax: new Vector3(1.0f, 1.0f, 1.0f),
            boundsRadius: 3.0f,
            embeddedSkinProfileCount: 0,
            embeddedSkinProfileOffset: 0,
            sequences:
            [
                new SyntheticSequence(AnimationId: 12, VariationIndex: 1, Duration: 1000u, MoveSpeed: 0f, Flags: 0u, Frequency: 0, ReplayMinimum: 0u, ReplayMaximum: 0u, BlendTimeIn: 0, BlendTimeOut: 0, BoundsMin: Vector3.Zero, BoundsMax: Vector3.One, BoundsRadius: 1.0f, VariationNext: -1, AliasNext: ushort.MaxValue),
                new SyntheticSequence(AnimationId: 12, VariationIndex: 2, Duration: 1000u, MoveSpeed: 0f, Flags: (uint)M2SequenceFlags.Alias, Frequency: 0, ReplayMinimum: 0u, ReplayMaximum: 0u, BlendTimeIn: 0, BlendTimeOut: 0, BoundsMin: Vector3.Zero, BoundsMax: Vector3.One, BoundsRadius: 1.0f, VariationNext: -1, AliasNext: 0),
            ])), "Creature\\SyntheticAnim\\SyntheticAnim.m2");

        M2ExternalAnimationRuntimeState chosen = M2ExternalAnimationRuntime.Choose(model, 1);
        M2ExternalAnimationDocument animation = M2AnimationReader.Read(new MemoryStream([1, 2, 3, 4]), "Creature\\SyntheticAnim\\SyntheticAnim0012-01.anim");

        M2ExternalAnimationRuntimeState loaded = M2ExternalAnimationRuntime.Load(chosen, animation);

        Assert.Equal(M2ExternalAnimationRuntimeStage.Loaded, loaded.Stage);
        Assert.Equal([0, 1], loaded.ReadySequenceIndices);
        Assert.NotNull(loaded.LoadedAnimation);
        Assert.Equal(4, loaded.LoadedAnimation!.PayloadSizeBytes);
    }

    [Fact]
    public void AnimationReader_ReadsAfm2ChunkPayload()
    {
        byte[] bytes = new byte[12];
        Encoding.ASCII.GetBytes("AFM2").CopyTo(bytes, 0);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4, 4), 4u);
        bytes[8] = 9;
        bytes[9] = 8;
        bytes[10] = 7;
        bytes[11] = 6;

        using MemoryStream stream = new(bytes);
        M2ExternalAnimationDocument document = M2AnimationReader.Read(stream, "Creature\\SyntheticAnim\\SyntheticAnim0012-01.anim");

        Assert.True(document.IsChunkedContainer);
        Assert.Equal("AFM2", document.ContainerSignature);
        Assert.Equal([9, 8, 7, 6], document.Payload);
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
            [new M2GeometryTextureLookup(textureId: 0)],
            [],
            [],
            [],
            []);

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
            batches: [new M2SkinBatch(flags: 0x2, priorityPlane: 3, shaderId: 4, skinSectionIndex: 0, geosetIndex: 5, colorIndex: -1, renderFlagsIndex: 0, materialLayer: 0, textureCount: 1, textureComboIndex: 0, textureCoordComboIndex: 0, transparencyComboIndex: 0, textureAnimationLookupIndex: 0)],
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
        Assert.Single(runtimeModel.StructuredSections);
        Assert.Equal((ushort)7, runtimeModel.Sections[0].SkinSectionId);
        Assert.Equal(4, runtimeModel.Sections[0].Vertices.Count);
        Assert.Equal(6, runtimeModel.Sections[0].Indices.Count);
        Assert.Equal(M2BlendMode.AlphaBlend, runtimeModel.Sections[0].Material.BlendMode);
        Assert.True(runtimeModel.Sections[0].Material.IsTransparent);
        Assert.True(runtimeModel.Sections[0].Material.IsTwoSided);
        Assert.Equal("Creature\\SyntheticRuntime\\synthetic.blp", runtimeModel.Sections[0].Material.TexturePath);
        Assert.Single(runtimeModel.Sections[0].Material.TextureBindings);
        Assert.Equal((ushort)0, runtimeModel.Sections[0].Material.TextureBindings[0].TextureId);
        Assert.Single(runtimeModel.StructuredSections[0].Passes);
        Assert.Equal(0, runtimeModel.StructuredSections[0].Passes[0].Material.MaterialLayer);
        Assert.Equal((ushort)4, runtimeModel.StructuredSections[0].Passes[0].Material.ShaderId);
        Assert.Equal("Diffuse_T1:Combiners_Decal", runtimeModel.StructuredSections[0].Passes[0].Material.EffectRecipe.RecipeKey);
        Assert.True(runtimeModel.StructuredSections[0].Passes[0].Material.EffectRecipe.IsAnimated);
        Assert.True(runtimeModel.StructuredSections[0].Passes[0].Material.EffectRecipe.UsesTextureTransformAnimation);
    }

    [Fact]
    public void StaticRenderModelBuilder_PreservesStructuredMultiPassMaterialRouting()
    {
        M2ModelDocument model = M2ModelReader.Read(new MemoryStream(CreateMd20Bytes(
            version: 0x108u,
            modelName: "SyntheticRouting",
            boundsMin: new Vector3(-2.0f, -2.0f, -2.0f),
            boundsMax: new Vector3(2.0f, 2.0f, 2.0f),
            boundsRadius: 4.0f,
            embeddedSkinProfileCount: 0,
            embeddedSkinProfileOffset: 0)), "Creature\\SyntheticRouting\\SyntheticRouting.m2");

        M2GeometryDocument geometry = new(
            model,
            [
                new M2GeometryVertex(new Vector3(0f, 0f, 0f), Vector3.UnitZ, new Vector2(0f, 0f), Vector2.Zero, Vector4.Zero, Vector4.Zero),
                new M2GeometryVertex(new Vector3(1f, 0f, 0f), Vector3.UnitZ, new Vector2(1f, 0f), Vector2.Zero, Vector4.Zero, Vector4.Zero),
                new M2GeometryVertex(new Vector3(1f, 1f, 0f), Vector3.UnitZ, new Vector2(1f, 1f), Vector2.Zero, Vector4.Zero, Vector4.Zero),
                new M2GeometryVertex(new Vector3(0f, 1f, 0f), Vector3.UnitZ, new Vector2(0f, 1f), Vector2.Zero, Vector4.Zero, Vector4.Zero),
            ],
            [
                new M2GeometryTexture("Creature\\SyntheticRouting\\base.blp", 0, 0),
                new M2GeometryTexture("Creature\\SyntheticRouting\\overlay.blp", 0, 1),
            ],
            [
                new M2GeometryRenderFlag(flags: 0x4, rawBlendMode: 2),
                new M2GeometryRenderFlag(flags: 0x1, rawBlendMode: 4),
            ],
            [
                new M2GeometryTextureLookup(textureId: 0),
                new M2GeometryTextureLookup(textureId: 1),
                new M2GeometryTextureLookup(textureId: 1),
            ],
            [
                new M2GeometryTextureUnitLookup(3),
                new M2GeometryTextureUnitLookup(5),
                new M2GeometryTextureUnitLookup(7),
            ],
            [
                new M2GeometryTransparencyLookup(11),
                new M2GeometryTransparencyLookup(13),
                new M2GeometryTransparencyLookup(17),
            ],
            [
                new M2GeometryTextureAnimationLookup(19),
                new M2GeometryTextureAnimationLookup(23),
                new M2GeometryTextureAnimationLookup(29),
            ],
            []);

        M2SkinDocument skin = new(
            sourcePath: "Creature\\SyntheticRouting\\SyntheticRouting00.skin",
            signature: "SKIN",
            vertexLookup: [0, 1, 2, 3],
            vertexLookupOffset: 0,
            triangleIndices: [0, 1, 2, 2, 3, 0],
            triangleIndexOffset: 0,
            boneLookup: [],
            boneLookupOffset: 0,
            submeshes: [new M2SkinSubmesh(skinSectionId: 9, level: 0, vertexStart: 0, vertexCount: 4, indexStart: 0, indexCount: 6)],
            submeshOffset: 0,
            batches:
            [
                new M2SkinBatch(flags: 0x2, priorityPlane: 3, shaderId: 4, skinSectionIndex: 0, geosetIndex: 8, colorIndex: -1, renderFlagsIndex: 0, materialLayer: 0, textureCount: 2, textureComboIndex: 0, textureCoordComboIndex: 0, transparencyComboIndex: 0, textureAnimationLookupIndex: 0),
                new M2SkinBatch(flags: 0x8, priorityPlane: 1, shaderId: 6, skinSectionIndex: 0, geosetIndex: 9, colorIndex: 2, renderFlagsIndex: 1, materialLayer: 1, textureCount: 1, textureComboIndex: 2, textureCoordComboIndex: 2, transparencyComboIndex: 2, textureAnimationLookupIndex: 2),
            ],
            batchOffset: 0,
            globalVertexOffset: 0,
            shadowBatchCount: 0,
            shadowBatchOffset: 0);

        M2SkinProfileSelection selection = new(0, skin.SourcePath);
        M2SkinProfileRuntimeState chosen = new(model, selection, M2SkinProfileStage.Chosen, loadedSkin: null, activeSkinProfile: null);
        M2SkinProfileRuntimeState loaded = M2SkinProfileRuntime.Load(chosen, skin);
        M2SkinProfileRuntimeState initialized = M2SkinProfileRuntime.Initialize(loaded);

        M2StaticRenderModel runtimeModel = M2StaticRenderModelBuilder.Build(geometry, initialized);

        Assert.Equal(2, runtimeModel.Sections.Count);
        Assert.Single(runtimeModel.StructuredSections);

        M2StructuredRenderSection structuredSection = runtimeModel.StructuredSections[0];
        Assert.Equal((ushort)9, structuredSection.SkinSectionId);
        Assert.Equal(2, structuredSection.PassCount);
        Assert.Equal(4, structuredSection.Vertices.Count);
        Assert.Equal(6, structuredSection.Indices.Count);

        M2StaticRenderMaterial firstPass = structuredSection.Passes[0].Material;
        Assert.Equal(0, firstPass.MaterialLayer);
        Assert.Equal((ushort)2, firstPass.TextureCount);
        Assert.Equal(M2BlendMode.AlphaBlend, firstPass.BlendMode);
        Assert.Equal(2, firstPass.TextureBindings.Count);
        Assert.Equal("Creature\\SyntheticRouting\\base.blp", firstPass.TextureBindings[0].TexturePath);
        Assert.Equal("Creature\\SyntheticRouting\\overlay.blp", firstPass.TextureBindings[1].TexturePath);
        Assert.Equal((ushort)3, firstPass.TextureBindings[0].TextureCoordLookupValue);
        Assert.Equal((ushort)5, firstPass.TextureBindings[1].TextureCoordLookupValue);
        Assert.Equal((ushort)11, firstPass.TextureBindings[0].TransparencyLookupValue);
        Assert.Equal((ushort)23, firstPass.TextureBindings[1].TextureAnimationLookupValue);
        Assert.Equal("Diffuse_T1_T2:Combiners_Decal", firstPass.EffectRecipe.RecipeKey);
        Assert.True(firstPass.EffectRecipe.UsesTransparencyAnimation);
        Assert.True(firstPass.EffectRecipe.UsesTextureTransformAnimation);
        Assert.False(firstPass.EffectRecipe.UsesColorAnimation);
        Assert.True(firstPass.EffectRecipe.IsAnimated);

        M2StaticRenderMaterial secondPass = structuredSection.Passes[1].Material;
        Assert.Equal(1, secondPass.MaterialLayer);
        Assert.Equal((ushort)1, secondPass.TextureCount);
        Assert.Equal(M2BlendMode.Add, secondPass.BlendMode);
        Assert.True(secondPass.IsTransparent);
        Assert.True(secondPass.IsUnshaded);
        Assert.Single(secondPass.TextureBindings);
        Assert.Equal("Creature\\SyntheticRouting\\overlay.blp", secondPass.TextureBindings[0].TexturePath);
        Assert.Equal((ushort)7, secondPass.TextureBindings[0].TextureCoordLookupValue);
        Assert.Equal("Diffuse_T1:Combiners_Add", secondPass.EffectRecipe.RecipeKey);
        Assert.True(secondPass.EffectRecipe.UsesColorAnimation);
        Assert.True(secondPass.EffectRecipe.UsesTransparencyAnimation);
        Assert.True(secondPass.EffectRecipe.UsesTextureTransformAnimation);
    }

    [Fact]
    public void StaticRenderModelBuilder_MarksProjectedRecipesFromBatchFlags()
    {
        M2ModelDocument model = M2ModelReader.Read(new MemoryStream(CreateMd20Bytes(
            version: 0x108u,
            modelName: "SyntheticProjected",
            boundsMin: new Vector3(-2.0f, -2.0f, -2.0f),
            boundsMax: new Vector3(2.0f, 2.0f, 2.0f),
            boundsRadius: 4.0f,
            embeddedSkinProfileCount: 0,
            embeddedSkinProfileOffset: 0)), "Creature\\SyntheticProjected\\SyntheticProjected.m2");

        M2GeometryDocument geometry = new(
            model,
            [new M2GeometryVertex(Vector3.Zero, Vector3.UnitZ, Vector2.Zero, Vector2.Zero, Vector4.Zero, Vector4.Zero)],
            [new M2GeometryTexture("Creature\\SyntheticProjected\\projected.blp", 0, 0)],
            [new M2GeometryRenderFlag(flags: 0x0, rawBlendMode: 0)],
            [new M2GeometryTextureLookup(textureId: 0)],
            [],
            [],
            [],
            []);

        M2SkinDocument skin = new(
            sourcePath: "Creature\\SyntheticProjected\\SyntheticProjected00.skin",
            signature: "SKIN",
            vertexLookup: [0],
            vertexLookupOffset: 0,
            triangleIndices: [0, 0, 0],
            triangleIndexOffset: 0,
            boneLookup: [],
            boneLookupOffset: 0,
            submeshes: [new M2SkinSubmesh(skinSectionId: 1, level: 0, vertexStart: 0, vertexCount: 1, indexStart: 0, indexCount: 3)],
            submeshOffset: 0,
            batches: [new M2SkinBatch(flags: 0x4, priorityPlane: 0, shaderId: 0, skinSectionIndex: 0, geosetIndex: 0, colorIndex: -1, renderFlagsIndex: 0, materialLayer: 0, textureCount: 1, textureComboIndex: 0, textureCoordComboIndex: 0, transparencyComboIndex: ushort.MaxValue, textureAnimationLookupIndex: ushort.MaxValue)],
            batchOffset: 0,
            globalVertexOffset: 0,
            shadowBatchCount: 0,
            shadowBatchOffset: 0);

        M2SkinProfileRuntimeState initialized = M2SkinProfileRuntime.Initialize(M2SkinProfileRuntime.Load(M2SkinProfileRuntime.Choose(model, 0), skin));
        M2StaticRenderModel runtimeModel = M2StaticRenderModelBuilder.Build(geometry, initialized);

        M2StaticRenderMaterial material = runtimeModel.StructuredSections[0].Passes[0].Material;
        Assert.True(material.EffectRecipe.IsProjected);
        Assert.Equal("Diffuse_Projected:Combiners_Opaque", material.EffectRecipe.RecipeKey);
    }

    private static byte[] CreateMd20Bytes(
        uint version,
        string modelName,
        Vector3 boundsMin,
        Vector3 boundsMax,
        float boundsRadius,
        uint embeddedSkinProfileCount,
        uint embeddedSkinProfileOffset,
        IReadOnlyList<uint>? globalLoops = null,
        IReadOnlyList<SyntheticSequence>? sequences = null,
        IReadOnlyList<short>? sequenceLookup = null)
    {
        globalLoops ??= [];
        sequences ??= [];
        sequenceLookup ??= [];

        byte[] nameBytes = Encoding.UTF8.GetBytes(modelName + "\0");
        int nameOffset = 0x110;
        int cursor = nameOffset + nameBytes.Length;
        int globalLoopOffset = Align(cursor, sizeof(uint));
        cursor = globalLoopOffset + (globalLoops.Count * sizeof(uint));
        int sequenceOffset = Align(cursor, 0x10);
        cursor = sequenceOffset + (sequences.Count * 0x40);
        int sequenceLookupOffset = Align(cursor, sizeof(short));
        cursor = sequenceLookupOffset + (sequenceLookup.Count * sizeof(short));

        cursor = Math.Max(cursor, 0x110);

        byte[] data = new byte[cursor];

        Encoding.ASCII.GetBytes("MD20").CopyTo(data, 0);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x04, 4), version);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x08, 4), (uint)nameBytes.Length);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x0C, 4), (uint)nameOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x14, 4), (uint)globalLoops.Count);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x18, 4), globalLoops.Count == 0 ? 0u : (uint)globalLoopOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x1C, 4), (uint)sequences.Count);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x20, 4), sequences.Count == 0 ? 0u : (uint)sequenceOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x24, 4), (uint)sequenceLookup.Count);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x28, 4), sequenceLookup.Count == 0 ? 0u : (uint)sequenceLookupOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x44, 4), embeddedSkinProfileCount);
        WriteVector3(data, 0xA0, boundsMin);
        WriteVector3(data, 0xAC, boundsMax);
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(0xB8, 4), BitConverter.SingleToInt32Bits(boundsRadius));
        nameBytes.CopyTo(data, nameOffset);

        for (int index = 0; index < globalLoops.Count; index++)
            BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(globalLoopOffset + (index * sizeof(uint)), sizeof(uint)), globalLoops[index]);

        for (int index = 0; index < sequences.Count; index++)
            WriteSequence(data, sequenceOffset + (index * 0x40), sequences[index]);

        for (int index = 0; index < sequenceLookup.Count; index++)
            BinaryPrimitives.WriteInt16LittleEndian(data.AsSpan(sequenceLookupOffset + (index * sizeof(short)), sizeof(short)), sequenceLookup[index]);

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

    private static byte[] CreateMd20BytesWithAnimatedTables()
    {
        byte[] data = CreateMd20Bytes(
            version: 0x108u,
            modelName: "SyntheticAnimated",
            boundsMin: new Vector3(-1.0f, -1.0f, -1.0f),
            boundsMax: new Vector3(1.0f, 1.0f, 1.0f),
            boundsRadius: 2.0f,
            embeddedSkinProfileCount: 1,
            embeddedSkinProfileOffset: 0,
            sequences:
            [
                new SyntheticSequence(AnimationId: 7, VariationIndex: 0, Duration: 1000u, MoveSpeed: 0f, Flags: (uint)M2SequenceFlags.StoredInline, Frequency: 0, ReplayMinimum: 0u, ReplayMaximum: 0u, BlendTimeIn: 0, BlendTimeOut: 0, BoundsMin: Vector3.Zero, BoundsMax: Vector3.One, BoundsRadius: 1.0f, VariationNext: -1, AliasNext: ushort.MaxValue),
            ],
            sequenceLookup: [(short)0]);

        int colorOffset = Align(data.Length, 4);
        int textureWeightOffset = colorOffset + 0x28;
        int textureTransformOffset = textureWeightOffset + 0x14;
        int lightOffset = textureTransformOffset + 0x3C;
        Array.Resize(ref data, lightOffset + 0x9C);

        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x48, 4), 1u);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x4C, 4), (uint)colorOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x58, 4), 1u);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x5C, 4), (uint)textureWeightOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x60, 4), 1u);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x64, 4), (uint)textureTransformOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x108, 4), 1u);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x10C, 4), (uint)lightOffset);

        WriteTrackHeader(data, colorOffset + 0x00, interpolation: 1, globalSequence: ushort.MaxValue, timestampArrayCount: 1, timestampArrayOffset: 0x200u, valueArrayCount: 1, valueArrayOffset: 0x240u);
        WriteTrackHeader(data, colorOffset + 0x14, interpolation: 0, globalSequence: ushort.MaxValue, timestampArrayCount: 1, timestampArrayOffset: 0x280u, valueArrayCount: 1, valueArrayOffset: 0x2C0u);
        WriteTrackHeader(data, textureWeightOffset + 0x00, interpolation: ushort.MaxValue, globalSequence: 1, timestampArrayCount: 1, timestampArrayOffset: 0x300u, valueArrayCount: 1, valueArrayOffset: 0x340u);
        WriteTrackHeader(data, textureTransformOffset + 0x00, interpolation: 1, globalSequence: ushort.MaxValue, timestampArrayCount: 1, timestampArrayOffset: 0x380u, valueArrayCount: 1, valueArrayOffset: 0x3C0u);
        WriteTrackHeader(data, textureTransformOffset + 0x14, interpolation: 1, globalSequence: ushort.MaxValue, timestampArrayCount: 1, timestampArrayOffset: 0x400u, valueArrayCount: 1, valueArrayOffset: 0x440u);
        WriteTrackHeader(data, textureTransformOffset + 0x28, interpolation: 1, globalSequence: ushort.MaxValue, timestampArrayCount: 1, timestampArrayOffset: 0x480u, valueArrayCount: 1, valueArrayOffset: 0x4C0u);

        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(lightOffset + 0x00, 2), 1);
        BinaryPrimitives.WriteInt16LittleEndian(data.AsSpan(lightOffset + 0x02, 2), -1);
        WriteVector3(data, lightOffset + 0x04, new Vector3(1.0f, 2.0f, 3.0f));
        WriteTrackHeader(data, lightOffset + 0x10, interpolation: 1, globalSequence: ushort.MaxValue, timestampArrayCount: 1, timestampArrayOffset: 0x500u, valueArrayCount: 1, valueArrayOffset: 0x540u);
        WriteTrackHeader(data, lightOffset + 0x24, interpolation: 1, globalSequence: ushort.MaxValue, timestampArrayCount: 1, timestampArrayOffset: 0x580u, valueArrayCount: 1, valueArrayOffset: 0x5C0u);
        WriteTrackHeader(data, lightOffset + 0x38, interpolation: 1, globalSequence: ushort.MaxValue, timestampArrayCount: 1, timestampArrayOffset: 0x600u, valueArrayCount: 1, valueArrayOffset: 0x640u);
        WriteTrackHeader(data, lightOffset + 0x4C, interpolation: 1, globalSequence: ushort.MaxValue, timestampArrayCount: 1, timestampArrayOffset: 0x680u, valueArrayCount: 1, valueArrayOffset: 0x6C0u);
        WriteTrackHeader(data, lightOffset + 0x60, interpolation: 1, globalSequence: ushort.MaxValue, timestampArrayCount: 1, timestampArrayOffset: 0x700u, valueArrayCount: 1, valueArrayOffset: 0x740u);
        WriteTrackHeader(data, lightOffset + 0x74, interpolation: 1, globalSequence: ushort.MaxValue, timestampArrayCount: 1, timestampArrayOffset: 0x780u, valueArrayCount: 1, valueArrayOffset: 0x7C0u);
        WriteTrackHeader(data, lightOffset + 0x88, interpolation: 0, globalSequence: ushort.MaxValue, timestampArrayCount: 1, timestampArrayOffset: 0x800u, valueArrayCount: 1, valueArrayOffset: 0x840u);

        return data;
    }

    private static void WriteTrackHeader(byte[] data, int offset, ushort interpolation, ushort globalSequence, uint timestampArrayCount, uint timestampArrayOffset, uint valueArrayCount, uint valueArrayOffset)
    {
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(offset + 0x00, 2), interpolation);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(offset + 0x02, 2), globalSequence);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset + 0x04, 4), timestampArrayCount);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset + 0x08, 4), timestampArrayOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset + 0x0C, 4), valueArrayCount);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset + 0x10, 4), valueArrayOffset);
    }

    private static SyntheticAnimatedFixture CreateAnimatedEvaluationFixture(bool useExternalPayload)
    {
        SyntheticTrackPayloadBuilder payloadBuilder = new();
        M2ColorDefinition color = new(
            0,
            payloadBuilder.AddTrack(M2TrackInterpolation.Linear, -1, [0u, 1000u], [new Vector3(1.0f, 0.0f, 0.0f), new Vector3(0.0f, 1.0f, 0.0f)]),
            payloadBuilder.AddTrack(M2TrackInterpolation.Linear, -1, [0u, 1000u], [(short)0x7FFF, (short)0]));
        M2TextureWeightDefinition textureWeight = new(
            0,
            payloadBuilder.AddTrack(M2TrackInterpolation.Linear, -1, [0u, 1000u], [(short)0x7FFF, (short)0x4000]));
        Quaternion quarterTurn = Quaternion.CreateFromAxisAngle(Vector3.UnitZ, MathF.PI / 2.0f);
        M2TextureTransformDefinition textureTransform = new(
            0,
            payloadBuilder.AddTrack(M2TrackInterpolation.Linear, -1, [0u, 1000u], [Vector3.Zero, new Vector3(1.0f, 2.0f, 0.0f)]),
            payloadBuilder.AddTrack(M2TrackInterpolation.Linear, -1, [0u, 1000u], [Quaternion.Identity, quarterTurn]),
            payloadBuilder.AddTrack(M2TrackInterpolation.Linear, -1, [0u, 1000u], [Vector3.One, new Vector3(2.0f, 2.0f, 2.0f)]));
        M2LightDefinition light = new(
            0,
            type: 1,
            boneIndex: -1,
            position: new Vector3(1.0f, 2.0f, 3.0f),
            ambientColorTrack: payloadBuilder.AddTrack(M2TrackInterpolation.Linear, -1, [0u, 1000u], [new Vector3(0.0f, 0.0f, 1.0f), Vector3.One]),
            ambientIntensityTrack: payloadBuilder.AddTrack(M2TrackInterpolation.Linear, -1, [0u, 1000u], [1.0f, 0.5f]),
            diffuseColorTrack: payloadBuilder.AddTrack(M2TrackInterpolation.Linear, -1, [0u, 1000u], [new Vector3(1.0f, 0.0f, 0.0f), new Vector3(0.0f, 1.0f, 0.0f)]),
            diffuseIntensityTrack: payloadBuilder.AddTrack(M2TrackInterpolation.Linear, -1, [0u, 1000u], [0.2f, 1.0f]),
            attenuationStartTrack: payloadBuilder.AddTrack(M2TrackInterpolation.Linear, -1, [0u, 1000u], [0.0f, 10.0f]),
            attenuationEndTrack: payloadBuilder.AddTrack(M2TrackInterpolation.Linear, -1, [0u, 1000u], [10.0f, 20.0f]),
            visibilityTrack: payloadBuilder.AddTrack(M2TrackInterpolation.None, -1, [0u], [(byte)1]));

        byte[] animatedPayload = payloadBuilder.ToArray();
        byte[] rootPayload = useExternalPayload ? [0] : animatedPayload;
        byte[] externalPayload = useExternalPayload ? animatedPayload : [];

        M2ModelDocument model = new(
            M2ModelIdentity.FromPath("Creature\\SyntheticAnimated\\SyntheticAnimated.m2"),
            rootPayload,
            "MD20",
            0x108u,
            0u,
            1u,
            "SyntheticAnimated",
            [],
            [new M2SequenceDefinition(0, animationId: 7, variationIndex: 0, duration: 1000u, moveSpeed: 0f, flags: useExternalPayload ? 0u : (uint)M2SequenceFlags.StoredInline, frequency: 0, replayMinimum: 0u, replayMaximum: 0u, blendTimeIn: 0, blendTimeOut: 0, boundsMin: Vector3.Zero, boundsMax: Vector3.One, boundsRadius: 1.0f, variationNext: -1, aliasNext: ushort.MaxValue)],
            [(short)0],
            [color],
            [textureWeight],
            [textureTransform],
            [light],
            new Vector3(-1.0f, -1.0f, -1.0f),
            new Vector3(1.0f, 1.0f, 1.0f),
            2.0f,
            0u,
            0u);

        M2GeometryDocument geometry = new(
            model,
            [
                new M2GeometryVertex(new Vector3(0f, 0f, 0f), Vector3.UnitZ, new Vector2(0f, 0f), Vector2.Zero, Vector4.Zero, Vector4.Zero),
                new M2GeometryVertex(new Vector3(1f, 0f, 0f), Vector3.UnitZ, new Vector2(1f, 0f), Vector2.Zero, Vector4.Zero, Vector4.Zero),
                new M2GeometryVertex(new Vector3(1f, 1f, 0f), Vector3.UnitZ, new Vector2(1f, 1f), Vector2.Zero, Vector4.Zero, Vector4.Zero),
                new M2GeometryVertex(new Vector3(0f, 1f, 0f), Vector3.UnitZ, new Vector2(0f, 1f), Vector2.Zero, Vector4.Zero, Vector4.Zero),
            ],
            [new M2GeometryTexture("Creature\\SyntheticAnimated\\animated.blp", 0, 0)],
            [new M2GeometryRenderFlag(flags: 0x4, rawBlendMode: 2)],
            [new M2GeometryTextureLookup(textureId: 0)],
            [new M2GeometryTextureUnitLookup(0)],
            [new M2GeometryTransparencyLookup(0)],
            [new M2GeometryTextureAnimationLookup(0)],
            []);

        M2SkinDocument skin = new(
            sourcePath: "Creature\\SyntheticAnimated\\SyntheticAnimated00.skin",
            signature: "SKIN",
            vertexLookup: [0, 1, 2, 3],
            vertexLookupOffset: 0,
            triangleIndices: [0, 1, 2, 2, 3, 0],
            triangleIndexOffset: 0,
            boneLookup: [],
            boneLookupOffset: 0,
            submeshes: [new M2SkinSubmesh(skinSectionId: 1, level: 0, vertexStart: 0, vertexCount: 4, indexStart: 0, indexCount: 6)],
            submeshOffset: 0,
            batches: [new M2SkinBatch(flags: 0x2, priorityPlane: 0, shaderId: 0, skinSectionIndex: 0, geosetIndex: 0, colorIndex: 0, renderFlagsIndex: 0, materialLayer: 0, textureCount: 1, textureComboIndex: 0, textureCoordComboIndex: 0, transparencyComboIndex: 0, textureAnimationLookupIndex: 0)],
            batchOffset: 0,
            globalVertexOffset: 0,
            shadowBatchCount: 0,
            shadowBatchOffset: 0);

        M2StaticRenderModel renderModel = M2StaticRenderModelBuilder.Build(geometry, M2SkinProfileRuntime.Initialize(M2SkinProfileRuntime.Load(M2SkinProfileRuntime.Choose(model, 0), skin)));
        return new SyntheticAnimatedFixture(model, renderModel, externalPayload);
    }

    private static void AssertVectorNear(Vector3 expected, Vector3 actual, float tolerance)
    {
        Assert.InRange(actual.X, expected.X - tolerance, expected.X + tolerance);
        Assert.InRange(actual.Y, expected.Y - tolerance, expected.Y + tolerance);
        Assert.InRange(actual.Z, expected.Z - tolerance, expected.Z + tolerance);
    }

    private static void WriteVector3(byte[] data, int offset, Vector3 value)
    {
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(offset + 0x00, 4), BitConverter.SingleToInt32Bits(value.X));
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(offset + 0x04, 4), BitConverter.SingleToInt32Bits(value.Y));
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(offset + 0x08, 4), BitConverter.SingleToInt32Bits(value.Z));
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

    private readonly record struct SyntheticAnimatedFixture(M2ModelDocument Model, M2StaticRenderModel RenderModel, byte[] ExternalPayload);

    private sealed class SyntheticTrackPayloadBuilder
    {
        private readonly MemoryStream _stream = new();

        public M2TrackDefinition<T> AddTrack<T>(M2TrackInterpolation interpolation, int globalSequenceIndex, IReadOnlyList<uint> times, IReadOnlyList<T> values)
        {
            ArgumentNullException.ThrowIfNull(times);
            ArgumentNullException.ThrowIfNull(values);
            if (times.Count != values.Count)
                throw new ArgumentException("Synthetic animated test tracks require matching time/value counts.");

            uint timestampArrayOffset = checked((uint)_stream.Position);
            WriteUInt32((uint)times.Count);
            long timestampDataPointerOffset = _stream.Position;
            WriteUInt32(0);

            uint valueArrayOffset = checked((uint)_stream.Position);
            WriteUInt32((uint)values.Count);
            long valueDataPointerOffset = _stream.Position;
            WriteUInt32(0);

            Align(4);
            uint timestampDataOffset = checked((uint)_stream.Position);
            foreach (uint time in times)
                WriteUInt32(time);
            PatchUInt32(timestampDataPointerOffset, timestampDataOffset);

            Align(4);
            uint valueDataOffset = checked((uint)_stream.Position);
            foreach (T value in values)
                WriteValue(value);
            PatchUInt32(valueDataPointerOffset, valueDataOffset);

            return new M2TrackDefinition<T>(
                interpolation,
                globalSequenceIndex,
                new M2TrackArrayReference(1u, timestampArrayOffset),
                new M2TrackArrayReference(1u, valueArrayOffset));
        }

        public byte[] ToArray()
        {
            return _stream.ToArray();
        }

        private void Align(int alignment)
        {
            while ((_stream.Position % alignment) != 0)
                _stream.WriteByte(0);
        }

        private void PatchUInt32(long offset, uint value)
        {
            long previousPosition = _stream.Position;
            _stream.Position = offset;
            WriteUInt32(value);
            _stream.Position = previousPosition;
        }

        private void WriteUInt32(uint value)
        {
            Span<byte> bytes = stackalloc byte[sizeof(uint)];
            BinaryPrimitives.WriteUInt32LittleEndian(bytes, value);
            _stream.Write(bytes);
        }

        private void WriteInt16(short value)
        {
            Span<byte> bytes = stackalloc byte[sizeof(short)];
            BinaryPrimitives.WriteInt16LittleEndian(bytes, value);
            _stream.Write(bytes);
        }

        private void WriteSingle(float value)
        {
            Span<byte> bytes = stackalloc byte[sizeof(float)];
            BinaryPrimitives.WriteInt32LittleEndian(bytes, BitConverter.SingleToInt32Bits(value));
            _stream.Write(bytes);
        }

        private void WriteValue<T>(T value)
        {
            switch (value)
            {
                case byte byteValue:
                    _stream.WriteByte(byteValue);
                    break;
                case short shortValue:
                    WriteInt16(shortValue);
                    break;
                case float floatValue:
                    WriteSingle(floatValue);
                    break;
                case Vector3 vectorValue:
                    WriteSingle(vectorValue.X);
                    WriteSingle(vectorValue.Y);
                    WriteSingle(vectorValue.Z);
                    break;
                case Quaternion quaternionValue:
                    WriteSingle(quaternionValue.X);
                    WriteSingle(quaternionValue.Y);
                    WriteSingle(quaternionValue.Z);
                    WriteSingle(quaternionValue.W);
                    break;
                default:
                    throw new NotSupportedException($"Unsupported synthetic M2 track value type '{typeof(T).FullName}'.");
            }
        }
    }
}