using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.IO.M2;
using WowViewer.Core.M2;
using WowViewer.Core.Runtime.M2;

namespace WowViewer.Core.Tests;

public sealed class M2RuntimeTests
{
    [Fact]
    public void ModelReader_ReadsBoneDefinitions()
    {
        byte[] bytes = CreateMd20WithOneBone();

        M2ModelDocument document = M2ModelReader.Read(new MemoryStream(bytes), "Creature\\SyntheticBones\\SyntheticBones.m2");

        Assert.Single(document.Bones);
        M2BoneDefinition bone = document.Bones[0];
        Assert.Equal(-1, bone.KeyBoneId);
        Assert.Equal(0x200u, bone.Flags);
        Assert.Equal((short)-1, bone.ParentBone);
        Assert.Equal((ushort)7, bone.SubmeshId);
        Assert.Equal(0x12345678u, bone.BoneNameCrc);
        Assert.Equal(new Vector3(3.0f, 4.0f, 5.0f), bone.Pivot);
        Assert.Equal(M2TrackInterpolation.Linear, bone.TranslationTrack.Interpolation);
    }

    [Fact]
    public void ModelReader_ReadsRibbonAndParticleDefinitions()
    {
        byte[] bytes = CreateMd20WithRibbonAndParticle();

        M2ModelDocument document = M2ModelReader.Read(new MemoryStream(bytes), "Creature\\SyntheticEffects\\SyntheticEffects.m2");

        M2RibbonDefinition ribbon = Assert.Single(document.Ribbons);
        Assert.Equal(42u, ribbon.RibbonId);
        Assert.Equal(3u, ribbon.BoneIndex);
        Assert.Equal([7], ribbon.TextureIndices);
        Assert.Equal([11], ribbon.MaterialIndices);
        Assert.Equal(12.0f, ribbon.EdgesPerSecond);
        Assert.Equal(0.5f, ribbon.EdgeLifetime);
        Assert.Equal((short)2, ribbon.PriorityPlane);
        Assert.Equal((sbyte)4, ribbon.RibbonColorIndex);

        M2ParticleDefinition particle = Assert.Single(document.Particles);
        Assert.Equal(99u, particle.ParticleId);
        Assert.Equal(0x80u, particle.Flags);
        Assert.Equal((ushort)4, particle.TextureIndex);
        Assert.Equal((ushort)4, particle.BlendingType);
        Assert.Equal((ushort)1, particle.EmitterType);
        Assert.Equal((ushort)6, particle.ParticleColorIndex);
        Assert.Equal("World\\Scale.m2", particle.GeometryModelPath);
        Assert.Equal("World\\Recursive.m2", particle.RecursionModelPath);
    }

    [Fact]
    public void BonePoseEvaluator_SolvesParentedPoseAndSkinnedVertices()
    {
        SyntheticTrackPayloadBuilder payload = new();
        M2ModelDocument model = CreateModel(
            payload.ToArray,
            bones:
            [
                new M2BoneDefinition(
                    0,
                    -1,
                    flags: 0,
                    parentBone: -1,
                    submeshId: 0,
                    boneNameCrc: 0,
                    payload.AddTrack(M2TrackInterpolation.Linear, -1, [0u, 1000u], [Vector3.Zero, new Vector3(2.0f, 0.0f, 0.0f)]),
                    EmptyCompressedRotationTrack(),
                    EmptyVectorTrack(),
                    Vector3.Zero),
                new M2BoneDefinition(
                    1,
                    -1,
                    flags: 0,
                    parentBone: 0,
                    submeshId: 0,
                    boneNameCrc: 0,
                    payload.AddTrack(M2TrackInterpolation.Linear, -1, [0u, 1000u], [Vector3.Zero, new Vector3(0.0f, 1.0f, 0.0f)]),
                    EmptyCompressedRotationTrack(),
                    EmptyVectorTrack(),
                    Vector3.Zero),
            ]);
        M2StaticRenderModel renderModel = CreateWeightedRenderModel(model);

        M2BonePoseState pose = M2BonePoseEvaluator.Evaluate(model, sequenceIndex: 0, timeMs: 500);
        M2SkinnedRenderModel skinned = M2SkinnedRenderModelBuilder.ApplyPose(renderModel, pose);

        Assert.Equal(2, pose.BoneCount);
        Assert.Equal(new Vector3(1.0f, 0.0f, 0.0f), pose.Bones[0].Translation);
        Assert.Equal(new Vector3(0.0f, 0.5f, 0.0f), pose.Bones[1].Translation);
        Assert.Single(skinned.Sections);
        AssertVectorNear(new Vector3(1.0f, 0.5f, 0.0f), skinned.Sections[0].Vertices[0].Position, 0.001f);
    }

    [Fact]
    public void ParticleRibbonRuntime_EvaluatesDescriptorsAndSubmissionEntries()
    {
        SyntheticTrackPayloadBuilder payload = new();
        M2ModelDocument model = CreateModel(
            payload.ToArray,
            ribbons:
            [
                new M2RibbonDefinition(
                    0,
                    ribbonId: 1,
                    boneIndex: 0,
                    position: new Vector3(0.0f, 0.0f, 2.0f),
                    textureIndices: [3],
                    materialIndices: [5],
                    colorTrack: payload.AddTrack(M2TrackInterpolation.None, -1, [0u], [new Vector3(0.4f, 0.8f, 1.0f)]),
                    alphaTrack: payload.AddTrack(M2TrackInterpolation.None, -1, [0u], [(short)32767]),
                    heightAboveTrack: payload.AddTrack(M2TrackInterpolation.None, -1, [0u], [0.25f]),
                    heightBelowTrack: payload.AddTrack(M2TrackInterpolation.None, -1, [0u], [0.25f]),
                    edgesPerSecond: 10.0f,
                    edgeLifetime: 0.5f,
                    gravity: 0.0f,
                    textureRows: 1,
                    textureColumns: 1,
                    textureSlotTrack: payload.AddTrack(M2TrackInterpolation.None, -1, [0u], [(ushort)0]),
                    visibilityTrack: payload.AddTrack(M2TrackInterpolation.None, -1, [0u], [(byte)1]),
                    priorityPlane: 0,
                    ribbonColorIndex: -1,
                    textureTransformLookupIndex: -1),
            ],
            particles:
            [
                new M2ParticleDefinition(
                    0,
                    particleId: 2,
                    flags: 0,
                    position: new Vector3(0.0f, 0.0f, 3.0f),
                    boneIndex: 0,
                    textureIndex: 4,
                    geometryModelPath: null,
                    recursionModelPath: null,
                    blendingType: 4,
                    emitterType: 1,
                    particleColorIndex: 0,
                    particleType: 0,
                    headOrTail: 0,
                    textureTileRotation: 0,
                    textureRows: 1,
                    textureColumns: 1,
                    emissionSpeedTrack: payload.AddTrack(M2TrackInterpolation.None, -1, [0u], [1.0f]),
                    speedVariationTrack: payload.AddTrack(M2TrackInterpolation.None, -1, [0u], [0.0f]),
                    verticalRangeTrack: payload.AddTrack(M2TrackInterpolation.None, -1, [0u], [0.0f]),
                    horizontalRangeTrack: payload.AddTrack(M2TrackInterpolation.None, -1, [0u], [0.0f]),
                    gravityTrack: payload.AddTrack(M2TrackInterpolation.None, -1, [0u], [0.0f]),
                    lifespanTrack: payload.AddTrack(M2TrackInterpolation.None, -1, [0u], [2.0f]),
                    emissionRateTrack: payload.AddTrack(M2TrackInterpolation.None, -1, [0u], [8.0f]),
                    emissionAreaLengthTrack: payload.AddTrack(M2TrackInterpolation.None, -1, [0u], [1.0f]),
                    emissionAreaWidthTrack: payload.AddTrack(M2TrackInterpolation.None, -1, [0u], [1.0f]),
                    zSourceTrack: payload.AddTrack(M2TrackInterpolation.None, -1, [0u], [0.0f]),
                    enabledTrack: payload.AddTrack(M2TrackInterpolation.None, -1, [0u], [(byte)1])),
            ]);

        M2EffectRuntimeState runtime = M2ParticleRibbonRuntimeEvaluator.Evaluate(model, sequenceIndex: 0, timeMs: 0);
        M2ParticleRuntimeState particle = Assert.Single(runtime.Particles);
        M2RibbonRuntimeState ribbon = Assert.Single(runtime.Ribbons);
        M2SceneSubmissionEntry[] entries = M2SceneSubmissionEntryBuilder.BuildParticleEntries(M2ParticleRibbonRuntimeEvaluator.BuildParticleSubmissionDescriptors(runtime, model.Identity.CanonicalModelPath))
            .Concat(M2SceneSubmissionEntryBuilder.BuildRibbonEntries(M2ParticleRibbonRuntimeEvaluator.BuildRibbonSubmissionDescriptors(runtime, model.Identity.CanonicalModelPath)))
            .ToArray();

        Assert.True(particle.Enabled);
        Assert.Equal(16, particle.EstimatedParticleCount);
        Assert.Equal("Particle_Additive", particle.EffectKey);
        Assert.True(particle.IsAdditive);
        Assert.True(ribbon.Visible);
        Assert.Equal(5, ribbon.EstimatedEdgeCount);
        Assert.Equal("Ribbon_Material_5", ribbon.EffectKey);
        Assert.Equal(2, entries.Length);
        Assert.Contains(entries, static entry => entry.Family == M2RenderEntryFamily.Particle && entry.VertexCount == 64 && entry.IndexCount == 96);
        Assert.Contains(entries, static entry => entry.Family == M2RenderEntryFamily.Ribbon && entry.VertexCount == 10 && entry.IndexCount == 24);
    }

    [Fact]
    public void CameraPathOverlayBuilder_BuildsSampledPathsFromCameraTracks()
    {
        SyntheticTrackPayloadBuilder payload = new();
        M2ModelDocument model = CreateModel(
            payload.ToArray,
            cameras:
            [
                new M2CameraDefinition(
                    index: 0,
                    type: -1,
                    staticFieldOfView: 1.2f,
                    farClip: 750.0f,
                    nearClip: 1.5f,
                    positionTrack: payload.AddTrack(M2TrackInterpolation.Linear, -1, [0u, 1000u], [Vector3.Zero, new Vector3(10.0f, 0.0f, 5.0f)]),
                    positionBase: new Vector3(100.0f, 200.0f, 300.0f),
                    targetPositionTrack: payload.AddTrack(M2TrackInterpolation.Linear, -1, [0u, 1000u], [Vector3.Zero, new Vector3(-5.0f, 10.0f, 0.0f)]),
                    targetPositionBase: new Vector3(150.0f, 250.0f, 325.0f),
                    rollTrack: EmptyFloatTrack()),
            ],
            viewCount: 0u);

        Assert.True(M2CameraPathOverlayBuilder.CanBuild(model));

        M2CameraPathVisualization visualization = M2CameraPathOverlayBuilder.Build(model);
        M2CameraPathOverlay overlay = Assert.Single(visualization.Overlays);

        Assert.Equal("flyby", overlay.TypeLabel);
        Assert.True(overlay.CameraSamples.Count >= 4);
        AssertVectorNear(new Vector3(100.0f, 200.0f, 300.0f), overlay.CameraSamples[0], 0.001f);
        AssertVectorNear(new Vector3(150.0f, 250.0f, 325.0f), overlay.TargetSamples[0], 0.001f);
        Assert.Contains(overlay.CameraSamples, static sample => sample.X > 100.0f && sample.Z > 300.0f);
        Assert.Contains(overlay.TargetSamples, static sample => sample.X < 150.0f && sample.Y > 250.0f);
        Assert.True(visualization.BoundsMin.X <= 100.0f && visualization.BoundsMax.X >= 150.0f);
        Assert.True(visualization.BoundsMin.Y <= 200.0f && visualization.BoundsMax.Y > 250.0f);
    }

    [Fact]
    public void CameraPathOverlayBuilder_AcceptsCamAssetsWithDummyViewCount()
    {
        SyntheticTrackPayloadBuilder payload = new();
        M2ModelDocument model = new(
            M2ModelIdentity.FromPath("Cameras\\Scry_cam.m2"),
            payload.ToArray(),
            "MD20",
            version: 0x108u,
            flags: 0u,
            viewCount: 1u,
            modelName: "Scry_cam",
            globalLoops: [],
            sequences:
            [
                new M2SequenceDefinition(0, animationId: 1, variationIndex: 0, duration: 1000u, moveSpeed: 0.0f, flags: (uint)M2SequenceFlags.StoredInline, frequency: 0, replayMinimum: 0u, replayMaximum: 0u, blendTimeIn: 0, blendTimeOut: 0, boundsMin: Vector3.Zero, boundsMax: Vector3.One, boundsRadius: 1.0f, variationNext: -1, aliasNext: ushort.MaxValue),
            ],
            sequenceLookup: [0],
            colors: [],
            textureWeights: [],
            textureTransforms: [],
            lights: [],
            cameras:
            [
                new M2CameraDefinition(
                    index: 0,
                    type: -1,
                    staticFieldOfView: 1.2f,
                    farClip: 750.0f,
                    nearClip: 1.5f,
                    positionTrack: EmptyVectorTrack(),
                    positionBase: new Vector3(100.0f, 200.0f, 300.0f),
                    targetPositionTrack: EmptyVectorTrack(),
                    targetPositionBase: new Vector3(150.0f, 250.0f, 325.0f),
                    rollTrack: EmptyFloatTrack()),
            ],
            boundsMin: Vector3.Zero,
            boundsMax: Vector3.One,
            boundsRadius: 1.0f,
            embeddedSkinProfileCount: 0u,
            embeddedSkinProfileOffset: 0u,
            bones:
            [
                new M2BoneDefinition(
                    index: 0,
                    keyBoneId: -1,
                    flags: 0u,
                    parentBone: -1,
                    submeshId: 0,
                    boneNameCrc: 0u,
                    translationTrack: EmptyVectorTrack(),
                    rotationTrack: EmptyCompressedRotationTrack(),
                    scalingTrack: EmptyVectorTrack(),
                    pivot: Vector3.Zero),
            ],
            ribbons: [],
            particles: []);

        Assert.True(M2CameraPathOverlayBuilder.CanBuild(model));

        M2CameraPathVisualization visualization = M2CameraPathOverlayBuilder.Build(model);
        M2CameraPathOverlay overlay = Assert.Single(visualization.Overlays);

        Assert.Single(overlay.CameraSamples);
        Assert.Single(overlay.TargetSamples);
        AssertVectorNear(new Vector3(100.0f, 200.0f, 300.0f), overlay.CameraSamples[0], 0.001f);
        AssertVectorNear(new Vector3(150.0f, 250.0f, 325.0f), overlay.TargetSamples[0], 0.001f);
    }

    [Fact]
    public void RenderConsumerFrameState_AppliesAnimatedPassAndModelLocalLightState()
    {
        M2StaticRenderModel renderModel = CreateConsumerRenderModel();
        M2AnimatedRenderState animatedState = new(
            requestedSequenceIndex: 0,
            resolvedSequenceIndex: 0,
            timeMs: 250,
            usesExternalPayload: false,
            passes:
            [
                new M2AnimatedRenderPassState(
                    sectionIndex: 0,
                    passIndex: 0,
                    batchIndex: 0,
                    color: new Vector3(0.25f, 0.5f, 0.75f),
                    colorAlpha: 0.8f,
                    combinedAlpha: 0.4f,
                    textureBindings:
                    [
                        new M2AnimatedTextureBindingState(0, 0.5f, new Vector3(1.0f, 0.0f, 0.0f), Quaternion.Identity, Vector3.One),
                    ]),
            ],
            lights:
            [
                new M2AnimatedLightState(
                    lightIndex: 0,
                    type: 1,
                    boneIndex: -1,
                    position: Vector3.Zero,
                    ambientColor: new Vector3(0.2f, 0.4f, 0.6f),
                    ambientIntensity: 0.5f,
                    diffuseColor: new Vector3(1.0f, 0.5f, 0.25f),
                    diffuseIntensity: 0.8f,
                    attenuationStart: 0.0f,
                    attenuationEnd: 10.0f,
                    visible: true),
            ]);

        M2RenderConsumerFrameState consumerState = M2RenderConsumerFrameStateBuilder.Build(renderModel, animatedState);

        Assert.Single(consumerState.Passes);
        Assert.Equal(1, consumerState.VisiblePassCount);
        AssertVectorNear(new Vector3(0.25f, 0.5f, 0.75f), consumerState.Passes[0].DiffuseColor, 0.001f);
        Assert.Equal(0.4f, consumerState.Passes[0].Alpha);
        Assert.True(consumerState.Passes[0].ReceivesLighting);
        AssertVectorNear(new Vector3(0.1f, 0.2f, 0.3f), consumerState.ModelAmbient, 0.001f);
        AssertVectorNear(new Vector3(0.8f, 0.4f, 0.2f), consumerState.ModelDiffuse, 0.001f);
        Assert.Equal("Creature\\Synthetic\\texture.blp", consumerState.Passes[0].Textures[0].TexturePath);
    }

    [Fact]
    public void EffectRegistry_ResolvesNativeEffectObjectAndRenderState()
    {
        M2StaticRenderModel renderModel = CreateConsumerRenderModel();
        M2StaticRenderMaterial material = renderModel.StructuredSections[0].Passes[0].Material;

        M2ResolvedEffect effect = M2EffectRegistry.Resolve(material);

        Assert.Equal("Diffuse_T1:Combiners_Decal", effect.RecipeKey);
        Assert.Equal("Diffuse_T1Combiners_Decal", effect.NativeEffectFamilyKey);
        Assert.Equal("Model2_Diffuse_T1Combiners_Decal", effect.EffectObjectKey);
        Assert.False(effect.DepthWrite);
        Assert.False(effect.AlphaTest);
        Assert.True(effect.IsTransparent);
        Assert.True(effect.ReceivesLighting);
        Assert.True(effect.IsHeuristic);
    }

    [Fact]
    public void SceneSubmissionCoordinator_GroupsCompatibleEntriesAndSplitsAtCapacity()
    {
        M2SceneSubmissionEntry[] entries =
        [
            new("a", "wolf", M2RenderEntryFamily.Doodad, "Diffuse_T1:Combiners_Opaque", textureSortKey: 1, stateBucket: 3, vertexCount: 4, indexCount: 6),
            new("b", "wolf", M2RenderEntryFamily.Doodad, "Diffuse_T1:Combiners_Opaque", textureSortKey: 1, stateBucket: 3, vertexCount: 4, indexCount: 6),
            new("c", "wolf", M2RenderEntryFamily.Doodad, "Diffuse_T1:Combiners_Opaque", textureSortKey: 1, stateBucket: 3, vertexCount: 4, indexCount: 6),
        ];

        M2SceneSubmissionPlan plan = M2SceneSubmissionCoordinator.BuildPlan(
            entries,
            M2RuntimeOptions.BatchDoodads,
            new M2SceneBatchLimits(maxVertices: 8, maxIndices: 12));

        Assert.Equal(2, plan.Batches.Count);
        Assert.False(plan.Batches[0].IsDirect);
        Assert.Equal("doodad-batch", plan.Batches[0].HandlerName);
        Assert.Equal(["a", "b"], plan.Batches[0].Entries.Select(static entry => entry.EntryKey).ToArray());
        Assert.Equal(["c"], plan.Batches[1].Entries.Select(static entry => entry.EntryKey).ToArray());
    }

    [Fact]
    public void SceneSubmissionCoordinator_LeavesParticlesDirectUnlessParticleBatchingIsEnabled()
    {
        M2SceneSubmissionEntry[] entries =
        [
            new("near", "spark", M2RenderEntryFamily.Particle, "Particle:Add", textureSortKey: 0, stateBucket: 1, vertexCount: 4, indexCount: 6, depthSortValue: 10.0f, isTransparent: true, isAdditive: true),
            new("far", "spark", M2RenderEntryFamily.Particle, "Particle:Add", textureSortKey: 0, stateBucket: 1, vertexCount: 4, indexCount: 6, depthSortValue: 100.0f, isTransparent: true, isAdditive: true),
        ];

        M2SceneSubmissionPlan directPlan = M2SceneSubmissionCoordinator.BuildPlan(entries, M2RuntimeOptions.None);
        M2SceneSubmissionPlan batchedPlan = M2SceneSubmissionCoordinator.BuildPlan(entries, M2RuntimeOptions.BatchParticles | M2RuntimeOptions.ForceAdditiveParticleSort);

        Assert.All(directPlan.Batches, static batch => Assert.True(batch.IsDirect));
        Assert.Single(batchedPlan.Batches);
        Assert.False(batchedPlan.Batches[0].IsDirect);
        Assert.Equal("particle-dispatch", batchedPlan.Batches[0].HandlerName);
        Assert.Equal(["far", "near"], batchedPlan.Batches[0].Entries.Select(static entry => entry.EntryKey).ToArray());
    }

    [Fact]
    public void SceneSubmissionEntryBuilder_KeepsParticleOptOutAndRibbonsOnDedicatedDirectHandlers()
    {
        M2SceneSubmissionEntry[] entries = M2SceneSubmissionEntryBuilder.BuildParticleEntries(
            [
                new M2ParticleSubmissionDescriptor("spark-a", "spark", "Particle_Add", 0, 1, 4, 6, depthSortValue: 5.0f, isAdditive: true),
                new M2ParticleSubmissionDescriptor("spark-b", "spark", "Particle_Add", 0, 1, 4, 6, depthSortValue: 4.0f, isAdditive: true, allowsBatching: false),
            ])
            .Concat(M2SceneSubmissionEntryBuilder.BuildRibbonEntries(
            [
                new M2RibbonSubmissionDescriptor("trail-a", "trail", "Ribbon_Mod", 0, 2, 8, 12, depthSortValue: 1.0f),
            ]))
            .ToArray();

        M2SceneSubmissionPlan plan = M2SceneSubmissionCoordinator.BuildPlan(entries, M2RuntimeOptions.BatchParticles);

        M2SceneSubmissionBatch forcedParticle = Assert.Single(plan.Batches, static batch => batch.Entries.Any(static entry => entry.EntryKey == "spark-b"));
        M2SceneSubmissionBatch ribbon = Assert.Single(plan.Batches, static batch => batch.Family == M2RenderEntryFamily.Ribbon);
        Assert.True(forcedParticle.IsDirect);
        Assert.Equal("particle-dispatch", forcedParticle.HandlerName);
        Assert.True(ribbon.IsDirect);
        Assert.True(ribbon.UsesDedicatedStateScope);
        Assert.Equal("ribbon-direct", ribbon.HandlerName);
    }

    [Fact]
    public void RuntimeGoldenFrameBuilder_ProducesDeterministicHashFromConsumerAndSubmissionState()
    {
        M2StaticRenderModel renderModel = CreateConsumerRenderModel();
        M2AnimatedRenderState animatedState = CreateConsumerAnimatedState();
        M2BonePoseState poseState = new(
            requestedSequenceIndex: 0,
            resolvedSequenceIndex: 0,
            timeMs: 250,
            usesExternalPayload: false,
            bones: []);
        M2RenderConsumerFrameState consumerState = M2RenderConsumerFrameStateBuilder.Build(renderModel, animatedState);
        M2SceneSubmissionPlan plan = M2SceneSubmissionCoordinator.BuildPlan(
            M2SceneSubmissionEntryBuilder.BuildRenderEntries(renderModel.Model, renderModel, consumerState),
            M2RuntimeOptions.BatchDoodads | M2RuntimeOptions.BatchParticles);

        M2RuntimeGoldenFrame first = M2RuntimeGoldenFrameBuilder.Build(renderModel.Model, animatedState, poseState, skinnedRenderModel: null, consumerState, plan);
        M2RuntimeGoldenFrame second = M2RuntimeGoldenFrameBuilder.Build(renderModel.Model, animatedState, poseState, skinnedRenderModel: null, consumerState, plan);

        Assert.Equal(first.RuntimeHash, second.RuntimeHash);
        Assert.Equal("Model2_Diffuse_T1Combiners_Decal", first.Effects[0].EffectObjectKey);
        Assert.Single(first.Batches);
        Assert.Equal("core-batch", first.Batches[0].Handler);
    }

    [Fact]
    public void RenderFrameAndSoftwareSnapshot_AreDeterministicAndNonBlank()
    {
        M2StaticRenderModel renderModel = CreateConsumerRenderModel();
        M2AnimatedRenderState animatedState = CreateConsumerAnimatedState();
        M2RenderConsumerFrameState consumerState = M2RenderConsumerFrameStateBuilder.Build(renderModel, animatedState);
        M2SceneSubmissionPlan submissionPlan = M2SceneSubmissionCoordinator.BuildPlan(
            M2SceneSubmissionEntryBuilder.BuildRenderEntries(renderModel.Model, renderModel, consumerState),
            M2RuntimeOptions.BatchDoodads | M2RuntimeOptions.BatchParticles);

        M2RenderFrame firstFrame = M2RenderFrameBuilder.Build(renderModel, skinnedRenderModel: null, consumerState, submissionPlan, timeMs: 250);
        M2RenderFrame secondFrame = M2RenderFrameBuilder.Build(renderModel, skinnedRenderModel: null, consumerState, submissionPlan, timeMs: 250);
        M2SoftwareVisualSnapshot firstSnapshot = M2SoftwareVisualSnapshotBuilder.Build(firstFrame, width: 64, height: 64);
        M2SoftwareVisualSnapshot secondSnapshot = M2SoftwareVisualSnapshotBuilder.Build(firstFrame, width: 64, height: 64);

        Assert.Equal(firstFrame.FrameHash, secondFrame.FrameHash);
        Assert.Single(firstFrame.DrawCommands);
        Assert.Equal(3, firstFrame.BackendVertexCount);
        Assert.Equal(3, firstFrame.BackendIndexCount);
        Assert.True(firstSnapshot.LitPixelCount > 0);
        Assert.Equal(firstSnapshot.VisualHash, secondSnapshot.VisualHash);
    }

    [Fact]
    public void RuntimeFramePipeline_BuildsSharedArtifactsForConsumerSurfaces()
    {
        M2StaticRenderModel renderModel = CreateConsumerRenderModel();

        M2RuntimeFrameResult first = M2RuntimeFramePipeline.Build(renderModel.Model, renderModel, sequenceIndex: 0, timeMs: 250, visualWidth: 64, visualHeight: 64);
        M2RuntimeFrameResult second = M2RuntimeFramePipeline.Build(renderModel.Model, renderModel, sequenceIndex: 0, timeMs: 250, visualWidth: 64, visualHeight: 64);

        Assert.Equal(first.GoldenFrame.RuntimeHash, second.GoldenFrame.RuntimeHash);
        Assert.Equal(first.RenderFrame.FrameHash, second.RenderFrame.FrameHash);
        Assert.Equal(first.VisualSnapshot.VisualHash, second.VisualSnapshot.VisualHash);
        Assert.Equal(1, first.ConsumerState.VisiblePassCount);
        Assert.Equal(1, first.RenderFrame.CommandCount);
        Assert.True(first.VisualSnapshot.LitPixelCount > 0);
        Assert.Empty(first.EffectRuntimeState.Particles);
        Assert.Empty(first.EffectRuntimeState.Ribbons);
    }

    private static M2ModelDocument CreateModel(
        Func<byte[]> payloadFactory,
        IReadOnlyList<M2BoneDefinition>? bones = null,
        IReadOnlyList<M2RibbonDefinition>? ribbons = null,
        IReadOnlyList<M2ParticleDefinition>? particles = null,
        IReadOnlyList<M2CameraDefinition>? cameras = null,
        uint viewCount = 1u)
    {
        return new M2ModelDocument(
            M2ModelIdentity.FromPath("Creature\\Synthetic\\Synthetic.m2"),
            payloadFactory(),
            "MD20",
            version: 0x108u,
            flags: 0u,
            viewCount: viewCount,
            modelName: "Synthetic",
            globalLoops: [],
            sequences:
            [
                new M2SequenceDefinition(0, animationId: 1, variationIndex: 0, duration: 1000u, moveSpeed: 0.0f, flags: (uint)M2SequenceFlags.StoredInline, frequency: 0, replayMinimum: 0u, replayMaximum: 0u, blendTimeIn: 0, blendTimeOut: 0, boundsMin: Vector3.Zero, boundsMax: Vector3.One, boundsRadius: 1.0f, variationNext: -1, aliasNext: ushort.MaxValue),
            ],
            sequenceLookup: [0],
            colors: [],
            textureWeights: [],
            textureTransforms: [],
            lights: [],
            cameras: cameras ?? [],
            boundsMin: Vector3.Zero,
            boundsMax: Vector3.One,
            boundsRadius: 1.0f,
            embeddedSkinProfileCount: 0u,
            embeddedSkinProfileOffset: 0u,
            bones,
            ribbons,
            particles);
    }

    private static M2StaticRenderModel CreateWeightedRenderModel(M2ModelDocument model)
    {
        M2GeometryDocument geometry = new(
            model,
            [
                new M2GeometryVertex(Vector3.Zero, Vector3.UnitZ, Vector2.Zero, Vector2.Zero, new Vector4(1, 0, 0, 0), new Vector4(1, 0, 0, 0)),
                new M2GeometryVertex(Vector3.UnitX, Vector3.UnitZ, Vector2.UnitX, Vector2.Zero, new Vector4(1, 0, 0, 0), new Vector4(1, 0, 0, 0)),
                new M2GeometryVertex(Vector3.UnitY, Vector3.UnitZ, Vector2.UnitY, Vector2.Zero, new Vector4(1, 0, 0, 0), new Vector4(1, 0, 0, 0)),
            ],
            [],
            [new M2GeometryRenderFlag(0, 0)],
            [],
            [],
            [],
            [],
            []);
        M2SkinDocument skin = new(
            sourcePath: "Creature\\Synthetic\\Synthetic00.skin",
            signature: "SKIN",
            vertexLookup: [0, 1, 2],
            vertexLookupOffset: 0,
            triangleIndices: [0, 1, 2],
            triangleIndexOffset: 0,
            boneEntries: [new M2SkinBoneEntry(0, 1, 0, 0)],
            boneEntryOffset: 0,
            submeshes: [new M2SkinSubmesh(1, 0, 0, 3, 0, 3, boneCount: 2, boneComboIndex: 0, boneInfluences: 1, centerBoneIndex: 1)],
            submeshOffset: 0,
            batches: [new M2SkinBatch(0, 0, 0, 0, 0, -1, 0, 0, 0, 0, ushort.MaxValue, ushort.MaxValue, ushort.MaxValue)],
            batchOffset: 0,
            globalVertexOffset: 0,
            shadowBatchCount: 0,
            shadowBatchOffset: 0);

        return M2StaticRenderModelBuilder.Build(geometry, M2SkinProfileRuntime.Initialize(M2SkinProfileRuntime.Load(M2SkinProfileRuntime.Choose(model), skin)));
    }

    private static M2StaticRenderModel CreateConsumerRenderModel()
    {
        M2ModelDocument model = CreateModel(() => []);
        M2StaticRenderTextureBinding texture = new(
            stageIndex: 0,
            textureLookupIndex: 0,
            textureId: 0,
            texturePath: "Creature\\Synthetic\\texture.blp",
            replaceableId: 0,
            textureFlags: 0,
            textureCoordLookupIndex: 0,
            textureCoordLookupValue: 0,
            transparencyLookupIndex: 0,
            transparencyLookupValue: 0,
            textureAnimationLookupIndex: 0,
            textureAnimationLookupValue: 0);
        M2StaticRenderMaterial material = new(
            batchIndex: 0,
            batchFlags: 0,
            priorityPlane: 0,
            shaderId: 0,
            geosetIndex: 0,
            colorIndex: 0,
            renderFlagsIndex: 0,
            materialLayer: 0,
            textureCount: 1,
            textureComboIndex: 0,
            textureCoordComboIndex: 0,
            transparencyComboIndex: 0,
            textureAnimationLookupIndex: 0,
            renderFlags: 0,
            rawBlendMode: 2,
            blendMode: M2BlendMode.AlphaBlend,
            texturePath: texture.TexturePath,
            replaceableId: 0,
            textureFlags: 0,
            textureBindings: [texture],
            effectRecipe: new M2EffectRecipe(M2DiffuseEffectFamily.T1, M2CombinerEffectFamily.Decal, isProjected: false, usesColorAnimation: true, usesTransparencyAnimation: true, usesTextureTransformAnimation: true, suppressCombinedTransparency: false, isHeuristic: true));
        M2StructuredRenderSection section = new(
            sectionIndex: 0,
            skinSectionId: 1,
            boneComboIndex: 0,
            boneCount: 0,
            boneInfluences: 0,
            centerBoneIndex: 0,
            vertices:
            [
                new M2StaticRenderVertex(Vector3.Zero, Vector3.UnitZ, Vector2.Zero, Vector2.Zero, Vector4.Zero, Vector4.Zero),
                new M2StaticRenderVertex(Vector3.UnitX, Vector3.UnitZ, Vector2.UnitX, Vector2.UnitX, Vector4.Zero, Vector4.Zero),
                new M2StaticRenderVertex(Vector3.UnitY, Vector3.UnitZ, Vector2.UnitY, Vector2.UnitY, Vector4.Zero, Vector4.Zero),
            ],
            indices: [0, 1, 2],
            passes: [new M2StructuredRenderPass(0, material)]);

        return new M2StaticRenderModel(model, [], [section], [], usesCompatibilityFallback: false);
    }

    private static M2AnimatedRenderState CreateConsumerAnimatedState()
    {
        return new M2AnimatedRenderState(
            requestedSequenceIndex: 0,
            resolvedSequenceIndex: 0,
            timeMs: 250,
            usesExternalPayload: false,
            passes:
            [
                new M2AnimatedRenderPassState(
                    sectionIndex: 0,
                    passIndex: 0,
                    batchIndex: 0,
                    color: new Vector3(0.25f, 0.5f, 0.75f),
                    colorAlpha: 0.8f,
                    combinedAlpha: 0.4f,
                    textureBindings:
                    [
                        new M2AnimatedTextureBindingState(0, 0.5f, new Vector3(1.0f, 0.0f, 0.0f), Quaternion.Identity, Vector3.One),
                    ]),
            ],
            lights: []);
    }

    private static byte[] CreateMd20WithOneBone()
    {
        byte[] data = new byte[0x240];
        Encoding.ASCII.GetBytes("MD20").CopyTo(data, 0);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x04, 4), 0x108u);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x2C, 4), 1u);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x30, 4), 0x140u);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x44, 4), 1u);
        WriteVector3(data, 0xA0, Vector3.Zero);
        WriteVector3(data, 0xAC, Vector3.One);
        WriteSingle(data, 0xB8, 1.0f);

        int boneOffset = 0x140;
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(boneOffset + 0x00, 4), -1);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(boneOffset + 0x04, 4), 0x200u);
        BinaryPrimitives.WriteInt16LittleEndian(data.AsSpan(boneOffset + 0x08, 2), -1);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(boneOffset + 0x0A, 2), 7);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(boneOffset + 0x0C, 4), 0x12345678u);
        WriteTrackHeader(data, boneOffset + 0x10, interpolation: 1);
        WriteTrackHeader(data, boneOffset + 0x24, interpolation: 0);
        WriteTrackHeader(data, boneOffset + 0x38, interpolation: 1);
        WriteVector3(data, boneOffset + 0x4C, new Vector3(3.0f, 4.0f, 5.0f));
        return data;
    }

    private static byte[] CreateMd20WithRibbonAndParticle()
    {
        byte[] data = new byte[0x700];
        Encoding.ASCII.GetBytes("MD20").CopyTo(data, 0);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x04, 4), 0x108u);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x44, 4), 1u);
        WriteVector3(data, 0xA0, Vector3.Zero);
        WriteVector3(data, 0xAC, Vector3.One);
        WriteSingle(data, 0xB8, 1.0f);

        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x120, 4), 1u);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x124, 4), 0x200u);
        int ribbonOffset = 0x200;
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(ribbonOffset + 0x00, 4), 42u);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(ribbonOffset + 0x04, 4), 3u);
        WriteVector3(data, ribbonOffset + 0x08, new Vector3(1.0f, 2.0f, 3.0f));
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(ribbonOffset + 0x14, 4), 1u);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(ribbonOffset + 0x18, 4), 0x500u);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(0x500, 2), 7);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(ribbonOffset + 0x1C, 4), 1u);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(ribbonOffset + 0x20, 4), 0x510u);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(0x510, 2), 11);
        WriteSingle(data, ribbonOffset + 0x74, 12.0f);
        WriteSingle(data, ribbonOffset + 0x78, 0.5f);
        WriteSingle(data, ribbonOffset + 0x7C, 0.25f);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(ribbonOffset + 0x80, 2), 2);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(ribbonOffset + 0x82, 2), 4);
        BinaryPrimitives.WriteInt16LittleEndian(data.AsSpan(ribbonOffset + 0xAC, 2), 2);
        data[ribbonOffset + 0xAE] = 4;
        data[ribbonOffset + 0xAF] = 5;

        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x128, 4), 1u);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x12C, 4), 0x300u);
        int particleOffset = 0x300;
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(particleOffset + 0x00, 4), 99u);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(particleOffset + 0x04, 4), 0x80u);
        WriteVector3(data, particleOffset + 0x08, new Vector3(4.0f, 5.0f, 6.0f));
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(particleOffset + 0x14, 2), 2);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(particleOffset + 0x16, 2), 4);
        WriteStringReference(data, particleOffset + 0x18, 0x520, "World\\Scale.m2");
        WriteStringReference(data, particleOffset + 0x20, 0x540, "World\\Recursive.m2");
        data[particleOffset + 0x28] = 4;
        data[particleOffset + 0x29] = 1;
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(particleOffset + 0x2A, 2), 6);
        data[particleOffset + 0x2C] = 1;
        data[particleOffset + 0x2D] = 2;
        BinaryPrimitives.WriteInt16LittleEndian(data.AsSpan(particleOffset + 0x2E, 2), 3);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(particleOffset + 0x30, 2), 8);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(particleOffset + 0x32, 2), 8);
        return data;
    }

    private static M2TrackDefinition<Vector3> EmptyVectorTrack()
    {
        return new M2TrackDefinition<Vector3>(M2TrackInterpolation.None, -1, new M2TrackArrayReference(0, 0), new M2TrackArrayReference(0, 0));
    }

    private static M2TrackDefinition<M2CompQuaternion> EmptyCompressedRotationTrack()
    {
        return new M2TrackDefinition<M2CompQuaternion>(M2TrackInterpolation.None, -1, new M2TrackArrayReference(0, 0), new M2TrackArrayReference(0, 0));
    }

    private static M2TrackDefinition<float> EmptyFloatTrack()
    {
        return new M2TrackDefinition<float>(M2TrackInterpolation.None, -1, new M2TrackArrayReference(0, 0), new M2TrackArrayReference(0, 0));
    }

    private static void WriteTrackHeader(byte[] data, int offset, ushort interpolation)
    {
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(offset + 0x00, 2), interpolation);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(offset + 0x02, 2), ushort.MaxValue);
    }

    private static void WriteVector3(byte[] data, int offset, Vector3 value)
    {
        WriteSingle(data, offset + 0x00, value.X);
        WriteSingle(data, offset + 0x04, value.Y);
        WriteSingle(data, offset + 0x08, value.Z);
    }

    private static void WriteSingle(byte[] data, int offset, float value)
    {
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(offset, 4), BitConverter.SingleToInt32Bits(value));
    }

    private static void WriteStringReference(byte[] data, int referenceOffset, int stringOffset, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value + '\0');
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(referenceOffset + 0x00, 4), checked((uint)bytes.Length));
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(referenceOffset + 0x04, 4), checked((uint)stringOffset));
        bytes.CopyTo(data.AsSpan(stringOffset, bytes.Length));
    }

    private static void AssertVectorNear(Vector3 expected, Vector3 actual, float tolerance)
    {
        Assert.InRange(actual.X, expected.X - tolerance, expected.X + tolerance);
        Assert.InRange(actual.Y, expected.Y - tolerance, expected.Y + tolerance);
        Assert.InRange(actual.Z, expected.Z - tolerance, expected.Z + tolerance);
    }

    private sealed class SyntheticTrackPayloadBuilder
    {
        private readonly MemoryStream _stream = new();

        public M2TrackDefinition<T> AddTrack<T>(M2TrackInterpolation interpolation, int globalSequenceIndex, IReadOnlyList<uint> times, IReadOnlyList<T> values)
        {
            if (times.Count != values.Count)
                throw new ArgumentException("Synthetic tracks require matching time/value counts.");

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

        private void WriteSingle(float value)
        {
            Span<byte> bytes = stackalloc byte[sizeof(float)];
            BinaryPrimitives.WriteInt32LittleEndian(bytes, BitConverter.SingleToInt32Bits(value));
            _stream.Write(bytes);
        }

        private void WriteInt16(short value)
        {
            Span<byte> bytes = stackalloc byte[sizeof(short)];
            BinaryPrimitives.WriteInt16LittleEndian(bytes, value);
            _stream.Write(bytes);
        }

        private void WriteUInt16(ushort value)
        {
            Span<byte> bytes = stackalloc byte[sizeof(ushort)];
            BinaryPrimitives.WriteUInt16LittleEndian(bytes, value);
            _stream.Write(bytes);
        }

        private void WriteValue<T>(T value)
        {
            switch (value)
            {
                case Vector3 vectorValue:
                    WriteSingle(vectorValue.X);
                    WriteSingle(vectorValue.Y);
                    WriteSingle(vectorValue.Z);
                    break;
                case float floatValue:
                    WriteSingle(floatValue);
                    break;
                case short int16Value:
                    WriteInt16(int16Value);
                    break;
                case ushort uint16Value:
                    WriteUInt16(uint16Value);
                    break;
                case byte byteValue:
                    _stream.WriteByte(byteValue);
                    break;
                default:
                    throw new NotSupportedException($"Unsupported synthetic track value type '{typeof(T).FullName}'.");
            }
        }
    }
}
