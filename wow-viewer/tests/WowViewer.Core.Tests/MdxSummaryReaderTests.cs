using System.Buffers.Binary;
using System.Numerics;
using System.Text;
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
            sequences:
            [
                ("Stand", 10, 40, 1.25f, 0x2u, 3.0f, 5, 35, -2.0f, -1.5f, -1.0f, 2.0f, 2.5f, 3.0f, 4.5f),
            ],
            pivotPoints:
            [
                new Vector3(1.0f, 2.0f, 3.0f),
                new Vector3(-4.0f, 5.5f, -6.25f),
            ],
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
        Assert.Equal(7, summary.ChunkCount);
        Assert.Equal(6, summary.KnownChunkCount);
        Assert.Equal(1, summary.UnknownChunkCount);
        Assert.Equal(1, summary.SequenceCount);
        Assert.Equal("Stand", summary.Sequences[0].Name);
        Assert.Equal(10, summary.Sequences[0].StartTime);
        Assert.Equal(40, summary.Sequences[0].EndTime);
        Assert.Equal(30, summary.Sequences[0].Duration);
        Assert.Equal(1.25f, summary.Sequences[0].MoveSpeed);
        Assert.Equal(0x2u, summary.Sequences[0].Flags);
        Assert.Equal(3.0f, summary.Sequences[0].Frequency);
        Assert.Equal(5, summary.Sequences[0].ReplayStart);
        Assert.Equal(35, summary.Sequences[0].ReplayEnd);
        Assert.Null(summary.Sequences[0].BlendTime);
        Assert.Equal(new Vector3(-2.0f, -1.5f, -1.0f), summary.Sequences[0].BoundsMin);
        Assert.Equal(new Vector3(2.0f, 2.5f, 3.0f), summary.Sequences[0].BoundsMax);
        Assert.Equal(4.5f, summary.Sequences[0].BoundsRadius);
        Assert.Equal(2, summary.PivotPointCount);
        Assert.Equal(new Vector3(1.0f, 2.0f, 3.0f), summary.PivotPoints[0].Position);
        Assert.Equal(new Vector3(-4.0f, 5.5f, -6.25f), summary.PivotPoints[1].Position);
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
        Assert.Equal(MdxChunkIds.Seqs, summary.Chunks[2].Id);
        Assert.Equal(MdxChunkIds.Pivt, summary.Chunks[3].Id);
        Assert.Equal(MdxChunkIds.Texs, summary.Chunks[4].Id);
        Assert.Equal(MdxChunkIds.Mtls, summary.Chunks[5].Id);
        Assert.Equal("UNKN", summary.Chunks[6].Id.ToString());
        Assert.False(summary.Chunks[6].IsKnownChunk);
    }

    [Fact]
    public void Read_SyntheticCountedNamed8CSeqs_ProducesExpectedSequenceSummary()
    {
        byte[] bytes = CreateMdxBytes(
            version: 1300,
            modelName: "SyntheticParticle",
            blendTime: 0,
            boundsMin: new Vector3(-1.0f, -1.0f, -1.0f),
            boundsMax: new Vector3(1.0f, 1.0f, 1.0f),
            sequences: [],
            pivotPoints: [],
            textures: [],
            materials: [],
            extraChunks:
            [
                CreateChunk("SEQS", CreateSeqsPayloadNamed8C(
                [
                    ("Birth", 0, 100, 0.5f, 0x4u, -1.0f, -2.0f, -3.0f, 1.0f, 2.0f, 3.0f, 10, 90, 25u),
                ])),
            ]);

        using MemoryStream stream = new(bytes);
        MdxSummary summary = MdxSummaryReader.Read(stream, "synthetic_8c.mdx");

        Assert.Equal(1, summary.SequenceCount);
        Assert.Equal("Birth", summary.Sequences[0].Name);
        Assert.Equal(0, summary.Sequences[0].StartTime);
        Assert.Equal(100, summary.Sequences[0].EndTime);
        Assert.Equal(0.5f, summary.Sequences[0].MoveSpeed);
        Assert.Equal(0x4u, summary.Sequences[0].Flags);
        Assert.Equal(1.0f, summary.Sequences[0].Frequency);
        Assert.Equal(10, summary.Sequences[0].ReplayStart);
        Assert.Equal(90, summary.Sequences[0].ReplayEnd);
        Assert.Equal(25u, summary.Sequences[0].BlendTime);
        Assert.Equal(new Vector3(-1.0f, -2.0f, -3.0f), summary.Sequences[0].BoundsMin);
        Assert.Equal(new Vector3(1.0f, 2.0f, 3.0f), summary.Sequences[0].BoundsMax);
        Assert.Null(summary.Sequences[0].BoundsRadius);
    }

    [Fact]
    public void Read_SyntheticGlbs_ProducesExpectedGlobalSequenceSummary()
    {
        byte[] bytes = CreateMdxBytes(
            version: 1300,
            modelName: "SyntheticGlbs",
            blendTime: 0,
            boundsMin: new Vector3(-1.0f, -1.0f, -1.0f),
            boundsMax: new Vector3(1.0f, 1.0f, 1.0f),
            sequences: [],
            pivotPoints: [],
            textures: [],
            materials: [],
            extraChunks:
            [
                CreateChunk("GLBS", CreateGlobalSequencesPayload([267u, 133u, 533u, 0u])),
            ]);

        using MemoryStream stream = new(bytes);
        MdxSummary summary = MdxSummaryReader.Read(stream, "synthetic_glbs.mdx");

        Assert.Equal(4, summary.GlobalSequenceCount);
        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Glbs);
        Assert.Equal(267u, summary.GlobalSequences[0].Duration);
        Assert.Equal(133u, summary.GlobalSequences[1].Duration);
        Assert.Equal(533u, summary.GlobalSequences[2].Duration);
        Assert.Equal(0u, summary.GlobalSequences[3].Duration);
    }

    [Fact]
    public void Read_SyntheticPivt_ProducesExpectedPivotSummary()
    {
        byte[] bytes = CreateMdxBytes(
            version: 1300,
            modelName: "SyntheticPivot",
            blendTime: 0,
            boundsMin: new Vector3(-1.0f, -1.0f, -1.0f),
            boundsMax: new Vector3(1.0f, 1.0f, 1.0f),
            sequences: [],
            pivotPoints:
            [
                new Vector3(0.0f, 0.0f, 0.0f),
                new Vector3(3.5f, -2.25f, 8.0f),
                new Vector3(-7.0f, 4.0f, 1.5f),
            ],
            textures: [],
            materials: [],
            extraChunks: []);

        using MemoryStream stream = new(bytes);
        MdxSummary summary = MdxSummaryReader.Read(stream, "synthetic_pivt.mdx");

        Assert.Equal(3, summary.PivotPointCount);
        Assert.Equal(new Vector3(0.0f, 0.0f, 0.0f), summary.PivotPoints[0].Position);
        Assert.Equal(new Vector3(3.5f, -2.25f, 8.0f), summary.PivotPoints[1].Position);
        Assert.Equal(new Vector3(-7.0f, 4.0f, 1.5f), summary.PivotPoints[2].Position);
        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Pivt);
    }

    [Fact]
    public void Read_SyntheticClassicGeos_ProducesExpectedGeosetSummary()
    {
        byte[] bytes = CreateMdxBytes(
            version: 1300,
            modelName: "SyntheticGeos",
            blendTime: 0,
            boundsMin: new Vector3(-1.0f, -1.0f, -1.0f),
            boundsMax: new Vector3(1.0f, 1.0f, 1.0f),
            sequences: [],
            pivotPoints: [],
            textures: [],
            materials: [],
            extraChunks:
            [
                CreateChunk("GEOS", CreateClassicGeosPayload(
                [
                    (3, 7, 2u, 0x10u, 5.5f, new Vector3(-1.0f, -2.0f, -3.0f), new Vector3(4.0f, 5.0f, 6.0f), 1),
                ])),
            ]);

        using MemoryStream stream = new(bytes);
        MdxSummary summary = MdxSummaryReader.Read(stream, "synthetic_geos.mdx");

        Assert.Equal(1, summary.GeosetCount);
        Assert.Equal(3, summary.Geosets[0].VertexCount);
        Assert.Equal(3, summary.Geosets[0].NormalCount);
        Assert.Equal(1, summary.Geosets[0].UvSetCount);
        Assert.Equal(3, summary.Geosets[0].PrimaryUvCount);
        Assert.Equal(3, summary.Geosets[0].IndexCount);
        Assert.Equal(1, summary.Geosets[0].TriangleCount);
        Assert.Equal(7, summary.Geosets[0].MaterialId);
        Assert.Equal(2u, summary.Geosets[0].SelectionGroup);
        Assert.Equal(0x10u, summary.Geosets[0].Flags);
        Assert.Equal(5.5f, summary.Geosets[0].BoundsRadius);
        Assert.Equal(new Vector3(-1.0f, -2.0f, -3.0f), summary.Geosets[0].BoundsMin);
        Assert.Equal(new Vector3(4.0f, 5.0f, 6.0f), summary.Geosets[0].BoundsMax);
        Assert.Equal(1, summary.Geosets[0].AnimationExtentCount);
        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Geos);
    }

    [Fact]
    public void Read_SyntheticClassicGeoa_ProducesExpectedGeosetAnimationSummary()
    {
        byte[] bytes = CreateMdxBytes(
            version: 1300,
            modelName: "SyntheticGeoa",
            blendTime: 0,
            boundsMin: new Vector3(-1.0f, -1.0f, -1.0f),
            boundsMax: new Vector3(1.0f, 1.0f, 1.0f),
            sequences: [],
            pivotPoints: [],
            textures: [],
            materials: [],
            extraChunks:
            [
                CreateChunk("GEOA", CreateGeoaPayload(
                [
                    (0u, 0.75f, new Vector3(1.0f, 0.5f, 0.25f), 0x1u, (1u, -1, new[] { 10, 40 }), (3u, 7, new[] { 15 })),
                    (uint.MaxValue, 1.0f, new Vector3(0.0f, 0.0f, 0.0f), 0x0u, null, null),
                ])),
            ]);

        using MemoryStream stream = new(bytes);
        MdxSummary summary = MdxSummaryReader.Read(stream, "synthetic_geoa.mdx");

        Assert.Equal(2, summary.GeosetAnimationCount);
        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Geoa);

        MdxGeosetAnimationSummary first = summary.GeosetAnimations[0];
        Assert.Equal(0u, first.GeosetId);
        Assert.Equal(0.75f, first.StaticAlpha);
        Assert.Equal(new Vector3(1.0f, 0.5f, 0.25f), first.StaticColor);
        Assert.Equal(0x1u, first.Flags);
        Assert.True(first.UsesStaticColor);
        Assert.NotNull(first.AlphaTrack);
        Assert.Equal("KGAO", first.AlphaTrack!.Tag);
        Assert.Equal(2, first.AlphaTrack.KeyCount);
        Assert.Equal(1u, first.AlphaTrack.InterpolationType);
        Assert.Equal(-1, first.AlphaTrack.GlobalSequenceId);
        Assert.Equal(10, first.AlphaTrack.FirstKeyTime);
        Assert.Equal(40, first.AlphaTrack.LastKeyTime);
        Assert.NotNull(first.ColorTrack);
        Assert.Equal("KGAC", first.ColorTrack!.Tag);
        Assert.Equal(1, first.ColorTrack.KeyCount);
        Assert.Equal(3u, first.ColorTrack.InterpolationType);
        Assert.Equal(7, first.ColorTrack.GlobalSequenceId);
        Assert.Equal(15, first.ColorTrack.FirstKeyTime);
        Assert.Equal(15, first.ColorTrack.LastKeyTime);

        MdxGeosetAnimationSummary second = summary.GeosetAnimations[1];
        Assert.Equal(uint.MaxValue, second.GeosetId);
        Assert.Equal(1.0f, second.StaticAlpha);
        Assert.False(second.UsesStaticColor);
        Assert.Null(second.AlphaTrack);
        Assert.Null(second.ColorTrack);
    }

    [Fact]
    public void Read_SyntheticClassicBone_ProducesExpectedBoneSummary()
    {
        byte[] bytes = CreateMdxBytes(
            version: 1300,
            modelName: "SyntheticBone",
            blendTime: 0,
            boundsMin: new Vector3(-1.0f, -1.0f, -1.0f),
            boundsMax: new Vector3(1.0f, 1.0f, 1.0f),
            sequences: [],
            pivotPoints: [],
            textures: [],
            materials: [],
            extraChunks:
            [
                CreateChunk("BONE", CreateBonePayload(
                [
                    ("Root", 0, -1, 0x80u, 0u, 1u, (1u, -1, new[] { 10, 40 }), (0u, -1, new[] { 20 }), null),
                    ("Child", 1, 0, 0x81u, uint.MaxValue, uint.MaxValue, null, null, (3u, 7, new[] { 15 })),
                ])),
            ]);

        using MemoryStream stream = new(bytes);
        MdxSummary summary = MdxSummaryReader.Read(stream, "synthetic_bone.mdx");

        Assert.Equal(2, summary.BoneCount);
        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Bone);

        MdxBoneSummary root = summary.Bones[0];
        Assert.Equal("Root", root.Name);
        Assert.Equal(0, root.ObjectId);
        Assert.False(root.HasParent);
        Assert.Equal(0x80u, root.Flags);
        Assert.True(root.UsesGeoset);
        Assert.Equal(0u, root.GeosetId);
        Assert.True(root.UsesGeosetAnimation);
        Assert.Equal(1u, root.GeosetAnimationId);
        Assert.NotNull(root.TranslationTrack);
        Assert.Equal("KGTR", root.TranslationTrack!.Tag);
        Assert.Equal(2, root.TranslationTrack.KeyCount);
        Assert.Equal(1u, root.TranslationTrack.InterpolationType);
        Assert.Equal(-1, root.TranslationTrack.GlobalSequenceId);
        Assert.Equal(10, root.TranslationTrack.FirstKeyTime);
        Assert.Equal(40, root.TranslationTrack.LastKeyTime);
        Assert.NotNull(root.RotationTrack);
        Assert.Equal("KGRT", root.RotationTrack!.Tag);
        Assert.Equal(1, root.RotationTrack.KeyCount);
        Assert.Equal(20, root.RotationTrack.FirstKeyTime);
        Assert.Equal(20, root.RotationTrack.LastKeyTime);
        Assert.Null(root.ScalingTrack);

        MdxBoneSummary child = summary.Bones[1];
        Assert.Equal("Child", child.Name);
        Assert.Equal(1, child.ObjectId);
        Assert.True(child.HasParent);
        Assert.Equal(0, child.ParentId);
        Assert.Equal(0x81u, child.Flags);
        Assert.False(child.UsesGeoset);
        Assert.False(child.UsesGeosetAnimation);
        Assert.Null(child.TranslationTrack);
        Assert.Null(child.RotationTrack);
        Assert.NotNull(child.ScalingTrack);
        Assert.Equal("KGSC", child.ScalingTrack!.Tag);
        Assert.Equal(1, child.ScalingTrack.KeyCount);
        Assert.Equal(3u, child.ScalingTrack.InterpolationType);
        Assert.Equal(7, child.ScalingTrack.GlobalSequenceId);
        Assert.Equal(15, child.ScalingTrack.FirstKeyTime);
        Assert.Equal(15, child.ScalingTrack.LastKeyTime);
    }

    [Fact]
    public void Read_SyntheticClassicHelp_ProducesExpectedHelperSummary()
    {
        byte[] bytes = CreateMdxBytes(
            version: 1300,
            modelName: "SyntheticHelp",
            blendTime: 0,
            boundsMin: new Vector3(-1.0f, -1.0f, -1.0f),
            boundsMax: new Vector3(1.0f, 1.0f, 1.0f),
            sequences: [],
            pivotPoints: [],
            textures: [],
            materials: [],
            extraChunks:
            [
                CreateChunk("HELP", CreateHelpPayload(
                [
                    ("HelperRoot", 6, -1, 0x20u, null, (0u, -1, new[] { 20 }), null),
                    ("HelperChild", 7, 6, 0x40u, (1u, 0, new[] { 10, 40 }), null, (3u, 7, new[] { 15 })),
                ])),
            ]);

        using MemoryStream stream = new(bytes);
        MdxSummary summary = MdxSummaryReader.Read(stream, "synthetic_help.mdx");

        Assert.Equal(2, summary.HelperCount);
        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Help);

        MdxHelperSummary root = summary.Helpers[0];
        Assert.Equal("HelperRoot", root.Name);
        Assert.Equal(6, root.ObjectId);
        Assert.False(root.HasParent);
        Assert.Equal(0x20u, root.Flags);
        Assert.Null(root.TranslationTrack);
        Assert.NotNull(root.RotationTrack);
        Assert.Equal("KGRT", root.RotationTrack!.Tag);
        Assert.Equal(1, root.RotationTrack.KeyCount);
        Assert.Equal(20, root.RotationTrack.FirstKeyTime);
        Assert.Equal(20, root.RotationTrack.LastKeyTime);
        Assert.Null(root.ScalingTrack);

        MdxHelperSummary child = summary.Helpers[1];
        Assert.Equal("HelperChild", child.Name);
        Assert.Equal(7, child.ObjectId);
        Assert.True(child.HasParent);
        Assert.Equal(6, child.ParentId);
        Assert.Equal(0x40u, child.Flags);
        Assert.NotNull(child.TranslationTrack);
        Assert.Equal("KGTR", child.TranslationTrack!.Tag);
        Assert.Equal(2, child.TranslationTrack.KeyCount);
        Assert.Equal(10, child.TranslationTrack.FirstKeyTime);
        Assert.Equal(40, child.TranslationTrack.LastKeyTime);
        Assert.Null(child.RotationTrack);
        Assert.NotNull(child.ScalingTrack);
        Assert.Equal("KGSC", child.ScalingTrack!.Tag);
        Assert.Equal(1, child.ScalingTrack.KeyCount);
        Assert.Equal(15, child.ScalingTrack.FirstKeyTime);
        Assert.Equal(15, child.ScalingTrack.LastKeyTime);
    }

    [Fact]
    public void Read_SyntheticClassicLite_ProducesExpectedLightSummary()
    {
        byte[] bytes = CreateMdxBytes(
            version: 1300,
            modelName: "SyntheticLite",
            blendTime: 0,
            boundsMin: new Vector3(-1.0f, -1.0f, -1.0f),
            boundsMax: new Vector3(1.0f, 1.0f, 1.0f),
            sequences: [],
            pivotPoints: [],
            textures: [],
            materials: [],
            extraChunks:
            [
                CreateChunk("LITE", CreateLightPayload(
                [
                    ("OmniLight", 8, 2, 0x100u, MdxLightType.Omni, 4.0f, 12.0f, new Vector3(1.0f, 0.5f, 0.25f), 2.0f, new Vector3(0.25f, 0.5f, 1.0f), 0.75f, (1u, -1, new[] { 10, 40 }), null, null, (0u, -1, new[] { 20 }), (1u, 3, new[] { 25, 45 }), null, (2u, 5, new[] { 30 }), (3u, 7, new[] { 35 }), null, (0u, -1, new[] { 50 })),
                    ("AmbientLight", 9, -1, 0x101u, MdxLightType.Ambient, 0.0f, 0.0f, new Vector3(0.1f, 0.2f, 0.3f), 1.0f, new Vector3(0.4f, 0.5f, 0.6f), 1.5f, null, (0u, -1, new[] { 15 }), (3u, 8, new[] { 25 }), null, null, (1u, 4, new[] { 55, 90 }), null, null, (2u, 6, new[] { 65 }), (1u, 2, new[] { 75 })),
                ])),
            ]);

        using MemoryStream stream = new(bytes);
        MdxSummary summary = MdxSummaryReader.Read(stream, "synthetic_lite.mdx");

        Assert.Equal(2, summary.LightCount);
        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Lite);

        MdxLightSummary first = summary.Lights[0];
        Assert.Equal("OmniLight", first.Name);
        Assert.Equal(8, first.ObjectId);
        Assert.True(first.HasParent);
        Assert.Equal(2, first.ParentId);
        Assert.Equal(0x100u, first.Flags);
        Assert.Equal(MdxLightType.Omni, first.LightType);
        Assert.Equal(4.0f, first.StaticAttenuationStart);
        Assert.Equal(12.0f, first.StaticAttenuationEnd);
        Assert.Equal(new Vector3(1.0f, 0.5f, 0.25f), first.StaticColor);
        Assert.Equal(2.0f, first.StaticIntensity);
        Assert.Equal(new Vector3(0.25f, 0.5f, 1.0f), first.StaticAmbientColor);
        Assert.Equal(0.75f, first.StaticAmbientIntensity);
        Assert.NotNull(first.TranslationTrack);
        Assert.Equal("KGTR", first.TranslationTrack!.Tag);
        Assert.Null(first.RotationTrack);
        Assert.Null(first.ScalingTrack);
        Assert.NotNull(first.AttenuationStartTrack);
        Assert.Equal("KLAS", first.AttenuationStartTrack!.Tag);
        Assert.NotNull(first.AttenuationEndTrack);
        Assert.Equal("KLAE", first.AttenuationEndTrack!.Tag);
        Assert.Null(first.ColorTrack);
        Assert.NotNull(first.IntensityTrack);
        Assert.Equal("KLAI", first.IntensityTrack!.Tag);
        Assert.NotNull(first.AmbientColorTrack);
        Assert.Equal("KLBC", first.AmbientColorTrack!.Tag);
        Assert.Null(first.AmbientIntensityTrack);
        Assert.NotNull(first.VisibilityTrack);
        Assert.Equal("KVIS", first.VisibilityTrack!.Tag);

        MdxLightSummary second = summary.Lights[1];
        Assert.Equal("AmbientLight", second.Name);
        Assert.False(second.HasParent);
        Assert.Equal(0x101u, second.Flags);
        Assert.Equal(MdxLightType.Ambient, second.LightType);
        Assert.Null(second.TranslationTrack);
        Assert.NotNull(second.RotationTrack);
        Assert.Equal("KGRT", second.RotationTrack!.Tag);
        Assert.NotNull(second.ScalingTrack);
        Assert.Equal("KGSC", second.ScalingTrack!.Tag);
        Assert.Null(second.AttenuationStartTrack);
        Assert.Null(second.AttenuationEndTrack);
        Assert.NotNull(second.ColorTrack);
        Assert.Equal("KLAC", second.ColorTrack!.Tag);
        Assert.Null(second.IntensityTrack);
        Assert.Null(second.AmbientColorTrack);
        Assert.NotNull(second.AmbientIntensityTrack);
        Assert.Equal("KLBI", second.AmbientIntensityTrack!.Tag);
        Assert.NotNull(second.VisibilityTrack);
        Assert.Equal("KVIS", second.VisibilityTrack!.Tag);
    }

    [Fact]
    public void Read_RealStandardArchiveMdx_WithLite_ProducesExpectedFixedSignals()
    {
        if (!Directory.Exists(MdxTestPaths.Standard060DataPath) || !File.Exists(MdxTestPaths.ListfilePath))
            return;

        using IArchiveCatalog catalog = new MpqArchiveCatalog();
        ArchiveCatalogBootstrapResult bootstrap = ArchiveCatalogBootstrapper.Bootstrap(catalog, [MdxTestPaths.Standard060DataPath], MdxTestPaths.ListfilePath);
        Assert.NotNull(bootstrap);

        if (!catalog.FileExists(MdxTestPaths.Standard060LightMdxVirtualPath))
            return;

        byte[]? bytes = catalog.ReadFile(MdxTestPaths.Standard060LightMdxVirtualPath);
        Assert.NotNull(bytes);

        using MemoryStream summaryStream = new(bytes);
        MdxSummary summary = MdxSummaryReader.Read(summaryStream, MdxTestPaths.Standard060LightMdxVirtualPath);

        Assert.Equal("DwarvenBrazier01", summary.ModelName);
        Assert.Equal(1, summary.LightCount);
        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Lite);

        MdxLightSummary light = summary.Lights[0];
        Assert.Equal("Omni02", light.Name);
        Assert.Equal(1, light.ObjectId);
        Assert.True(light.HasParent);
        Assert.Equal(0, light.ParentId);
        Assert.Equal(0x00000100u, light.Flags);
        Assert.Equal(MdxLightType.Omni, light.LightType);
        Assert.Equal(0.8333333f, light.StaticAttenuationStart, 6);
        Assert.Equal(0.9722222f, light.StaticAttenuationEnd, 6);
        Assert.Equal(0.97647065f, light.StaticColor.X, 6);
        Assert.Equal(0.77647066f, light.StaticColor.Y, 6);
        Assert.Equal(0.23529413f, light.StaticColor.Z, 6);
        Assert.Equal(0f, light.StaticIntensity, 6);
        Assert.Equal(1f, light.StaticAmbientColor.X, 6);
        Assert.Equal(1f, light.StaticAmbientColor.Y, 6);
        Assert.Equal(1f, light.StaticAmbientColor.Z, 6);
        Assert.Equal(0f, light.StaticAmbientIntensity, 6);
        Assert.Null(light.TranslationTrack);
        Assert.Null(light.RotationTrack);
        Assert.Null(light.ScalingTrack);
        Assert.Null(light.AttenuationStartTrack);
        Assert.Null(light.AttenuationEndTrack);
        Assert.Null(light.ColorTrack);
        Assert.NotNull(light.IntensityTrack);
        Assert.Equal("KLAI", light.IntensityTrack!.Tag);
        Assert.Equal(26, light.IntensityTrack.KeyCount);
        Assert.Equal(3u, light.IntensityTrack.InterpolationType);
        Assert.Equal(-1, light.IntensityTrack.GlobalSequenceId);
        Assert.Equal(0, light.IntensityTrack.FirstKeyTime);
        Assert.Equal(3333, light.IntensityTrack.LastKeyTime);
        Assert.Null(light.AmbientColorTrack);
        Assert.Null(light.AmbientIntensityTrack);
        Assert.Null(light.VisibilityTrack);
    }

    [Fact]
    public void Read_RealAlpha053Corpus_CurrentSample_ParsesAndReportsNoLite()
    {
        if (!Directory.Exists(MdxTestPaths.Alpha053TreePath))
            return;

        string[] inputPaths = Directory
            .EnumerateFiles(MdxTestPaths.Alpha053TreePath, "*.mdx", SearchOption.AllDirectories)
            .Concat(Directory.EnumerateFiles(MdxTestPaths.Alpha053TreePath, "*.MDX", SearchOption.AllDirectories))
            .OrderBy(static path => path, StringComparer.OrdinalIgnoreCase)
            .ToArray();

        Assert.NotEmpty(inputPaths);

        foreach (string inputPath in inputPaths)
        {
            MdxSummary summary = MdxSummaryReader.Read(inputPath);

            Assert.Equal("MDLX", summary.Signature);
            Assert.True(summary.Version is 1300u or 1400u, $"Unexpected MDX version {summary.Version} for {inputPath}");
            Assert.True(summary.LightCount == 0, $"Expected no lights in current 0.5.3 sample file {inputPath}");
            Assert.DoesNotContain(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Lite);
        }
    }

    [Fact]
    public void Read_SyntheticClassicAtch_ProducesExpectedAttachmentSummary()
    {
        byte[] bytes = CreateMdxBytes(
            version: 1300,
            modelName: "SyntheticAtch",
            blendTime: 0,
            boundsMin: new Vector3(-1.0f, -1.0f, -1.0f),
            boundsMax: new Vector3(1.0f, 1.0f, 1.0f),
            sequences: [],
            pivotPoints: [],
            textures: [],
            materials: [],
            extraChunks:
            [
                CreateChunk("ATCH", CreateAttachmentPayload(
                [
                    ("HandRight", 10, 6, 0x400u, 1u, "Abilities\\Weapons\\Sword.mdx", (1u, 3, new[] { 10, 40 }), null, null, (0u, -1, new[] { 20 })),
                    ("Shield", 11, -1, 0x401u, 0u, string.Empty, null, (2u, 5, new[] { 15, 45 }), (3u, 7, new[] { 25 }), null),
                ])),
            ]);

        using MemoryStream stream = new(bytes);
        MdxSummary summary = MdxSummaryReader.Read(stream, "synthetic_atch.mdx");

        Assert.Equal(2, summary.AttachmentCount);
        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Atch);

        MdxAttachmentSummary rightHand = summary.Attachments[0];
        Assert.Equal("HandRight", rightHand.Name);
        Assert.Equal(10, rightHand.ObjectId);
        Assert.True(rightHand.HasParent);
        Assert.Equal(6, rightHand.ParentId);
        Assert.Equal(0x400u, rightHand.Flags);
        Assert.Equal(1u, rightHand.AttachmentId);
        Assert.Equal("Abilities\\Weapons\\Sword.mdx", rightHand.Path);
        Assert.NotNull(rightHand.TranslationTrack);
        Assert.Equal("KGTR", rightHand.TranslationTrack!.Tag);
        Assert.Equal(2, rightHand.TranslationTrack.KeyCount);
        Assert.Equal(10, rightHand.TranslationTrack.FirstKeyTime);
        Assert.Equal(40, rightHand.TranslationTrack.LastKeyTime);
        Assert.Null(rightHand.RotationTrack);
        Assert.Null(rightHand.ScalingTrack);
        Assert.NotNull(rightHand.VisibilityTrack);
        Assert.Equal("KVIS", rightHand.VisibilityTrack!.Tag);
        Assert.Equal(1, rightHand.VisibilityTrack.KeyCount);
        Assert.Equal(20, rightHand.VisibilityTrack.FirstKeyTime);
        Assert.Equal(20, rightHand.VisibilityTrack.LastKeyTime);

        MdxAttachmentSummary shield = summary.Attachments[1];
        Assert.Equal("Shield", shield.Name);
        Assert.Equal(11, shield.ObjectId);
        Assert.False(shield.HasParent);
        Assert.Equal(0x401u, shield.Flags);
        Assert.Equal(0u, shield.AttachmentId);
        Assert.Null(shield.Path);
        Assert.Null(shield.TranslationTrack);
        Assert.NotNull(shield.RotationTrack);
        Assert.Equal("KGRT", shield.RotationTrack!.Tag);
        Assert.Equal(2, shield.RotationTrack.KeyCount);
        Assert.Equal(15, shield.RotationTrack.FirstKeyTime);
        Assert.Equal(45, shield.RotationTrack.LastKeyTime);
        Assert.NotNull(shield.ScalingTrack);
        Assert.Equal("KGSC", shield.ScalingTrack!.Tag);
        Assert.Equal(1, shield.ScalingTrack.KeyCount);
        Assert.Equal(25, shield.ScalingTrack.FirstKeyTime);
        Assert.Equal(25, shield.ScalingTrack.LastKeyTime);
        Assert.Null(shield.VisibilityTrack);
    }

    [Fact]
    public void Read_SyntheticClassicPre2_ProducesExpectedParticleEmitterSummary()
    {
        byte[] bytes = CreateMdxBytes(
            version: 1300,
            modelName: "SyntheticPre2",
            blendTime: 0,
            boundsMin: new Vector3(-1.0f, -1.0f, -1.0f),
            boundsMax: new Vector3(1.0f, 1.0f, 1.0f),
            sequences: [],
            pivotPoints: [],
            textures: [],
            materials: [],
            extraChunks:
            [
                CreateChunk("PRE2", CreatePre2Payload(
                [
                    ("ParticleA", 14, 6, 0x0200u, 1, 2.5f, 0.25f, 1.5f, 3.0f, -0.5f, 0.75f, 4.0f, 8.0f, 0.125f, 0.5f, 2u, 3u, 1u, 1.25f, 0.4f, new Vector3(1.0f, 0.5f, 0.25f), new Vector3(0.75f, 0.5f, 0.25f), new Vector3(0.5f, 0.25f, 0.0f), (byte)10, (byte)20, (byte)30, 0.1f, 0.2f, 0.3f, 3u, 4, 2, 5u, "Objects\\ParticleA.mdl", string.Empty, [], 1, (1u, -1, new[] { 10, 40 }), null, null, (0u, -1, new[] { 20 }), (1u, 7, new[] { 30, 60 }), null, (2u, 5, new[] { 35 }), null, null, (0u, -1, new[] { 50 }), (3u, 9, new[] { 70 }), null, (1u, 4, new[] { 80, 100 }), null),
                    ("ParticleB", 15, -1, 0x0201u, 2, 6.0f, 0.0f, 0.25f, 0.5f, 1.0f, 0.0f, 2.0f, 12.0f, 3.0f, 4.0f, 1u, 1u, 2u, 0.75f, 0.6f, new Vector3(0.25f, 0.5f, 1.0f), new Vector3(0.5f, 0.75f, 1.0f), new Vector3(1.0f, 1.0f, 1.0f), byte.MaxValue, (byte)128, (byte)0, 1.0f, 1.5f, 2.0f, 1u, -1, 0, 0u, string.Empty, "Rec\\ParticleB.mdl", [new Vector3(1.0f, 2.0f, 3.0f)], 0, null, (0u, -1, new[] { 15 }), (3u, 8, new[] { 25 }), null, null, (1u, 3, new[] { 45, 90 }), null, (0u, -1, new[] { 55 }), (2u, 6, new[] { 65 }), null, null, (1u, 2, new[] { 75 }), null, (0u, -1, new[] { 85 })),
                ])),
            ]);

        using MemoryStream stream = new(bytes);
        MdxSummary summary = MdxSummaryReader.Read(stream, "synthetic_pre2.mdx");

        Assert.Equal(2, summary.ParticleEmitter2Count);
        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Pre2);

        MdxParticleEmitter2Summary first = summary.ParticleEmitters2[0];
        Assert.Equal("ParticleA", first.Name);
        Assert.Equal(14, first.ObjectId);
        Assert.True(first.HasParent);
        Assert.Equal(6, first.ParentId);
        Assert.Equal(0x0200u, first.Flags);
        Assert.Equal(1, first.EmitterType);
        Assert.Equal(2.5f, first.StaticSpeed);
        Assert.Equal(0.25f, first.StaticVariation);
        Assert.Equal(1.5f, first.StaticLatitude);
        Assert.Equal(3.0f, first.StaticLongitude);
        Assert.Equal(-0.5f, first.StaticGravity);
        Assert.Equal(0.75f, first.StaticZSource);
        Assert.Equal(4.0f, first.StaticLife);
        Assert.Equal(8.0f, first.StaticEmissionRate);
        Assert.Equal(0.125f, first.StaticLength);
        Assert.Equal(0.5f, first.StaticWidth);
        Assert.Equal(2u, first.Rows);
        Assert.Equal(3u, first.Columns);
        Assert.Equal(1u, first.ParticleType);
        Assert.Equal(1.25f, first.TailLength);
        Assert.Equal(0.4f, first.MiddleTime);
        Assert.Equal(new Vector3(1.0f, 0.5f, 0.25f), first.StartColor);
        Assert.Equal(new Vector3(0.75f, 0.5f, 0.25f), first.MiddleColor);
        Assert.Equal(new Vector3(0.5f, 0.25f, 0.0f), first.EndColor);
        Assert.Equal(10, first.StartAlpha);
        Assert.Equal(20, first.MiddleAlpha);
        Assert.Equal(30, first.EndAlpha);
        Assert.Equal(0.1f, first.StartScale);
        Assert.Equal(0.2f, first.MiddleScale);
        Assert.Equal(0.3f, first.EndScale);
        Assert.Equal(3u, first.BlendMode);
        Assert.Equal(4, first.TextureId);
        Assert.Equal(2, first.PriorityPlane);
        Assert.Equal(5u, first.ReplaceableId);
        Assert.Equal("Objects\\ParticleA.mdl", first.GeometryModel);
        Assert.Null(first.RecursionModel);
        Assert.Equal(0u, first.SplineCount);
        Assert.Equal(1, first.Squirts);
        Assert.NotNull(first.TranslationTrack);
        Assert.Equal("KGTR", first.TranslationTrack!.Tag);
        Assert.Null(first.RotationTrack);
        Assert.Null(first.ScalingTrack);
        Assert.NotNull(first.VisibilityTrack);
        Assert.Equal("KVIS", first.VisibilityTrack!.Tag);
        Assert.NotNull(first.SpeedTrack);
        Assert.Equal("KP2S", first.SpeedTrack!.Tag);
        Assert.NotNull(first.LatitudeTrack);
        Assert.Equal("KP2L", first.LatitudeTrack!.Tag);
        Assert.NotNull(first.LifeTrack);
        Assert.Equal("KLIF", first.LifeTrack!.Tag);
        Assert.NotNull(first.EmissionRateTrack);
        Assert.Equal("KP2E", first.EmissionRateTrack!.Tag);
        Assert.NotNull(first.LengthTrack);
        Assert.Equal("KP2N", first.LengthTrack!.Tag);
        Assert.Null(first.VariationTrack);
        Assert.Null(first.LongitudeTrack);
        Assert.Null(first.GravityTrack);
        Assert.Null(first.WidthTrack);
        Assert.Null(first.ZSourceTrack);

        MdxParticleEmitter2Summary second = summary.ParticleEmitters2[1];
        Assert.Equal("ParticleB", second.Name);
        Assert.False(second.HasParent);
        Assert.Equal(0x0201u, second.Flags);
        Assert.Equal(2, second.EmitterType);
        Assert.Equal(1u, second.Rows);
        Assert.Equal(1u, second.Columns);
        Assert.Equal(2u, second.ParticleType);
        Assert.Equal(-1, second.TextureId);
        Assert.Equal(0, second.PriorityPlane);
        Assert.Equal(0u, second.ReplaceableId);
        Assert.Null(second.GeometryModel);
        Assert.Equal("Rec\\ParticleB.mdl", second.RecursionModel);
        Assert.Equal(1u, second.SplineCount);
        Assert.Equal(0, second.Squirts);
        Assert.Null(second.TranslationTrack);
        Assert.NotNull(second.RotationTrack);
        Assert.Equal("KGRT", second.RotationTrack!.Tag);
        Assert.NotNull(second.ScalingTrack);
        Assert.Equal("KGSC", second.ScalingTrack!.Tag);
        Assert.Null(second.VisibilityTrack);
        Assert.Null(second.SpeedTrack);
        Assert.NotNull(second.VariationTrack);
        Assert.Equal("KP2R", second.VariationTrack!.Tag);
        Assert.NotNull(second.LongitudeTrack);
        Assert.Equal("KPLN", second.LongitudeTrack!.Tag);
        Assert.NotNull(second.GravityTrack);
        Assert.Equal("KP2G", second.GravityTrack!.Tag);
        Assert.NotNull(second.WidthTrack);
        Assert.Equal("KP2W", second.WidthTrack!.Tag);
        Assert.NotNull(second.ZSourceTrack);
        Assert.Equal("KP2Z", second.ZSourceTrack!.Tag);
    }

    [Fact]
    public void Read_RealAlpha053WispMdx_WithPre2_ProducesExpectedFixedSignals()
    {
        if (!File.Exists(MdxTestPaths.Alpha053WispMdxPath))
            return;

        MdxSummary summary = MdxSummaryReader.Read(MdxTestPaths.Alpha053WispMdxPath);

        Assert.Equal("Wisp", summary.ModelName);
        Assert.Equal(11, summary.ParticleEmitter2Count);
        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Pre2);

        MdxParticleEmitter2Summary firstEmitter = summary.ParticleEmitters2[0];
        Assert.Equal("BlizParticle01", firstEmitter.Name);
        Assert.Equal(34, firstEmitter.ObjectId);
        Assert.True(firstEmitter.HasParent);
        Assert.Equal(16, firstEmitter.ParentId);
        Assert.Equal(0x020C8800u, firstEmitter.Flags);
        Assert.Equal(1, firstEmitter.EmitterType);
        Assert.Equal(2.5f, firstEmitter.StaticSpeed, 6);
        Assert.Equal(0f, firstEmitter.StaticVariation, 6);
        Assert.Equal(3.1415927f, firstEmitter.StaticLatitude, 6);
        Assert.Equal(6.2831855f, firstEmitter.StaticLongitude, 6);
        Assert.Equal(0f, firstEmitter.StaticGravity, 6);
        Assert.Equal(0f, firstEmitter.StaticZSource, 6);
        Assert.Equal(0.4f, firstEmitter.StaticLife, 6);
        Assert.Equal(100f, firstEmitter.StaticEmissionRate, 6);
        Assert.Equal(0f, firstEmitter.StaticLength, 6);
        Assert.Equal(0f, firstEmitter.StaticWidth, 6);
        Assert.Equal(1u, firstEmitter.Rows);
        Assert.Equal(1u, firstEmitter.Columns);
        Assert.Equal(1u, firstEmitter.ParticleType);
        Assert.Equal(3f, firstEmitter.TailLength, 6);
        Assert.Equal(0.5f, firstEmitter.MiddleTime, 6);
        Assert.Equal(1f, firstEmitter.StartColor.X, 6);
        Assert.Equal(1f, firstEmitter.StartColor.Y, 6);
        Assert.Equal(1f, firstEmitter.StartColor.Z, 6);
        Assert.Equal(0.7372549f, firstEmitter.MiddleColor.X, 6);
        Assert.Equal(0.8392158f, firstEmitter.MiddleColor.Y, 6);
        Assert.Equal(0.9686275f, firstEmitter.MiddleColor.Z, 6);
        Assert.Equal(0.5254902f, firstEmitter.EndColor.X, 6);
        Assert.Equal(0.7098039f, firstEmitter.EndColor.Y, 6);
        Assert.Equal(0.94117653f, firstEmitter.EndColor.Z, 6);
        Assert.Equal(100, firstEmitter.StartAlpha);
        Assert.Equal(175, firstEmitter.MiddleAlpha);
        Assert.Equal(0, firstEmitter.EndAlpha);
        Assert.Equal(0.02777778f, firstEmitter.StartScale, 6);
        Assert.Equal(0.2222222f, firstEmitter.MiddleScale, 6);
        Assert.Equal(0.01388889f, firstEmitter.EndScale, 6);
        Assert.Equal(1u, firstEmitter.BlendMode);
        Assert.Equal(2, firstEmitter.TextureId);
        Assert.Equal(1, firstEmitter.PriorityPlane);
        Assert.Equal(2u, firstEmitter.ReplaceableId);
        Assert.Null(firstEmitter.GeometryModel);
        Assert.Null(firstEmitter.RecursionModel);
        Assert.Equal(0u, firstEmitter.SplineCount);
        Assert.Equal(0, firstEmitter.Squirts);
        Assert.Null(firstEmitter.TranslationTrack);
        Assert.NotNull(firstEmitter.RotationTrack);
        Assert.Equal("KGRT", firstEmitter.RotationTrack!.Tag);
        Assert.Equal(1, firstEmitter.RotationTrack.KeyCount);
        Assert.Equal(2u, firstEmitter.RotationTrack.InterpolationType);
        Assert.Equal(-1, firstEmitter.RotationTrack.GlobalSequenceId);
        Assert.Equal(33, firstEmitter.RotationTrack.FirstKeyTime);
        Assert.Equal(33, firstEmitter.RotationTrack.LastKeyTime);
        Assert.Null(firstEmitter.ScalingTrack);
        Assert.NotNull(firstEmitter.VisibilityTrack);
        Assert.Equal("KVIS", firstEmitter.VisibilityTrack!.Tag);
        Assert.Equal(2, firstEmitter.VisibilityTrack.KeyCount);
        Assert.Equal(0u, firstEmitter.VisibilityTrack.InterpolationType);
        Assert.Equal(-1, firstEmitter.VisibilityTrack.GlobalSequenceId);
        Assert.Equal(1167, firstEmitter.VisibilityTrack.FirstKeyTime);
        Assert.Equal(1833, firstEmitter.VisibilityTrack.LastKeyTime);

        MdxParticleEmitter2Summary deathEmitter = summary.ParticleEmitters2[5];
        Assert.Equal("BlizParticleBlackDeath", deathEmitter.Name);
        Assert.Equal(39, deathEmitter.ObjectId);
        Assert.Equal(16, deathEmitter.ParentId);
        Assert.NotNull(deathEmitter.SpeedTrack);
        Assert.Equal("KP2S", deathEmitter.SpeedTrack!.Tag);
        Assert.Equal(5, deathEmitter.SpeedTrack.KeyCount);
        Assert.Equal(3u, deathEmitter.SpeedTrack.InterpolationType);
        Assert.Equal(-1, deathEmitter.SpeedTrack.GlobalSequenceId);
        Assert.Equal(0, deathEmitter.SpeedTrack.FirstKeyTime);
        Assert.Equal(3333, deathEmitter.SpeedTrack.LastKeyTime);
        Assert.NotNull(deathEmitter.EmissionRateTrack);
        Assert.Equal("KP2E", deathEmitter.EmissionRateTrack!.Tag);
        Assert.Equal(7, deathEmitter.EmissionRateTrack.KeyCount);
        Assert.Equal(3u, deathEmitter.EmissionRateTrack.InterpolationType);
        Assert.Equal(-1, deathEmitter.EmissionRateTrack.GlobalSequenceId);
        Assert.Equal(0, deathEmitter.EmissionRateTrack.FirstKeyTime);
        Assert.Equal(3333, deathEmitter.EmissionRateTrack.LastKeyTime);
    }

    [Fact]
    public void Read_SyntheticClassicRibb_ProducesExpectedRibbonSummary()
    {
        byte[] bytes = CreateMdxBytes(
            version: 1300,
            modelName: "SyntheticRibb",
            blendTime: 0,
            boundsMin: new Vector3(-1.0f, -1.0f, -1.0f),
            boundsMax: new Vector3(1.0f, 1.0f, 1.0f),
            sequences: [],
            pivotPoints: [],
            textures: [],
            materials: [],
            extraChunks:
            [
                CreateChunk("RIBB", CreateRibbonPayload(
                [
                    ("RibbonA", 12, 6, 0x2000u, 0.25f, 0.5f, 0.75f, new Vector3(1.0f, 0.5f, 0.25f), 1.25f, 0u, 16u, 2u, 3u, 4u, 0.125f, (1u, -1, new[] { 10, 40 }), null, null, (0u, -1, new[] { 20 }), null, (2u, 7, new[] { 30 }), (3u, 9, new[] { 32 }), (1u, 3, new[] { 35, 55 }), (0u, -1, new[] { 45 })),
                    ("RibbonB", 13, -1, 0x2001u, 0.125f, 0.375f, 1.0f, new Vector3(0.0f, 0.25f, 1.0f), 0.75f, 1u, 8u, 1u, 1u, 5u, -0.25f, null, (0u, -1, new[] { 15 }), (3u, 8, new[] { 25 }), null, (1u, 4, new[] { 50, 90 }), null, null, null, null),
                ])),
            ]);

        using MemoryStream stream = new(bytes);
        MdxSummary summary = MdxSummaryReader.Read(stream, "synthetic_ribb.mdx");

        Assert.Equal(2, summary.RibbonCount);
        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Ribb);

        MdxRibbonEmitterSummary first = summary.Ribbons[0];
        Assert.Equal("RibbonA", first.Name);
        Assert.Equal(12, first.ObjectId);
        Assert.True(first.HasParent);
        Assert.Equal(6, first.ParentId);
        Assert.Equal(0x2000u, first.Flags);
        Assert.Equal(0.25f, first.StaticHeightAbove);
        Assert.Equal(0.5f, first.StaticHeightBelow);
        Assert.Equal(0.75f, first.StaticAlpha);
        Assert.Equal(new Vector3(1.0f, 0.5f, 0.25f), first.StaticColor);
        Assert.Equal(1.25f, first.EdgeLifetime);
        Assert.Equal(0u, first.StaticTextureSlot);
        Assert.Equal(16u, first.EdgesPerSecond);
        Assert.Equal(2u, first.TextureRows);
        Assert.Equal(3u, first.TextureColumns);
        Assert.Equal(4u, first.MaterialId);
        Assert.Equal(0.125f, first.Gravity);
        Assert.NotNull(first.TranslationTrack);
        Assert.Equal("KGTR", first.TranslationTrack!.Tag);
        Assert.Equal(2, first.TranslationTrack.KeyCount);
        Assert.NotNull(first.HeightAboveTrack);
        Assert.Equal("KRHA", first.HeightAboveTrack!.Tag);
        Assert.Equal(1, first.HeightAboveTrack.KeyCount);
        Assert.NotNull(first.AlphaTrack);
        Assert.Equal("KRAL", first.AlphaTrack!.Tag);
        Assert.Equal(1, first.AlphaTrack.KeyCount);
        Assert.NotNull(first.ColorTrack);
        Assert.Equal("KRCO", first.ColorTrack!.Tag);
        Assert.Equal(1, first.ColorTrack.KeyCount);
        Assert.NotNull(first.TextureSlotTrack);
        Assert.Equal("KRTX", first.TextureSlotTrack!.Tag);
        Assert.Equal(2, first.TextureSlotTrack.KeyCount);
        Assert.NotNull(first.VisibilityTrack);
        Assert.Equal("KVIS", first.VisibilityTrack!.Tag);
        Assert.Equal(1, first.VisibilityTrack.KeyCount);

        MdxRibbonEmitterSummary second = summary.Ribbons[1];
        Assert.Equal("RibbonB", second.Name);
        Assert.False(second.HasParent);
        Assert.Equal(0x2001u, second.Flags);
        Assert.Null(second.TranslationTrack);
        Assert.NotNull(second.RotationTrack);
        Assert.Equal("KGRT", second.RotationTrack!.Tag);
        Assert.NotNull(second.ScalingTrack);
        Assert.Equal("KGSC", second.ScalingTrack!.Tag);
        Assert.NotNull(second.HeightBelowTrack);
        Assert.Equal("KRHB", second.HeightBelowTrack!.Tag);
        Assert.Equal(2, second.HeightBelowTrack.KeyCount);
        Assert.Null(second.AlphaTrack);
        Assert.Null(second.ColorTrack);
        Assert.Null(second.TextureSlotTrack);
        Assert.Null(second.VisibilityTrack);
    }

    [Fact]
    public void Read_RealAlpha053WispMdx_WithRibb_ProducesExpectedFixedSignals()
    {
        if (!File.Exists(MdxTestPaths.Alpha053WispMdxPath))
            return;

        MdxSummary summary = MdxSummaryReader.Read(MdxTestPaths.Alpha053WispMdxPath);

        Assert.Equal("Wisp", summary.ModelName);
        Assert.Equal(3, summary.RibbonCount);
        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Ribb);

        MdxRibbonEmitterSummary firstRibbon = summary.Ribbons[0];
        Assert.Equal("BlizRibbon01", firstRibbon.Name);
        Assert.Equal(45, firstRibbon.ObjectId);
        Assert.True(firstRibbon.HasParent);
        Assert.Equal(18, firstRibbon.ParentId);
        Assert.Equal(0x2000u, firstRibbon.Flags);
        Assert.Equal(0.055555556f, firstRibbon.StaticHeightAbove, 6);
        Assert.Equal(0.055555556f, firstRibbon.StaticHeightBelow, 6);
        Assert.Equal(0.2f, firstRibbon.StaticAlpha, 6);
        Assert.Equal(0.047058824f, firstRibbon.StaticColor.X, 6);
        Assert.Equal(0.41960785f, firstRibbon.StaticColor.Y, 6);
        Assert.Equal(0.88235295f, firstRibbon.StaticColor.Z, 6);
        Assert.Equal(0.2f, firstRibbon.EdgeLifetime, 6);
        Assert.Equal(0u, firstRibbon.StaticTextureSlot);
        Assert.Equal(40u, firstRibbon.EdgesPerSecond);
        Assert.Equal(1u, firstRibbon.TextureRows);
        Assert.Equal(1u, firstRibbon.TextureColumns);
        Assert.Equal(2u, firstRibbon.MaterialId);
        Assert.Equal(0f, firstRibbon.Gravity, 6);
        Assert.Null(firstRibbon.TranslationTrack);
        Assert.Null(firstRibbon.RotationTrack);
        Assert.Null(firstRibbon.ScalingTrack);
        Assert.Null(firstRibbon.HeightAboveTrack);
        Assert.Null(firstRibbon.HeightBelowTrack);
        Assert.Null(firstRibbon.AlphaTrack);
        Assert.Null(firstRibbon.ColorTrack);
        Assert.Null(firstRibbon.TextureSlotTrack);
        Assert.NotNull(firstRibbon.VisibilityTrack);
        Assert.Equal("KVIS", firstRibbon.VisibilityTrack!.Tag);
        Assert.Equal(2, firstRibbon.VisibilityTrack.KeyCount);
        Assert.Equal(0u, firstRibbon.VisibilityTrack.InterpolationType);
        Assert.Equal(-1, firstRibbon.VisibilityTrack.GlobalSequenceId);
        Assert.Equal(1167, firstRibbon.VisibilityTrack.FirstKeyTime);
        Assert.Equal(1833, firstRibbon.VisibilityTrack.LastKeyTime);
    }

    [Fact]
    public void Read_SyntheticClassicCams_ProducesExpectedCameraSummary()
    {
        byte[] bytes = CreateMdxBytes(
            version: 1300,
            modelName: "SyntheticCams",
            blendTime: 0,
            boundsMin: new Vector3(-1.0f, -1.0f, -1.0f),
            boundsMax: new Vector3(1.0f, 1.0f, 1.0f),
            sequences: [],
            pivotPoints: [],
            textures: [],
            materials: [],
            extraChunks:
            [
                CreateChunk("CAMS", CreateCameraPayload(
                [
                    ("Portrait", new Vector3(1.0f, 2.0f, 3.0f), 0.95f, 27.777778f, 0.22222222f, new Vector3(4.0f, 5.0f, 6.0f), (1u, -1, new[] { 10, 40 }), (3u, 7, new[] { 20 }), (0u, -1, new[] { 30, 60 }), (2u, 9, new[] { 35 })),
                    ("Paperdoll", new Vector3(-1.0f, -2.0f, -3.0f), 1.1f, 50.0f, 0.5f, new Vector3(-4.0f, -5.0f, -6.0f), null, null, null, null),
                ])),
            ]);

        using MemoryStream stream = new(bytes);
        MdxSummary summary = MdxSummaryReader.Read(stream, "synthetic_cams.mdx");

        Assert.Equal(2, summary.CameraCount);
        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Cams);

        MdxCameraSummary first = summary.Cameras[0];
        Assert.Equal("Portrait", first.Name);
        Assert.Equal(new Vector3(1.0f, 2.0f, 3.0f), first.PivotPoint);
        Assert.Equal(0.95f, first.FieldOfView);
        Assert.Equal(27.777778f, first.FarClip);
        Assert.Equal(0.22222222f, first.NearClip);
        Assert.Equal(new Vector3(4.0f, 5.0f, 6.0f), first.TargetPivotPoint);
        Assert.NotNull(first.PositionTrack);
        Assert.Equal("KCTR", first.PositionTrack!.Tag);
        Assert.Equal(2, first.PositionTrack.KeyCount);
        Assert.NotNull(first.RollTrack);
        Assert.Equal("KCRL", first.RollTrack!.Tag);
        Assert.Equal(1, first.RollTrack.KeyCount);
        Assert.NotNull(first.VisibilityTrack);
        Assert.Equal("KVIS", first.VisibilityTrack!.Tag);
        Assert.Equal(2, first.VisibilityTrack.KeyCount);
        Assert.NotNull(first.TargetPositionTrack);
        Assert.Equal("KTTR", first.TargetPositionTrack!.Tag);
        Assert.Equal(1, first.TargetPositionTrack.KeyCount);

        MdxCameraSummary second = summary.Cameras[1];
        Assert.Equal("Paperdoll", second.Name);
        Assert.Equal(new Vector3(-1.0f, -2.0f, -3.0f), second.PivotPoint);
        Assert.Equal(1.1f, second.FieldOfView);
        Assert.Equal(50.0f, second.FarClip);
        Assert.Equal(0.5f, second.NearClip);
        Assert.Equal(new Vector3(-4.0f, -5.0f, -6.0f), second.TargetPivotPoint);
        Assert.Null(second.PositionTrack);
        Assert.Null(second.RollTrack);
        Assert.Null(second.VisibilityTrack);
        Assert.Null(second.TargetPositionTrack);
    }

    [Fact]
    public void Read_RealAlpha053WispMdx_WithCams_ProducesExpectedFixedSignals()
    {
        if (!File.Exists(MdxTestPaths.Alpha053WispMdxPath))
            return;

        MdxSummary summary = MdxSummaryReader.Read(MdxTestPaths.Alpha053WispMdxPath);

        Assert.Equal("Wisp", summary.ModelName);
        Assert.Equal(1, summary.CameraCount);
        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Cams);

        MdxCameraSummary camera = summary.Cameras[0];
        Assert.Equal("Portrait", camera.Name);
        Assert.Equal(0.8041092f, camera.PivotPoint.X, 6);
        Assert.Equal(0.48125452f, camera.PivotPoint.Y, 6);
        Assert.Equal(1.89845f, camera.PivotPoint.Z, 6);
        Assert.Equal(0.9500215f, camera.FieldOfView, 6);
        Assert.Equal(27.7777786f, camera.FarClip, 6);
        Assert.Equal(0.2222222f, camera.NearClip, 6);
        Assert.Equal(0.4076503f, camera.TargetPivotPoint.X, 6);
        Assert.Equal(0.0007732428f, camera.TargetPivotPoint.Y, 6);
        Assert.Equal(1.88286f, camera.TargetPivotPoint.Z, 6);
        Assert.Null(camera.PositionTrack);
        Assert.Null(camera.RollTrack);
        Assert.Null(camera.VisibilityTrack);
        Assert.Null(camera.TargetPositionTrack);
    }

    [Fact]
    public void Read_SyntheticClassicEvts_ProducesExpectedEventSummary()
    {
        byte[] bytes = CreateMdxBytes(
            version: 1300,
            modelName: "SyntheticEvts",
            blendTime: 0,
            boundsMin: new Vector3(-1.0f, -1.0f, -1.0f),
            boundsMax: new Vector3(1.0f, 1.0f, 1.0f),
            sequences: [],
            pivotPoints: [],
            textures: [],
            materials: [],
            extraChunks:
            [
                CreateChunk("EVTS", CreateEventPayload(
                [
                    ("$SND", 12, 4, 0x200u, (1u, -1, new[] { 10, 40 }), (3u, 7, new[] { 20 }), null, (-1, new[] { 15, 45 })),
                    ("$DTH", 13, -1, 0x202u, null, null, null, null),
                ])),
            ]);

        using MemoryStream stream = new(bytes);
        MdxSummary summary = MdxSummaryReader.Read(stream, "synthetic_evts.mdx");

        Assert.Equal(2, summary.EventCount);
        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Evts);

        MdxEventSummary first = summary.Events[0];
        Assert.Equal("$SND", first.Name);
        Assert.Equal(12, first.ObjectId);
        Assert.True(first.HasParent);
        Assert.Equal(4, first.ParentId);
        Assert.Equal(0x200u, first.Flags);
        Assert.NotNull(first.TranslationTrack);
        Assert.Equal("KGTR", first.TranslationTrack!.Tag);
        Assert.Equal(2, first.TranslationTrack.KeyCount);
        Assert.NotNull(first.RotationTrack);
        Assert.Equal("KGRT", first.RotationTrack!.Tag);
        Assert.Equal(1, first.RotationTrack.KeyCount);
        Assert.Null(first.ScalingTrack);
        Assert.NotNull(first.EventTrack);
        Assert.Equal("KEVT", first.EventTrack!.Tag);
        Assert.Equal(2, first.EventTrack.KeyCount);
        Assert.Equal(-1, first.EventTrack.GlobalSequenceId);
        Assert.Equal(15, first.EventTrack.FirstKeyTime);
        Assert.Equal(45, first.EventTrack.LastKeyTime);

        MdxEventSummary second = summary.Events[1];
        Assert.Equal("$DTH", second.Name);
        Assert.Equal(13, second.ObjectId);
        Assert.False(second.HasParent);
        Assert.Equal(-1, second.ParentId);
        Assert.Equal(0x202u, second.Flags);
        Assert.Null(second.TranslationTrack);
        Assert.Null(second.RotationTrack);
        Assert.Null(second.ScalingTrack);
        Assert.Null(second.EventTrack);
    }

    [Fact]
    public void Read_RealAlpha053WispMdx_WithEvts_ProducesExpectedFixedSignals()
    {
        if (!File.Exists(MdxTestPaths.Alpha053WispMdxPath))
            return;

        MdxSummary summary = MdxSummaryReader.Read(MdxTestPaths.Alpha053WispMdxPath);

        Assert.Equal("Wisp", summary.ModelName);
        Assert.Equal(3, summary.EventCount);
        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Evts);

        MdxEventSummary first = summary.Events[0];
        Assert.Equal("$CCH", first.Name);
        Assert.Equal(48, first.ObjectId);
        Assert.Equal(16, first.ParentId);
        Assert.Equal(0x200u, first.Flags);
        Assert.Null(first.TranslationTrack);
        Assert.Null(first.RotationTrack);
        Assert.Null(first.ScalingTrack);
        Assert.Null(first.EventTrack);

        MdxEventSummary second = summary.Events[1];
        Assert.Equal("$CHD", second.Name);
        Assert.Equal(49, second.ObjectId);
        Assert.Equal(16, second.ParentId);
        Assert.Equal(0x202u, second.Flags);
        Assert.Null(second.TranslationTrack);
        Assert.Null(second.RotationTrack);
        Assert.Null(second.ScalingTrack);
        Assert.Null(second.EventTrack);

        MdxEventSummary third = summary.Events[2];
        Assert.Equal("$DTH", third.Name);
        Assert.Equal(50, third.ObjectId);
        Assert.Equal(-1, third.ParentId);
        Assert.Equal(0x200u, third.Flags);
        Assert.Null(third.TranslationTrack);
        Assert.Null(third.RotationTrack);
        Assert.Null(third.ScalingTrack);
        Assert.NotNull(third.EventTrack);
        Assert.Equal("KEVT", third.EventTrack!.Tag);
        Assert.Equal(1, third.EventTrack.KeyCount);
        Assert.Equal(-1, third.EventTrack.GlobalSequenceId);
        Assert.Equal(1667, third.EventTrack.FirstKeyTime);
        Assert.Equal(1667, third.EventTrack.LastKeyTime);
    }

    [Fact]
    public void Read_SyntheticClassicHtst_ProducesExpectedHitTestShapeSummary()
    {
        byte[] bytes = CreateMdxBytes(
            version: 1300,
            modelName: "SyntheticHtst",
            blendTime: 0,
            boundsMin: new Vector3(-1.0f, -1.0f, -1.0f),
            boundsMax: new Vector3(1.0f, 1.0f, 1.0f),
            sequences: [],
            pivotPoints: [],
            textures: [],
            materials: [],
            extraChunks:
            [
                CreateChunk("HTST", CreateHitTestShapePayload(
                [
                    ("HitBox", 21, 7, 0x1000u, MdxGeometryShapeType.Box, new Vector3(-1.0f, -2.0f, -3.0f), new Vector3(4.0f, 5.0f, 6.0f), 0.0f, 0.0f, (1u, -1, new[] { 10, 40 }), null, null),
                    ("HitCylinder", 22, 7, 0x1001u, MdxGeometryShapeType.Cylinder, new Vector3(1.5f, 2.5f, 3.5f), Vector3.Zero, 7.0f, 0.75f, null, null, null),
                    ("HitSphere", 23, -1, 0x1002u, MdxGeometryShapeType.Sphere, new Vector3(0.25f, 0.5f, 0.75f), Vector3.Zero, 1.25f, 0.0f, null, (3u, 7, new[] { 15 }), null),
                    ("HitPlane", 24, -1, 0x1003u, MdxGeometryShapeType.Plane, Vector3.Zero, Vector3.Zero, 9.0f, 4.5f, null, null, (2u, 5, new[] { 20 })),
                ])),
            ]);

        using MemoryStream stream = new(bytes);
        MdxSummary summary = MdxSummaryReader.Read(stream, "synthetic_htst.mdx");

        Assert.Equal(4, summary.HitTestShapeCount);
        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Htst);

        MdxHitTestShapeSummary box = summary.HitTestShapes[0];
        Assert.Equal("HitBox", box.Name);
        Assert.Equal(MdxGeometryShapeType.Box, box.ShapeType);
        Assert.Equal(new Vector3(-1.0f, -2.0f, -3.0f), box.Minimum);
        Assert.Equal(new Vector3(4.0f, 5.0f, 6.0f), box.Maximum);
        Assert.NotNull(box.TranslationTrack);
        Assert.Equal("KGTR", box.TranslationTrack!.Tag);
        Assert.Null(box.RotationTrack);
        Assert.Null(box.ScalingTrack);

        MdxHitTestShapeSummary cylinder = summary.HitTestShapes[1];
        Assert.Equal(MdxGeometryShapeType.Cylinder, cylinder.ShapeType);
        Assert.Equal(new Vector3(1.5f, 2.5f, 3.5f), cylinder.BasePoint);
        Assert.Equal(7.0f, cylinder.Height);
        Assert.Equal(0.75f, cylinder.Radius);

        MdxHitTestShapeSummary sphere = summary.HitTestShapes[2];
        Assert.Equal(MdxGeometryShapeType.Sphere, sphere.ShapeType);
        Assert.Equal(new Vector3(0.25f, 0.5f, 0.75f), sphere.Center);
        Assert.Equal(1.25f, sphere.Radius);
        Assert.NotNull(sphere.RotationTrack);
        Assert.Equal("KGRT", sphere.RotationTrack!.Tag);

        MdxHitTestShapeSummary plane = summary.HitTestShapes[3];
        Assert.Equal(MdxGeometryShapeType.Plane, plane.ShapeType);
        Assert.Equal(9.0f, plane.Length);
        Assert.Equal(4.5f, plane.Width);
        Assert.NotNull(plane.ScalingTrack);
        Assert.Equal("KGSC", plane.ScalingTrack!.Tag);
    }

    [Fact]
    public void Read_RealAlpha053WispMdx_WithHtst_ProducesExpectedFixedSignals()
    {
        if (!File.Exists(MdxTestPaths.Alpha053WispMdxPath))
            return;

        MdxSummary summary = MdxSummaryReader.Read(MdxTestPaths.Alpha053WispMdxPath);

        Assert.Equal("Wisp", summary.ModelName);
        Assert.Equal(1, summary.HitTestShapeCount);
        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Htst);

        MdxHitTestShapeSummary shape = summary.HitTestShapes[0];
        Assert.Equal("HIT01", shape.Name);
        Assert.Equal(51, shape.ObjectId);
        Assert.Equal(24, shape.ParentId);
        Assert.Equal(0x1002u, shape.Flags);
        Assert.Equal(MdxGeometryShapeType.Sphere, shape.ShapeType);
        Assert.Equal(0.3661754f, shape.Center!.Value.X, 6);
        Assert.Equal(0.008944444f, shape.Center!.Value.Y, 6);
        Assert.Equal(1.889694f, shape.Center!.Value.Z, 6);
        Assert.Equal(0.8333333f, shape.Radius!.Value, 6);
        Assert.Null(shape.Minimum);
        Assert.Null(shape.Maximum);
        Assert.Null(shape.BasePoint);
        Assert.Null(shape.Height);
        Assert.Null(shape.Length);
        Assert.Null(shape.Width);
        Assert.Null(shape.TranslationTrack);
        Assert.Null(shape.RotationTrack);
        Assert.Null(shape.ScalingTrack);
    }

    [Fact]
    public void Read_SyntheticClassicClid_ProducesExpectedCollisionSummary()
    {
        byte[] bytes = CreateMdxBytes(
            version: 1300,
            modelName: "SyntheticClid",
            blendTime: 0,
            boundsMin: new Vector3(-1.0f, -1.0f, -1.0f),
            boundsMax: new Vector3(1.0f, 1.0f, 1.0f),
            sequences: [],
            pivotPoints: [],
            textures: [],
            materials: [],
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
        MdxSummary summary = MdxSummaryReader.Read(stream, "synthetic_clid.mdx");

        Assert.NotNull(summary.Collision);
        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Clid);

        MdxCollisionSummary collision = summary.Collision!;
        Assert.Equal(4, collision.VertexCount);
        Assert.Equal(6, collision.TriangleIndexCount);
        Assert.Equal(2, collision.TriangleCount);
        Assert.Equal(2, collision.FacetNormalCount);
        Assert.Equal(3, collision.MaxTriangleIndex);
        Assert.Equal(new Vector3(-1.0f, -2.0f, -3.0f), collision.BoundsMin);
        Assert.Equal(new Vector3(4.0f, 5.0f, 6.0f), collision.BoundsMax);
    }

    [Fact]
    public void Read_SyntheticClassicClid_WithNonTriangleMultipleIndexCount_ThrowsInvalidDataException()
    {
        byte[] bytes = CreateMdxBytes(
            version: 1300,
            modelName: "SyntheticClidBadTriCount",
            blendTime: 0,
            boundsMin: new Vector3(-1.0f, -1.0f, -1.0f),
            boundsMax: new Vector3(1.0f, 1.0f, 1.0f),
            sequences: [],
            pivotPoints: [],
            textures: [],
            materials: [],
            extraChunks:
            [
                CreateChunk("CLID", CreateCollisionPayload(
                    [
                        new Vector3(0.0f, 0.0f, 0.0f),
                        new Vector3(1.0f, 0.0f, 0.0f),
                        new Vector3(0.0f, 1.0f, 0.0f),
                    ],
                    [0, 1, 2, 0],
                    [new Vector3(0.0f, 0.0f, 1.0f)])),
            ]);

        using MemoryStream stream = new(bytes);

        InvalidDataException exception = Assert.Throws<InvalidDataException>(() => MdxSummaryReader.Read(stream, "synthetic_clid_bad_tri_count.mdx"));
        Assert.Contains("TRI count must be divisible by 3", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Read_SyntheticClassicClid_WithOutOfRangeTriangleIndex_ThrowsInvalidDataException()
    {
        byte[] bytes = CreateMdxBytes(
            version: 1300,
            modelName: "SyntheticClidBadIndex",
            blendTime: 0,
            boundsMin: new Vector3(-1.0f, -1.0f, -1.0f),
            boundsMax: new Vector3(1.0f, 1.0f, 1.0f),
            sequences: [],
            pivotPoints: [],
            textures: [],
            materials: [],
            extraChunks:
            [
                CreateChunk("CLID", CreateCollisionPayload(
                    [
                        new Vector3(0.0f, 0.0f, 0.0f),
                        new Vector3(1.0f, 0.0f, 0.0f),
                        new Vector3(0.0f, 1.0f, 0.0f),
                    ],
                    [0, 1, 3],
                    [new Vector3(0.0f, 0.0f, 1.0f)])),
            ]);

        using MemoryStream stream = new(bytes);

        InvalidDataException exception = Assert.Throws<InvalidDataException>(() => MdxSummaryReader.Read(stream, "synthetic_clid_bad_index.mdx"));
        Assert.Contains("TRI index 3 exceeded VRTX count 3", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Read_RealAlpha053WispMdx_WithClid_ProducesExpectedFixedSignals()
    {
        if (!File.Exists(MdxTestPaths.Alpha053WispMdxPath))
            return;

        MdxSummary summary = MdxSummaryReader.Read(MdxTestPaths.Alpha053WispMdxPath);

        Assert.Equal("Wisp", summary.ModelName);
        Assert.NotNull(summary.Collision);
        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Clid);

        MdxCollisionSummary collision = summary.Collision!;
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
    }

    [Fact]
    public void Read_RealAlpha053WispMdx_WithGlbs_ProducesExpectedFixedSignals()
    {
        if (!File.Exists(MdxTestPaths.Alpha053WispMdxPath))
            return;

        MdxSummary summary = MdxSummaryReader.Read(MdxTestPaths.Alpha053WispMdxPath);

        Assert.Equal("Wisp", summary.ModelName);
        Assert.Equal(11, summary.GlobalSequenceCount);
        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Glbs);

        uint[] expectedDurations = [267u, 133u, 533u, 0u, 567u, 900u, 1167u, 667u, 467u, 933u, 300u];
        Assert.Equal(expectedDurations.Length, summary.GlobalSequences.Count);
        for (int index = 0; index < expectedDurations.Length; index++)
        {
            Assert.Equal(index, summary.GlobalSequences[index].Index);
            Assert.Equal(expectedDurations[index], summary.GlobalSequences[index].Duration);
        }
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

    [Fact]
    public void Read_RealStandardArchiveAnimatedMdx_ProducesSequenceSignals()
    {
        if (!Directory.Exists(MdxTestPaths.Standard060DataPath) || !File.Exists(MdxTestPaths.ListfilePath))
            return;

        using IArchiveCatalog catalog = new MpqArchiveCatalog();
        ArchiveCatalogBootstrapResult bootstrap = ArchiveCatalogBootstrapper.Bootstrap(catalog, [MdxTestPaths.Standard060DataPath], MdxTestPaths.ListfilePath);
        Assert.NotNull(bootstrap);

        string? virtualPath = MdxTestPaths.Standard060AnimatedMdxCandidates.FirstOrDefault(catalog.FileExists);
        if (virtualPath is null)
            return;

        byte[]? bytes = catalog.ReadFile(virtualPath);
        Assert.NotNull(bytes);

        using MemoryStream summaryStream = new(bytes);
        MdxSummary summary = MdxSummaryReader.Read(summaryStream, virtualPath);

        if (summary.SequenceCount == 0)
            return;

        Assert.All(summary.Sequences, static sequence => Assert.False(string.IsNullOrWhiteSpace(sequence.Name)));
        Assert.All(summary.Sequences, static sequence => Assert.True(sequence.EndTime >= sequence.StartTime));
        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Seqs);
    }

    [Fact]
    public void Read_RealStandardArchiveMdx_WithGeos_ProducesGeosetSignals()
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

        using MemoryStream summaryStream = new(bytes);
        MdxSummary summary = MdxSummaryReader.Read(summaryStream, MdxTestPaths.Standard060MdxVirtualPath);

        Assert.Equal(2, summary.GeosetCount);
        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Geos);
        Assert.Equal(6, summary.Geosets[0].IndexCount);
        Assert.Equal(102, summary.Geosets[1].IndexCount);
        Assert.Equal(1, summary.Geosets[0].MaterialId);
        Assert.Equal(0, summary.Geosets[1].MaterialId);
        Assert.Equal(4, summary.Geosets[1].AnimationExtentCount);
    }

    [Fact]
    public void Read_RealStandardArchiveMdx_WithPivt_ProducesPivotSignals()
    {
        if (!Directory.Exists(MdxTestPaths.Standard060DataPath) || !File.Exists(MdxTestPaths.ListfilePath))
            return;

        using IArchiveCatalog catalog = new MpqArchiveCatalog();
        ArchiveCatalogBootstrapResult bootstrap = ArchiveCatalogBootstrapper.Bootstrap(catalog, [MdxTestPaths.Standard060DataPath], MdxTestPaths.ListfilePath);
        Assert.NotNull(bootstrap);

        string? virtualPath = MdxTestPaths.Standard060PivotMdxCandidates.FirstOrDefault(catalog.FileExists);
        if (virtualPath is null)
            return;

        byte[]? bytes = catalog.ReadFile(virtualPath);
        Assert.NotNull(bytes);

        using MemoryStream summaryStream = new(bytes);
        MdxSummary summary = MdxSummaryReader.Read(summaryStream, virtualPath);

        if (summary.PivotPointCount == 0)
            return;

        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Pivt);
        Assert.All(summary.PivotPoints, static pivot =>
        {
            Assert.True(float.IsFinite(pivot.Position.X));
            Assert.True(float.IsFinite(pivot.Position.Y));
            Assert.True(float.IsFinite(pivot.Position.Z));
        });
    }

    [Fact]
    public void Read_RealStandardArchiveMdx_WithGeoa_ProducesGeosetAnimationSignals()
    {
        if (!Directory.Exists(MdxTestPaths.Standard060DataPath) || !File.Exists(MdxTestPaths.ListfilePath))
            return;

        using IArchiveCatalog catalog = new MpqArchiveCatalog();
        ArchiveCatalogBootstrapResult bootstrap = ArchiveCatalogBootstrapper.Bootstrap(catalog, [MdxTestPaths.Standard060DataPath], MdxTestPaths.ListfilePath);
        Assert.NotNull(bootstrap);

        string? virtualPath = MdxTestPaths.Standard060GeoaMdxCandidates.FirstOrDefault(catalog.FileExists);
        if (virtualPath is null)
            return;

        byte[]? bytes = catalog.ReadFile(virtualPath);
        Assert.NotNull(bytes);

        using MemoryStream summaryStream = new(bytes);
        MdxSummary summary = MdxSummaryReader.Read(summaryStream, virtualPath);

        if (summary.GeosetAnimationCount == 0)
            return;

        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Geoa);
        foreach (MdxGeosetAnimationSummary geosetAnimation in summary.GeosetAnimations)
        {
            Assert.True(float.IsFinite(geosetAnimation.StaticAlpha));
            Assert.True(float.IsFinite(geosetAnimation.StaticColor.X));
            Assert.True(float.IsFinite(geosetAnimation.StaticColor.Y));
            Assert.True(float.IsFinite(geosetAnimation.StaticColor.Z));
            Assert.True(
                geosetAnimation.GeosetId == uint.MaxValue || geosetAnimation.GeosetId < summary.GeosetCount,
                $"Expected GEOA geoset id {geosetAnimation.GeosetId} to reference a real geoset or be 0xFFFFFFFF.");

            if (geosetAnimation.AlphaTrack is not null)
                Assert.True(geosetAnimation.AlphaTrack.LastKeyTime >= geosetAnimation.AlphaTrack.FirstKeyTime);

            if (geosetAnimation.ColorTrack is not null)
                Assert.True(geosetAnimation.ColorTrack.LastKeyTime >= geosetAnimation.ColorTrack.FirstKeyTime);
        }
    }

    [Fact]
    public void Read_RealAlpha053Mdx_WithGeoa_ProducesGeosetAnimationSignals()
    {
        string? inputPath = MdxTestPaths.Alpha053GeoaMdxCandidates.FirstOrDefault(File.Exists);
        if (inputPath is null)
            return;

        MdxSummary summary = MdxSummaryReader.Read(inputPath);

        Assert.True(summary.GeosetAnimationCount > 0);
        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Geoa);
        Assert.All(summary.GeosetAnimations, geosetAnimation =>
        {
            Assert.True(float.IsFinite(geosetAnimation.StaticAlpha));
            Assert.True(float.IsFinite(geosetAnimation.StaticColor.X));
            Assert.True(float.IsFinite(geosetAnimation.StaticColor.Y));
            Assert.True(float.IsFinite(geosetAnimation.StaticColor.Z));
            Assert.True(
                geosetAnimation.GeosetId == uint.MaxValue || geosetAnimation.GeosetId < summary.GeosetCount,
                $"Expected GEOA geoset id {geosetAnimation.GeosetId} to reference a real geoset or be 0xFFFFFFFF.");

            if (geosetAnimation.AlphaTrack is not null)
            {
                Assert.True(geosetAnimation.AlphaTrack.KeyCount > 0);
                Assert.True(geosetAnimation.AlphaTrack.LastKeyTime >= geosetAnimation.AlphaTrack.FirstKeyTime);
            }

            if (geosetAnimation.ColorTrack is not null)
            {
                Assert.True(geosetAnimation.ColorTrack.KeyCount > 0);
                Assert.True(geosetAnimation.ColorTrack.LastKeyTime >= geosetAnimation.ColorTrack.FirstKeyTime);
            }
        });
    }

    [Fact]
    public void Read_RealAlpha053WispMdx_WithGeoa_ProducesExpectedFixedSignals()
    {
        if (!File.Exists(MdxTestPaths.Alpha053WispMdxPath))
            return;

        MdxSummary summary = MdxSummaryReader.Read(MdxTestPaths.Alpha053WispMdxPath);

        Assert.Equal("Wisp", summary.ModelName);
        Assert.Equal(4, summary.GeosetCount);
        Assert.Equal(4, summary.GeosetAnimationCount);
        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Geoa);

        Assert.Collection(summary.GeosetAnimations,
            geosetAnimation => AssertExpectedWispGeoa(geosetAnimation, 0u, 0x0u, usesStaticColor: false),
            geosetAnimation => AssertExpectedWispGeoa(geosetAnimation, 1u, 0x1u, usesStaticColor: true),
            geosetAnimation => AssertExpectedWispGeoa(geosetAnimation, 2u, 0x0u, usesStaticColor: false),
            geosetAnimation => AssertExpectedWispGeoa(geosetAnimation, 3u, 0x0u, usesStaticColor: false));
    }

    [Fact]
    public void Read_RealAlpha053WispMdx_WithBone_ProducesExpectedFixedSignals()
    {
        if (!File.Exists(MdxTestPaths.Alpha053WispMdxPath))
            return;

        MdxSummary summary = MdxSummaryReader.Read(MdxTestPaths.Alpha053WispMdxPath);

        Assert.Equal("Wisp", summary.ModelName);
        Assert.Equal(16, summary.BoneCount);
        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Bone);

        MdxBoneSummary firstBone = summary.Bones[0];
        Assert.Equal("Plane02", firstBone.Name);
        Assert.Equal(0, firstBone.ObjectId);
        Assert.True(firstBone.HasParent);
        Assert.Equal(16, firstBone.ParentId);
        Assert.Equal(0x8u, firstBone.Flags);
        Assert.True(firstBone.UsesGeoset);
        Assert.Equal(1u, firstBone.GeosetId);
        Assert.True(firstBone.UsesGeosetAnimation);
        Assert.Equal(1u, firstBone.GeosetAnimationId);
        Assert.Null(firstBone.TranslationTrack);
        Assert.Null(firstBone.RotationTrack);
        Assert.NotNull(firstBone.ScalingTrack);
        Assert.Equal("KGSC", firstBone.ScalingTrack!.Tag);
        Assert.Equal(3, firstBone.ScalingTrack.KeyCount);
        Assert.Equal(3u, firstBone.ScalingTrack.InterpolationType);
        Assert.Equal(0, firstBone.ScalingTrack.GlobalSequenceId);
        Assert.Equal(0, firstBone.ScalingTrack.FirstKeyTime);
        Assert.Equal(267, firstBone.ScalingTrack.LastKeyTime);
    }

    [Fact]
    public void Read_RealAlpha053WispMdx_WithHelp_ProducesExpectedFixedSignals()
    {
        if (!File.Exists(MdxTestPaths.Alpha053WispMdxPath))
            return;

        MdxSummary summary = MdxSummaryReader.Read(MdxTestPaths.Alpha053WispMdxPath);

        Assert.Equal("Wisp", summary.ModelName);
        Assert.Equal(9, summary.HelperCount);
        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Help);

        MdxHelperSummary rootHelper = summary.Helpers[0];
        Assert.Equal("Root", rootHelper.Name);
        Assert.Equal(16, rootHelper.ObjectId);
        Assert.False(rootHelper.HasParent);
        Assert.Equal(0u, rootHelper.Flags);

        Assert.NotNull(rootHelper.TranslationTrack);
        Assert.Equal("KGTR", rootHelper.TranslationTrack!.Tag);
        Assert.Equal(26, rootHelper.TranslationTrack.KeyCount);
        Assert.Equal(3u, rootHelper.TranslationTrack.InterpolationType);
        Assert.Equal(-1, rootHelper.TranslationTrack.GlobalSequenceId);
        Assert.Equal(33, rootHelper.TranslationTrack.FirstKeyTime);
        Assert.Equal(20833, rootHelper.TranslationTrack.LastKeyTime);

        Assert.NotNull(rootHelper.RotationTrack);
        Assert.Equal("KGRT", rootHelper.RotationTrack!.Tag);
        Assert.Equal(4, rootHelper.RotationTrack.KeyCount);
        Assert.Equal(2u, rootHelper.RotationTrack.InterpolationType);
        Assert.Equal(-1, rootHelper.RotationTrack.GlobalSequenceId);
        Assert.Equal(20000, rootHelper.RotationTrack.FirstKeyTime);
        Assert.Equal(20833, rootHelper.RotationTrack.LastKeyTime);

        Assert.NotNull(rootHelper.ScalingTrack);
        Assert.Equal("KGSC", rootHelper.ScalingTrack!.Tag);
        Assert.Equal(4, rootHelper.ScalingTrack.KeyCount);
        Assert.Equal(3u, rootHelper.ScalingTrack.InterpolationType);
        Assert.Equal(-1, rootHelper.ScalingTrack.GlobalSequenceId);
        Assert.Equal(1167, rootHelper.ScalingTrack.FirstKeyTime);
        Assert.Equal(3333, rootHelper.ScalingTrack.LastKeyTime);
    }

    [Fact]
    public void Read_RealAlpha053WispMdx_WithAtch_ProducesExpectedFixedSignals()
    {
        if (!File.Exists(MdxTestPaths.Alpha053WispMdxPath))
            return;

        MdxSummary summary = MdxSummaryReader.Read(MdxTestPaths.Alpha053WispMdxPath);

        Assert.Equal("Wisp", summary.ModelName);
        Assert.Equal(9, summary.AttachmentCount);
        Assert.Contains(summary.Chunks, static chunk => chunk.Id == MdxChunkIds.Atch);

        MdxAttachmentSummary firstAttachment = summary.Attachments[0];
        Assert.Equal("_BloodFront", firstAttachment.Name);
        Assert.Equal(25, firstAttachment.ObjectId);
        Assert.True(firstAttachment.HasParent);
        Assert.Equal(16, firstAttachment.ParentId);
        Assert.Equal(0x402u, firstAttachment.Flags);
        Assert.Equal(15u, firstAttachment.AttachmentId);
        Assert.Null(firstAttachment.Path);
        Assert.Null(firstAttachment.TranslationTrack);
        Assert.Null(firstAttachment.RotationTrack);
        Assert.Null(firstAttachment.ScalingTrack);
        Assert.NotNull(firstAttachment.VisibilityTrack);
        Assert.Equal("KVIS", firstAttachment.VisibilityTrack!.Tag);
        Assert.Equal(2, firstAttachment.VisibilityTrack.KeyCount);
        Assert.Equal(0u, firstAttachment.VisibilityTrack.InterpolationType);
        Assert.Equal(-1, firstAttachment.VisibilityTrack.GlobalSequenceId);
        Assert.Equal(1167, firstAttachment.VisibilityTrack.FirstKeyTime);
        Assert.Equal(1833, firstAttachment.VisibilityTrack.LastKeyTime);

        MdxAttachmentSummary secondAttachment = summary.Attachments[1];
        Assert.Equal("_BloodBack", secondAttachment.Name);
        Assert.NotNull(secondAttachment.RotationTrack);
        Assert.Equal("KGRT", secondAttachment.RotationTrack!.Tag);
        Assert.Equal(1, secondAttachment.RotationTrack.KeyCount);
        Assert.Equal(0u, secondAttachment.RotationTrack.InterpolationType);
        Assert.Equal(3, secondAttachment.RotationTrack.GlobalSequenceId);
        Assert.Equal(0, secondAttachment.RotationTrack.FirstKeyTime);
        Assert.Equal(0, secondAttachment.RotationTrack.LastKeyTime);
    }

    private static byte[] CreateMdxBytes(uint version, string modelName, uint blendTime, Vector3 boundsMin, Vector3 boundsMax, IReadOnlyList<(string Name, int StartTime, int EndTime, float MoveSpeed, uint Flags, float Frequency, int ReplayStart, int ReplayEnd, float BoundsMinX, float BoundsMinY, float BoundsMinZ, float BoundsMaxX, float BoundsMaxY, float BoundsMaxZ, float BoundsRadius)> sequences, IReadOnlyList<Vector3> pivotPoints, IReadOnlyList<(uint ReplaceableId, string Path, uint Flags)> textures, IReadOnlyList<(int PriorityPlane, IReadOnlyList<(uint BlendMode, uint Flags, int TextureId, int TransformId, int CoordId, float StaticAlpha)> Layers)> materials, IReadOnlyList<byte[]> extraChunks)
    {
        List<byte> bytes =
        [
            (byte)'M', (byte)'D', (byte)'L', (byte)'X',
        ];

        bytes.AddRange(CreateChunk("VERS", CreateUInt32Payload(version)));
        bytes.AddRange(CreateChunk("MODL", CreateModlPayload(modelName, blendTime, boundsMin, boundsMax)));
        bytes.AddRange(CreateChunk("SEQS", CreateSeqsPayload(sequences)));
        bytes.AddRange(CreateChunk("PIVT", CreatePivtPayload(pivotPoints)));
        bytes.AddRange(CreateChunk("TEXS", CreateTexsPayload(textures)));
        bytes.AddRange(CreateChunk("MTLS", CreateMtlsPayload(materials)));
        foreach (byte[] chunk in extraChunks)
            bytes.AddRange(chunk);

        return [.. bytes];
    }

    private static byte[] CreateBonePayload(IReadOnlyList<(string Name, int ObjectId, int ParentId, uint Flags, uint GeosetId, uint GeosetAnimationId, (uint InterpolationType, int GlobalSequenceId, int[] Times)? TranslationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? RotationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? ScalingTrack)> bones)
    {
        List<byte> payload = [];
        payload.AddRange(CreateUInt32Payload((uint)bones.Count));

        foreach ((string name, int objectId, int parentId, uint flags, uint geosetId, uint geosetAnimationId, (uint InterpolationType, int GlobalSequenceId, int[] Times)? translationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? rotationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? scalingTrack) in bones)
        {
            List<byte> nodePayload = [];
            nodePayload.AddRange(CreateFixedAsciiPayload(name, 0x50));
            nodePayload.AddRange(CreateInt32Payload(objectId));
            nodePayload.AddRange(CreateInt32Payload(parentId));
            nodePayload.AddRange(CreateUInt32Payload(flags));

            if (translationTrack is not null)
                nodePayload.AddRange(CreateNodeTrackChunk("KGTR", translationTrack.Value.InterpolationType, translationTrack.Value.GlobalSequenceId, translationTrack.Value.Times, TrackValueKind.Vector3));

            if (rotationTrack is not null)
                nodePayload.AddRange(CreateNodeTrackChunk("KGRT", rotationTrack.Value.InterpolationType, rotationTrack.Value.GlobalSequenceId, rotationTrack.Value.Times, TrackValueKind.Quaternion));

            if (scalingTrack is not null)
                nodePayload.AddRange(CreateNodeTrackChunk("KGSC", scalingTrack.Value.InterpolationType, scalingTrack.Value.GlobalSequenceId, scalingTrack.Value.Times, TrackValueKind.Vector3));

            payload.AddRange(CreateSizedPayload(nodePayload));
            payload.AddRange(CreateUInt32Payload(geosetId));
            payload.AddRange(CreateUInt32Payload(geosetAnimationId));
        }

        return [.. payload];
    }

    private static byte[] CreateHelpPayload(IReadOnlyList<(string Name, int ObjectId, int ParentId, uint Flags, (uint InterpolationType, int GlobalSequenceId, int[] Times)? TranslationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? RotationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? ScalingTrack)> helpers)
    {
        List<byte> payload = [];
        payload.AddRange(CreateUInt32Payload((uint)helpers.Count));

        foreach ((string name, int objectId, int parentId, uint flags, (uint InterpolationType, int GlobalSequenceId, int[] Times)? translationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? rotationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? scalingTrack) in helpers)
        {
            List<byte> nodePayload = [];
            nodePayload.AddRange(CreateFixedAsciiPayload(name, 0x50));
            nodePayload.AddRange(CreateInt32Payload(objectId));
            nodePayload.AddRange(CreateInt32Payload(parentId));
            nodePayload.AddRange(CreateUInt32Payload(flags));

            if (translationTrack is not null)
                nodePayload.AddRange(CreateNodeTrackChunk("KGTR", translationTrack.Value.InterpolationType, translationTrack.Value.GlobalSequenceId, translationTrack.Value.Times, TrackValueKind.Vector3));

            if (rotationTrack is not null)
                nodePayload.AddRange(CreateNodeTrackChunk("KGRT", rotationTrack.Value.InterpolationType, rotationTrack.Value.GlobalSequenceId, rotationTrack.Value.Times, TrackValueKind.Quaternion));

            if (scalingTrack is not null)
                nodePayload.AddRange(CreateNodeTrackChunk("KGSC", scalingTrack.Value.InterpolationType, scalingTrack.Value.GlobalSequenceId, scalingTrack.Value.Times, TrackValueKind.Vector3));

            payload.AddRange(CreateSizedPayload(nodePayload));
        }

        return [.. payload];
    }

    private static byte[] CreateLightPayload(IReadOnlyList<(string Name, int ObjectId, int ParentId, uint Flags, MdxLightType LightType, float StaticAttenuationStart, float StaticAttenuationEnd, Vector3 StaticColor, float StaticIntensity, Vector3 StaticAmbientColor, float StaticAmbientIntensity, (uint InterpolationType, int GlobalSequenceId, int[] Times)? TranslationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? RotationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? ScalingTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? AttenuationStartTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? AttenuationEndTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? ColorTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? IntensityTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? AmbientColorTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? AmbientIntensityTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? VisibilityTrack)> lights)
    {
        List<byte> payload = [];
        payload.AddRange(CreateUInt32Payload((uint)lights.Count));

        foreach ((string name, int objectId, int parentId, uint flags, MdxLightType lightType, float staticAttenuationStart, float staticAttenuationEnd, Vector3 staticColor, float staticIntensity, Vector3 staticAmbientColor, float staticAmbientIntensity, (uint InterpolationType, int GlobalSequenceId, int[] Times)? translationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? rotationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? scalingTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? attenuationStartTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? attenuationEndTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? colorTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? intensityTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? ambientColorTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? ambientIntensityTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? visibilityTrack) in lights)
        {
            List<byte> entryPayload = [];
            List<byte> nodePayload = [];
            nodePayload.AddRange(CreateFixedAsciiPayload(name, 0x50));
            nodePayload.AddRange(CreateInt32Payload(objectId));
            nodePayload.AddRange(CreateInt32Payload(parentId));
            nodePayload.AddRange(CreateUInt32Payload(flags));

            if (translationTrack is not null)
                nodePayload.AddRange(CreateNodeTrackChunk("KGTR", translationTrack.Value.InterpolationType, translationTrack.Value.GlobalSequenceId, translationTrack.Value.Times, TrackValueKind.Vector3));

            if (rotationTrack is not null)
                nodePayload.AddRange(CreateNodeTrackChunk("KGRT", rotationTrack.Value.InterpolationType, rotationTrack.Value.GlobalSequenceId, rotationTrack.Value.Times, TrackValueKind.Quaternion));

            if (scalingTrack is not null)
                nodePayload.AddRange(CreateNodeTrackChunk("KGSC", scalingTrack.Value.InterpolationType, scalingTrack.Value.GlobalSequenceId, scalingTrack.Value.Times, TrackValueKind.Vector3));

            entryPayload.AddRange(CreateSizedPayload(nodePayload));
            entryPayload.AddRange(CreateUInt32Payload((uint)lightType));
            entryPayload.AddRange(CreateSinglePayload(staticAttenuationStart));
            entryPayload.AddRange(CreateSinglePayload(staticAttenuationEnd));
            entryPayload.AddRange(CreateVector3Payload(staticColor));
            entryPayload.AddRange(CreateSinglePayload(staticIntensity));
            entryPayload.AddRange(CreateVector3Payload(staticAmbientColor));
            entryPayload.AddRange(CreateSinglePayload(staticAmbientIntensity));

            if (attenuationStartTrack is not null)
                entryPayload.AddRange(CreateNodeTrackChunk("KLAS", attenuationStartTrack.Value.InterpolationType, attenuationStartTrack.Value.GlobalSequenceId, attenuationStartTrack.Value.Times, TrackValueKind.Scalar));

            if (attenuationEndTrack is not null)
                entryPayload.AddRange(CreateNodeTrackChunk("KLAE", attenuationEndTrack.Value.InterpolationType, attenuationEndTrack.Value.GlobalSequenceId, attenuationEndTrack.Value.Times, TrackValueKind.Scalar));

            if (colorTrack is not null)
                entryPayload.AddRange(CreateNodeTrackChunk("KLAC", colorTrack.Value.InterpolationType, colorTrack.Value.GlobalSequenceId, colorTrack.Value.Times, TrackValueKind.Vector3));

            if (intensityTrack is not null)
                entryPayload.AddRange(CreateNodeTrackChunk("KLAI", intensityTrack.Value.InterpolationType, intensityTrack.Value.GlobalSequenceId, intensityTrack.Value.Times, TrackValueKind.Scalar));

            if (ambientColorTrack is not null)
                entryPayload.AddRange(CreateNodeTrackChunk("KLBC", ambientColorTrack.Value.InterpolationType, ambientColorTrack.Value.GlobalSequenceId, ambientColorTrack.Value.Times, TrackValueKind.Vector3));

            if (ambientIntensityTrack is not null)
                entryPayload.AddRange(CreateNodeTrackChunk("KLBI", ambientIntensityTrack.Value.InterpolationType, ambientIntensityTrack.Value.GlobalSequenceId, ambientIntensityTrack.Value.Times, TrackValueKind.Scalar));

            if (visibilityTrack is not null)
                entryPayload.AddRange(CreateNodeTrackChunk("KVIS", visibilityTrack.Value.InterpolationType, visibilityTrack.Value.GlobalSequenceId, visibilityTrack.Value.Times, TrackValueKind.Scalar));

            payload.AddRange(CreateSizedPayload(entryPayload));
        }

        return [.. payload];
    }

    private static byte[] CreateEventPayload(IReadOnlyList<(string Name, int ObjectId, int ParentId, uint Flags, (uint InterpolationType, int GlobalSequenceId, int[] Times)? TranslationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? RotationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? ScalingTrack, (int GlobalSequenceId, int[] Times)? EventTrack)> events)
    {
        List<byte> payload = [];
        payload.AddRange(CreateUInt32Payload((uint)events.Count));

        foreach ((string name, int objectId, int parentId, uint flags, (uint InterpolationType, int GlobalSequenceId, int[] Times)? translationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? rotationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? scalingTrack, (int GlobalSequenceId, int[] Times)? eventTrack) in events)
        {
            List<byte> entryPayload = [];
            List<byte> nodePayload = [];
            nodePayload.AddRange(CreateFixedAsciiPayload(name, 0x50));
            nodePayload.AddRange(CreateInt32Payload(objectId));
            nodePayload.AddRange(CreateInt32Payload(parentId));
            nodePayload.AddRange(CreateUInt32Payload(flags));

            if (translationTrack is not null)
                nodePayload.AddRange(CreateNodeTrackChunk("KGTR", translationTrack.Value.InterpolationType, translationTrack.Value.GlobalSequenceId, translationTrack.Value.Times, TrackValueKind.Vector3));

            if (rotationTrack is not null)
                nodePayload.AddRange(CreateNodeTrackChunk("KGRT", rotationTrack.Value.InterpolationType, rotationTrack.Value.GlobalSequenceId, rotationTrack.Value.Times, TrackValueKind.Quaternion));

            if (scalingTrack is not null)
                nodePayload.AddRange(CreateNodeTrackChunk("KGSC", scalingTrack.Value.InterpolationType, scalingTrack.Value.GlobalSequenceId, scalingTrack.Value.Times, TrackValueKind.Vector3));

            entryPayload.AddRange(CreateSizedPayload(nodePayload));

            if (eventTrack is not null)
                entryPayload.AddRange(CreateSimpleEventTrackChunk("KEVT", eventTrack.Value.GlobalSequenceId, eventTrack.Value.Times));

            payload.AddRange(CreateSizedPayload(entryPayload));
        }

        return [.. payload];
    }

    private static byte[] CreateHitTestShapePayload(IReadOnlyList<(string Name, int ObjectId, int ParentId, uint Flags, MdxGeometryShapeType ShapeType, Vector3 PrimaryVector, Vector3 SecondaryVector, float PrimaryScalar, float SecondaryScalar, (uint InterpolationType, int GlobalSequenceId, int[] Times)? TranslationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? RotationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? ScalingTrack)> shapes)
    {
        List<byte> payload = [];
        payload.AddRange(CreateUInt32Payload((uint)shapes.Count));

        foreach ((string name, int objectId, int parentId, uint flags, MdxGeometryShapeType shapeType, Vector3 primaryVector, Vector3 secondaryVector, float primaryScalar, float secondaryScalar, (uint InterpolationType, int GlobalSequenceId, int[] Times)? translationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? rotationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? scalingTrack) in shapes)
        {
            List<byte> entryPayload = [];
            List<byte> nodePayload = [];
            nodePayload.AddRange(CreateFixedAsciiPayload(name, 0x50));
            nodePayload.AddRange(CreateInt32Payload(objectId));
            nodePayload.AddRange(CreateInt32Payload(parentId));
            nodePayload.AddRange(CreateUInt32Payload(flags));

            if (translationTrack is not null)
                nodePayload.AddRange(CreateNodeTrackChunk("KGTR", translationTrack.Value.InterpolationType, translationTrack.Value.GlobalSequenceId, translationTrack.Value.Times, TrackValueKind.Vector3));

            if (rotationTrack is not null)
                nodePayload.AddRange(CreateNodeTrackChunk("KGRT", rotationTrack.Value.InterpolationType, rotationTrack.Value.GlobalSequenceId, rotationTrack.Value.Times, TrackValueKind.Quaternion));

            if (scalingTrack is not null)
                nodePayload.AddRange(CreateNodeTrackChunk("KGSC", scalingTrack.Value.InterpolationType, scalingTrack.Value.GlobalSequenceId, scalingTrack.Value.Times, TrackValueKind.Vector3));

            entryPayload.AddRange(CreateSizedPayload(nodePayload));
            entryPayload.Add((byte)shapeType);

            switch (shapeType)
            {
                case MdxGeometryShapeType.Box:
                    entryPayload.AddRange(CreateVector3Payload(primaryVector));
                    entryPayload.AddRange(CreateVector3Payload(secondaryVector));
                    break;
                case MdxGeometryShapeType.Cylinder:
                    entryPayload.AddRange(CreateVector3Payload(primaryVector));
                    entryPayload.AddRange(CreateSinglePayload(primaryScalar));
                    entryPayload.AddRange(CreateSinglePayload(secondaryScalar));
                    break;
                case MdxGeometryShapeType.Sphere:
                    entryPayload.AddRange(CreateVector3Payload(primaryVector));
                    entryPayload.AddRange(CreateSinglePayload(primaryScalar));
                    break;
                case MdxGeometryShapeType.Plane:
                    entryPayload.AddRange(CreateSinglePayload(primaryScalar));
                    entryPayload.AddRange(CreateSinglePayload(secondaryScalar));
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(shapeType), shapeType, null);
            }

            payload.AddRange(CreateSizedPayload(entryPayload));
        }

        return [.. payload];
    }

    private static byte[] CreateCollisionPayload(IReadOnlyList<Vector3> vertices, IReadOnlyList<ushort> triangleIndices, IReadOnlyList<Vector3> facetNormals)
    {
        List<byte> payload = [];
        payload.AddRange(Encoding.ASCII.GetBytes("VRTX"));
        payload.AddRange(CreateUInt32Payload((uint)vertices.Count));
        foreach (Vector3 vertex in vertices)
            payload.AddRange(CreateVector3Payload(vertex));

        payload.AddRange(Encoding.ASCII.GetBytes("TRI "));
        payload.AddRange(CreateUInt32Payload((uint)triangleIndices.Count));
        foreach (ushort triangleIndex in triangleIndices)
            payload.AddRange(BitConverter.GetBytes(triangleIndex));

        payload.AddRange(Encoding.ASCII.GetBytes("NRMS"));
        payload.AddRange(CreateUInt32Payload((uint)facetNormals.Count));
        foreach (Vector3 normal in facetNormals)
            payload.AddRange(CreateVector3Payload(normal));

        return [.. payload];
    }

    private static byte[] CreateGlobalSequencesPayload(IReadOnlyList<uint> durations)
    {
        List<byte> payload = [];
        foreach (uint duration in durations)
            payload.AddRange(CreateUInt32Payload(duration));

        return [.. payload];
    }

    private static byte[] CreateAttachmentPayload(IReadOnlyList<(string Name, int ObjectId, int ParentId, uint Flags, uint AttachmentId, string Path, (uint InterpolationType, int GlobalSequenceId, int[] Times)? TranslationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? RotationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? ScalingTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? VisibilityTrack)> attachments)
    {
        List<byte> payload = [];
        payload.AddRange(CreateUInt32Payload((uint)attachments.Count));
        payload.AddRange(CreateUInt32Payload(0u));

        foreach ((string name, int objectId, int parentId, uint flags, uint attachmentId, string path, (uint InterpolationType, int GlobalSequenceId, int[] Times)? translationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? rotationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? scalingTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? visibilityTrack) in attachments)
        {
            List<byte> entryPayload = [];
            List<byte> nodePayload = [];
            nodePayload.AddRange(CreateFixedAsciiPayload(name, 0x50));
            nodePayload.AddRange(CreateInt32Payload(objectId));
            nodePayload.AddRange(CreateInt32Payload(parentId));
            nodePayload.AddRange(CreateUInt32Payload(flags));

            if (translationTrack is not null)
                nodePayload.AddRange(CreateNodeTrackChunk("KGTR", translationTrack.Value.InterpolationType, translationTrack.Value.GlobalSequenceId, translationTrack.Value.Times, TrackValueKind.Vector3));

            if (rotationTrack is not null)
                nodePayload.AddRange(CreateNodeTrackChunk("KGRT", rotationTrack.Value.InterpolationType, rotationTrack.Value.GlobalSequenceId, rotationTrack.Value.Times, TrackValueKind.Quaternion));

            if (scalingTrack is not null)
                nodePayload.AddRange(CreateNodeTrackChunk("KGSC", scalingTrack.Value.InterpolationType, scalingTrack.Value.GlobalSequenceId, scalingTrack.Value.Times, TrackValueKind.Vector3));

            entryPayload.AddRange(CreateSizedPayload(nodePayload));
            entryPayload.AddRange(CreateUInt32Payload(attachmentId));
            entryPayload.Add(0);
            entryPayload.AddRange(CreateFixedAsciiPayload(path, 0x104));

            if (visibilityTrack is not null)
                entryPayload.AddRange(CreateNodeTrackChunk("KVIS", visibilityTrack.Value.InterpolationType, visibilityTrack.Value.GlobalSequenceId, visibilityTrack.Value.Times, TrackValueKind.Scalar));

            payload.AddRange(CreateSizedPayload(entryPayload));
        }

        return [.. payload];
    }

    private static byte[] CreatePre2Payload(IReadOnlyList<(string Name, int ObjectId, int ParentId, uint Flags, int EmitterType, float StaticSpeed, float StaticVariation, float StaticLatitude, float StaticLongitude, float StaticGravity, float StaticZSource, float StaticLife, float StaticEmissionRate, float StaticLength, float StaticWidth, uint Rows, uint Columns, uint ParticleType, float TailLength, float MiddleTime, Vector3 StartColor, Vector3 MiddleColor, Vector3 EndColor, byte StartAlpha, byte MiddleAlpha, byte EndAlpha, float StartScale, float MiddleScale, float EndScale, uint BlendMode, int TextureId, int PriorityPlane, uint ReplaceableId, string GeometryModel, string RecursionModel, IReadOnlyList<Vector3> Splines, int Squirts, (uint InterpolationType, int GlobalSequenceId, int[] Times)? TranslationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? RotationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? ScalingTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? VisibilityTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? SpeedTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? VariationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? LatitudeTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? LongitudeTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? GravityTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? LifeTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? EmissionRateTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? WidthTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? LengthTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? ZSourceTrack)> particleEmitters)
    {
        List<byte> payload = [];
        payload.AddRange(CreateUInt32Payload((uint)particleEmitters.Count));

        foreach ((string name, int objectId, int parentId, uint flags, int emitterType, float staticSpeed, float staticVariation, float staticLatitude, float staticLongitude, float staticGravity, float staticZSource, float staticLife, float staticEmissionRate, float staticLength, float staticWidth, uint rows, uint columns, uint particleType, float tailLength, float middleTime, Vector3 startColor, Vector3 middleColor, Vector3 endColor, byte startAlpha, byte middleAlpha, byte endAlpha, float startScale, float middleScale, float endScale, uint blendMode, int textureId, int priorityPlane, uint replaceableId, string geometryModel, string recursionModel, IReadOnlyList<Vector3> splines, int squirts, (uint InterpolationType, int GlobalSequenceId, int[] Times)? translationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? rotationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? scalingTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? visibilityTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? speedTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? variationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? latitudeTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? longitudeTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? gravityTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? lifeTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? emissionRateTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? widthTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? lengthTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? zSourceTrack) in particleEmitters)
        {
            List<byte> entryPayload = [];
            List<byte> nodePayload = [];
            nodePayload.AddRange(CreateFixedAsciiPayload(name, 0x50));
            nodePayload.AddRange(CreateInt32Payload(objectId));
            nodePayload.AddRange(CreateInt32Payload(parentId));
            nodePayload.AddRange(CreateUInt32Payload(flags));

            if (translationTrack is not null)
                nodePayload.AddRange(CreateNodeTrackChunk("KGTR", translationTrack.Value.InterpolationType, translationTrack.Value.GlobalSequenceId, translationTrack.Value.Times, TrackValueKind.Vector3));

            if (rotationTrack is not null)
                nodePayload.AddRange(CreateNodeTrackChunk("KGRT", rotationTrack.Value.InterpolationType, rotationTrack.Value.GlobalSequenceId, rotationTrack.Value.Times, TrackValueKind.Quaternion));

            if (scalingTrack is not null)
                nodePayload.AddRange(CreateNodeTrackChunk("KGSC", scalingTrack.Value.InterpolationType, scalingTrack.Value.GlobalSequenceId, scalingTrack.Value.Times, TrackValueKind.Vector3));

            entryPayload.AddRange(CreateSizedPayload(nodePayload));

            List<byte> emitterPayload = [];
            emitterPayload.AddRange(CreateInt32Payload(emitterType));
            emitterPayload.AddRange(CreateSinglePayload(staticSpeed));
            emitterPayload.AddRange(CreateSinglePayload(staticVariation));
            emitterPayload.AddRange(CreateSinglePayload(staticLatitude));
            emitterPayload.AddRange(CreateSinglePayload(staticLongitude));
            emitterPayload.AddRange(CreateSinglePayload(staticGravity));
            emitterPayload.AddRange(CreateSinglePayload(staticZSource));
            emitterPayload.AddRange(CreateSinglePayload(staticLife));
            emitterPayload.AddRange(CreateSinglePayload(staticEmissionRate));
            emitterPayload.AddRange(CreateSinglePayload(staticLength));
            emitterPayload.AddRange(CreateSinglePayload(staticWidth));
            emitterPayload.AddRange(CreateUInt32Payload(rows));
            emitterPayload.AddRange(CreateUInt32Payload(columns));
            emitterPayload.AddRange(CreateUInt32Payload(particleType));
            emitterPayload.AddRange(CreateSinglePayload(tailLength));
            emitterPayload.AddRange(CreateSinglePayload(middleTime));
            emitterPayload.AddRange(CreateVector3Payload(startColor));
            emitterPayload.AddRange(CreateVector3Payload(middleColor));
            emitterPayload.AddRange(CreateVector3Payload(endColor));
            emitterPayload.Add(startAlpha);
            emitterPayload.Add(middleAlpha);
            emitterPayload.Add(endAlpha);
            emitterPayload.AddRange(CreateSinglePayload(startScale));
            emitterPayload.AddRange(CreateSinglePayload(middleScale));
            emitterPayload.AddRange(CreateSinglePayload(endScale));

            for (int intervalIndex = 0; intervalIndex < 4; intervalIndex++)
            {
                emitterPayload.AddRange(CreateUInt32Payload(0u));
                emitterPayload.AddRange(CreateUInt32Payload(0u));
                emitterPayload.AddRange(CreateUInt32Payload(1u));
            }

            emitterPayload.AddRange(CreateUInt32Payload(blendMode));
            emitterPayload.AddRange(CreateInt32Payload(textureId));
            emitterPayload.AddRange(CreateInt32Payload(priorityPlane));
            emitterPayload.AddRange(CreateUInt32Payload(replaceableId));
            emitterPayload.AddRange(CreateFixedAsciiPayload(geometryModel, 0x104));
            emitterPayload.AddRange(CreateFixedAsciiPayload(recursionModel, 0x104));
            emitterPayload.AddRange(CreateSinglePayload(10.0f));
            emitterPayload.AddRange(CreateSinglePayload(1.0f));
            emitterPayload.AddRange(CreateSinglePayload(1.0f));
            emitterPayload.AddRange(CreateSinglePayload(1.0f));
            emitterPayload.AddRange(CreateSinglePayload(1.0f));

            for (int tumbleIndex = 0; tumbleIndex < 6; tumbleIndex++)
                emitterPayload.AddRange(CreateSinglePayload(0.0f));

            emitterPayload.AddRange(CreateSinglePayload(0.0f));
            emitterPayload.AddRange(CreateSinglePayload(0.0f));
            emitterPayload.AddRange(CreateVector3Payload(Vector3.Zero));
            emitterPayload.AddRange(CreateSinglePayload(0.0f));
            emitterPayload.AddRange(CreateSinglePayload(2.5f));
            emitterPayload.AddRange(CreateSinglePayload(0.7f));
            emitterPayload.AddRange(CreateSinglePayload(7.0f));
            emitterPayload.AddRange(CreateSinglePayload(0.9f));
            emitterPayload.AddRange(CreateUInt32Payload((uint)splines.Count));
            foreach (Vector3 spline in splines)
                emitterPayload.AddRange(CreateVector3Payload(spline));

            emitterPayload.AddRange(CreateInt32Payload(squirts));

            entryPayload.AddRange(CreateUInt32Payload((uint)emitterPayload.Count));
            entryPayload.AddRange(emitterPayload);

            if (visibilityTrack is not null)
                entryPayload.AddRange(CreateNodeTrackChunk("KVIS", visibilityTrack.Value.InterpolationType, visibilityTrack.Value.GlobalSequenceId, visibilityTrack.Value.Times, TrackValueKind.Scalar));

            if (speedTrack is not null)
                entryPayload.AddRange(CreateNodeTrackChunk("KP2S", speedTrack.Value.InterpolationType, speedTrack.Value.GlobalSequenceId, speedTrack.Value.Times, TrackValueKind.Scalar));

            if (variationTrack is not null)
                entryPayload.AddRange(CreateNodeTrackChunk("KP2R", variationTrack.Value.InterpolationType, variationTrack.Value.GlobalSequenceId, variationTrack.Value.Times, TrackValueKind.Scalar));

            if (latitudeTrack is not null)
                entryPayload.AddRange(CreateNodeTrackChunk("KP2L", latitudeTrack.Value.InterpolationType, latitudeTrack.Value.GlobalSequenceId, latitudeTrack.Value.Times, TrackValueKind.Scalar));

            if (longitudeTrack is not null)
                entryPayload.AddRange(CreateNodeTrackChunk("KPLN", longitudeTrack.Value.InterpolationType, longitudeTrack.Value.GlobalSequenceId, longitudeTrack.Value.Times, TrackValueKind.Scalar));

            if (gravityTrack is not null)
                entryPayload.AddRange(CreateNodeTrackChunk("KP2G", gravityTrack.Value.InterpolationType, gravityTrack.Value.GlobalSequenceId, gravityTrack.Value.Times, TrackValueKind.Scalar));

            if (lifeTrack is not null)
                entryPayload.AddRange(CreateNodeTrackChunk("KLIF", lifeTrack.Value.InterpolationType, lifeTrack.Value.GlobalSequenceId, lifeTrack.Value.Times, TrackValueKind.Scalar));

            if (emissionRateTrack is not null)
                entryPayload.AddRange(CreateNodeTrackChunk("KP2E", emissionRateTrack.Value.InterpolationType, emissionRateTrack.Value.GlobalSequenceId, emissionRateTrack.Value.Times, TrackValueKind.Scalar));

            if (widthTrack is not null)
                entryPayload.AddRange(CreateNodeTrackChunk("KP2W", widthTrack.Value.InterpolationType, widthTrack.Value.GlobalSequenceId, widthTrack.Value.Times, TrackValueKind.Scalar));

            if (lengthTrack is not null)
                entryPayload.AddRange(CreateNodeTrackChunk("KP2N", lengthTrack.Value.InterpolationType, lengthTrack.Value.GlobalSequenceId, lengthTrack.Value.Times, TrackValueKind.Scalar));

            if (zSourceTrack is not null)
                entryPayload.AddRange(CreateNodeTrackChunk("KP2Z", zSourceTrack.Value.InterpolationType, zSourceTrack.Value.GlobalSequenceId, zSourceTrack.Value.Times, TrackValueKind.Scalar));

            payload.AddRange(CreateSizedPayload(entryPayload));
        }

        return [.. payload];
    }

    private static byte[] CreateRibbonPayload(IReadOnlyList<(string Name, int ObjectId, int ParentId, uint Flags, float StaticHeightAbove, float StaticHeightBelow, float StaticAlpha, Vector3 StaticColor, float EdgeLifetime, uint StaticTextureSlot, uint EdgesPerSecond, uint TextureRows, uint TextureColumns, uint MaterialId, float Gravity, (uint InterpolationType, int GlobalSequenceId, int[] Times)? TranslationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? RotationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? ScalingTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? HeightAboveTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? HeightBelowTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? AlphaTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? ColorTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? TextureSlotTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? VisibilityTrack)> ribbons)
    {
        List<byte> payload = [];
        payload.AddRange(CreateUInt32Payload((uint)ribbons.Count));

        foreach ((string name, int objectId, int parentId, uint flags, float staticHeightAbove, float staticHeightBelow, float staticAlpha, Vector3 staticColor, float edgeLifetime, uint staticTextureSlot, uint edgesPerSecond, uint textureRows, uint textureColumns, uint materialId, float gravity, (uint InterpolationType, int GlobalSequenceId, int[] Times)? translationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? rotationTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? scalingTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? heightAboveTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? heightBelowTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? alphaTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? colorTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? textureSlotTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? visibilityTrack) in ribbons)
        {
            List<byte> entryPayload = [];
            List<byte> nodePayload = [];
            nodePayload.AddRange(CreateFixedAsciiPayload(name, 0x50));
            nodePayload.AddRange(CreateInt32Payload(objectId));
            nodePayload.AddRange(CreateInt32Payload(parentId));
            nodePayload.AddRange(CreateUInt32Payload(flags));

            if (translationTrack is not null)
                nodePayload.AddRange(CreateNodeTrackChunk("KGTR", translationTrack.Value.InterpolationType, translationTrack.Value.GlobalSequenceId, translationTrack.Value.Times, TrackValueKind.Vector3));

            if (rotationTrack is not null)
                nodePayload.AddRange(CreateNodeTrackChunk("KGRT", rotationTrack.Value.InterpolationType, rotationTrack.Value.GlobalSequenceId, rotationTrack.Value.Times, TrackValueKind.Quaternion));

            if (scalingTrack is not null)
                nodePayload.AddRange(CreateNodeTrackChunk("KGSC", scalingTrack.Value.InterpolationType, scalingTrack.Value.GlobalSequenceId, scalingTrack.Value.Times, TrackValueKind.Vector3));

            entryPayload.AddRange(CreateSizedPayload(nodePayload));
            entryPayload.AddRange(CreateUInt32Payload(56u));
            entryPayload.AddRange(CreateSinglePayload(staticHeightAbove));
            entryPayload.AddRange(CreateSinglePayload(staticHeightBelow));
            entryPayload.AddRange(CreateSinglePayload(staticAlpha));
            entryPayload.AddRange(CreateVector3Payload(staticColor));
            entryPayload.AddRange(CreateSinglePayload(edgeLifetime));
            entryPayload.AddRange(CreateUInt32Payload(staticTextureSlot));
            entryPayload.AddRange(CreateUInt32Payload(edgesPerSecond));
            entryPayload.AddRange(CreateUInt32Payload(textureRows));
            entryPayload.AddRange(CreateUInt32Payload(textureColumns));
            entryPayload.AddRange(CreateUInt32Payload(materialId));
            entryPayload.AddRange(CreateSinglePayload(gravity));

            if (heightAboveTrack is not null)
                entryPayload.AddRange(CreateNodeTrackChunk("KRHA", heightAboveTrack.Value.InterpolationType, heightAboveTrack.Value.GlobalSequenceId, heightAboveTrack.Value.Times, TrackValueKind.Scalar));

            if (heightBelowTrack is not null)
                entryPayload.AddRange(CreateNodeTrackChunk("KRHB", heightBelowTrack.Value.InterpolationType, heightBelowTrack.Value.GlobalSequenceId, heightBelowTrack.Value.Times, TrackValueKind.Scalar));

            if (alphaTrack is not null)
                entryPayload.AddRange(CreateNodeTrackChunk("KRAL", alphaTrack.Value.InterpolationType, alphaTrack.Value.GlobalSequenceId, alphaTrack.Value.Times, TrackValueKind.Scalar));

            if (colorTrack is not null)
                entryPayload.AddRange(CreateNodeTrackChunk("KRCO", colorTrack.Value.InterpolationType, colorTrack.Value.GlobalSequenceId, colorTrack.Value.Times, TrackValueKind.Vector3));

            if (textureSlotTrack is not null)
                entryPayload.AddRange(CreateNodeTrackChunk("KRTX", textureSlotTrack.Value.InterpolationType, textureSlotTrack.Value.GlobalSequenceId, textureSlotTrack.Value.Times, TrackValueKind.Int32));

            if (visibilityTrack is not null)
                entryPayload.AddRange(CreateNodeTrackChunk("KVIS", visibilityTrack.Value.InterpolationType, visibilityTrack.Value.GlobalSequenceId, visibilityTrack.Value.Times, TrackValueKind.Scalar));

            payload.AddRange(CreateSizedPayload(entryPayload));
        }

        return [.. payload];
    }

    private static byte[] CreateCameraPayload(IReadOnlyList<(string Name, Vector3 PivotPoint, float FieldOfView, float FarClip, float NearClip, Vector3 TargetPivotPoint, (uint InterpolationType, int GlobalSequenceId, int[] Times)? PositionTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? RollTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? VisibilityTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? TargetPositionTrack)> cameras)
    {
        List<byte> payload = [];
        payload.AddRange(CreateUInt32Payload((uint)cameras.Count));

        foreach ((string name, Vector3 pivotPoint, float fieldOfView, float farClip, float nearClip, Vector3 targetPivotPoint, (uint InterpolationType, int GlobalSequenceId, int[] Times)? positionTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? rollTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? visibilityTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? targetPositionTrack) in cameras)
        {
            List<byte> cameraPayload = [];
            cameraPayload.AddRange(CreateFixedAsciiPayload(name, 0x50));
            cameraPayload.AddRange(CreateVector3Payload(pivotPoint));
            cameraPayload.AddRange(CreateSinglePayload(fieldOfView));
            cameraPayload.AddRange(CreateSinglePayload(farClip));
            cameraPayload.AddRange(CreateSinglePayload(nearClip));
            cameraPayload.AddRange(CreateVector3Payload(targetPivotPoint));

            if (positionTrack is not null)
                cameraPayload.AddRange(CreateNodeTrackChunk("KCTR", positionTrack.Value.InterpolationType, positionTrack.Value.GlobalSequenceId, positionTrack.Value.Times, TrackValueKind.Vector3));

            if (rollTrack is not null)
                cameraPayload.AddRange(CreateNodeTrackChunk("KCRL", rollTrack.Value.InterpolationType, rollTrack.Value.GlobalSequenceId, rollTrack.Value.Times, TrackValueKind.Scalar));

            if (visibilityTrack is not null)
                cameraPayload.AddRange(CreateNodeTrackChunk("KVIS", visibilityTrack.Value.InterpolationType, visibilityTrack.Value.GlobalSequenceId, visibilityTrack.Value.Times, TrackValueKind.Scalar));

            if (targetPositionTrack is not null)
                cameraPayload.AddRange(CreateNodeTrackChunk("KTTR", targetPositionTrack.Value.InterpolationType, targetPositionTrack.Value.GlobalSequenceId, targetPositionTrack.Value.Times, TrackValueKind.Vector3));

            payload.AddRange(CreateSizedPayload(cameraPayload));
        }

        return [.. payload];
    }

    private static void AssertExpectedWispGeoa(MdxGeosetAnimationSummary geosetAnimation, uint expectedGeosetId, uint expectedFlags, bool usesStaticColor)
    {
        Assert.Equal(expectedGeosetId, geosetAnimation.GeosetId);
        Assert.Equal(1.0f, geosetAnimation.StaticAlpha);
        Assert.Equal(new Vector3(1.0f, 1.0f, 1.0f), geosetAnimation.StaticColor);
        Assert.Equal(expectedFlags, geosetAnimation.Flags);
        Assert.Equal(usesStaticColor, geosetAnimation.UsesStaticColor);
        Assert.Null(geosetAnimation.ColorTrack);
        Assert.NotNull(geosetAnimation.AlphaTrack);
        Assert.Equal("KGAO", geosetAnimation.AlphaTrack!.Tag);
        Assert.Equal(2, geosetAnimation.AlphaTrack.KeyCount);
        Assert.Equal(0u, geosetAnimation.AlphaTrack.InterpolationType);
        Assert.Equal(-1, geosetAnimation.AlphaTrack.GlobalSequenceId);
        Assert.Equal(1167, geosetAnimation.AlphaTrack.FirstKeyTime);
        Assert.Equal(1833, geosetAnimation.AlphaTrack.LastKeyTime);
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

    private static byte[] CreatePivtPayload(IReadOnlyList<Vector3> pivotPoints)
    {
        byte[] payload = new byte[pivotPoints.Count * 12];
        for (int index = 0; index < pivotPoints.Count; index++)
        {
            int offset = index * 12;
            BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(offset, 4), pivotPoints[index].X);
            BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(offset + 4, 4), pivotPoints[index].Y);
            BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(offset + 8, 4), pivotPoints[index].Z);
        }

        return payload;
    }

    private static byte[] CreateGeoaPayload(IReadOnlyList<(uint GeosetId, float StaticAlpha, Vector3 StaticColor, uint Flags, (uint InterpolationType, int GlobalSequenceId, int[] Times)? AlphaTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? ColorTrack)> geosetAnimations)
    {
        List<byte> payload = [];
        payload.AddRange(CreateUInt32Payload((uint)geosetAnimations.Count));

        foreach ((uint geosetId, float staticAlpha, Vector3 staticColor, uint flags, (uint InterpolationType, int GlobalSequenceId, int[] Times)? alphaTrack, (uint InterpolationType, int GlobalSequenceId, int[] Times)? colorTrack) in geosetAnimations)
        {
            List<byte> geosetAnimationPayload = [];
            geosetAnimationPayload.AddRange(CreateUInt32Payload(geosetId));
            geosetAnimationPayload.AddRange(CreateSinglePayload(staticAlpha));
            geosetAnimationPayload.AddRange(CreateVector3Payload(staticColor));
            geosetAnimationPayload.AddRange(CreateUInt32Payload(flags));

            if (alphaTrack is not null)
                geosetAnimationPayload.AddRange(CreateGeoaTrackChunk("KGAO", alphaTrack.Value.InterpolationType, alphaTrack.Value.GlobalSequenceId, alphaTrack.Value.Times, isColorTrack: false));

            if (colorTrack is not null)
                geosetAnimationPayload.AddRange(CreateGeoaTrackChunk("KGAC", colorTrack.Value.InterpolationType, colorTrack.Value.GlobalSequenceId, colorTrack.Value.Times, isColorTrack: true));

            payload.AddRange(CreateSizedPayload(geosetAnimationPayload));
        }

        return [.. payload];
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

            byte[] geosetBytes = new byte[4 + geosetPayload.Count];
            BinaryPrimitives.WriteUInt32LittleEndian(geosetBytes.AsSpan(0, 4), (uint)(4 + geosetPayload.Count));
            geosetPayload.CopyTo(geosetBytes, 4);
            payload.AddRange(geosetBytes);
        }

        return [.. payload];
    }

    private static byte[] CreateSeqsPayload(IReadOnlyList<(string Name, int StartTime, int EndTime, float MoveSpeed, uint Flags, float Frequency, int ReplayStart, int ReplayEnd, float BoundsMinX, float BoundsMinY, float BoundsMinZ, float BoundsMaxX, float BoundsMaxY, float BoundsMaxZ, float BoundsRadius)> sequences)
    {
        const int entrySize = 136;

        byte[] payload = new byte[4 + (sequences.Count * entrySize)];
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(0, 4), (uint)sequences.Count);
        for (int index = 0; index < sequences.Count; index++)
        {
            int offset = 4 + (index * entrySize);
            WriteFixedAscii(payload, offset, 0x50, sequences[index].Name);
            BinaryPrimitives.WriteInt32LittleEndian(payload.AsSpan(offset + 0x50, 4), sequences[index].StartTime);
            BinaryPrimitives.WriteInt32LittleEndian(payload.AsSpan(offset + 0x54, 4), sequences[index].EndTime);
            BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(offset + 0x58, 4), sequences[index].MoveSpeed);
            BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(offset + 0x5C, 4), sequences[index].Flags);
            BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(offset + 0x60, 4), sequences[index].Frequency);
            BinaryPrimitives.WriteInt32LittleEndian(payload.AsSpan(offset + 0x64, 4), sequences[index].ReplayStart);
            BinaryPrimitives.WriteInt32LittleEndian(payload.AsSpan(offset + 0x68, 4), sequences[index].ReplayEnd);
            BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(offset + 0x6C, 4), sequences[index].BoundsRadius);
            BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(offset + 0x70, 4), sequences[index].BoundsMinX);
            BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(offset + 0x74, 4), sequences[index].BoundsMinY);
            BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(offset + 0x78, 4), sequences[index].BoundsMinZ);
            BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(offset + 0x7C, 4), sequences[index].BoundsMaxX);
            BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(offset + 0x80, 4), sequences[index].BoundsMaxY);
            BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(offset + 0x84, 4), sequences[index].BoundsMaxZ);
        }

        return payload;
    }

    private static byte[] CreateSeqsPayloadNamed8C(IReadOnlyList<(string Name, int StartTime, int EndTime, float MoveSpeed, uint Flags, float BoundsMinX, float BoundsMinY, float BoundsMinZ, float BoundsMaxX, float BoundsMaxY, float BoundsMaxZ, int ReplayStart, int ReplayEnd, uint BlendTime)> sequences)
    {
        const int entrySize = 0x8C;

        byte[] payload = new byte[4 + (sequences.Count * entrySize)];
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(0, 4), (uint)sequences.Count);
        for (int index = 0; index < sequences.Count; index++)
        {
            int offset = 4 + (index * entrySize);
            WriteFixedAscii(payload, offset, 0x50, sequences[index].Name);
            BinaryPrimitives.WriteInt32LittleEndian(payload.AsSpan(offset + 0x50, 4), sequences[index].StartTime);
            BinaryPrimitives.WriteInt32LittleEndian(payload.AsSpan(offset + 0x54, 4), sequences[index].EndTime);
            BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(offset + 0x58, 4), sequences[index].MoveSpeed);
            BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(offset + 0x5C, 4), sequences[index].Flags);
            BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(offset + 0x60, 4), sequences[index].BoundsMinX);
            BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(offset + 0x64, 4), sequences[index].BoundsMinY);
            BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(offset + 0x68, 4), sequences[index].BoundsMinZ);
            BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(offset + 0x6C, 4), sequences[index].BoundsMaxX);
            BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(offset + 0x70, 4), sequences[index].BoundsMaxY);
            BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(offset + 0x74, 4), sequences[index].BoundsMaxZ);
            BinaryPrimitives.WriteInt32LittleEndian(payload.AsSpan(offset + 0x80, 4), sequences[index].ReplayStart);
            BinaryPrimitives.WriteInt32LittleEndian(payload.AsSpan(offset + 0x84, 4), sequences[index].ReplayEnd);
            BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(offset + 0x88, 4), sequences[index].BlendTime);
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

    private static byte[] CreateVector3Payload(Vector3 value)
    {
        byte[] bytes = new byte[12];
        BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(0, 4), value.X);
        BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(4, 4), value.Y);
        BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(8, 4), value.Z);
        return bytes;
    }

    private static byte[] CreateFixedAsciiPayload(string value, int length)
    {
        byte[] bytes = new byte[length];
        WriteFixedAscii(bytes, 0, length, value);
        return bytes;
    }

    private static byte[] CreateSizedPayload(List<byte> payload)
    {
        byte[] bytes = new byte[4 + payload.Count];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0, 4), (uint)(4 + payload.Count));
        payload.CopyTo(bytes, 4);
        return bytes;
    }

    private static byte[] CreateGeoaTrackChunk(string tag, uint interpolationType, int globalSequenceId, IReadOnlyList<int> times, bool isColorTrack)
    {
        List<byte> payload = [];
        payload.AddRange(Encoding.ASCII.GetBytes(tag));
        payload.AddRange(CreateUInt32Payload((uint)times.Count));
        payload.AddRange(CreateUInt32Payload(interpolationType));
        payload.AddRange(CreateInt32Payload(globalSequenceId));

        foreach (int time in times)
        {
            payload.AddRange(CreateInt32Payload(time));
            if (isColorTrack)
            {
                payload.AddRange(CreateVector3Payload(new Vector3(1.0f, 1.0f, 1.0f)));
                if (interpolationType >= 2u)
                {
                    payload.AddRange(CreateVector3Payload(Vector3.Zero));
                    payload.AddRange(CreateVector3Payload(Vector3.Zero));
                }
            }
            else
            {
                payload.AddRange(CreateSinglePayload(1.0f));
                if (interpolationType >= 2u)
                {
                    payload.AddRange(CreateSinglePayload(0.0f));
                    payload.AddRange(CreateSinglePayload(0.0f));
                }
            }
        }

        return [.. payload];
    }

    private static byte[] CreateNodeTrackChunk(string tag, uint interpolationType, int globalSequenceId, IReadOnlyList<int> times, TrackValueKind valueKind)
    {
        List<byte> payload = [];
        payload.AddRange(Encoding.ASCII.GetBytes(tag));
        payload.AddRange(CreateUInt32Payload((uint)times.Count));
        payload.AddRange(CreateUInt32Payload(interpolationType));
        payload.AddRange(CreateInt32Payload(globalSequenceId));

        foreach (int time in times)
        {
            payload.AddRange(CreateInt32Payload(time));
            payload.AddRange(CreateBoneTrackValuePayload(valueKind));
            if (interpolationType >= 2u)
            {
                payload.AddRange(CreateBoneTrackValuePayload(valueKind));
                payload.AddRange(CreateBoneTrackValuePayload(valueKind));
            }
        }

        return [.. payload];
    }

    private static byte[] CreateSimpleEventTrackChunk(string tag, int globalSequenceId, IReadOnlyList<int> times)
    {
        List<byte> payload = [];
        payload.AddRange(Encoding.ASCII.GetBytes(tag));
        payload.AddRange(CreateUInt32Payload((uint)times.Count));
        payload.AddRange(CreateInt32Payload(globalSequenceId));

        foreach (int time in times)
            payload.AddRange(CreateInt32Payload(time));

        return [.. payload];
    }

    private static byte[] CreateBoneTrackValuePayload(TrackValueKind valueKind)
    {
        return valueKind switch
        {
            TrackValueKind.Vector3 => CreateVector3Payload(new Vector3(1.0f, 2.0f, 3.0f)),
            TrackValueKind.Quaternion => [1, 2, 3, 4, 5, 6, 7, 8],
            TrackValueKind.Scalar => CreateSinglePayload(1.0f),
            TrackValueKind.Int32 => CreateInt32Payload(1),
            _ => throw new ArgumentOutOfRangeException(nameof(valueKind), valueKind, null),
        };
    }

    private static void WriteTagAndCount(List<byte> bytes, string tag, int count)
    {
        bytes.AddRange(Encoding.ASCII.GetBytes(tag));
        bytes.AddRange(CreateInt32Payload(count));
    }
}

internal enum TrackValueKind
{
    Vector3,
    Quaternion,
    Scalar,
    Int32,
}

internal static class MdxTestPaths
{
    public const string Standard060MdxVirtualPath = "world/generic/activedoodads/chest01/chest01.mdx";
    public const string Standard060LightMdxVirtualPath = "world/generic/dwarf/passive doodads/braziers/dwarvenbrazier01.mdx";

    public static readonly string Alpha053TreePath = Path.Combine(GetWowViewerRoot(), "testdata", "0.5.3", "tree");
    public static readonly string Alpha053WispMdxPath = Path.Combine(GetWowViewerRoot(), "testdata", "0.5.3", "tree", "Creature", "Wisp", "Wisp.mdx");

    public static readonly string[] Standard060AnimatedMdxCandidates =
    [
        "world/generic/passivedoodads/particleemitters/greengroundfog.mdx",
        "world/azeroth/burningsteppes/passivedoodads/fallingembers/fallingembers.mdx",
        "world/azeroth/burningsteppes/passivedoodads/lavafalls/lavafallsblackrock01.mdx",
        "world/azeroth/burningsteppes/passivedoodads/volcanicvents/volcanicventlarge01.mdx",
    ];

    public static readonly string[] Standard060PivotMdxCandidates =
    [
        Standard060MdxVirtualPath,
        "world/generic/passivedoodads/particleemitters/greengroundfog.mdx",
        "world/azeroth/burningsteppes/passivedoodads/lavafalls/lavafallsblackrock01.mdx",
    ];

    public static readonly string[] Standard060GeoaMdxCandidates =
    [
        "world/generic/passivedoodads/lights/freestandingtorch01.mdx",
        "world/generic/dwarf/passive doodads/braziers/dwarvenbrazier01.mdx",
        "world/generic/human/passive doodads/smokestack/smokestack.mdx",
        "world/azeroth/burningsteppes/passivedoodads/smoke/ashtreesmoke01.mdx",
        "world/generic/ogre/passive doodads/ogremoundvent/ogresmokevent01.mdx",
    ];

    public static readonly string[] Alpha053GeoaMdxCandidates =
    [
        Alpha053WispMdxPath,
        Path.Combine(GetWowViewerRoot(), "testdata", "0.5.3", "tree", "Creature", "WaterElemental", "WaterElemental.mdx"),
        Path.Combine(GetWowViewerRoot(), "testdata", "0.5.3", "tree", "Creature", "Banshee", "Banshee.mdx"),
        Path.Combine(GetWowViewerRoot(), "testdata", "0.5.3", "tree", "Creature", "Spells", "HealingTotem.mdx"),
    ];

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