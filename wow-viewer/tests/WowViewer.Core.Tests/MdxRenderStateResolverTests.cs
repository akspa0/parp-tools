using System.Numerics;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.Tests;

public sealed class MdxRenderStateResolverTests
{
    [Fact]
    public void ResolveMaterial_TransparentKeyLayer_ProducesExpectedCutoutState()
    {
        MdxSummary summary = CreateSummary(
            textures:
            [
                new MdxTextureSummary(0, 11u, null, 0u),
            ],
            materials:
            [
                new MdxMaterialSummary(0, 0,
                [
                    new MdxMaterialLayerSummary(0, 1u, 0u, 0, 2, 3, 0.5f),
                ]),
            ]);

        MdxResolvedMaterialState state = MdxRenderStateResolver.ResolveMaterial(summary, 0);

        Assert.Null(state.TexturePath);
        Assert.Equal(11u, state.ReplaceableId);
        Assert.Equal(2, state.TransformId);
        Assert.Equal(3, state.CoordId);
        Assert.True(state.IsTransparent);
        Assert.False(state.IsAdditive);
        Assert.True(state.DepthWrite);
        Assert.True(state.AlphaCutout);
        Assert.Equal(0.5f, state.Alpha, 4);
        Assert.Equal(1u, state.BlendMode);
    }

    [Fact]
    public void ResolveMaterial_AdditiveLayer_DisablesDepthWrite()
    {
        MdxSummary summary = CreateSummary(
            textures:
            [
                new MdxTextureSummary(0, 0u, @"Textures\Torch.blp", 0u),
            ],
            materials:
            [
                new MdxMaterialSummary(0, 0,
                [
                    new MdxMaterialLayerSummary(0, 3u, 0u, 0, -1, 0, 1.0f),
                ]),
            ]);

        MdxResolvedMaterialState state = MdxRenderStateResolver.ResolveMaterial(summary, 0);

        Assert.Equal(@"Textures\Torch.blp", state.TexturePath);
        Assert.True(state.IsTransparent);
        Assert.True(state.IsAdditive);
        Assert.False(state.DepthWrite);
        Assert.False(state.AlphaCutout);
        Assert.Equal(1.0f, state.Alpha, 4);
        Assert.Equal(3u, state.BlendMode);
    }

    [Fact]
    public void ResolveGeosetRenderState_RuntimeAnimationAndFlags_ProduceExpectedState()
    {
        MdxSummary summary = CreateSummary(
            sequences:
            [
                new MdxSequenceSummary(0, "Stand", 0, 100, 0.0f, 0u, 0.0f, 0, 100, null, null, null, null),
            ]);
        MdxResolvedMaterialState material = new(
            TexturePath: null,
            ReplaceableId: 0u,
            TransformId: -1,
            CoordId: 0,
            IsTransparent: false,
            IsAdditive: false,
            DepthWrite: true,
            AlphaCutout: false,
            Alpha: 0.8f,
            BlendMode: 0u);
        MdxGeosetGeometry geoset = CreateGeoset(index: 0, flags: 0x1u | 0x40u | 0x80u);
        MdxGeosetAnimationFile geosetAnimations = new(
            "synthetic.mdx",
            "MDLX",
            1300u,
            "Synthetic",
            [
                new MdxGeosetAnimation(
                    0,
                    0u,
                    1.0f,
                    new Vector3(1.0f, 1.0f, 1.0f),
                    0x1u,
                    new MdxScalarTrack(
                        "KGAO",
                        MdxTrackInterpolationType.Linear,
                        -1,
                        [
                            new MdxScalarKeyframe(0, 0.25f, null, null),
                            new MdxScalarKeyframe(100, 0.75f, null, null),
                        ]),
                    new MdxColorTrack(
                        "KGAC",
                        MdxTrackInterpolationType.Linear,
                        -1,
                        [
                            new MdxColorKeyframe(0, new Vector3(1.0f, 1.0f, 1.0f), null, null),
                            new MdxColorKeyframe(100, new Vector3(0.2f, 0.4f, 0.6f), null, null),
                        ])),
            ]);

        MdxResolvedGeosetRenderState state = MdxRenderStateResolver.ResolveGeosetRenderState(summary, geosetAnimations, 0, 50, geoset, material);

        Assert.False(state.ReceivesLighting);
        Assert.False(state.DepthTest);
        Assert.False(state.DepthWrite);
        Assert.Equal(0.4f, state.Alpha, 4);
        AssertVector3Equal(new Vector3(0.6f, 0.7f, 0.8f), state.BaseColor);
    }

    [Fact]
    public void ResolveGeosetRenderState_FallsBackToSummaryStaticSignalsWhenRuntimeAnimationIsMissing()
    {
        MdxSummary summary = CreateSummary(
            geosetAnimations:
            [
                new MdxGeosetAnimationSummary(0, 0u, 0.25f, new Vector3(0.4f, 0.6f, 0.8f), 0x1u, null, null),
            ]);
        MdxResolvedMaterialState material = new(
            TexturePath: null,
            ReplaceableId: 0u,
            TransformId: -1,
            CoordId: 0,
            IsTransparent: false,
            IsAdditive: false,
            DepthWrite: true,
            AlphaCutout: false,
            Alpha: 0.5f,
            BlendMode: 0u);
        MdxGeosetGeometry geoset = CreateGeoset(index: 0, flags: 0u);

        MdxResolvedGeosetRenderState state = MdxRenderStateResolver.ResolveGeosetRenderState(summary, null, 0, 0, geoset, material);

        Assert.True(state.ReceivesLighting);
        Assert.True(state.DepthTest);
        Assert.True(state.DepthWrite);
        Assert.Equal(0.125f, state.Alpha, 4);
        AssertVector3Equal(new Vector3(0.4f, 0.6f, 0.8f), state.BaseColor);
    }

    [Fact]
    public void ResolveTextureTransform_AnimatedTracks_ProduceExpectedTransform()
    {
        MdxSummary summary = CreateSummary(
            sequences:
            [
                new MdxSequenceSummary(0, "Stand", 0, 100, 0.0f, 0u, 0.0f, 0, 100, null, null, null, null),
            ]);
        MdxResolvedMaterialState material = new(
            TexturePath: null,
            ReplaceableId: 0u,
            TransformId: 0,
            CoordId: 0,
            IsTransparent: false,
            IsAdditive: false,
            DepthWrite: true,
            AlphaCutout: false,
            Alpha: 1.0f,
            BlendMode: 0u);
        Quaternion rotation = Quaternion.CreateFromAxisAngle(Vector3.UnitZ, MathF.PI / 2.0f);
        MdxTextureAnimationFile textureAnimations = new(
            "synthetic.mdx",
            "MDLX",
            1300u,
            "Synthetic",
            [
                new MdxTextureAnimation(
                    0,
                    new MdxVector3NodeTrack(
                        "KTAT",
                        MdxTrackInterpolationType.None,
                        -1,
                        [
                            new MdxVector3Keyframe(0, new Vector3(2.0f, 3.0f, 0.0f), null, null),
                        ]),
                    new MdxQuaternionNodeTrack(
                        "KTAR",
                        MdxTrackInterpolationType.None,
                        -1,
                        [
                            new MdxQuaternionKeyframe(0, rotation, null, null),
                        ]),
                    new MdxVector3NodeTrack(
                        "KTAS",
                        MdxTrackInterpolationType.None,
                        -1,
                        [
                            new MdxVector3Keyframe(0, new Vector3(0.75f, 1.5f, 1.0f), null, null),
                        ])),
            ]);
        Matrix4x4 expectedRotationMatrix = Matrix4x4.CreateFromQuaternion(rotation);

        MdxResolvedTextureTransform state = MdxRenderStateResolver.ResolveTextureTransform(summary, textureAnimations, 0, 0, material);

        Assert.True(state.UsesTransform);
        AssertVector2Equal(new Vector2(2.0f, 3.0f), state.Translation);
        AssertVector2Equal(new Vector2(0.75f, 1.5f), state.Scale);
        AssertVector2Equal(new Vector2(expectedRotationMatrix.M11, expectedRotationMatrix.M12), state.RotationRow0);
        AssertVector2Equal(new Vector2(expectedRotationMatrix.M21, expectedRotationMatrix.M22), state.RotationRow1);
    }

    private static MdxSummary CreateSummary(
        IReadOnlyList<MdxMaterialSummary>? materials = null,
        IReadOnlyList<MdxTextureSummary>? textures = null,
        IReadOnlyList<MdxSequenceSummary>? sequences = null,
        IReadOnlyList<MdxGlobalSequenceSummary>? globalSequences = null,
        IReadOnlyList<MdxGeosetAnimationSummary>? geosetAnimations = null)
    {
        return new MdxSummary(
            "synthetic.mdx",
            "MDLX",
            1300u,
            "Synthetic",
            0u,
            null,
            null,
            globalSequences ?? [],
            sequences ?? [],
            [],
            geosetAnimations ?? [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            null,
            [],
            textures ?? [],
            materials ?? [],
            [],
            0,
            0);
    }

    private static MdxGeosetGeometry CreateGeoset(int index, uint flags)
    {
        return new MdxGeosetGeometry(
            index,
            [Vector3.Zero, Vector3.UnitX, Vector3.UnitY],
            [Vector3.UnitZ, Vector3.UnitZ, Vector3.UnitZ],
            [[Vector2.Zero, Vector2.UnitX, Vector2.UnitY]],
            [(byte)4],
            [3],
            [(ushort)0, (ushort)1, (ushort)2],
            [],
            [],
            [],
            [],
            [],
            0,
            0u,
            flags,
            null,
            null,
            null,
            0);
    }

    private static void AssertVector2Equal(Vector2 expected, Vector2 actual, float tolerance = 0.0001f)
    {
        Assert.InRange(actual.X, expected.X - tolerance, expected.X + tolerance);
        Assert.InRange(actual.Y, expected.Y - tolerance, expected.Y + tolerance);
    }

    private static void AssertVector3Equal(Vector3 expected, Vector3 actual, float tolerance = 0.0001f)
    {
        Assert.InRange(actual.X, expected.X - tolerance, expected.X + tolerance);
        Assert.InRange(actual.Y, expected.Y - tolerance, expected.Y + tolerance);
        Assert.InRange(actual.Z, expected.Z - tolerance, expected.Z + tolerance);
    }
}
