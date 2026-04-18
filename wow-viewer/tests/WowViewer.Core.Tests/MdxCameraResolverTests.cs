using System.Numerics;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.Tests;

public sealed class MdxCameraResolverTests
{
    [Fact]
    public void Resolve_AddsAnimatedOffsetsToStaticCameraPivots()
    {
        MdxSummary summary = CreateSummary();
        MdxCamera camera = new(
            0,
            "Portrait",
            new Vector3(10.0f, 0.0f, 0.0f),
            0.95f,
            27.0f,
            0.2f,
            new Vector3(10.0f, 1.0f, 0.0f),
            new MdxVector3NodeTrack(
                "KCTR",
                MdxTrackInterpolationType.Linear,
                -1,
                [
                    new MdxVector3Keyframe(100, Vector3.Zero, null, null),
                    new MdxVector3Keyframe(200, new Vector3(2.0f, 0.0f, 0.0f), null, null),
                ]),
            null,
            null,
            new MdxVector3NodeTrack(
                "KTTR",
                MdxTrackInterpolationType.Linear,
                -1,
                [
                    new MdxVector3Keyframe(100, Vector3.Zero, null, null),
                    new MdxVector3Keyframe(200, new Vector3(0.0f, 3.0f, 0.0f), null, null),
                ]));

        MdxResolvedCameraState state = MdxCameraResolver.Resolve(summary, camera, sequenceIndex: 0, timeMs: 50);

        AssertVector3Equal(new Vector3(11.0f, 0.0f, 0.0f), state.Position);
        AssertVector3Equal(new Vector3(10.0f, 2.5f, 0.0f), state.Target);
        Assert.True(state.Visible);
    }

    [Fact]
    public void Resolve_RollTrackRotatesUpVectorAroundForwardAxis()
    {
        MdxSummary summary = CreateSummary();
        MdxCamera camera = new(
            0,
            "Portrait",
            Vector3.Zero,
            0.95f,
            27.0f,
            0.2f,
            Vector3.UnitY,
            null,
            new MdxScalarTrack(
                "KCRL",
                MdxTrackInterpolationType.None,
                -1,
                [
                    new MdxScalarKeyframe(100, MathF.PI * 0.5f, null, null),
                ]),
            null,
            null);

        MdxResolvedCameraState state = MdxCameraResolver.Resolve(summary, camera, sequenceIndex: 0, timeMs: 0);

        AssertVector3Equal(Vector3.UnitX, state.Up);
    }

    [Fact]
    public void Resolve_VisibilityTrackCanHideCamera()
    {
        MdxSummary summary = CreateSummary();
        MdxCamera camera = new(
            0,
            "Portrait",
            Vector3.Zero,
            0.95f,
            27.0f,
            0.2f,
            Vector3.UnitY,
            null,
            null,
            new MdxScalarTrack(
                "KVIS",
                MdxTrackInterpolationType.None,
                -1,
                [
                    new MdxScalarKeyframe(100, 0.0f, null, null),
                ]),
            null);

        MdxResolvedCameraState state = MdxCameraResolver.Resolve(summary, camera, sequenceIndex: 0, timeMs: 0);

        Assert.False(state.Visible);
    }

    private static MdxSummary CreateSummary()
    {
        return new MdxSummary(
            "synthetic.mdx",
            "MDLX",
            1300u,
            "Synthetic",
            0u,
            null,
            null,
            [],
            [new MdxSequenceSummary(0, "Stand", 100, 200, 0.0f, 0u, 0.0f, 0, 100, null, null, null, null)],
            [],
            [],
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
            [],
            [],
            [],
            0,
            0);
    }

    private static void AssertVector3Equal(Vector3 expected, Vector3 actual)
    {
        Assert.Equal(expected.X, actual.X, 4);
        Assert.Equal(expected.Y, actual.Y, 4);
        Assert.Equal(expected.Z, actual.Z, 4);
    }
}
