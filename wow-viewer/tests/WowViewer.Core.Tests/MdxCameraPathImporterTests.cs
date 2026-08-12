using System.Numerics;
using WowViewer.Core.Mdx;
using WowViewer.Core.Runtime.M2;

namespace WowViewer.Core.Tests;

public sealed class MdxCameraPathImporterTests
{
    [Fact]
    public void Import_SamplesClassicCameraTracksIntoPathKeys()
    {
        MdxCamera camera = new(
            0,
            "Flyby",
            new Vector3(10f, 20f, 30f),
            0.95f,
            500f,
            0.1f,
            new Vector3(10f, 20f, 25f),
            new MdxVector3NodeTrack(
                "KCTR",
                MdxTrackInterpolationType.Linear,
                -1,
                [
                    new MdxVector3Keyframe(0, Vector3.Zero, null, null),
                    new MdxVector3Keyframe(100, new Vector3(10f, 0f, 0f), null, null),
                ]),
            null,
            null,
            new MdxVector3NodeTrack(
                "KTTR",
                MdxTrackInterpolationType.Linear,
                -1,
                [
                    new MdxVector3Keyframe(0, Vector3.Zero, null, null),
                    new MdxVector3Keyframe(100, new Vector3(0f, 10f, 0f), null, null),
                ]));

        MdxCameraFile cameraFile = new(
            "World\\Cameras\\Flyby.mdx",
            "MDLX",
            1300u,
            "Flyby",
            [camera]);
        MdxSummary summary = CreateSummary();

        M2CameraPathDocument path = MdxCameraPathImporter.Import(cameraFile, summary, sampleIntervalMs: 50);

        Assert.Equal("Flyby", path.Name);
        Assert.Equal(2, path.Keyframes.Count);
        Assert.Equal(new Vector3(10f, 20f, 30f), path.Keyframes[0].Position);
        Assert.True(Vector3.Distance(path.Keyframes[1].Position, new Vector3(19.9f, 20f, 30f)) < 0.01f);
        Assert.Equal(new Vector3(10f, 20f, 25f), path.Keyframes[0].Target);
        Assert.Equal(0.95f * (180f / MathF.PI), path.Keyframes[0].FovDegrees, 3);
    }

    private static MdxSummary CreateSummary()
        => new(
            "World\\Cameras\\Flyby.mdx",
            "MDLX",
            1300u,
            "Flyby",
            0u,
            null,
            null,
            [],
            [new MdxSequenceSummary(0, "Stand", 0, 100, 0f, 0u, 0f, 0, 100, null, null, null, null)],
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
