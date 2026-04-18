using System.Numerics;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.Tests;

public sealed class MdxAnimationSamplerTests
{
    [Fact]
    public void SampleVector3Track_SequenceBound_UsesSequenceStartPlusWrappedTime()
    {
        MdxSummary summary = CreateSummary();
        MdxVector3NodeTrack track = new(
            "KTAT",
            MdxTrackInterpolationType.Linear,
            globalSequenceId: -1,
            [
                new MdxVector3Keyframe(100, Vector3.Zero, null, null),
                new MdxVector3Keyframe(200, new Vector3(10.0f, 0.0f, 0.0f), null, null),
            ]);

        Vector3 sampled = MdxAnimationSampler.SampleVector3Track(track, summary, sequenceIndex: 0, timeMs: 50, defaultValue: new Vector3(-1.0f));

        Assert.Equal(new Vector3(5.0f, 0.0f, 0.0f), sampled);
    }

    [Fact]
    public void SampleScalarTrack_GlobalSequence_UsesGlobalDurationWrapping()
    {
        MdxSummary summary = CreateSummary();
        MdxScalarTrack track = new(
            "KGAO",
            MdxTrackInterpolationType.Linear,
            globalSequenceId: 0,
            [
                new MdxScalarKeyframe(0, 0.0f, null, null),
                new MdxScalarKeyframe(100, 1.0f, null, null),
            ]);

        float sampled = MdxAnimationSampler.SampleScalarTrack(track, summary, sequenceIndex: 0, timeMs: 250, defaultValue: -1.0f);

        Assert.Equal(0.5f, sampled, 4);
    }

    [Fact]
    public void SampleQuaternionTrack_Linear_UsesSlerp()
    {
        MdxSummary summary = CreateSummary();
        Quaternion end = Quaternion.CreateFromAxisAngle(Vector3.UnitZ, MathF.PI);
        MdxQuaternionNodeTrack track = new(
            "KTAR",
            MdxTrackInterpolationType.Linear,
            globalSequenceId: -1,
            [
                new MdxQuaternionKeyframe(100, Quaternion.Identity, null, null),
                new MdxQuaternionKeyframe(200, end, null, null),
            ]);

        Quaternion sampled = MdxAnimationSampler.SampleQuaternionTrack(track, summary, sequenceIndex: 0, timeMs: 50, defaultValue: Quaternion.Identity);
        Vector3 rotated = Vector3.Transform(Vector3.UnitX, sampled);

        Assert.Equal(0.0f, rotated.X, 4);
        Assert.Equal(-1.0f, rotated.Y, 4);
    }

    private static MdxSummary CreateSummary()
    {
        return new MdxSummary(
            sourcePath: "synthetic.mdx",
            signature: "MDLX",
            version: 1300,
            modelName: "Synthetic",
            blendTime: 0,
            boundsMin: Vector3.Zero,
            boundsMax: Vector3.One,
            globalSequences: [new MdxGlobalSequenceSummary(0, 200)],
            sequences: [new MdxSequenceSummary(0, "Stand", 100, 200, 0.0f, 0u, 1.0f, 0, 0, 0u, Vector3.Zero, Vector3.One, 1.0f)],
            geosets: [],
            geosetAnimations: [],
            bones: [],
            lights: [],
            helpers: [],
            attachments: [],
            particleEmitters2: [],
            ribbons: [],
            cameras: [],
            events: [],
            hitTestShapes: [],
            collision: null,
            pivotPoints: [],
            textures: [],
            materials: [],
            chunks: [],
            knownChunkCount: 0,
            unknownChunkCount: 0);
    }
}
