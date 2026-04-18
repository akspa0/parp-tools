using System.Numerics;
using WowViewer.Core.Mdx;
using WowViewer.Core.Runtime.Mdx;

namespace WowViewer.Core.Tests;

public sealed class MdxEffectRuntimeEvaluatorTests
{
    [Fact]
    public void Evaluate_SamplesPre2RibbonAndEventRuntimeState()
    {
        MdxSummary summary = CreateSummary();
        MdxEventFile events = new(
            "synthetic.mdx",
            "MDLX",
            1300,
            "Synthetic",
            [
                new MdxEvent(
                    0,
                    "Footstep",
                    10,
                    -1,
                    0,
                    new Vector3(1.0f, 2.0f, 3.0f),
                    new MdxVector3NodeTrack("KGTR", MdxTrackInterpolationType.Linear, -1, [new MdxVector3Keyframe(0, Vector3.Zero, null, null), new MdxVector3Keyframe(100, new Vector3(2.0f, 0.0f, 0.0f), null, null)]),
                    null,
                    null,
                    new MdxEventTrack("KEVT", -1, [50, 100, 150]))
            ]);

        MdxParticleEmitter2File particles = new(
            "synthetic.mdx",
            "MDLX",
            1300,
            "Synthetic",
            [
                new MdxParticleEmitter2(
                    0,
                    "Dust",
                    20,
                    -1,
                    0,
                    new Vector3(4.0f, 0.0f, 0.0f),
                    1,
                    5.0f,
                    0.0f,
                    0.25f,
                    0.5f,
                    -2.0f,
                    0.0f,
                    2.0f,
                    1.0f,
                    3.0f,
                    4.0f,
                    1,
                    1,
                    0,
                    0.0f,
                    0.5f,
                    new Vector3(1.0f, 0.0f, 0.0f),
                    new Vector3(0.0f, 1.0f, 0.0f),
                    new Vector3(0.0f, 0.0f, 1.0f),
                    255,
                    255,
                    255,
                    1.0f,
                    1.0f,
                    1.0f,
                    Array.Empty<uint>(),
                    3,
                    7,
                    0,
                    0,
                    null,
                    null,
                    Array.Empty<float>(),
                    Array.Empty<float>(),
                    Array.Empty<float>(),
                    Vector3.Zero,
                    Array.Empty<float>(),
                    0,
                    Array.Empty<Vector3>(),
                    0,
                    new MdxVector3NodeTrack("KGTR", MdxTrackInterpolationType.Linear, -1, [new MdxVector3Keyframe(0, Vector3.Zero, null, null), new MdxVector3Keyframe(100, new Vector3(0.0f, 3.0f, 0.0f), null, null)]),
                    null,
                    null,
                    new MdxScalarTrack("KVIS", MdxTrackInterpolationType.Linear, -1, [new MdxScalarKeyframe(0, 0.0f, null, null), new MdxScalarKeyframe(100, 1.0f, null, null)]),
                    new MdxScalarTrack("KP2S", MdxTrackInterpolationType.Linear, -1, [new MdxScalarKeyframe(0, 2.0f, null, null), new MdxScalarKeyframe(100, 6.0f, null, null)]),
                    null,
                    null,
                    null,
                    null,
                    new MdxScalarTrack("KLIF", MdxTrackInterpolationType.Linear, -1, [new MdxScalarKeyframe(0, 1.0f, null, null), new MdxScalarKeyframe(100, 2.0f, null, null)]),
                    new MdxScalarTrack("KP2E", MdxTrackInterpolationType.Linear, -1, [new MdxScalarKeyframe(0, 1.0f, null, null), new MdxScalarKeyframe(100, 3.0f, null, null)]),
                    null,
                    null,
                    null)
            ]);

        MdxRibbonEmitterFile ribbons = new(
            "synthetic.mdx",
            "MDLX",
            1300,
            "Synthetic",
            [
                new MdxRibbonEmitter(
                    0,
                    "Trail",
                    30,
                    -1,
                    0,
                    new Vector3(0.0f, 1.0f, 0.0f),
                    2.0f,
                    1.0f,
                    0.25f,
                    new Vector3(0.5f, 0.5f, 0.5f),
                    2.0f,
                    0,
                    4,
                    2,
                    3,
                    9,
                    -1.0f,
                    new MdxVector3NodeTrack("KGTR", MdxTrackInterpolationType.Linear, -1, [new MdxVector3Keyframe(0, Vector3.Zero, null, null), new MdxVector3Keyframe(100, new Vector3(1.0f, 1.0f, 0.0f), null, null)]),
                    null,
                    null,
                    null,
                    null,
                    new MdxScalarTrack("KRAL", MdxTrackInterpolationType.Linear, -1, [new MdxScalarKeyframe(0, 0.0f, null, null), new MdxScalarKeyframe(100, 0.75f, null, null)]),
                    new MdxColorTrack("KRCO", MdxTrackInterpolationType.Linear, -1, [new MdxColorKeyframe(0, Vector3.Zero, null, null), new MdxColorKeyframe(100, new Vector3(0.2f, 0.4f, 0.6f), null, null)]),
                    new MdxIntTrack("KRTX", MdxTrackInterpolationType.None, -1, [new MdxIntKeyframe(0, 1, null, null), new MdxIntKeyframe(100, 2, null, null)]),
                    new MdxScalarTrack("KVIS", MdxTrackInterpolationType.Linear, -1, [new MdxScalarKeyframe(0, 0.0f, null, null), new MdxScalarKeyframe(100, 1.0f, null, null)]))
            ]);

        MdxEffectRuntimeState runtime = MdxEffectRuntimeEvaluator.Evaluate(summary, events, particles, ribbons, sequenceIndex: 0, timeMs: 100);

        Assert.Equal(1, runtime.TriggeredEventCount);
        Assert.Equal(1, runtime.VisibleParticleEmitterCount);
        Assert.Equal(1, runtime.VisibleRibbonEmitterCount);

        MdxEventRuntimeState effectEvent = Assert.Single(runtime.Events);
        Assert.True(effectEvent.Triggered);
        Assert.Equal("KEVT", effectEvent.Tag);
        Assert.Equal(new Vector3(3.0f, 2.0f, 3.0f), effectEvent.Position);

        MdxParticleEmitter2RuntimeState particle = Assert.Single(runtime.Particles);
        Assert.True(particle.Enabled);
        Assert.Equal(new Vector3(4.0f, 3.0f, 0.0f), particle.Position);
        Assert.Equal(6.0f, particle.Speed, 5);
        Assert.Equal(3.0f, particle.EmissionRate, 5);
        Assert.Equal(2.0f, particle.Life, 5);
        Assert.Equal(6, particle.EstimatedParticleCount);
        Assert.Equal("Particle_Additive", particle.EffectKey);

        MdxRibbonRuntimeState ribbon = Assert.Single(runtime.Ribbons);
        Assert.True(ribbon.Visible);
        Assert.Equal(new Vector3(1.0f, 2.0f, 0.0f), ribbon.Position);
        Assert.Equal(new Vector3(0.2f, 0.4f, 0.6f), ribbon.Color);
        Assert.Equal(0.75f, ribbon.Alpha, 5);
        Assert.Equal(2, ribbon.TextureSlot);
        Assert.Equal(8, ribbon.EstimatedEdgeCount);
        Assert.Equal("Ribbon_Material_9", ribbon.EffectKey);
    }

    private static MdxSummary CreateSummary()
    {
        return new MdxSummary(
            "synthetic.mdx",
            "MDLX",
            1300,
            "Synthetic",
            null,
            null,
            null,
            Array.Empty<MdxGlobalSequenceSummary>(),
            Array.Empty<MdxSequenceSummary>(),
            Array.Empty<MdxGeosetSummary>(),
            Array.Empty<MdxGeosetAnimationSummary>(),
            Array.Empty<MdxBoneSummary>(),
            Array.Empty<MdxLightSummary>(),
            Array.Empty<MdxHelperSummary>(),
            Array.Empty<MdxAttachmentSummary>(),
            Array.Empty<MdxParticleEmitter2Summary>(),
            Array.Empty<MdxRibbonEmitterSummary>(),
            Array.Empty<MdxCameraSummary>(),
            Array.Empty<MdxEventSummary>(),
            Array.Empty<MdxHitTestShapeSummary>(),
            null,
            Array.Empty<MdxPivotPointSummary>(),
            Array.Empty<MdxTextureSummary>(),
            Array.Empty<MdxMaterialSummary>(),
            Array.Empty<MdxChunkSummary>(),
            0,
            0);
    }
}