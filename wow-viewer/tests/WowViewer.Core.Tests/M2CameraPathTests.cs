using System.Numerics;
using WowViewer.Core.IO.M2;
using WowViewer.Core.Runtime.M2;

namespace WowViewer.Core.Tests;

public sealed class M2CameraPathTests
{
    [Fact]
    public void Evaluator_InterpolatesPositionTargetFovAndRoll()
    {
        M2CameraPathDocument path = new()
        {
            Interpolation = M2CameraPathInterpolation.Linear,
            Keyframes =
            [
                new() { TimeMs = 0, Position = Vector3.Zero, Target = Vector3.UnitY, FovDegrees = 40f, RollDegrees = 0f },
                new() { TimeMs = 1000, Position = new Vector3(10f, 0f, 5f), Target = new Vector3(0f, 10f, 5f), FovDegrees = 60f, RollDegrees = 20f },
            ],
        };

        M2CameraPathSample sample = M2CameraPathEvaluator.Sample(path, 500);

        Assert.Equal(new Vector3(5f, 0f, 2.5f), sample.Position);
        Assert.Equal(new Vector3(0f, 5.5f, 2.5f), sample.Target);
        Assert.Equal(50f, sample.FovDegrees);
        Assert.Equal(10f, sample.RollDegrees);
    }

    [Fact]
    public void Evaluator_CatmullRomUsesNeighborKeysAndLoops()
    {
        M2CameraPathDocument path = new()
        {
            Keyframes =
            [
                new() { TimeMs = 0, Position = new Vector3(0f, 0f, 0f), Target = Vector3.UnitY },
                new() { TimeMs = 100, Position = new Vector3(10f, 0f, 0f), Target = Vector3.UnitY },
                new() { TimeMs = 200, Position = new Vector3(20f, 10f, 0f), Target = Vector3.UnitY },
                new() { TimeMs = 300, Position = new Vector3(30f, 10f, 0f), Target = Vector3.UnitY },
            ],
        };

        M2CameraPathSample middle = M2CameraPathEvaluator.Sample(path, 150);
        M2CameraPathSample looped = M2CameraPathEvaluator.Sample(path, 300, loop: true);

        Assert.True(middle.Position.Y > 0f && middle.Position.Y < 10f);
        Assert.Equal(path.Keyframes[0].Position, looped.Position);
    }

    [Fact]
    public void NativeWriter_RoundTripsCameraTrackDataThroughExistingReader()
    {
        M2CameraPathDocument path = new()
        {
            Name = "StormwindFlyby",
            MapName = "Azeroth",
            BuildVersion = "4.0.0.11927",
            Keyframes =
            [
                new() { TimeMs = 0, Position = new Vector3(1f, 2f, 3f), Target = new Vector3(4f, 5f, 6f), FovDegrees = 35f },
                new() { TimeMs = 500, Position = new Vector3(11f, 12f, 13f), Target = new Vector3(14f, 15f, 16f), FovDegrees = 55f },
            ],
        };

        byte[] bytes = M2CameraPathWriter.Build(path);
        using MemoryStream stream = new(bytes, writable: false);
        var model = WowViewer.Core.IO.M2.M2ModelReader.Read(stream, "Cameras\\StormwindFlyby.m2");

        Assert.Equal(1, model.CameraCount);
        M2CameraPathDocument imported = M2CameraPathImporter.Import(model, sampleIntervalMs: 500);
        Assert.Equal(2, imported.Keyframes.Count);
        Assert.Equal(path.Keyframes[0].Position, imported.Keyframes[0].Position);
        Assert.Equal(path.Keyframes[1].Target, imported.Keyframes[1].Target);
        Assert.Equal(35f, imported.Keyframes[0].FovDegrees, 2);
    }
}
