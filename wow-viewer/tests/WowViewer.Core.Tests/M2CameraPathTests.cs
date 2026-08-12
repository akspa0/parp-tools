using System.Numerics;
using System.Text.Json;
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

    [Fact]
    public void JsonOptions_PersistVectorCameraPositionTargetAndRoll()
    {
        M2CameraPathDocument path = new()
        {
            Keyframes =
            [
                new()
                {
                    TimeMs = 250,
                    Position = new Vector3(1.25f, -2.5f, 3.75f),
                    Target = new Vector3(4.5f, 5.5f, 6.5f),
                    FovDegrees = 52f,
                    RollDegrees = -17.5f,
                },
            ],
        };

        string json = JsonSerializer.Serialize(path, M2CameraPathJson.CreateOptions());
        M2CameraPathDocument? roundTrip = JsonSerializer.Deserialize<M2CameraPathDocument>(json, M2CameraPathJson.CreateOptions());

        Assert.NotNull(roundTrip);
        M2CameraPathKeyframe key = Assert.Single(roundTrip!.Keyframes);
        Assert.Equal(path.Keyframes[0].Position, key.Position);
        Assert.Equal(path.Keyframes[0].Target, key.Target);
        Assert.Equal(path.Keyframes[0].RollDegrees, key.RollDegrees);
        Assert.Contains("\"X\":1.25", json, StringComparison.Ordinal);
    }

    [Fact]
    public void Placement_AppliesCinematicCameraOriginAndFacingOnce()
    {
        M2CameraPathDocument path = new()
        {
            CoordinatesAreWorldSpace = false,
            Keyframes =
            [
                new()
                {
                    Position = new Vector3(1f, 0f, 2f),
                    Target = new Vector3(1f, 1f, 2f),
                },
            ],
        };

        M2CameraPathPlacement.ApplyCinematicCameraOrigin(
            path,
            cameraId: 2,
            modelPath: "Cameras\\FlybyUndead.mdx",
            origin: new Vector3(1658.58f, 1662.91f, 141.234f),
            facingRadians: MathF.PI,
            tileX: 28,
            tileY: 28);

        M2CameraPathKeyframe key = Assert.Single(path.Keyframes);
        Assert.Equal(new Vector3(1657.58f, 1662.91f, 143.234f), key.Position, new Vector3EqualityComparer(0.001f));
        Assert.Equal(new Vector3(1657.58f, 1661.91f, 143.234f), key.Target, new Vector3EqualityComparer(0.001f));
        Assert.True(path.CoordinatesAreWorldSpace);
        Assert.True(path.HasCinematicCameraOrigin);
        Assert.Equal(28, path.CinematicCameraOriginTileX);
        Assert.Equal(28, path.CinematicCameraOriginTileY);

        M2CameraPathPlacement.ApplyCinematicCameraOrigin(
            path,
            cameraId: 2,
            modelPath: "Cameras\\FlybyUndead.mdx",
            origin: new Vector3(999f),
            facingRadians: 0f,
            tileX: 0,
            tileY: 0);

        Assert.Equal(new Vector3(1657.58f, 1662.91f, 143.234f), key.Position, new Vector3EqualityComparer(0.001f));
    }

    private sealed class Vector3EqualityComparer(float tolerance) : IEqualityComparer<Vector3>
    {
        public bool Equals(Vector3 left, Vector3 right)
            => Vector3.DistanceSquared(left, right) <= tolerance * tolerance;

        public int GetHashCode(Vector3 value) => value.GetHashCode();
    }
}
