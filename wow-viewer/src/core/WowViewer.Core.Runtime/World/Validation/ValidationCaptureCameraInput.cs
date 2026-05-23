using System.Numerics;

namespace WowViewer.Core.Runtime.World.Validation;

public readonly record struct ValidationCaptureCameraInput(
    int TileX,
    int TileY,
    float AspectRatio,
    float GroundHeight,
    float MapOrigin,
    float TileWorldSize,
    float DesiredSpan,
    float EyeHeightOffset,
    float NearPlane,
    float FarPlane,
    Vector3 Up);