using System.Numerics;

namespace WowViewer.Core.Runtime.World.Validation;

public readonly record struct ValidationCaptureCameraFrame(
    Vector3 Eye,
    Vector3 Target,
    Vector3 Up,
    float WorldSpanX,
    float WorldSpanY,
    Matrix4x4 View,
    Matrix4x4 Projection);