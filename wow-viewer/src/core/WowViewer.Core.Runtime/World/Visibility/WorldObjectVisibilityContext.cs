using System.Numerics;

namespace WowViewer.Core.Runtime.World.Visibility;

public enum WorldObjectVisibilityProfile
{
    Quality,
    Balanced,
    Performance,
}

public readonly record struct WorldObjectVisibilityContext(
    Vector3 CameraPosition,
    Vector3 CameraForward,
    float FogEnd,
    float ObjectStreamingRangeMultiplier,
    bool CullSmallDoodadsOnly,
    bool CountAsTaxiActor,
    float VerticalFieldOfViewRadians,
    WorldObjectVisibilityProfile VisibilityProfile);