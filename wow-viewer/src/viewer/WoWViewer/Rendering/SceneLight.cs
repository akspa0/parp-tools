using System.Numerics;

namespace WoWViewer.Rendering;

/// <summary>
/// World-space point light emitted by scene assets such as WMO MOLT entries and MDX/M2 LITE data.
/// </summary>
public readonly record struct SceneLight(
    Vector3 Position,
    Vector3 Color,
    float Intensity,
    float AttenuationStart,
    float AttenuationEnd,
    string SourceKind,
    string SourceKey)
{
    public float Radius => AttenuationEnd;
}
