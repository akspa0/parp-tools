using System.Numerics;

namespace WoWViewer.Terrain;

/// <summary>
/// Combines terrain (WDT/ADT), WMO placements (MODF), and MDX placements (MDDF)
/// into a single world scene — the same way the game client renders a map.
/// 
/// Uses <see cref="WorldAssetManager"/> to ensure each model is loaded exactly once.
/// Instances are lightweight structs holding only a model key + transform.
/// </summary>
public readonly record struct TaxiActorPose(
    int RouteId,
    Vector3 Position,
    Vector3 Forward,
    float YawRadians,
    float Scale,
    string ModelPath);
