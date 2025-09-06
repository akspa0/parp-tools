using System.Collections.Generic;

namespace GillijimProject.Next.Core.PM4;

/// <summary>
/// Minimal PM4 scene container for port scaffolding.
/// </summary>
public class Pm4Scene
{
    public readonly List<P3> Vertices = new();
    public readonly List<(int A, int B, int C)> Triangles = new();
    public readonly List<int> Indices = new();

    // Optional MSCN anchor points (world-ish coordinates). Kept simple for now.
    public readonly List<P3> MscnAnchors = new();

    // Optional: when available from loader, parallel tile IDs (Y*64 + X) for each MSCN anchor.
    // Enables sidecar counts grouped by tile.
    public readonly List<int> MscnTileIds = new();
}

/// <summary>
/// Simple immutable 3D point (float) used for PM4 scaffolding.
/// </summary>
public readonly struct P3
{
    public readonly float X;
    public readonly float Y;
    public readonly float Z;
    public P3(float x, float y, float z)
    {
        X = x; Y = y; Z = z;
    }
}
