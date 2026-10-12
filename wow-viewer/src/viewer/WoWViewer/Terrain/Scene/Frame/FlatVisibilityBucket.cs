using System.Numerics;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using static WoWViewer.Terrain.WorldScene;

namespace WoWViewer.Terrain;

// Moved from WorldScene (Spec 255 W0): formerly a private nested class; body unchanged.
// AreFiniteOrderedBounds still lives on WorldScene (made internal; reached through the using static above).
internal sealed class FlatVisibilityBucket
{
    public List<ObjectInstance> Instances { get; } = new();
    public Vector3 Min { get; private set; } = new(float.MaxValue);
    public Vector3 Max { get; private set; } = new(float.MinValue);
    public bool BoundsKnown { get; private set; } = true;

    public void Add(in ObjectInstance instance)
    {
        Instances.Add(instance);
        if (!instance.BoundsResolved
            || !AreFiniteOrderedBounds(instance.BoundsMin, instance.BoundsMax))
        {
            BoundsKnown = false;
            return;
        }

        Min = Vector3.Min(Min, instance.BoundsMin);
        Max = Vector3.Max(Max, instance.BoundsMax);
    }
}
