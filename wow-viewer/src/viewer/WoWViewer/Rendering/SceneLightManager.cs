using System.Numerics;

namespace WoWViewer.Rendering;

/// <summary>
/// Collects active scene-emitted point lights and selects the nearest bounded set for a render target.
/// </summary>
public sealed class SceneLightManager
{
    public const int MaxShaderLights = 8;

    private readonly List<SceneLight> _lights = new();
    private readonly List<(SceneLight Light, float DistanceSq)> _selectionScratch = new(MaxShaderLights * 2);

    public int Count => _lights.Count;

    public IReadOnlyList<SceneLight> Lights => _lights;

    public void Clear() => _lights.Clear();

    public void Add(SceneLight light)
    {
        if (!IsValidLight(light))
            return;

        _lights.Add(light);
    }

    public void AddRange(IEnumerable<SceneLight> lights)
    {
        ArgumentNullException.ThrowIfNull(lights);

        foreach (SceneLight light in lights)
            Add(light);
    }

    public int QueryAffecting(Vector3 boundsMin, Vector3 boundsMax, Span<SceneLight> destination)
    {
        if (destination.Length == 0 || _lights.Count == 0)
            return 0;

        _selectionScratch.Clear();
        for (int i = 0; i < _lights.Count; i++)
        {
            SceneLight light = _lights[i];
            float distanceSq = DistanceSquaredPointToAabb(light.Position, boundsMin, boundsMax);
            float radius = MathF.Max(light.AttenuationEnd, 0.0f);
            if (distanceSq > radius * radius)
                continue;

            _selectionScratch.Add((light, distanceSq));
        }

        if (_selectionScratch.Count == 0)
            return 0;

        _selectionScratch.Sort(static (left, right) => left.DistanceSq.CompareTo(right.DistanceSq));

        int count = Math.Min(destination.Length, Math.Min(MaxShaderLights, _selectionScratch.Count));
        for (int i = 0; i < count; i++)
            destination[i] = _selectionScratch[i].Light;

        return count;
    }

    private static bool IsValidLight(SceneLight light)
    {
        return IsFinite(light.Position)
            && IsFinite(light.Color)
            && float.IsFinite(light.Intensity)
            && float.IsFinite(light.AttenuationStart)
            && float.IsFinite(light.AttenuationEnd)
            && light.Intensity > 0.0f
            && light.AttenuationEnd > 0.0f;
    }

    private static float DistanceSquaredPointToAabb(Vector3 point, Vector3 min, Vector3 max)
    {
        float dx = DistanceOutsideAxis(point.X, min.X, max.X);
        float dy = DistanceOutsideAxis(point.Y, min.Y, max.Y);
        float dz = DistanceOutsideAxis(point.Z, min.Z, max.Z);
        return dx * dx + dy * dy + dz * dz;
    }

    private static float DistanceOutsideAxis(float value, float min, float max)
    {
        float axisMin = MathF.Min(min, max);
        float axisMax = MathF.Max(min, max);
        if (value < axisMin)
            return axisMin - value;
        if (value > axisMax)
            return value - axisMax;
        return 0.0f;
    }

    private static bool IsFinite(Vector3 value)
        => float.IsFinite(value.X) && float.IsFinite(value.Y) && float.IsFinite(value.Z);
}
