using System.Numerics;
using WowViewer.Core.Runtime.World;

namespace WoWViewer.Rendering;

/// <summary>
/// Outdoor ambient/directional lighting for the current frame, sourced from the active
/// weather/LIT/DBC profile. Point-light casting is layered on top of this base lighting.
/// </summary>
public readonly record struct SceneAmbientLight(
    Vector3 Direction,
    Vector3 LightColor,
    Vector3 AmbientColor);

/// <summary>
/// Collects active scene-emitted point lights and selects the nearest bounded set for a render target.
/// Also carries the frame's outdoor ambient/sun representation so every lit surface can share one
/// base-lighting contract.
/// </summary>
/// <remarks>
/// Epic 249 R-10d: queries run against a uniform XY grid built lazily after the light set changes,
/// instead of scanning every light for every placement, chunk and doodad. The result is the same
/// light set in the same order as a full scan: nearest first, ties broken by insertion order.
/// </remarks>
public sealed class SceneLightManager
{
    public const int MaxShaderLights = 8;

    /// <summary>
    /// World-unit slack added to a light's radius when culling it against the view frustum, so a light
    /// entering view while the camera moves is not missed by passes that use the previous frame's set.
    /// </summary>
    public const float FrustumCullMargin = 256.0f;

    /// <summary>Grid cell edge in world units.</summary>
    internal const float CellSize = 64.0f;

    /// <summary>Lights spanning more cells than this per axis are kept in a list every query scans.</summary>
    private const int MaxCellsPerAxis = 16;

    private readonly List<SceneLight> _lights = new();
    private readonly List<(int Index, float DistanceSq)> _selectionScratch = new(MaxShaderLights * 2);
    private readonly Dictionary<long, List<int>> _cells = new();
    private readonly Stack<List<int>> _cellListPool = new();
    private readonly List<int> _largeLights = new();
    private readonly List<int> _candidateScratch = new();
    private bool[] _isLargeLight = Array.Empty<bool>();
    private int[] _queryStamp = Array.Empty<int>();
    private int _queryGeneration;
    private bool _gridDirty = true;
    private SceneAmbientLight? _ambient;

    private int _lightsCollected;
    private int _queryCount;
    private long _candidatesTested;
    private int _wmoBatched;
    private int _wmoLitFallback;
    private int _wmoSelfLit;

    public int Count => _lights.Count;

    public IReadOnlyList<SceneLight> Lights => _lights;

    /// <summary>
    /// Frame outdoor ambient/sun lighting, or <c>null</c> when no profile has supplied one this frame.
    /// </summary>
    public SceneAmbientLight? Ambient => _ambient;

    /// <summary>
    /// Returns this frame's workload and restarts the query counters. Called once per frame when the
    /// frame stats are taken, so queries issued before the light rebuild (terrain) are included.
    /// </summary>
    public SceneLightingFrameStats TakeFrameCounters()
    {
        var stats = new SceneLightingFrameStats(
            _lightsCollected,
            _lights.Count,
            _queryCount,
            _candidatesTested,
            _wmoBatched,
            _wmoLitFallback,
            _wmoSelfLit);
        _queryCount = 0;
        _candidatesTested = 0;
        return stats;
    }

    /// <summary>
    /// Sets the frame outdoor ambient/sun representation. Non-finite or zero-radius inputs are
    /// rejected so a corrupt profile cannot poison every lit surface.
    /// </summary>
    public void SetAmbient(Vector3 direction, Vector3 lightColor, Vector3 ambientColor)
    {
        if (!IsFinite(direction) || !IsFinite(lightColor) || !IsFinite(ambientColor))
            return;

        _ambient = new SceneAmbientLight(direction, lightColor, ambientColor);
    }

    public void Clear()
    {
        _lights.Clear();
        _ambient = null;
        _gridDirty = true;
        _lightsCollected = 0;
        _wmoBatched = 0;
        _wmoLitFallback = 0;
        _wmoSelfLit = 0;
    }

    public void Add(SceneLight light)
    {
        _lightsCollected++;
        if (!IsValidLight(light))
            return;

        _lights.Add(light);
        _gridDirty = true;
    }

    public void AddRange(IEnumerable<SceneLight> lights)
    {
        ArgumentNullException.ThrowIfNull(lights);

        foreach (SceneLight light in lights)
            Add(light);
    }

    /// <summary>
    /// Adds only the lights <paramref name="keep"/> accepts; rejected lights still count as collected.
    /// </summary>
    public void AddRange(IEnumerable<SceneLight> lights, Func<SceneLight, bool> keep)
    {
        ArgumentNullException.ThrowIfNull(lights);
        ArgumentNullException.ThrowIfNull(keep);

        foreach (SceneLight light in lights)
        {
            if (keep(light))
            {
                Add(light);
            }
            else
            {
                _lightsCollected++;
            }
        }
    }

    /// <summary>Records how this frame's opaque WMO placements were partitioned.</summary>
    public void RecordWmoPartition(int batched, int litFallback, int selfLit)
    {
        _wmoBatched = batched;
        _wmoLitFallback = litFallback;
        _wmoSelfLit = selfLit;
    }

    public int QueryAffecting(Vector3 boundsMin, Vector3 boundsMax, Span<SceneLight> destination)
    {
        _queryCount++;
        if (destination.Length == 0 || _lights.Count == 0)
            return 0;

        CollectCandidates(boundsMin, boundsMax);
        _selectionScratch.Clear();
        for (int i = 0; i < _candidateScratch.Count; i++)
        {
            int index = _candidateScratch[i];
            SceneLight light = _lights[index];
            float distanceSq = DistanceSquaredPointToAabb(light.Position, boundsMin, boundsMax);
            float radius = MathF.Max(light.AttenuationEnd, 0.0f);
            if (distanceSq <= radius * radius)
                _selectionScratch.Add((index, distanceSq));
        }

        if (_selectionScratch.Count == 0)
            return 0;

        _selectionScratch.Sort(NearestThenInsertionOrderComparison);

        int count = Math.Min(destination.Length, Math.Min(MaxShaderLights, _selectionScratch.Count));
        for (int i = 0; i < count; i++)
            destination[i] = _lights[_selectionScratch[i].Index];

        return count;
    }

    /// <summary>Whether any active light's attenuation reaches the bounds.</summary>
    public bool AnyAffecting(Vector3 boundsMin, Vector3 boundsMax)
    {
        _queryCount++;
        if (_lights.Count == 0)
            return false;

        CollectCandidates(boundsMin, boundsMax);
        for (int i = 0; i < _candidateScratch.Count; i++)
        {
            SceneLight light = _lights[_candidateScratch[i]];
            float radius = MathF.Max(light.AttenuationEnd, 0.0f);
            if (DistanceSquaredPointToAabb(light.Position, boundsMin, boundsMax) <= radius * radius)
                return true;
        }

        return false;
    }

    // Cached so the per-query sort does not allocate a delegate.
    private static readonly Comparison<(int Index, float DistanceSq)> NearestThenInsertionOrderComparison = NearestThenInsertionOrder;

    private static int NearestThenInsertionOrder((int Index, float DistanceSq) left, (int Index, float DistanceSq) right)
    {
        int byDistance = left.DistanceSq.CompareTo(right.DistanceSq);
        return byDistance != 0 ? byDistance : left.Index.CompareTo(right.Index);
    }

    /// <summary>
    /// Fills <see cref="_candidateScratch"/> with every light whose grid footprint overlaps the bounds,
    /// each once — a superset of the lights that reach them. Z is not indexed; callers apply the exact test.
    /// </summary>
    private void CollectCandidates(Vector3 boundsMin, Vector3 boundsMax)
    {
        EnsureGrid();
        _candidateScratch.Clear();
        _candidateScratch.AddRange(_largeLights);

        int cellMinX = CellOf(MathF.Min(boundsMin.X, boundsMax.X));
        int cellMaxX = CellOf(MathF.Max(boundsMin.X, boundsMax.X));
        int cellMinY = CellOf(MathF.Min(boundsMin.Y, boundsMax.Y));
        int cellMaxY = CellOf(MathF.Max(boundsMin.Y, boundsMax.Y));

        // A query spanning more cells than the grid holds is cheaper as a full scan; both are exact.
        if ((long)(cellMaxX - cellMinX + 1) * (cellMaxY - cellMinY + 1) > _cells.Count)
        {
            for (int index = 0; index < _lights.Count; index++)
            {
                if (!_isLargeLight[index])
                    _candidateScratch.Add(index);
            }

            _candidatesTested += _candidateScratch.Count;
            return;
        }

        int generation = NextQueryGeneration();
        for (int cx = cellMinX; cx <= cellMaxX; cx++)
        {
            for (int cy = cellMinY; cy <= cellMaxY; cy++)
            {
                if (!_cells.TryGetValue(CellKey(cx, cy), out List<int>? cell))
                    continue;

                for (int i = 0; i < cell.Count; i++)
                {
                    int index = cell[i];
                    if (_queryStamp[index] == generation)
                        continue;

                    _queryStamp[index] = generation;
                    _candidateScratch.Add(index);
                }
            }
        }

        _candidatesTested += _candidateScratch.Count;
    }

    private void EnsureGrid()
    {
        if (!_gridDirty)
            return;

        _gridDirty = false;
        foreach (List<int> cell in _cells.Values)
        {
            cell.Clear();
            _cellListPool.Push(cell);
        }

        _cells.Clear();
        _largeLights.Clear();
        if (_queryStamp.Length < _lights.Count)
        {
            int size = Math.Max(_lights.Count, _queryStamp.Length * 2);
            _queryStamp = new int[size];
            _isLargeLight = new bool[size];
        }

        Array.Clear(_queryStamp);
        Array.Clear(_isLargeLight);
        _queryGeneration = 0;

        for (int index = 0; index < _lights.Count; index++)
        {
            SceneLight light = _lights[index];
            float radius = MathF.Max(light.AttenuationEnd, 0.0f);
            int cellMinX = CellOf(light.Position.X - radius);
            int cellMaxX = CellOf(light.Position.X + radius);
            int cellMinY = CellOf(light.Position.Y - radius);
            int cellMaxY = CellOf(light.Position.Y + radius);
            if (cellMaxX - cellMinX >= MaxCellsPerAxis || cellMaxY - cellMinY >= MaxCellsPerAxis)
            {
                _largeLights.Add(index);
                _isLargeLight[index] = true;
                continue;
            }

            for (int cx = cellMinX; cx <= cellMaxX; cx++)
            {
                for (int cy = cellMinY; cy <= cellMaxY; cy++)
                {
                    long key = CellKey(cx, cy);
                    if (!_cells.TryGetValue(key, out List<int>? cell))
                    {
                        cell = _cellListPool.Count > 0 ? _cellListPool.Pop() : new List<int>();
                        _cells.Add(key, cell);
                    }

                    cell.Add(index);
                }
            }
        }
    }

    private int NextQueryGeneration()
    {
        _queryGeneration++;
        if (_queryGeneration == int.MaxValue)
        {
            // Stamps wrap after ~2 billion queries; reset so a stale stamp cannot suppress a light.
            Array.Clear(_queryStamp);
            _queryGeneration = 1;
        }

        return _queryGeneration;
    }

    private static int CellOf(float value)
    {
        float cell = MathF.Floor(value / CellSize);
        return (int)Math.Clamp(cell, -1_000_000f, 1_000_000f);
    }

    private static long CellKey(int x, int y) => ((long)x << 32) | (uint)y;

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
