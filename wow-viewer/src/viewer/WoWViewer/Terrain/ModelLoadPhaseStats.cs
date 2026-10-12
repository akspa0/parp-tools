using System.Diagnostics;

namespace WoWViewer.Terrain;

/// <summary>Phases of one world-model load, timed by <see cref="ModelLoadPhaseStats"/> (Spec 256 P0).</summary>
public enum ModelLoadPhase
{
    /// <summary>Path resolution and reading the model file's bytes.</summary>
    ModelRead = 0,

    /// <summary>Building the skin candidate list (includes <see cref="SkinListScan"/> when it runs here).</summary>
    SkinCandidates = 1,

    /// <summary>Reading a skin file's bytes.</summary>
    SkinRead = 2,

    /// <summary>Building the native M2 runtime model.</summary>
    NativeParse = 3,

    /// <summary>Building the M2-to-MDX adapter model.</summary>
    AdapterParse = 4,

    /// <summary>Creating the renderer (GL buffers, shaders).</summary>
    GpuCreate = 5,

    /// <summary>Parsing a WMO root and its groups.</summary>
    WmoParse = 6,

    /// <summary>Creating a WMO renderer.</summary>
    WmoGpuCreate = 7,

    /// <summary>
    /// One scan of the data source's whole <c>.skin</c> list by <c>ResolveBestSkinPath</c>, whoever called it
    /// (a model load, or queueing a model for prefetch). Counted on its own, not only inside a load.
    /// </summary>
    SkinListScan = 8,

    /// <summary>
    /// Loading WMO doodad models in one WMO's deferred doodad step (Spec 256 D5); usually one model, cache
    /// hits are free and not counted.
    /// </summary>
    WmoDoodadLoad = 9,
}

/// <summary>
/// Running totals per <see cref="ModelLoadPhase"/>: count, total, worst. Written on the render thread
/// only; read by the Runtime Stats panel. Allocation-free.
/// </summary>
public sealed class ModelLoadPhaseStats
{
    public const int PhaseCount = 10;

    private readonly long[] _count = new long[PhaseCount];
    private readonly double[] _totalMs = new double[PhaseCount];
    private readonly double[] _maxMs = new double[PhaseCount];

    /// <summary>
    /// Records the time since <paramref name="startTimestamp"/> against <paramref name="phase"/> and
    /// returns the current timestamp, so consecutive phases can be chained.
    /// </summary>
    public long Mark(ModelLoadPhase phase, long startTimestamp)
    {
        long now = Stopwatch.GetTimestamp();
        double ms = (now - startTimestamp) * 1000.0 / Stopwatch.Frequency;
        int index = (int)phase;
        _count[index]++;
        _totalMs[index] += ms;
        if (ms > _maxMs[index])
            _maxMs[index] = ms;
        return now;
    }

    public long Count(ModelLoadPhase phase) => _count[(int)phase];

    public double TotalMs(ModelLoadPhase phase) => _totalMs[(int)phase];

    public double MaxMs(ModelLoadPhase phase) => _maxMs[(int)phase];

    public double AverageMs(ModelLoadPhase phase)
    {
        long count = _count[(int)phase];
        return count == 0 ? 0 : _totalMs[(int)phase] / count;
    }

    public void Reset()
    {
        Array.Clear(_count);
        Array.Clear(_totalMs);
        Array.Clear(_maxMs);
    }
}
