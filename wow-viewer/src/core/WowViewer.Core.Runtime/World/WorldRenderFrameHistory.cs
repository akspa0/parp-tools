namespace WowViewer.Core.Runtime.World;

/// <summary>
/// The per-stage timers carried by <see cref="WorldRenderFrameStats"/>, in a fixed order so that
/// history can be stored as a flat stride-indexed array instead of one object per frame.
/// </summary>
public enum WorldRenderStage
{
    DeferredAssetLoads = 0,
    TaxiActorUpdate = 1,
    Lighting = 2,
    Sky = 3,
    SkyboxBackdrop = 4,
    Wdl = 5,
    Terrain = 6,
    WmoVisibility = 7,
    WmoSubmission = 8,
    WmoTransparentSubmission = 9,
    MdxAnimation = 10,
    MdxVisibility = 11,
    MdxOpaqueSubmission = 12,
    Liquid = 13,
    MdxTransparentSort = 14,
    MdxTransparentSubmission = 15,
    Overlay = 16,
    SceneMaintenance = 17,

    /// <summary>
    /// Object-phase preparation: camera resolution, audio residency, PM4 overlay window, frustum
    /// update, opaque GL state. Added because it was the one pass in <c>WorldFramePasses</c> with no
    /// timer, so a ~212 ms periodic stall sat in the unaccounted pass gap for an entire investigation.
    /// </summary>
    PrepareObjectPhase = 18,
}

/// <summary>Distribution of one timing series across the retained window.</summary>
public readonly record struct WorldRenderTimingDistribution(
    int SampleCount,
    double MedianMs,
    double P95Ms,
    double P99Ms,
    double MaxMs,
    double MeanMs,
    int OverThresholdCount)
{
    public static WorldRenderTimingDistribution Empty { get; } = new(0, 0, 0, 0, 0, 0, 0);
}

/// <summary>
/// One frame that exceeded the hitch threshold.
/// <para>
/// <see cref="UnaccountedMs"/> is total frame time minus the sum of every instrumented stage. When
/// it dominates, the cost is somewhere no stage timer covers, and naming a stage would be actively
/// misleading — a 350 ms frame whose largest stage is 0.2 ms is not a story about that stage.
/// </para>
/// </summary>
public readonly record struct WorldRenderHitch(
    long FrameIndex,
    double TotalCpuMs,
    WorldRenderStage DominantStage,
    double DominantStageMs,
    double UnaccountedMs)
{
    /// <summary>True when uninstrumented time exceeds the largest instrumented stage.</summary>
    public bool IsDominatedByUnaccountedTime => UnaccountedMs > DominantStageMs;

    /// <summary>Human-readable cause: either a stage name or an explicit "not instrumented".</summary>
    public string DominantCause => IsDominatedByUnaccountedTime
        ? "UNACCOUNTED (no stage timer covers it)"
        : DominantStage.ToString();

    public double DominantCauseMs => IsDominatedByUnaccountedTime ? UnaccountedMs : DominantStageMs;
}

/// <summary>
/// A point-in-time read of the retained window. Allocation happens here, on demand, never on the
/// render path.
/// </summary>
public sealed record WorldRenderFrameHistorySnapshot(
    int FrameCount,
    long FirstFrameIndex,
    long LastFrameIndex,
    double HitchThresholdMs,
    bool CameraMovedDuringWindow,
    WorldRenderTimingDistribution Total,
    WorldRenderTimingDistribution Unaccounted,
    IReadOnlyDictionary<WorldRenderStage, WorldRenderTimingDistribution> Stages,
    IReadOnlyList<WorldRenderHitch> Hitches,
    double RecorderOverheadMsPerFrame)
{
    /// <summary>
    /// Stages ordered by worst observed frame, not by p99.
    /// <para>
    /// Ranking by p99 structurally hides the profile a hitch hunt is looking for: a stage that is
    /// rare (so its p99 is near zero) but enormous when it does fire. Sorting by max surfaces it.
    /// </para>
    /// </summary>
    public IEnumerable<KeyValuePair<WorldRenderStage, WorldRenderTimingDistribution>> StagesByMaxDescending()
        => Stages.OrderByDescending(static kvp => kvp.Value.MaxMs);

    /// <summary>
    /// A stationary window cannot demonstrate movement-induced behavior. Callers must not read a
    /// stationary result as evidence about the gallop.
    /// </summary>
    public bool CanDemonstrateMovementBehavior => CameraMovedDuringWindow;

    public IEnumerable<KeyValuePair<WorldRenderStage, WorldRenderTimingDistribution>> StagesByP99Descending()
        => Stages.OrderByDescending(static kvp => kvp.Value.P99Ms);
}

/// <summary>
/// Fixed-capacity rolling history of per-frame render timings.
/// <para>
/// This exists because <c>LastRenderFrameStats</c> retains exactly one frame, so nothing in the
/// viewer could observe behavior <em>over time</em> — which is the only way a periodic hitch is
/// visible. The instrumentation was already being produced every frame and then discarded.
/// </para>
/// <para>
/// The buffer is allocated once and never grows. <see cref="Record"/> performs no allocation, so
/// the recorder cannot become a churn source of the kind it exists to measure.
/// </para>
/// </summary>
public sealed class WorldRenderFrameHistory
{
    public const int StageCount = 19;
    public const int DefaultCapacity = 2048;

    private readonly int _capacity;
    private readonly double[] _totalMs;
    private readonly double[] _stageMs;      // _capacity * StageCount, stride-indexed
    private readonly double[] _unaccountedMs; // total minus the sum of all stages, per frame
    private readonly bool[] _cameraMoved;
    private readonly long[] _frameIndex;
    private readonly double[] _scratch;      // reused by percentile queries; never touched by Record

    private int _count;
    private int _next;                        // write cursor
    private long _nextFrameIndex;
    private double _recorderTicks;
    private long _recordedFrames;

    public WorldRenderFrameHistory(int capacity = DefaultCapacity)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(capacity, 2);
        _capacity = capacity;
        _totalMs = new double[capacity];
        _stageMs = new double[capacity * StageCount];
        _unaccountedMs = new double[capacity];
        _cameraMoved = new bool[capacity];
        _frameIndex = new long[capacity];
        _scratch = new double[capacity];
    }

    public int Capacity => _capacity;

    public int Count => _count;

    /// <summary>Frames observed since construction, including ones already evicted.</summary>
    public long TotalFramesRecorded => _nextFrameIndex;

    /// <summary>Mean cost of <see cref="Record"/> itself, so the instrument is separable from what it measures.</summary>
    public double RecorderOverheadMsPerFrame =>
        _recordedFrames == 0 ? 0 : _recorderTicks / _recordedFrames * 1000.0 / System.Diagnostics.Stopwatch.Frequency;

    /// <summary>
    /// Append one frame. No allocation. <paramref name="cameraMoved"/> records whether the camera
    /// changed since the previous frame, so a stationary window can be labelled as such.
    /// </summary>
    public void Record(in WorldRenderFrameStats stats, bool cameraMoved)
    {
        long start = System.Diagnostics.Stopwatch.GetTimestamp();

        int slot = _next;
        _totalMs[slot] = stats.TotalCpuMs;
        _cameraMoved[slot] = cameraMoved;
        _frameIndex[slot] = _nextFrameIndex++;

        int baseIndex = slot * StageCount;
        _stageMs[baseIndex + (int)WorldRenderStage.DeferredAssetLoads] = stats.DeferredAssetLoads.DurationMs;
        _stageMs[baseIndex + (int)WorldRenderStage.TaxiActorUpdate] = stats.TaxiActorUpdate.DurationMs;
        _stageMs[baseIndex + (int)WorldRenderStage.Lighting] = stats.Lighting.DurationMs;
        _stageMs[baseIndex + (int)WorldRenderStage.Sky] = stats.Sky.DurationMs;
        _stageMs[baseIndex + (int)WorldRenderStage.SkyboxBackdrop] = stats.SkyboxBackdrop.DurationMs;
        _stageMs[baseIndex + (int)WorldRenderStage.Wdl] = stats.Wdl.DurationMs;
        _stageMs[baseIndex + (int)WorldRenderStage.Terrain] = stats.Terrain.DurationMs;
        _stageMs[baseIndex + (int)WorldRenderStage.WmoVisibility] = stats.WmoVisibility.DurationMs;
        _stageMs[baseIndex + (int)WorldRenderStage.WmoSubmission] = stats.WmoSubmission.DurationMs;
        _stageMs[baseIndex + (int)WorldRenderStage.WmoTransparentSubmission] = stats.WmoTransparentSubmission.DurationMs;
        _stageMs[baseIndex + (int)WorldRenderStage.MdxAnimation] = stats.MdxAnimation.DurationMs;
        _stageMs[baseIndex + (int)WorldRenderStage.MdxVisibility] = stats.MdxVisibility.DurationMs;
        _stageMs[baseIndex + (int)WorldRenderStage.MdxOpaqueSubmission] = stats.MdxOpaqueSubmission.DurationMs;
        _stageMs[baseIndex + (int)WorldRenderStage.Liquid] = stats.Liquid.DurationMs;
        _stageMs[baseIndex + (int)WorldRenderStage.MdxTransparentSort] = stats.MdxTransparentSort.DurationMs;
        _stageMs[baseIndex + (int)WorldRenderStage.MdxTransparentSubmission] = stats.MdxTransparentSubmission.DurationMs;
        _stageMs[baseIndex + (int)WorldRenderStage.Overlay] = stats.Overlay.DurationMs;
        _stageMs[baseIndex + (int)WorldRenderStage.SceneMaintenance] = stats.SceneMaintenance.DurationMs;
        _stageMs[baseIndex + (int)WorldRenderStage.PrepareObjectPhase] = stats.PrepareObjectPhase.DurationMs;

        // Uninstrumented remainder. If this dominates a hitch, the cost is somewhere no stage timer
        // covers, and reporting the largest stage would point at the wrong place entirely.
        double stageSum = 0;
        for (int stage = 0; stage < StageCount; stage++)
            stageSum += _stageMs[baseIndex + stage];
        _unaccountedMs[slot] = Math.Max(0, stats.TotalCpuMs - stageSum);

        _next = slot + 1 == _capacity ? 0 : slot + 1;
        if (_count < _capacity)
            _count++;

        _recorderTicks += System.Diagnostics.Stopwatch.GetTimestamp() - start;
        _recordedFrames++;
    }

    /// <summary>
    /// Copy the most recent total-frame-time samples into <paramref name="destination"/>, oldest
    /// first, and return how many were written. Allocation-free, so a plot can read the real series
    /// every UI frame instead of reconstructing an approximation of it.
    /// </summary>
    public int CopyRecentTotalMs(float[] destination)
    {
        ArgumentNullException.ThrowIfNull(destination);
        int take = Math.Min(destination.Length, _count);
        if (take == 0)
            return 0;

        // _next is one past the newest sample; walk back `take` entries from there.
        int start = ((_next - take) % _capacity + _capacity) % _capacity;
        for (int i = 0; i < take; i++)
            destination[i] = (float)_totalMs[(start + i) % _capacity];

        return take;
    }

    public void Clear()
    {
        _count = 0;
        _next = 0;
        _recorderTicks = 0;
        _recordedFrames = 0;
    }

    /// <summary>
    /// Read the window. Allocates a dictionary and a hitch list, so this is a deliberate on-demand
    /// read: call it from diagnostics, tests, or a throttled UI refresh — never from the render path
    /// and never once per UI frame. The series summaries themselves are computed without allocating.
    /// </summary>
    public WorldRenderFrameHistorySnapshot Snapshot(double hitchThresholdMs)
    {
        if (_count == 0)
        {
            return new WorldRenderFrameHistorySnapshot(
                0, 0, 0, hitchThresholdMs, false,
                WorldRenderTimingDistribution.Empty,
                WorldRenderTimingDistribution.Empty,
                new Dictionary<WorldRenderStage, WorldRenderTimingDistribution>(),
                Array.Empty<WorldRenderHitch>(),
                RecorderOverheadMsPerFrame);
        }

        int oldest = _count == _capacity ? _next : 0;
        bool cameraMoved = false;
        for (int i = 0; i < _count; i++)
        {
            if (_cameraMoved[(oldest + i) % _capacity])
            {
                cameraMoved = true;
                break;
            }
        }

        WorldRenderTimingDistribution total = Describe(_totalMs, 0, 1, oldest, hitchThresholdMs);
        WorldRenderTimingDistribution unaccounted =
            Describe(_unaccountedMs, 0, 1, oldest, hitchThresholdMs);

        var stages = new Dictionary<WorldRenderStage, WorldRenderTimingDistribution>(StageCount);
        for (int stage = 0; stage < StageCount; stage++)
        {
            stages[(WorldRenderStage)stage] =
                Describe(_stageMs, stage, StageCount, oldest, hitchThresholdMs);
        }

        var hitches = new List<WorldRenderHitch>();
        for (int i = 0; i < _count; i++)
        {
            int slot = (oldest + i) % _capacity;
            double frameMs = _totalMs[slot];
            if (frameMs < hitchThresholdMs)
                continue;

            int baseIndex = slot * StageCount;
            int dominant = 0;
            double dominantMs = _stageMs[baseIndex];
            for (int stage = 1; stage < StageCount; stage++)
            {
                double value = _stageMs[baseIndex + stage];
                if (value > dominantMs)
                {
                    dominantMs = value;
                    dominant = stage;
                }
            }

            hitches.Add(new WorldRenderHitch(
                _frameIndex[slot], frameMs, (WorldRenderStage)dominant, dominantMs, _unaccountedMs[slot]));
        }

        return new WorldRenderFrameHistorySnapshot(
            _count,
            _frameIndex[oldest],
            _frameIndex[(oldest + _count - 1) % _capacity],
            hitchThresholdMs,
            cameraMoved,
            total,
            unaccounted,
            stages,
            hitches,
            RecorderOverheadMsPerFrame);
    }

    /// <summary>
    /// Summarize one series without allocating. <paramref name="stride"/> and
    /// <paramref name="offset"/> address either the flat total array (stride 1) or one stage lane
    /// inside the interleaved stage array (stride <see cref="StageCount"/>), so no per-stage
    /// delegate is created.
    /// </summary>
    private WorldRenderTimingDistribution Describe(
        double[] source, int offset, int stride, int oldest, double thresholdMs)
    {
        double sum = 0;
        double max = 0;
        int over = 0;
        for (int i = 0; i < _count; i++)
        {
            int slot = (oldest + i) % _capacity;
            double value = source[slot * stride + offset];
            _scratch[i] = value;
            sum += value;
            if (value > max)
                max = value;
            if (value >= thresholdMs)
                over++;
        }

        Array.Sort(_scratch, 0, _count);
        return new WorldRenderTimingDistribution(
            _count,
            Percentile(_scratch, _count, 0.50),
            Percentile(_scratch, _count, 0.95),
            Percentile(_scratch, _count, 0.99),
            max,
            sum / _count,
            over);
    }

    /// <summary>Nearest-rank percentile over an already-sorted prefix.</summary>
    private static double Percentile(double[] sorted, int count, double fraction)
    {
        if (count == 0)
            return 0;
        int rank = (int)Math.Ceiling(fraction * count) - 1;
        return sorted[Math.Clamp(rank, 0, count - 1)];
    }
}
