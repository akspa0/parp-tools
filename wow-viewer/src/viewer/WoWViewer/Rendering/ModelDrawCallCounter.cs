namespace WoWViewer.Rendering;

/// <summary>
/// Counts the GL draw calls model renderers actually issue, so a pass can report draw calls
/// rather than instances.
/// </summary>
/// <remarks>
/// Spec 202 Phase 0 T003. Draw calls are the thing being optimised and instances are not, and
/// the two are not convertible: a model draws once per geoset (legacy MDX) or once per section
/// (native M2), so no arithmetic over the instance counters can produce this number. Counting at
/// the call site is the only honest source.
/// <para>
/// Callers read <see cref="Count"/> either side of a pass and subtract. Plain static state with
/// no interlocking: every increment happens on the GL thread inside a draw call, and making this
/// atomic would put a lock-prefixed instruction in the hottest loop in the renderer to protect
/// against a caller that cannot exist.
/// </para>
/// </remarks>
public static class ModelDrawCallCounter
{
    /// <summary>Draw calls issued by model renderers since process start.</summary>
    public static long Count { get; private set; }

    /// <summary>Records one issued draw call. Call immediately after the GL draw.</summary>
    public static void Record() => Count++;

    /// <summary>Draw calls issued since <paramref name="startCount"/> was sampled.</summary>
    public static int Since(long startCount)
    {
        long delta = Count - startCount;
        return delta <= 0 ? 0 : delta > int.MaxValue ? int.MaxValue : (int)delta;
    }
}
