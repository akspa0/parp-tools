using System.Collections.Concurrent;

namespace WowViewer.Core.IO.Files;

internal static class MpqDiagnostics
{
    private static readonly ConcurrentDictionary<string, long> Counters = new();

    public static void Increment(string name, long delta = 1)
    {
        Counters.AddOrUpdate(name, delta, (_, oldValue) => oldValue + delta);
    }

    public static long Get(string name)
    {
        return Counters.TryGetValue(name, out long value) ? value : 0;
    }

    public static IReadOnlyDictionary<string, long> Snapshot()
    {
        return new Dictionary<string, long>(Counters);
    }

    public static void Reset()
    {
        Counters.Clear();
    }
}