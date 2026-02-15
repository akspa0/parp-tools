using System.Collections.Concurrent;
using System.Threading;

namespace WoWMapConverter.Core.Diagnostics;

public static class Build335Diagnostics
{
    private static readonly ConcurrentDictionary<string, long> Counters = new();

    public static void Increment(string name, long delta = 1)
    {
        Counters.AddOrUpdate(name, delta, (_, oldValue) => oldValue + delta);
    }

    public static long Get(string name)
    {
        return Counters.TryGetValue(name, out var value) ? value : 0;
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
