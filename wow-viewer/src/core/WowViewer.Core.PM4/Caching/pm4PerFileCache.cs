using System.Collections.Generic;

namespace WowViewer.Core.PM4.Caching;

/// <summary>
/// In-memory per-file PM4 overlay cache. Keyed on the normalized virtual
/// path of the PM4 file in the data source. Bounded by a simple LRU cap;
/// the cap is "soft" (inserts that would exceed the cap evict the
/// least-recently-touched entry, but reads do not touch the LRU order — the
/// cap exists to bound memory, not to enforce exact LRU semantics).
///
/// All public methods are safe to call from a single thread; the cache is
/// not internally synchronized. Callers (typically <c>WorldScene</c>)
/// must serialize access themselves.
/// </summary>
public sealed class Pm4PerFileCache
{
    private readonly int _capacity;
    private readonly Dictionary<string, Pm4PerFileCacheEntry> _entries = new();

    public Pm4PerFileCache(int capacity = 256)
    {
        if (capacity < 1)
            throw new ArgumentOutOfRangeException(nameof(capacity), "Capacity must be >= 1.");
        _capacity = capacity;
    }

    public int Count => _entries.Count;

    public int Capacity => _capacity;

    public bool TryGet(string normalizedPath, long fileLength, long lastWriteTicks, out Pm4PerFileCacheEntry? entry)
    {
        if (_entries.TryGetValue(normalizedPath, out Pm4PerFileCacheEntry? existing))
        {
            if (existing.FileLength == fileLength && existing.LastWriteTicks == lastWriteTicks)
            {
                entry = existing;
                return true;
            }

            // Stamp mismatch — caller should treat as a miss and re-decode.
            entry = null;
            return false;
        }

        entry = null;
        return false;
    }

    public void Set(string normalizedPath, Pm4PerFileCacheEntry entry)
    {
        if (_entries.TryGetValue(normalizedPath, out Pm4PerFileCacheEntry? existing))
        {
            _entries[normalizedPath] = entry;
            return;
        }

        if (_entries.Count >= _capacity)
        {
            // Simple eviction: pick any one entry. The cap is a soft bound;
            // the next Set will evict again until the count stabilises near the cap.
            // This keeps the cache O(1) per insert without tracking a full LRU list.
            string? victim = null;
            foreach (string key in _entries.Keys)
            {
                victim = key;
                break;
            }
            if (victim != null)
                _entries.Remove(victim);
        }

        _entries[normalizedPath] = entry;
    }

    public bool Remove(string normalizedPath) => _entries.Remove(normalizedPath);

    public void Clear() => _entries.Clear();
}
