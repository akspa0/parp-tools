namespace WoWViewer.Rendering;

/// <summary>
/// Spec 256 amendment D: WMO doodad models shared by every world <see cref="WmoRenderer"/> of one
/// <c>WorldAssetManager</c>. Before this each WMO loaded its own copy, so a bench in two buildings was read,
/// parsed and uploaded twice. Entries are reference-counted per WMO; the last release disposes the renderer.
/// A failed load is cached too (null renderer), so other WMOs do not retry it. Render thread only.
/// </summary>
internal sealed class WmoDoodadModelShare
{
    private sealed class Entry
    {
        public IModelRenderer? Renderer;
        public int RefCount;
    }

    private readonly Dictionary<string, Entry> _entries = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>Spec 256 D1: the <c>.skin</c> lookup the world model path already uses (P1).</summary>
    public SkinPathIndex SkinIndex { get; } = new();

    public int ModelCount => _entries.Count;

    public long SharedHits { get; private set; }

    /// <summary>Adds a reference to the model stored under <paramref name="key"/>, if any.</summary>
    public bool TryAcquire(string key, out IModelRenderer? renderer)
    {
        renderer = null;
        if (!_entries.TryGetValue(key, out Entry? entry))
            return false;

        entry.RefCount++;
        SharedHits++;
        renderer = entry.Renderer;
        return true;
    }

    /// <summary>Stores a newly loaded model (or a failed load) with one reference.</summary>
    public void Add(string key, IModelRenderer? renderer)
    {
        if (_entries.TryGetValue(key, out Entry? existing))
        {
            // Cannot happen on the single render thread; keep the stored model and drop the duplicate.
            existing.RefCount++;
            if (!ReferenceEquals(existing.Renderer, renderer))
                renderer?.Dispose();
            return;
        }

        _entries.Add(key, new Entry { Renderer = renderer, RefCount = 1 });
    }

    /// <summary>Drops one reference; the last one disposes the renderer.</summary>
    public void Release(string key)
    {
        if (!_entries.TryGetValue(key, out Entry? entry))
            return;

        if (--entry.RefCount > 0)
            return;

        _entries.Remove(key);
        entry.Renderer?.Dispose();
    }
}
