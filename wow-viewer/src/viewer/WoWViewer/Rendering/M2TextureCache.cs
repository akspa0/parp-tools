using System.Runtime.CompilerServices;
using Silk.NET.OpenGL;
using WoWViewer.DataSources;

namespace WoWViewer.Rendering;

/// <summary>
/// Spec 256 P2a: GL textures shared by every native <see cref="M2Renderer"/> on the same data source.
/// Before this each renderer decoded and uploaded its own copy, so a texture used by 40 models was
/// decoded 40 times. Entries are reference-counted per renderer; the last release deletes the texture.
/// </summary>
/// <remarks>
/// Keys, so a shared texture is always the one the renderer would have loaded itself:
/// <list type="bullet">
/// <item><b>request</b> — model directory + requested path + clamp flags. Resolution of a requested path
/// depends only on these (and the data source), so a hit needs no read.</item>
/// <item><b>resolved</b> — resolved path + clamp flags: two requests that resolve to the same file share it.</item>
/// </list>
/// One table per <see cref="IDataSource"/> instance, so opening another client never reuses a texture from
/// the previous one. Render thread only.
/// </remarks>
internal static class M2TextureCache
{
    private sealed class Entry
    {
        public uint TextureId;
        public int RefCount;
        public readonly List<string> Keys = new(2);
    }

    private sealed class Table
    {
        public readonly Dictionary<string, Entry> ByKey = new(StringComparer.OrdinalIgnoreCase);
        public readonly Dictionary<uint, Entry> ById = new();
    }

    private static readonly ConditionalWeakTable<IDataSource, Table> TablesBySource = new();
    private static readonly Table NoSourceTable = new();

    private static Table TableFor(IDataSource? dataSource)
        => dataSource is null ? NoSourceTable : TablesBySource.GetValue(dataSource, static _ => new Table());

    /// <summary>Adds a reference to the texture stored under <paramref name="key"/>, if any.</summary>
    public static bool TryAcquire(IDataSource? dataSource, string key, out uint textureId)
    {
        textureId = 0;
        Table table = TableFor(dataSource);
        if (!table.ByKey.TryGetValue(key, out Entry? entry))
            return false;

        entry.RefCount++;
        textureId = entry.TextureId;
        return true;
    }

    /// <summary>Registers another key for a texture already in the cache (no reference change).</summary>
    public static void AddAlias(IDataSource? dataSource, string key, uint textureId)
    {
        Table table = TableFor(dataSource);
        if (table.ByKey.ContainsKey(key) || !table.ById.TryGetValue(textureId, out Entry? entry))
            return;

        table.ByKey[key] = entry;
        entry.Keys.Add(key);
    }

    /// <summary>Stores a newly uploaded texture with one reference, under every given key.</summary>
    public static void Add(IDataSource? dataSource, uint textureId, params string[] keys)
    {
        Table table = TableFor(dataSource);
        var entry = new Entry { TextureId = textureId, RefCount = 1 };
        table.ById[textureId] = entry;
        foreach (string key in keys)
        {
            if (table.ByKey.TryAdd(key, entry))
                entry.Keys.Add(key);
        }
    }

    /// <summary>Drops one reference; the last one deletes the GL texture.</summary>
    public static void Release(GL gl, IDataSource? dataSource, uint textureId)
    {
        Table table = TableFor(dataSource);
        if (!table.ById.TryGetValue(textureId, out Entry? entry))
        {
            gl.DeleteTexture(textureId); // not shared (cannot happen for tracked ids; kept for safety)
            return;
        }

        if (--entry.RefCount > 0)
            return;

        table.ById.Remove(textureId);
        foreach (string key in entry.Keys)
            table.ByKey.Remove(key);

        gl.DeleteTexture(textureId);
    }
}
