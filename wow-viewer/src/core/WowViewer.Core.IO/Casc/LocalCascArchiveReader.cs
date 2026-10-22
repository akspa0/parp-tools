using System.Collections.Concurrent;
using System.Globalization;
using Microsoft.Win32.SafeHandles;
using TACTSharp;

namespace WowViewer.Core.IO.Casc;

/// <summary>
/// Spec 275 CASC-1: reads encoded (BLTE) entries out of a local install's <c>Data/data/data.NNN</c>
/// archives through one cached read-only handle per archive.
/// <para>
/// TACTSharp's own local read (<c>CDN.TryGetLocalFile</c>) stats the archive, maps the whole archive
/// file, creates a view over it, copies the entry out, unmaps and closes - for every file - and
/// then copies the result a second time. Measured on a 768 MB scratch file with a warm OS cache:
/// ~250-385 us per read whatever the entry size, and 8 threads only reach ~1.5x one thread, against
/// ~7-18 us per read for 4-64 KB entries with a cached handle. A streaming scene reads tens of
/// thousands of small M2/skin/BLP/WMO-group files, so the fixed cost dominates.
/// </para>
/// <para>
/// This type reuses TACTSharp's public <see cref="CASCIndexInstance"/> for the index lookup (same
/// bucket rule, same highest-version selection, same offset/size arithmetic), so entries resolve to
/// exactly the bytes TACTSharp would return; only the I/O differs. The handles are opened with
/// <c>FileShare.ReadWrite | FileShare.Delete</c> so the Battle.net agent can still update the files.
/// A miss (entry not in the local index, archive absent, short read) returns false and the caller
/// falls back to the stock path, which also owns CDN fill.
/// </para>
/// </summary>
public sealed class LocalCascArchiveReader : IDisposable
{
    /// <summary>Largest single entry accepted; anything bigger falls back to the stock path.</summary>
    public const int MaxEntryBytes = 512 * 1024 * 1024;

    // A local index keys entries by the first 9 bytes of the encoding key.
    private const int KeyBytesInBucketHash = 9;

    private readonly string _dataDirectory;
    private readonly Dictionary<byte, CASCIndexInstance> _indices;
    private readonly ConcurrentDictionary<int, SafeFileHandle> _archives = new();
    private volatile bool _disposed;

    private LocalCascArchiveReader(string dataDirectory, Dictionary<byte, CASCIndexInstance> indices)
    {
        _dataDirectory = dataDirectory;
        _indices = indices;
    }

    /// <summary>Number of index buckets loaded (a full install has 16).</summary>
    public int BucketCount => _indices.Count;

    /// <summary>
    /// Opens the newest <c>.idx</c> of every bucket under <c>&lt;installDir&gt;/Data/data</c>, the same
    /// selection TACTSharp makes. Returns null when the folder or its indices are absent or unreadable;
    /// the caller then keeps using the stock reader.
    /// </summary>
    public static LocalCascArchiveReader? TryOpen(string installDir)
    {
        if (string.IsNullOrWhiteSpace(installDir))
            return null;

        string dataDirectory = Path.Combine(installDir, "Data", "data");
        if (!Directory.Exists(dataDirectory))
            return null;

        var newest = new Dictionary<byte, (int Version, string Path)>();
        try
        {
            foreach (string file in Directory.EnumerateFiles(dataDirectory, "*.idx"))
            {
                string name = Path.GetFileNameWithoutExtension(file);
                if (name.Contains("tempfile", StringComparison.OrdinalIgnoreCase) || name.Length < 3)
                    continue;

                if (!byte.TryParse(name.AsSpan(0, 2), NumberStyles.HexNumber, CultureInfo.InvariantCulture, out byte bucket)
                    || !int.TryParse(name.AsSpan(2), NumberStyles.HexNumber, CultureInfo.InvariantCulture, out int version))
                {
                    continue;
                }

                if (!newest.TryGetValue(bucket, out var current) || version > current.Version)
                    newest[bucket] = (version, file);
            }

            var indices = new Dictionary<byte, CASCIndexInstance>(newest.Count);
            foreach (var (bucket, entry) in newest)
                indices[bucket] = new CASCIndexInstance(entry.Path);

            return indices.Count == 0 ? null : new LocalCascArchiveReader(dataDirectory, indices);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or InvalidOperationException or ArgumentException)
        {
            return null;
        }
    }

    /// <summary>
    /// Reads the still-encoded (BLTE) bytes of the entry whose encoding key is
    /// <paramref name="eKey"/> (16 bytes). False when the local index does not hold it or the archive
    /// cannot supply all its bytes.
    /// </summary>
    public bool TryReadEncoded(ReadOnlySpan<byte> eKey, out byte[]? encoded)
    {
        encoded = null;
        if (_disposed || eKey.Length < KeyBytesInBucketHash)
            return false;

        int fold = 0;
        for (int i = 0; i < KeyBytesInBucketHash; i++)
            fold ^= eKey[i];
        byte bucket = (byte)((fold & 0xF) ^ (fold >> 4));

        if (!_indices.TryGetValue(bucket, out CASCIndexInstance? index))
            return false;

        // GetIndexInfo takes a Span and slices the leading key bytes; give it a private copy.
        Span<byte> probe = stackalloc byte[eKey.Length];
        eKey.CopyTo(probe);
        (int offset, int size, int archiveIndex) = index.GetIndexInfo(probe);
        if (offset < 0 || size <= 0 || size > MaxEntryBytes)
            return false;

        SafeFileHandle? archive = GetArchive(archiveIndex);
        if (archive is null)
            return false;

        byte[] buffer = GC.AllocateUninitializedArray<byte>(size);
        int total = 0;
        while (total < size)
        {
            int read = RandomAccess.Read(archive, buffer.AsSpan(total), (long)offset + total);
            if (read <= 0)
                return false; // archive shorter than the index claims: let the stock path decide
            total += read;
        }

        encoded = buffer;
        return true;
    }

    private SafeFileHandle? GetArchive(int archiveIndex)
    {
        if (_archives.TryGetValue(archiveIndex, out SafeFileHandle? existing))
            return existing;

        string path = Path.Combine(_dataDirectory, "data." + archiveIndex.ToString(CultureInfo.InvariantCulture).PadLeft(3, '0'));
        try
        {
            SafeFileHandle opened = File.OpenHandle(path, FileMode.Open, FileAccess.Read,
                FileShare.ReadWrite | FileShare.Delete, FileOptions.RandomAccess);
            if (_archives.TryAdd(archiveIndex, opened))
                return opened;

            opened.Dispose();
            return _archives[archiveIndex];
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    public void Dispose()
    {
        _disposed = true;
        foreach (SafeFileHandle handle in _archives.Values)
            handle.Dispose();
        _archives.Clear();
    }
}
