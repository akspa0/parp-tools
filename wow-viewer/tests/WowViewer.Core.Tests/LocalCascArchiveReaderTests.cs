using System.Buffers.Binary;
using TACTSharp;
using WowViewer.Core.IO.Casc;

namespace WowViewer.Core.Tests;

/// <summary>
/// Spec 275 CASC-1: the cached-handle local reader must return exactly the bytes TACTSharp's own
/// local read (<c>CDN.TryGetLocalFile</c>) returns. These tests build a synthetic install with the
/// real <c>.idx</c> layout (40-byte header, 18-byte entries, 30-byte archive entry headers) so the
/// comparison runs without game data; real-install parity is checked by <c>inspect casc bench</c>.
/// </summary>
public sealed class LocalCascArchiveReaderTests : IDisposable
{
    private const int EntryHeaderBytes = 30;
    private readonly string _root = Path.Combine(AppContext.BaseDirectory, "test-output", "casc-local-" + Guid.NewGuid().ToString("N"));

    private sealed record Entry(byte[] EKey, byte[] Payload, int Archive, int Offset);

    public LocalCascArchiveReaderTests() => Directory.CreateDirectory(DataDirectory);

    private string DataDirectory => Path.Combine(_root, "Data", "data");

    public void Dispose()
    {
        // Index files stay memory-mapped until the runtime releases them; leftover folders sit under bin/.
        try { Directory.Delete(_root, recursive: true); } catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
    }

    private static int BucketOf(byte[] eKey)
    {
        int fold = 0;
        for (int i = 0; i < 9; i++) fold ^= eKey[i];
        return (fold & 0xF) ^ (fold >> 4);
    }

    /// <summary>Writes <c>data.NNN</c> archives and one <c>.idx</c> per bucket (all at <paramref name="version"/>).</summary>
    private List<Entry> BuildInstall(int count, int archives, int seed, int maxPayload = 70_000, int version = 1)
    {
        var random = new Random(seed);
        var entries = new List<Entry>();
        var archiveStreams = new List<MemoryStream>();
        for (int a = 0; a < archives; a++) archiveStreams.Add(new MemoryStream());

        var seen = new HashSet<string>();
        while (entries.Count < count)
        {
            var key = new byte[16];
            random.NextBytes(key);
            if (key[0] == 0) key[0] = 1; // keep the all-zero sentinel (below) sorted first
            if (!seen.Add(Convert.ToHexString(key.AsSpan(0, 9)))) continue;

            int size = entries.Count % 37 == 0 ? 2_000_000 : 1 + random.Next(maxPayload);
            var payload = new byte[size];
            random.NextBytes(payload);

            int archive = entries.Count % archives;
            MemoryStream stream = archiveStreams[archive];
            int offset = (int)stream.Position;
            var header = new byte[EntryHeaderBytes];
            random.NextBytes(header);
            stream.Write(header);
            stream.Write(payload);
            entries.Add(new Entry(key, payload, archive, offset));
        }

        for (int a = 0; a < archives; a++)
            File.WriteAllBytes(Path.Combine(DataDirectory, $"data.{a:000}"), archiveStreams[a].ToArray());

        // TACTSharp's CASCIndexInstance.GetIndexInfo reports "not found" whenever the lower bound lands on
        // the first entry of an index (a quirk that is negligible at thousands of entries per bucket but
        // would hide one entry per bucket here). A leading sentinel keeps every real entry findable.
        var sentinel = new Entry(new byte[16], [], 0, 0);
        foreach (IGrouping<int, Entry> bucket in entries.GroupBy(e => BucketOf(e.EKey)))
            WriteIndex(bucket.Key, version, bucket.Append(sentinel));

        return entries;
    }

    private void WriteIndex(int bucket, int version, IEnumerable<Entry> entries, int offsetShift = 0)
    {
        // The index is binary-searched on the first 9 key bytes, so entries must be sorted by them.
        List<Entry> sorted = entries
            .OrderBy(e => e.EKey.AsSpan(0, 9).ToArray(), Comparer<byte[]>.Create((x, y) => x.AsSpan().SequenceCompareTo(y.AsSpan())))
            .ToList();
        const int entrySize = 18;
        var bytes = new byte[40 + sorted.Count * entrySize];
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(8), 7);       // version
        bytes[10] = (byte)bucket;                                           // bucket index
        bytes[12] = 4; bytes[13] = 5; bytes[14] = 9; bytes[15] = 30;        // size/offset/key bytes, offset bits
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(32), (uint)(sorted.Count * entrySize));

        for (int i = 0; i < sorted.Count; i++)
        {
            Entry e = sorted[i];
            Span<byte> row = bytes.AsSpan(40 + i * entrySize, entrySize);
            e.EKey.AsSpan(0, 9).CopyTo(row);
            row[9] = (byte)(e.Archive >> 2);
            uint low = ((uint)(e.Archive & 3) << 30) | (uint)(e.Offset + offsetShift);
            BinaryPrimitives.WriteUInt32BigEndian(row.Slice(10, 4), low);
            BinaryPrimitives.WriteInt32LittleEndian(row.Slice(14, 4), e.Payload.Length + EntryHeaderBytes);
        }

        // TACTSharp's naming: two hex digits of bucket, then the version as hex padded to 8 digits.
        File.WriteAllBytes(Path.Combine(DataDirectory, $"{bucket:x2}{version.ToString("x2").PadLeft(8, '0')}.idx"), bytes);
    }

    [Fact]
    public void Reads_EveryEntryAcrossArchives_AndMatchesTactSharp()
    {
        List<Entry> entries = BuildInstall(count: 220, archives: 3, seed: 11);
        using LocalCascArchiveReader reader = LocalCascArchiveReader.TryOpen(_root)!;
        Assert.NotNull(reader);

        var cdn = new CDN(new Settings { BaseDir = _root, TryCDN = false });
        cdn.OpenLocal();
        Assert.True(cdn.HasLocal);

        foreach (Entry entry in entries)
        {
            Assert.True(reader.TryReadEncoded(entry.EKey, out byte[]? mine));
            Assert.True(entry.Payload.AsSpan().SequenceEqual(mine), "reader bytes differ from the archive payload");

            Assert.True(cdn.TryGetLocalFile(Convert.ToHexStringLower(entry.EKey), out ReadOnlySpan<byte> stock));
            Assert.True(stock.SequenceEqual(mine), "reader bytes differ from TACTSharp's local read");
        }
    }

    [Fact]
    public void ParallelReads_ReturnIdenticalBytes()
    {
        List<Entry> entries = BuildInstall(count: 120, archives: 2, seed: 5);
        using LocalCascArchiveReader reader = LocalCascArchiveReader.TryOpen(_root)!;

        int failures = 0;
        Parallel.For(0, entries.Count * 6, new ParallelOptions { MaxDegreeOfParallelism = 8 }, i =>
        {
            Entry entry = entries[i % entries.Count];
            if (!reader.TryReadEncoded(entry.EKey, out byte[]? bytes) || !entry.Payload.AsSpan().SequenceEqual(bytes))
                Interlocked.Increment(ref failures);
        });

        Assert.Equal(0, failures);
    }

    [Fact]
    public void UnknownKey_ReturnsFalse()
    {
        BuildInstall(count: 30, archives: 1, seed: 3);
        using LocalCascArchiveReader reader = LocalCascArchiveReader.TryOpen(_root)!;

        Assert.False(reader.TryReadEncoded(new byte[16].Select((_, i) => (byte)(0xA0 + i)).ToArray(), out byte[]? bytes));
        Assert.Null(bytes);
        Assert.False(reader.TryReadEncoded(new byte[4], out _));
    }

    [Fact]
    public void TruncatedArchive_FailsOnlyTheEntriesThatNoLongerFit()
    {
        List<Entry> entries = BuildInstall(count: 60, archives: 1, seed: 9);
        string archive = Path.Combine(DataDirectory, "data.000");
        long cut = new FileInfo(archive).Length / 2;
        using (var fs = new FileStream(archive, FileMode.Open, FileAccess.Write)) fs.SetLength(cut);

        using LocalCascArchiveReader reader = LocalCascArchiveReader.TryOpen(_root)!;
        foreach (Entry entry in entries)
        {
            bool fits = entry.Offset + EntryHeaderBytes + entry.Payload.Length <= cut;
            Assert.Equal(fits, reader.TryReadEncoded(entry.EKey, out byte[]? bytes));
            if (fits) Assert.True(entry.Payload.AsSpan().SequenceEqual(bytes));
        }
    }

    [Fact]
    public void MissingArchive_FailsOnlyItsEntries()
    {
        List<Entry> entries = BuildInstall(count: 40, archives: 2, seed: 21);
        File.Delete(Path.Combine(DataDirectory, "data.001"));

        using LocalCascArchiveReader reader = LocalCascArchiveReader.TryOpen(_root)!;
        foreach (Entry entry in entries)
            Assert.Equal(entry.Archive == 0, reader.TryReadEncoded(entry.EKey, out _));
    }

    [Fact]
    public void TryOpen_WithoutIndices_ReturnsNull()
    {
        Assert.Null(LocalCascArchiveReader.TryOpen(_root));                                 // Data/data exists but is empty
        Assert.Null(LocalCascArchiveReader.TryOpen(Path.Combine(_root, "does-not-exist")));
        Assert.Null(LocalCascArchiveReader.TryOpen(""));
    }

    [Fact]
    public void NewestIndexVersion_WinsPerBucket_LikeTactSharp()
    {
        // Version 1 carries a stale offset; version 2 is correct. Both readers must pick version 2.
        List<Entry> entries = BuildInstall(count: 50, archives: 1, seed: 33, version: 2);
        foreach (IGrouping<int, Entry> bucket in entries.GroupBy(e => BucketOf(e.EKey)))
            WriteIndex(bucket.Key, 1, bucket, offsetShift: 7);

        using LocalCascArchiveReader reader = LocalCascArchiveReader.TryOpen(_root)!;
        var cdn = new CDN(new Settings { BaseDir = _root, TryCDN = false });
        cdn.OpenLocal();

        foreach (Entry entry in entries)
        {
            Assert.True(reader.TryReadEncoded(entry.EKey, out byte[]? mine));
            Assert.True(entry.Payload.AsSpan().SequenceEqual(mine));
            Assert.True(cdn.TryGetLocalFile(Convert.ToHexStringLower(entry.EKey), out ReadOnlySpan<byte> stock));
            Assert.True(stock.SequenceEqual(mine));
        }
    }

    [Fact]
    public void CachedHandles_DoNotBlockAnotherWriter()
    {
        List<Entry> entries = BuildInstall(count: 10, archives: 1, seed: 4);
        using LocalCascArchiveReader reader = LocalCascArchiveReader.TryOpen(_root)!;
        Assert.True(reader.TryReadEncoded(entries[0].EKey, out _)); // opens and caches the handle

        // The Battle.net agent appends to and replaces these files while the viewer runs.
        string archive = Path.Combine(DataDirectory, "data.000");
        using (var writer = new FileStream(archive, FileMode.Append, FileAccess.Write, FileShare.ReadWrite | FileShare.Delete))
            writer.Write(new byte[] { 1, 2, 3 });

        Assert.True(reader.TryReadEncoded(entries[1].EKey, out byte[]? bytes));
        Assert.True(entries[1].Payload.AsSpan().SequenceEqual(bytes));
    }

    [Fact]
    public void ReadAfterDispose_ReturnsFalse()
    {
        List<Entry> entries = BuildInstall(count: 5, archives: 1, seed: 8);
        LocalCascArchiveReader reader = LocalCascArchiveReader.TryOpen(_root)!;
        Assert.True(reader.TryReadEncoded(entries[0].EKey, out _));

        reader.Dispose();

        Assert.False(reader.TryReadEncoded(entries[0].EKey, out byte[]? bytes));
        Assert.Null(bytes);
    }
}
