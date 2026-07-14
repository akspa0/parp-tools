using System.Buffers.Binary;
using System.Text;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Binary V22 enrichment stream format.
///
/// Layout:
///   Header: "V22E" + version uint32 (1)
///   Entries: zero or more "ENTRY" records
///   Terminator: "ENDS"
///
/// Each ENTRY:
///   "ENTRY" magic  (5 bytes)
///   path_len       (uint32)
///   path_utf8      (path_len bytes)
///   kind           (uint8: 0=unknown, 1=M2, 2=WMO, 3=BLP)
///   load_error     (uint8: 0=ok, 1=fail)
///   array_count    (uint32)
///   for each array:
///     name_len     (uint32)
///     name_utf8    (name_len bytes)
///     ndim         (uint32)
///     shape        (ndim × uint32)
///     dtype        (8 bytes ASCII, null-padded)
///     data_len     (int64)
///     data         (data_len bytes)
///
/// All multi-byte values are little-endian.
/// </summary>
public sealed class EnrichmentStreamWriter : IDisposable
{
    private readonly Stream _stream;
    private bool _headerWritten;
    private bool _disposed;

    /// <param name="stream">Writable stream. The writer takes ownership of the stream.</param>
    public EnrichmentStreamWriter(Stream stream)
    {
        _stream = stream ?? throw new ArgumentNullException(nameof(stream));
        _headerWritten = false;
    }

    /// <summary>
    /// Write the header and prepare for entries. Must be called once before <see cref="WriteEntry"/>.
    /// </summary>
    public void WriteHeader()
    {
        if (_headerWritten)
            return;
        _headerWritten = true;

        // Magic "V22E"
        _stream.Write("V22E"u8);
        // Version
        Span<byte> versionBuf = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(versionBuf, 1);
        _stream.Write(versionBuf);
    }

    /// <summary>
    /// Write one asset entry.
    /// </summary>
    public void WriteEntry(EnrichmentEntry entry)
    {
        ArgumentNullException.ThrowIfNull(entry);
        if (!_headerWritten)
            WriteHeader();

        // Magic "ENTRY"
        _stream.Write("ENTRY"u8);

        // Path
        byte[] pathBytes = Encoding.UTF8.GetBytes(entry.CanonicalPath);
        WriteUInt32((uint)pathBytes.Length);
        _stream.Write(pathBytes);

        // Kind + load_error
        _stream.WriteByte((byte)entry.Kind);
        _stream.WriteByte((byte)entry.LoadError);

        // Array count
        WriteUInt32((uint)entry.Arrays.Count);

        // Each array
        foreach (EnrichmentArray array in entry.Arrays)
        {
            // Name
            byte[] nameBytes = Encoding.UTF8.GetBytes(array.Name);
            WriteUInt32((uint)nameBytes.Length);
            _stream.Write(nameBytes);

            // Ndim + shape
            int ndim = array.Rank;
            WriteUInt32((uint)ndim);
            for (int i = 0; i < ndim; i++)
                WriteUInt32((uint)array.Shape[i]);

            // Dtype (8 bytes ASCII, null-padded)
            string dtypeStr = GetDtypeString(array.DataType);
            byte[] dtypeBytes = Encoding.ASCII.GetBytes(dtypeStr);
            Span<byte> dtypeBuf = stackalloc byte[8];
            dtypeBuf.Clear();
            int copyLen = Math.Min(dtypeBytes.Length, 8);
            dtypeBytes.AsSpan(0, copyLen).CopyTo(dtypeBuf);
            _stream.Write(dtypeBuf);

            // Data
            WriteInt64(array.Data.LongLength);
            _stream.Write(array.Data);
        }
    }

    /// <summary>
    /// Write the ENDS terminator and flush. Call once after all entries.
    /// </summary>
    public void WriteEnds()
    {
        if (!_headerWritten)
            WriteHeader();
        _stream.Write("ENDS"u8);
        _stream.Flush();
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _stream.Dispose();
        }
    }

    private void WriteUInt32(uint value)
    {
        Span<byte> buf = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(buf, value);
        _stream.Write(buf);
    }

    private void WriteInt64(long value)
    {
        Span<byte> buf = stackalloc byte[8];
        BinaryPrimitives.WriteInt64LittleEndian(buf, value);
        _stream.Write(buf);
    }

    private static string GetDtypeString(Type type)
    {
        if (type == typeof(float)) return "<f4";
        if (type == typeof(double)) return "<f8";
        if (type == typeof(int)) return "<i4";
        if (type == typeof(uint)) return "<u4";
        if (type == typeof(short)) return "<i2";
        if (type == typeof(ushort)) return "<u2";
        if (type == typeof(byte)) return "|u1";
        if (type == typeof(sbyte)) return "|i1";
        if (type == typeof(bool)) return "|b1";
        return "|u1";
    }
}

/// <summary>
/// Reads a V22 enrichment stream. Forward-only, no random access.
/// </summary>
public sealed class EnrichmentStreamReader : IDisposable
{
    private readonly Stream _stream;
    private bool _disposed;

    /// <param name="stream">Readable stream. The reader takes ownership of the stream.</param>
    public EnrichmentStreamReader(Stream stream)
    {
        _stream = stream ?? throw new ArgumentNullException(nameof(stream));
    }

    /// <summary>
    /// Read the header. Must be called before <see cref="ReadEntries"/>.
    /// Returns the version number. Throws if magic is missing.
    /// </summary>
    public uint ReadHeader()
    {
        Span<byte> magicBuf = stackalloc byte[4];
        ReadExact(magicBuf);
        if (!magicBuf.SequenceEqual("V22E"u8))
            throw new InvalidDataException($"Expected V22E magic, got {Encoding.ASCII.GetString(magicBuf)}");

        Span<byte> versionBuf = stackalloc byte[4];
        ReadExact(versionBuf);
        return BinaryPrimitives.ReadUInt32LittleEndian(versionBuf);
    }

    /// <summary>
    /// Enumerate all entries in the stream. Stops when ENDS is encountered.
    /// </summary>
    public IEnumerable<EnrichmentEntry> ReadEntries()
    {
        while (true)
        {
            Span<byte> magicBuf = stackalloc byte[4];
            int read = _stream.Read(magicBuf);
            if (read < 4)
                yield break;

            if (magicBuf.SequenceEqual("ENDS"u8))
                yield break;

            Span<byte> entryBuf = stackalloc byte[5];
            magicBuf.CopyTo(entryBuf);
            int suffix = _stream.ReadByte();
            if (suffix < 0)
                throw new InvalidDataException("Unexpected end of stream reading ENTRY magic.");
            entryBuf[4] = (byte)suffix;

            if (!entryBuf.SequenceEqual("ENTRY"u8))
                throw new InvalidDataException($"Expected ENTRY or ENDS, got {Encoding.ASCII.GetString(entryBuf)}");

            // Path
            uint pathLen = ReadUInt32();
            string path = ReadString((int)pathLen);

            // Kind + load_error
            int kindByte = _stream.ReadByte();
            int loadErrorByte = _stream.ReadByte();
            if (kindByte < 0 || loadErrorByte < 0)
                throw new InvalidDataException("Unexpected end of stream reading entry header.");
            var kind = (AssetKind)kindByte;
            int loadError = loadErrorByte;

            // Array count
            uint arrayCount = ReadUInt32();
            var arrays = new List<EnrichmentArray>((int)arrayCount);

            for (int i = 0; i < arrayCount; i++)
            {
                // Name
                uint nameLen = ReadUInt32();
                string name = ReadString((int)nameLen);

                // Ndim + shape
                uint ndim = ReadUInt32();
                int[] shape = new int[ndim];
                for (int j = 0; j < ndim; j++)
                    shape[j] = (int)ReadUInt32();

                // Dtype
                Span<byte> dtypeBuf = stackalloc byte[8];
                ReadExact(dtypeBuf);
                string dtypeStr = Encoding.ASCII.GetString(dtypeBuf).TrimEnd('\0');
                Type dataType = ParseDtype(dtypeStr);

                // Data
                long dataLen = ReadInt64();
                byte[] data = new byte[dataLen];
                ReadExact(data);

                arrays.Add(new EnrichmentArray(name, shape, dataType, data));
            }

            yield return new EnrichmentEntry(path, kind, loadError, arrays);
        }
    }

    /// <summary>
    /// Read the ENDS terminator. Should be called after enumerating entries to confirm clean termination.
    /// </summary>
    public void ReadEnds()
    {
        Span<byte> magicBuf = stackalloc byte[4];
        ReadExact(magicBuf);
        if (!magicBuf.SequenceEqual("ENDS"u8))
            throw new InvalidDataException($"Expected ENDS terminator, got {Encoding.ASCII.GetString(magicBuf)}");
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _stream.Dispose();
        }
    }

    private uint ReadUInt32()
    {
        Span<byte> buf = stackalloc byte[4];
        ReadExact(buf);
        return BinaryPrimitives.ReadUInt32LittleEndian(buf);
    }

    private long ReadInt64()
    {
        Span<byte> buf = stackalloc byte[8];
        ReadExact(buf);
        return BinaryPrimitives.ReadInt64LittleEndian(buf);
    }

    private void ReadExact(Span<byte> buf)
    {
        int offset = 0;
        while (offset < buf.Length)
        {
            int read = _stream.Read(buf[offset..]);
            if (read <= 0)
                throw new InvalidDataException("Unexpected end of stream.");
            offset += read;
        }
    }

    private string ReadString(int length)
    {
        byte[] buf = new byte[length];
        ReadExact(buf);
        return Encoding.UTF8.GetString(buf);
    }

    private static Type ParseDtype(string dtype)
    {
        return dtype switch
        {
            "<f4" => typeof(float),
            "<f8" => typeof(double),
            "<i4" => typeof(int),
            "<u4" => typeof(uint),
            "<i2" => typeof(short),
            "<u2" => typeof(ushort),
            "|u1" => typeof(byte),
            "|i1" => typeof(sbyte),
            "|b1" => typeof(bool),
            _ => typeof(byte),
        };
    }
}
