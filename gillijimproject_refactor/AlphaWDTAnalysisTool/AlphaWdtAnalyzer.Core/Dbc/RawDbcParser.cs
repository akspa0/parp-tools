using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace AlphaWdtAnalyzer.Core.Dbc;

public sealed class RawDbcTable
{
    public int RecordCount { get; init; }
    public int FieldCount { get; init; }
    public int RecordSize { get; init; }
    public int StringBlockSize { get; init; }
    public List<RawDbcRow> Rows { get; } = new();
}

public sealed class RawDbcRow
{
    public uint[] Fields { get; }
    public string?[] GuessedStrings { get; }

    public RawDbcRow(uint[] fields, string?[] guessedStrings)
    {
        Fields = fields;
        GuessedStrings = guessedStrings;
    }
}

public sealed class DbcTable
{
    public int RecordCount { get; init; }
    public int FieldCount { get; init; }
    public int RecordSize { get; init; }
    public int StringBlockSize { get; init; }
    public List<DbcRow> Rows { get; } = new();
}

public sealed class DbcRow
{
    public int[] Fields { get; init; } = Array.Empty<int>();
    public string?[] GuessedStrings { get; init; } = Array.Empty<string?>();
}

public static class RawDbcParser
{
    public static RawDbcTable Parse(string path)
    {
        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs, Encoding.ASCII, leaveOpen: false);

        var magic = br.ReadBytes(4);
        if (magic.Length != 4 || magic[0] != (byte)'W' || magic[1] != (byte)'D' || magic[2] != (byte)'B' || magic[3] != (byte)'C')
        {
            throw new InvalidDataException("Not a WDBC file");
        }

        int recordCount = br.ReadInt32();
        int fieldCount = br.ReadInt32();
        int recordSize = br.ReadInt32();
        int stringBlockSize = br.ReadInt32();

        var table = new RawDbcTable
        {
            RecordCount = recordCount,
            FieldCount = fieldCount,
            RecordSize = recordSize,
            StringBlockSize = stringBlockSize
        };

        // read records
        var recordsData = br.ReadBytes(recordCount * recordSize);
        var stringBlock = br.ReadBytes(stringBlockSize);

        for (int i = 0; i < recordCount; i++)
        {
            var fields = new uint[fieldCount];
            int baseOffset = i * recordSize;
            for (int f = 0; f < fieldCount; f++)
            {
                fields[f] = BitConverter.ToUInt32(recordsData, baseOffset + (f * 4));
            }
            var guessed = new string?[fieldCount];
            for (int f = 0; f < fieldCount; f++)
            {
                var val = fields[f];
                // If value is within string block, read a C-string
                if (val < stringBlock.Length)
                {
                    var s = ReadCString(stringBlock, (int)val);
                    if (!string.IsNullOrEmpty(s))
                    {
                        guessed[f] = s;
                    }
                }
            }
            table.Rows.Add(new RawDbcRow(fields, guessed));
        }

        return table;
    }

    private static string ReadCString(byte[] data, int offset)
    {
        int end = Array.IndexOf<byte>(data, 0, offset);
        if (end < 0) end = data.Length;
        int len = end - offset;
        if (len <= 0) return string.Empty;
        return Encoding.ASCII.GetString(data, offset, len);
    }
}

public static class DbcParser
{
    public static DbcTable Parse(string path)
    {
        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs, Encoding.UTF8, leaveOpen: false);

        var magic = br.ReadBytes(4);
        if (magic.Length != 4) throw new InvalidDataException("DBC: unexpected EOF reading magic");
        var magicStr = Encoding.ASCII.GetString(magic);
        if (magicStr != "WDBC" && magicStr != "WDB2")
        {
            throw new InvalidDataException($"Unsupported DBC magic: {magicStr}");
        }

        // Many DBC/WDB2 variants share the first 4 ints in this order.
        // This is a pragmatic reader for our use case.
        int recordCount = br.ReadInt32();
        int fieldCount = br.ReadInt32();
        int recordSize = br.ReadInt32();
        int stringBlockSize = br.ReadInt32();

        // For WDB2 with extended header, skip the remainder of the header if present.
        long headerRead = 4 + 16; // magic + 4 ints
        long toData = fs.Position;

        long recordsBytes = (long)recordCount * recordSize;
        if (toData + recordsBytes + stringBlockSize > fs.Length)
        {
            // Some WDB2 add extra header ints; try to align by scanning forward to plausible data start.
            // Fallback: assume current position is data.
        }

        var rows = new List<DbcRow>(recordCount);
        byte[] recordBuf = new byte[recordSize];

        for (int i = 0; i < recordCount; i++)
        {
            int read = br.Read(recordBuf, 0, recordBuf.Length);
            if (read != recordBuf.Length) throw new InvalidDataException("DBC: truncated record data");
            var ints = new int[fieldCount];
            for (int f = 0; f < fieldCount; f++)
            {
                int off = f * 4;
                if (off + 4 <= recordBuf.Length)
                {
                    ints[f] = BitConverter.ToInt32(recordBuf, off);
                }
                else
                {
                    ints[f] = 0;
                }
            }
            rows.Add(new DbcRow { Fields = ints });
        }

        // Read string block
        var stringBlockStart = fs.Position;
        var stringBlock = br.ReadBytes(stringBlockSize);

        // Populate guessed strings per row by treating any 32-bit field as an offset into the string block
        foreach (var row in rows)
        {
            var strs = new string?[fieldCount];
            for (int f = 0; f < fieldCount; f++)
            {
                int off = row.Fields[f];
                if (off <= 0 || off >= stringBlockSize) continue;
                strs[f] = ReadCString(stringBlock, off);
            }
            row.GuessedStrings = strs;
        }

        return new DbcTable
        {
            RecordCount = recordCount,
            FieldCount = fieldCount,
            RecordSize = recordSize,
            StringBlockSize = stringBlockSize,
            Rows = rows
        };
    }

    private static string ReadCString(byte[] block, int offset)
    {
        int end = offset;
        while (end < block.Length && block[end] != 0) end++;
        if (end <= offset) return string.Empty;
        return Encoding.UTF8.GetString(block, offset, end - offset);
    }
}
