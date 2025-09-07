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
                // Heuristic: if value is within string block, try to read a C-string
                if (val < stringBlock.Length)
                {
                    var s = ReadCString(stringBlock, (int)val);
                    if (!string.IsNullOrWhiteSpace(s) && LooksLikeText(s))
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

    private static bool LooksLikeText(string s)
    {
        // Accept printable ASCII and common path chars, and also filter out very short junk
        if (s.Length < 2) return false;
        foreach (var ch in s)
        {
            if (ch < 0x20 || ch > 0x7E) return false;
        }
        return true;
    }
}
