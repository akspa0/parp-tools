using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace GillijimProject.Next.Core.WowFiles;

/// <summary>
/// Helpers for reading and writing FourCC chunked files (WDT/ADT/WDL).
/// Conventions: FourCCs are forward in memory and reversed on disk.
/// </summary>
public static class ChunkIO
{
    public const int HeaderSize = 8; // fourcc + size

    public static bool TryReadHeader(FileStream fs, BinaryReader br, out string id, out uint size, out long dataOffset)
    {
        id = string.Empty; size = 0; dataOffset = 0;
        if (fs.Position + HeaderSize > fs.Length) return false;
        var four = br.ReadBytes(4);
        id = Encoding.ASCII.GetString(four);
        size = br.ReadUInt32();
        dataOffset = fs.Position;
        return true;
    }

    public static void SkipChunk(FileStream fs, uint size)
    {
        long dataEnd = fs.Position + size;
        fs.Position = dataEnd;
        if ((size & 1) == 1 && fs.Position < fs.Length) fs.Position++;
    }

    public static string ReverseFourCC(string s)
    {
        if (s is null || s.Length != 4) throw new ArgumentException("FourCC must be 4 chars", nameof(s));
        return new string(new[] { s[3], s[2], s[1], s[0] });
    }

    public static bool Matches(string id, string expected)
    {
        if (id.Equals(expected, StringComparison.OrdinalIgnoreCase)) return true;
        var rev = ReverseFourCC(expected);
        return id.Equals(rev, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Parse zero-terminated ASCII strings from a byte buffer.
    /// </summary>
    public static IReadOnlyList<string> ParseZeroTerminatedStrings(byte[] buffer)
    {
        var list = new List<string>();
        int start = 0;
        for (int i = 0; i < buffer.Length; i++)
        {
            if (buffer[i] == 0)
            {
                if (i > start)
                {
                    list.Add(Encoding.ASCII.GetString(buffer, start, i - start));
                }
                start = i + 1;
            }
        }
        // If buffer does not end with 0, include trailing
        if (start < buffer.Length)
        {
            list.Add(Encoding.ASCII.GetString(buffer, start, buffer.Length - start));
        }
        return list;
    }
}
