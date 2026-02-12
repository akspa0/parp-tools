using System;
using System.Text;

public static class ChunkUtils
{
    /// <summary>
    /// Converts a big-endian chunk ID as read from file to its little-endian documentation format.
    /// For example: "REVM" -> "MVER"
    /// </summary>
    public static string ConvertChunkIdToDocFormat(string bigEndianId)
    {
        if (string.IsNullOrEmpty(bigEndianId) || bigEndianId.Length != 4)
        {
            throw new ArgumentException("Chunk ID must be exactly 4 characters", nameof(bigEndianId));
        }
        
        // Reverse the characters to convert from big-endian to little-endian
        char[] chars = bigEndianId.ToCharArray();
        Array.Reverse(chars);
        return new string(chars);
    }

    /// <summary>
    /// Converts a documentation format (little-endian) chunk ID to its big-endian file format.
    /// For example: "MVER" -> "REVM"
    /// </summary>
    public static string ConvertChunkIdToFileFormat(string docFormatId)
    {
        if (string.IsNullOrEmpty(docFormatId) || docFormatId.Length != 4)
        {
            throw new ArgumentException("Chunk ID must be exactly 4 characters", nameof(docFormatId));
        }
        
        // Reverse the characters to convert from little-endian to big-endian
        char[] chars = docFormatId.ToCharArray();
        Array.Reverse(chars);
        return new string(chars);
    }

    /// <summary>
    /// Reads a chunk ID from a binary reader and converts it to documentation format.
    /// </summary>
    public static string ReadChunkId(BinaryReader reader)
    {
        byte[] idBytes = reader.ReadBytes(4);
        string bigEndianId = Encoding.ASCII.GetString(idBytes);
        return ConvertChunkIdToDocFormat(bigEndianId);
    }

    /// <summary>
    /// Writes a documentation format chunk ID to a binary writer in file format (big-endian).
    /// </summary>
    public static void WriteChunkId(BinaryWriter writer, string docFormatId)
    {
        string fileFormatId = ConvertChunkIdToFileFormat(docFormatId);
        byte[] idBytes = Encoding.ASCII.GetBytes(fileFormatId);
        writer.Write(idBytes);
    }
} 