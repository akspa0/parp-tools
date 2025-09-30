using System;
using System.Collections.Generic;
using System.IO;

namespace WoWRollback.Core.Services;

/// <summary>
/// Reads MDDF/MODF placement data from converted Lich King ADT files.
/// These files have proper world coordinates, unlike Alpha 0.5.3 which stores (0,0,0).
/// </summary>
public class LkAdtReader
{
    public record PlacementData(
        int NameIndex,
        int UniqueId,
        float WorldX,
        float WorldY,
        float WorldZ,
        string AssetType // "M2" or "WMO"
    );

    /// <summary>
    /// Reads all MDDF (M2/MDX) placements from a LK ADT file.
    /// </summary>
    public static List<PlacementData> ReadMddf(string adtPath)
    {
        var placements = new List<PlacementData>();
        
        if (!File.Exists(adtPath))
            return placements;

        try
        {
            var bytes = File.ReadAllBytes(adtPath);
            
            // Find FDDM chunk (MDDF reversed on disk)
            var chunkStart = FindChunk(bytes, "FDDM");
            if (chunkStart == -1)
                return placements;

            // Read chunk size (little-endian, 4 bytes after FourCC)
            var chunkSize = BitConverter.ToInt32(bytes, chunkStart + 4);
            var dataStart = chunkStart + 8;
            
            // Parse MDDF entries (36 bytes each)
            const int entrySize = 36;
            for (int offset = 0; offset + entrySize <= chunkSize; offset += entrySize)
            {
                var entryStart = dataStart + offset;
                
                int nameIndex = BitConverter.ToInt32(bytes, entryStart + 0);
                int uniqueId = BitConverter.ToInt32(bytes, entryStart + 4);
                float worldX = BitConverter.ToSingle(bytes, entryStart + 8);
                float worldZ = BitConverter.ToSingle(bytes, entryStart + 12);
                float worldY = BitConverter.ToSingle(bytes, entryStart + 16);
                
                placements.Add(new PlacementData(nameIndex, uniqueId, worldX, worldY, worldZ, "M2"));
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[LkAdtReader] Error reading MDDF from {adtPath}: {ex.Message}");
        }

        return placements;
    }

    /// <summary>
    /// Reads all MODF (WMO) placements from a LK ADT file.
    /// </summary>
    public static List<PlacementData> ReadModf(string adtPath)
    {
        var placements = new List<PlacementData>();
        
        if (!File.Exists(adtPath))
            return placements;

        try
        {
            var bytes = File.ReadAllBytes(adtPath);
            
            // Find FDOM chunk (MODF reversed on disk)
            var chunkStart = FindChunk(bytes, "FDOM");
            if (chunkStart == -1)
                return placements;

            // Read chunk size
            var chunkSize = BitConverter.ToInt32(bytes, chunkStart + 4);
            var dataStart = chunkStart + 8;
            
            // Parse MODF entries (64 bytes each)
            const int entrySize = 64;
            for (int offset = 0; offset + entrySize <= chunkSize; offset += entrySize)
            {
                var entryStart = dataStart + offset;
                
                int nameIndex = BitConverter.ToInt32(bytes, entryStart + 0);
                int uniqueId = BitConverter.ToInt32(bytes, entryStart + 4);
                float worldX = BitConverter.ToSingle(bytes, entryStart + 8);
                float worldZ = BitConverter.ToSingle(bytes, entryStart + 12);
                float worldY = BitConverter.ToSingle(bytes, entryStart + 16);
                
                placements.Add(new PlacementData(nameIndex, uniqueId, worldX, worldY, worldZ, "WMO"));
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[LkAdtReader] Error reading MODF from {adtPath}: {ex.Message}");
        }

        return placements;
    }

    /// <summary>
    /// Reads MMDX (M2 filename table) from a LK ADT file.
    /// </summary>
    public static List<string> ReadMmdx(string adtPath)
    {
        var names = new List<string>();
        
        if (!File.Exists(adtPath))
            return names;

        try
        {
            var bytes = File.ReadAllBytes(adtPath);
            
            // Find XDMM chunk (MMDX reversed)
            var chunkStart = FindChunk(bytes, "XDMM");
            if (chunkStart == -1)
                return names;

            var chunkSize = BitConverter.ToInt32(bytes, chunkStart + 4);
            var dataStart = chunkStart + 8;
            
            // Parse null-terminated strings
            names = ParseStringTable(bytes, dataStart, chunkSize);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[LkAdtReader] Error reading MMDX from {adtPath}: {ex.Message}");
        }

        return names;
    }

    /// <summary>
    /// Reads MWMO (WMO filename table) from a LK ADT file.
    /// </summary>
    public static List<string> ReadMwmo(string adtPath)
    {
        var names = new List<string>();
        
        if (!File.Exists(adtPath))
            return names;

        try
        {
            var bytes = File.ReadAllBytes(adtPath);
            
            // Find OMWM chunk (MWMO reversed)
            var chunkStart = FindChunk(bytes, "OMWM");
            if (chunkStart == -1)
                return names;

            var chunkSize = BitConverter.ToInt32(bytes, chunkStart + 4);
            var dataStart = chunkStart + 8;
            
            names = ParseStringTable(bytes, dataStart, chunkSize);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[LkAdtReader] Error reading MWMO from {adtPath}: {ex.Message}");
        }

        return names;
    }

    private static int FindChunk(byte[] bytes, string reversedFourCC)
    {
        var pattern = System.Text.Encoding.ASCII.GetBytes(reversedFourCC);
        
        for (int i = 0; i < bytes.Length - 4; i++)
        {
            if (bytes[i] == pattern[0] &&
                bytes[i + 1] == pattern[1] &&
                bytes[i + 2] == pattern[2] &&
                bytes[i + 3] == pattern[3])
            {
                return i;
            }
        }
        
        return -1;
    }

    private static List<string> ParseStringTable(byte[] bytes, int start, int length)
    {
        var strings = new List<string>();
        var currentString = new List<byte>();
        
        for (int i = 0; i < length; i++)
        {
            byte b = bytes[start + i];
            if (b == 0)
            {
                if (currentString.Count > 0)
                {
                    strings.Add(System.Text.Encoding.UTF8.GetString(currentString.ToArray()));
                    currentString.Clear();
                }
            }
            else
            {
                currentString.Add(b);
            }
        }
        
        // Add last string if no trailing null
        if (currentString.Count > 0)
        {
            strings.Add(System.Text.Encoding.UTF8.GetString(currentString.ToArray()));
        }
        
        return strings;
    }
}
