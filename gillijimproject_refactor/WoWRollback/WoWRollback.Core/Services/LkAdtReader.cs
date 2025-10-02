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
        float RotX,
        float RotY,
        float RotZ,
        float Scale,
        ushort Flags,
        ushort DoodadSet,
        ushort NameSet,
        string AssetType // "M2" or "WMO"
    );

    /// <summary>
    /// Reads all MDDF (M2/MDX) placements from a LK ADT file.
    /// </summary>
    public static List<PlacementData> ReadMddf(string adtPath) => ReadMddf(adtPath, out _);

    /// <summary>
    /// Reads all MDDF (M2/MDX) placements from a LK ADT file and reports the detected entry size.
    /// Supports both LK (36-byte) and early 0.6.0 (32-byte) layouts.
    /// </summary>
    public static List<PlacementData> ReadMddf(string adtPath, out int entrySize)
    {
        var placements = new List<PlacementData>();
        entrySize = 0;

        if (!File.Exists(adtPath))
            return placements;

        try
        {
            var bytes = File.ReadAllBytes(adtPath);

            // Find MDDF chunk (support forward and reversed on disk)
            var chunkStart = FindChunkEither(bytes, "MDDF", "FDDM");
            if (chunkStart == -1)
            {
                Console.WriteLine($"[LkAdtReader] MDDF not found in {adtPath}");
                return placements;
            }

            // Read chunk size (little-endian, 4 bytes after FourCC)
            var chunkSize = BitConverter.ToInt32(bytes, chunkStart + 4);
            var dataStart = chunkStart + 8;
            var available = Math.Max(0, Math.Min(chunkSize, bytes.Length - dataStart));

            if (available <= 0)
            {
                return placements;
            }

            entrySize = DetermineEntrySize(available, 36, 32);
            if (entrySize == 0)
            {
                Console.WriteLine($"[LkAdtReader] Unsupported MDDF entry size in {adtPath} (chunkSize={chunkSize}).");
                return placements;
            }

            for (int offset = 0; offset + entrySize <= available; offset += entrySize)
            {
                var entryStart = dataStart + offset;
                if (entryStart + entrySize > bytes.Length)
                {
                    break;
                }

                int nameIndex = BitConverter.ToInt32(bytes, entryStart + 0);
                int uniqueId = BitConverter.ToInt32(bytes, entryStart + 4);

                // MDDF coordinates are relative to map corner, convert to world coords
                // Formula from wowdev.wiki: posx = 32 * TILESIZE - mddf.position[0]
                const float TILESIZE = 533.33333f;
                const float MAP_HALF_SIZE = 32.0f * TILESIZE; // 17066.66656

                float rawX = BitConverter.ToSingle(bytes, entryStart + 8);   // X (east-west)
                float rawZ = BitConverter.ToSingle(bytes, entryStart + 12);  // Z (north-south)
                float rawY = BitConverter.ToSingle(bytes, entryStart + 16);  // Y (height)
                float rotX = BitConverter.ToSingle(bytes, entryStart + 20);
                float rotY = BitConverter.ToSingle(bytes, entryStart + 24);
                float rotZ = BitConverter.ToSingle(bytes, entryStart + 28);
                ushort scaleU = entrySize >= 34 ? BitConverter.ToUInt16(bytes, entryStart + 32) : (ushort)1024;
                ushort flags = entrySize >= 36 ? BitConverter.ToUInt16(bytes, entryStart + 34) : (ushort)0;

                // Convert from map-corner-relative to world coordinates (centered at origin)
                // WoW convention: Y is up (height). The horizontal plane is X/Z.
                // World horizontal coordinates are measured from map center:
                //   X_world = 32*TILESIZE - rawX
                //   Z_world = 32*TILESIZE - rawZ
                // Height stays as is: Y_world = rawY
                float worldX = MAP_HALF_SIZE - rawX;
                float worldY = MAP_HALF_SIZE - rawZ; // north-south
                float worldZ = rawY;                 // height

                // LK MDDF scale typically stored as ushort where 1024 == 1.0f
                float scale = scaleU > 0 ? scaleU / 1024.0f : 1.0f;

                placements.Add(new PlacementData(nameIndex, uniqueId, worldX, worldY, worldZ,
                    rotX, rotY, rotZ, scale, flags, 0, 0, "M2"));
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
    public static List<PlacementData> ReadModf(string adtPath) => ReadModf(adtPath, out _);

    /// <summary>
    /// Reads all MODF (WMO) placements and reports the detected entry size.
    /// Supports both LK (64-byte) and early 0.6.0 (56-byte) layouts.
    /// </summary>
    public static List<PlacementData> ReadModf(string adtPath, out int entrySize)
    {
        var placements = new List<PlacementData>();
        entrySize = 0;

        if (!File.Exists(adtPath))
            return placements;

        try
        {
            var bytes = File.ReadAllBytes(adtPath);

            // Find MODF chunk (support forward and reversed on disk)
            var chunkStart = FindChunkEither(bytes, "MODF", "FDOM");
            if (chunkStart == -1)
            {
                Console.WriteLine($"[LkAdtReader] MODF not found in {adtPath}");
                return placements;
            }

            // Read chunk size
            var chunkSize = BitConverter.ToInt32(bytes, chunkStart + 4);
            var dataStart = chunkStart + 8;
            var available = Math.Max(0, Math.Min(chunkSize, bytes.Length - dataStart));

            if (available <= 0)
            {
                return placements;
            }

            entrySize = DetermineEntrySize(available, 64, 56);
            if (entrySize == 0)
            {
                Console.WriteLine($"[LkAdtReader] Unsupported MODF entry size in {adtPath} (chunkSize={chunkSize}).");
                return placements;
            }

            for (int offset = 0; offset + entrySize <= available; offset += entrySize)
            {
                var entryStart = dataStart + offset;
                if (entryStart + entrySize > bytes.Length)
                {
                    break;
                }

                int nameIndex = BitConverter.ToInt32(bytes, entryStart + 0);
                int uniqueId = BitConverter.ToInt32(bytes, entryStart + 4);

                // MODF coordinates are relative to map corner, convert to world coords
                const float TILESIZE = 533.33333f;
                const float MAP_HALF_SIZE = 32.0f * TILESIZE; // 17066.66656

                float rawX = BitConverter.ToSingle(bytes, entryStart + 8);   // X (east-west)
                float rawZ = BitConverter.ToSingle(bytes, entryStart + 12);  // Z (north-south)
                float rawY = BitConverter.ToSingle(bytes, entryStart + 16);  // Y (height)
                float rotX = BitConverter.ToSingle(bytes, entryStart + 20);
                float rotY = BitConverter.ToSingle(bytes, entryStart + 24);
                float rotZ = BitConverter.ToSingle(bytes, entryStart + 28);
                // Bounding box at 32..56 (ignored here)
                ushort flags = entrySize >= 58 ? BitConverter.ToUInt16(bytes, entryStart + 56) : (ushort)0;
                ushort doodadSet = entrySize >= 60 ? BitConverter.ToUInt16(bytes, entryStart + 58) : (ushort)0;
                ushort nameSet = entrySize >= 62 ? BitConverter.ToUInt16(bytes, entryStart + 60) : (ushort)0;
                ushort scaleU = entrySize >= 64 ? BitConverter.ToUInt16(bytes, entryStart + 62) : (ushort)1024;

                // Convert from map-corner-relative to world coordinates (centered at origin)
                // WoW convention: Y is up (height). The horizontal plane is X/Z.
                float worldX = MAP_HALF_SIZE - rawX;
                float worldY = MAP_HALF_SIZE - rawZ; // north-south
                float worldZ = rawY;                 // height

                float scale = scaleU > 0 ? scaleU / 1024.0f : 1.0f;

                placements.Add(new PlacementData(nameIndex, uniqueId, worldX, worldY, worldZ,
                    rotX, rotY, rotZ, scale, flags, doodadSet, nameSet, "WMO"));
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
            
            // Find MMDX chunk (support forward and reversed)
            var chunkStart = FindChunkEither(bytes, "MMDX", "XDMM");
            if (chunkStart == -1)
            {
                Console.WriteLine($"[LkAdtReader] MMDX not found in {adtPath}");
                return names;
            }

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

    private static int DetermineEntrySize(int availableBytes, params int[] candidates)
    {
        if (availableBytes <= 0)
            return 0;

        foreach (var size in candidates)
        {
            if (size > 0 && availableBytes % size == 0)
                return size;
        }

        foreach (var size in candidates)
        {
            if (size > 0 && availableBytes >= size)
                return size;
        }

        return 0;
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
            
            // Find MWMO chunk (support forward and reversed)
            var chunkStart = FindChunkEither(bytes, "MWMO", "OMWM");
            if (chunkStart == -1)
            {
                Console.WriteLine($"[LkAdtReader] MWMO not found in {adtPath}");
                return names;
            }

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

    private static int FindChunk(byte[] bytes, string fourCC)
    {
        var pattern = System.Text.Encoding.ASCII.GetBytes(fourCC);
        for (int i = 0; i <= bytes.Length - 4; i++)
        {
            if (bytes[i] == pattern[0] && bytes[i + 1] == pattern[1] && bytes[i + 2] == pattern[2] && bytes[i + 3] == pattern[3])
                return i;
        }
        return -1;
    }

    private static int FindChunkEither(byte[] bytes, string forwardFourCC, string reversedFourCC)
    {
        int pos = FindChunk(bytes, forwardFourCC);
        if (pos != -1) return pos;
        return FindChunk(bytes, reversedFourCC);
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
