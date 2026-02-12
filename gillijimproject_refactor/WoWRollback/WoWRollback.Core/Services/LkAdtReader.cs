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
    /// Reads MDDF (M2 filename table) from a LK ADT file.
    /// </summary>
    public static List<PlacementData> ReadMddf(string adtPath)
    {
        var placements = new List<PlacementData>();
        
        if (!File.Exists(adtPath))
            return placements;

        // Debug: Log to file for first ADT only
        bool isFirstFile = adtPath.Contains("_0_0_obj0.adt");
        StreamWriter? debugLog = null;
        if (isFirstFile)
        {
            debugLog = new StreamWriter("lk_adt_debug.txt", append: false);
            debugLog.WriteLine($"=== DEBUG: {Path.GetFileName(adtPath)} ===");
        }

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
                
                // MDDF coordinates are relative to map corner, convert to world coords
                // Formula from wowdev.wiki: posx = 32 * TILESIZE - mddf.position[0]
                const float TILESIZE = 533.33333f;
                const float MAP_HALF_SIZE = 32.0f * TILESIZE; // 17066.66656
                
                float rawX = BitConverter.ToSingle(bytes, entryStart + 8);   // X (east-west)
                // CATA 4.0+ CHANGED ORDER: offset 12 is now Y(height), offset 16 is now Z(north-south)!
                float rawY = BitConverter.ToSingle(bytes, entryStart + 12);  // Y (height) - SWAPPED in Cata!
                float rawZ = BitConverter.ToSingle(bytes, entryStart + 16);  // Z (north-south) - SWAPPED in Cata!
                float rotX = BitConverter.ToSingle(bytes, entryStart + 20);
                float rotY = BitConverter.ToSingle(bytes, entryStart + 24);
                float rotZ = BitConverter.ToSingle(bytes, entryStart + 28);
                ushort scaleU = BitConverter.ToUInt16(bytes, entryStart + 32);
                ushort flags = BitConverter.ToUInt16(bytes, entryStart + 34);
                
                // DEBUG: Log first 10 placements to file
                if (debugLog != null && placements.Count < 10)
                {
                    debugLog.WriteLine($"MDDF #{placements.Count}: rawX={rawX:F1}, rawZ={rawZ:F1}, rawY={rawY:F1} | uniqueId={uniqueId}");
                    Console.WriteLine($"[LkAdtReader] MDDF #{placements.Count}: rawX={rawX:F1}, rawZ={rawZ:F1}, rawY={rawY:F1} | uniqueId={uniqueId}");
                }
                
                // LK ADT coordinates are ALREADY in world space - no conversion needed!
                // WoW convention: Y is up (height), X/Z form the horizontal plane
                // Just assign directly (swap Y/Z for our 2D visualization which expects X/Y as horizontal)
                float worldX = rawX;  // east-west
                float worldY = rawZ;  // north-south (horizontal in our visualization)
                float worldZ = rawY;  // height (vertical in game, not used in 2D viz)
                
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
        finally
        {
            debugLog?.Close();
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
                
                // MODF coordinates are relative to map corner, convert to world coords
                const float TILESIZE = 533.33333f;
                const float MAP_HALF_SIZE = 32.0f * TILESIZE; // 17066.66656
                
                float rawX = BitConverter.ToSingle(bytes, entryStart + 8);   // X (east-west)
                // CATA 4.0+ CHANGED ORDER: offset 12 is now Y(height), offset 16 is now Z(north-south)!
                float rawY = BitConverter.ToSingle(bytes, entryStart + 12);  // Y (height) - SWAPPED in Cata!
                float rawZ = BitConverter.ToSingle(bytes, entryStart + 16);  // Z (north-south) - SWAPPED in Cata!
                float rotX = BitConverter.ToSingle(bytes, entryStart + 20);
                float rotY = BitConverter.ToSingle(bytes, entryStart + 24);
                float rotZ = BitConverter.ToSingle(bytes, entryStart + 28);
                // Bounding box at 32..56 (ignored here)
                ushort flags = BitConverter.ToUInt16(bytes, entryStart + 56);
                ushort doodadSet = BitConverter.ToUInt16(bytes, entryStart + 58);
                ushort nameSet = BitConverter.ToUInt16(bytes, entryStart + 60);
                ushort scaleU = BitConverter.ToUInt16(bytes, entryStart + 62);
                
                // DEBUG: Log first 3 placements to verify byte offsets
                if (placements.Count < 3)
                {
                    Console.WriteLine($"[LkAdtReader] MODF #{placements.Count}: rawX={rawX:F1}, rawZ={rawZ:F1}, rawY={rawY:F1} | uniqueId={uniqueId}");
                }
                
                // LK ADT coordinates are ALREADY in world space - no conversion needed!
                // WoW convention: Y is up (height), X/Z form the horizontal plane
                float worldX = rawX;  // east-west
                float worldY = rawZ;  // north-south (horizontal in our visualization)
                float worldZ = rawY;  // height (vertical in game, not used in 2D viz)
                
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

    public static int? ReadTileMajorAreaId(string adtPath)
    {
        try
        {
            if (!File.Exists(adtPath)) return null;
            var bytes = File.ReadAllBytes(adtPath);
            var mcin = FindChunk(bytes, "NICM");
            if (mcin < 0) return null;
            var size = BitConverter.ToInt32(bytes, mcin + 4);
            var dataStart = mcin + 8;
            var counts = new Dictionary<int, int>();
            const int entrySize = 16;
            int entries = Math.Min(256, size / entrySize);
            for (int i = 0; i < entries; i++)
            {
                int ofs = BitConverter.ToInt32(bytes, dataStart + i * entrySize + 0);
                if (ofs <= 0 || ofs + 8 + 0x80 >= bytes.Length) continue;
                int areaId = BitConverter.ToInt32(bytes, ofs + 8 + 0x7C);
                if (areaId <= 0) continue;
                counts[areaId] = counts.GetValueOrDefault(areaId) + 1;
            }
            if (counts.Count == 0) return null;
            int best = -1, bestC = -1;
            foreach (var kv in counts)
            {
                if (kv.Value > bestC) { best = kv.Key; bestC = kv.Value; }
            }
            return best;
        }
        catch
        {
            return null;
        }
    }
}
