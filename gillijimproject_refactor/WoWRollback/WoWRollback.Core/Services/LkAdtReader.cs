using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

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

    public record McnkData(
        int ChunkX,
        int ChunkY,
        int AreaId,
        uint Flags,
        float PositionX,
        float PositionY,
        float PositionZ,
        bool HasMcsh,
        bool HasMccv
    );

    /// <summary>
    /// Reads MCNK chunk headers from a LK ADT file.
    /// Returns metadata needed for terrain overlays (area IDs, flags, positions).
    /// </summary>
    public static List<McnkData> ReadMcnkChunks(string adtPath)
    {
        var chunks = new List<McnkData>();

        if (!File.Exists(adtPath))
            return chunks;

        try
        {
            var bytes = File.ReadAllBytes(adtPath);

            var searchPos = 0;
            while (searchPos < bytes.Length)
            {
                var chunkStart = FindChunk(bytes, "KNCM", searchPos);
                if (chunkStart == -1)
                    break;

                var chunkSize = BitConverter.ToInt32(bytes, chunkStart + 4);
                var headerStart = chunkStart + 8;

                uint flags = BitConverter.ToUInt32(bytes, headerStart + 0x00);
                int indexX = (int)BitConverter.ToUInt32(bytes, headerStart + 0x04);
                int indexY = (int)BitConverter.ToUInt32(bytes, headerStart + 0x08);
                int areaId = BitConverter.ToInt32(bytes, headerStart + 0x34);

                float posX = BitConverter.ToSingle(bytes, headerStart + 0x40);
                float posY = BitConverter.ToSingle(bytes, headerStart + 0x44);
                float posZ = BitConverter.ToSingle(bytes, headerStart + 0x48);

                bool hasMcsh = HasSubChunk(bytes, chunkStart, chunkSize, "HSCM");
                bool hasMccv = HasSubChunk(bytes, chunkStart, chunkSize, "VCCM");

                chunks.Add(new McnkData(
                    ChunkX: indexX,
                    ChunkY: indexY,
                    AreaId: areaId,
                    Flags: flags,
                    PositionX: posX,
                    PositionY: posY,
                    PositionZ: posZ,
                    HasMcsh: hasMcsh,
                    HasMccv: hasMccv
                ));

                searchPos = chunkStart + 8 + chunkSize;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[LkAdtReader] Error reading MCNK from {adtPath}: {ex.Message}");
        }

        return chunks;
    }

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
                ushort scaleU = BitConverter.ToUInt16(bytes, entryStart + 32);
                ushort flags = BitConverter.ToUInt16(bytes, entryStart + 34);
                
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
                float rawZ = BitConverter.ToSingle(bytes, entryStart + 12);  // Z (north-south)
                float rawY = BitConverter.ToSingle(bytes, entryStart + 16);  // Y (height)
                float rotX = BitConverter.ToSingle(bytes, entryStart + 20);
                float rotY = BitConverter.ToSingle(bytes, entryStart + 24);
                float rotZ = BitConverter.ToSingle(bytes, entryStart + 28);
                // Bounding box at 32..56 (ignored here)
                ushort flags = BitConverter.ToUInt16(bytes, entryStart + 56);
                ushort doodadSet = BitConverter.ToUInt16(bytes, entryStart + 58);
                ushort nameSet = BitConverter.ToUInt16(bytes, entryStart + 60);
                ushort scaleU = BitConverter.ToUInt16(bytes, entryStart + 62);
                
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

    private static bool HasSubChunk(byte[] bytes, int chunkStart, int chunkSize, string reversedFourCC)
    {
        var end = chunkStart + 8 + chunkSize;
        var pattern = Encoding.ASCII.GetBytes(reversedFourCC);

        for (int i = chunkStart + 8; i <= end - 4 && i <= bytes.Length - 4; i++)
        {
            if (bytes[i] == pattern[0] &&
                bytes[i + 1] == pattern[1] &&
                bytes[i + 2] == pattern[2] &&
                bytes[i + 3] == pattern[3])
            {
                return true;
            }
        }

        return false;
    }

    private static int FindChunk(byte[] bytes, string reversedFourCC, int startIndex = 0)
    {
        var pattern = Encoding.ASCII.GetBytes(reversedFourCC);

        for (int i = startIndex; i < bytes.Length - 4; i++)
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
