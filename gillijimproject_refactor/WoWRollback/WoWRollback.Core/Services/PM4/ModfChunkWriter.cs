using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;

namespace WoWRollback.Core.Services.PM4;

/// <summary>
/// Writes MWMO and MODF chunks for ADT _obj files from reconstructed placement data.
/// </summary>
public sealed class ModfChunkWriter
{
    /// <summary>
    /// MODF entry structure (64 bytes) for ADT files.
    /// </summary>
    public struct ModfEntry
    {
        public uint NameId;           // Index into MWMO string block
        public uint UniqueId;         // Unique identifier for this placement
        public Vector3 Position;      // World position
        public Vector3 Rotation;      // Rotation in degrees (X, Y, Z)
        public Vector3 BoundsMin;     // Bounding box min (world space)
        public Vector3 BoundsMax;     // Bounding box max (world space)
        public ushort Flags;          // Placement flags
        public ushort DoodadSet;      // Doodad set index
        public ushort NameSet;        // Name set index
        public ushort Scale;          // Scale (1024 = 1.0) - always 1024 for WMOs
    }

    /// <summary>
    /// Write MWMO chunk (WMO path strings, null-terminated).
    /// </summary>
    public byte[] WriteMwmoChunk(List<string> wmoPaths)
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        // Write chunk header
        bw.Write(Encoding.ASCII.GetBytes("MWMO"));
        
        // Calculate data size
        int dataSize = wmoPaths.Sum(p => Encoding.ASCII.GetByteCount(p) + 1); // +1 for null terminator
        bw.Write(dataSize);

        // Write null-terminated strings
        foreach (var path in wmoPaths)
        {
            bw.Write(Encoding.ASCII.GetBytes(path));
            bw.Write((byte)0); // Null terminator
        }

        return ms.ToArray();
    }

    /// <summary>
    /// Write MODF chunk (WMO placement entries, 64 bytes each).
    /// </summary>
    public byte[] WriteModfChunk(List<ModfEntry> entries)
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        // Write chunk header
        bw.Write(Encoding.ASCII.GetBytes("MODF"));
        bw.Write(entries.Count * 64); // Each entry is 64 bytes

        foreach (var entry in entries)
        {
            bw.Write(entry.NameId);
            bw.Write(entry.UniqueId);
            
            // Position (3 floats)
            bw.Write(entry.Position.X);
            bw.Write(entry.Position.Y);
            bw.Write(entry.Position.Z);
            
            // Rotation (3 floats) - in degrees
            bw.Write(entry.Rotation.X);
            bw.Write(entry.Rotation.Y);
            bw.Write(entry.Rotation.Z);
            
            // Bounding box min (3 floats)
            bw.Write(entry.BoundsMin.X);
            bw.Write(entry.BoundsMin.Y);
            bw.Write(entry.BoundsMin.Z);
            
            // Bounding box max (3 floats)
            bw.Write(entry.BoundsMax.X);
            bw.Write(entry.BoundsMax.Y);
            bw.Write(entry.BoundsMax.Z);
            
            // Flags, DoodadSet, NameSet, Scale
            bw.Write(entry.Flags);
            bw.Write(entry.DoodadSet);
            bw.Write(entry.NameSet);
            bw.Write(entry.Scale);
        }

        return ms.ToArray();
    }

    /// <summary>
    /// Build MWMO string table and get offsets for each path.
    /// </summary>
    public (byte[] mwmoData, Dictionary<string, uint> pathToOffset) BuildMwmoTable(List<string> wmoPaths)
    {
        var pathToOffset = new Dictionary<string, uint>();
        using var ms = new MemoryStream();

        uint offset = 0;
        foreach (var path in wmoPaths.Distinct())
        {
            pathToOffset[path] = offset;
            var bytes = Encoding.ASCII.GetBytes(path);
            ms.Write(bytes, 0, bytes.Length);
            ms.WriteByte(0); // Null terminator
            offset += (uint)(bytes.Length + 1);
        }

        return (ms.ToArray(), pathToOffset);
    }

    /// <summary>
    /// Convert reconstruction results to binary MODF data.
    /// </summary>
    public (byte[] mwmoChunk, byte[] modfChunk) ConvertToBinary(Pm4ModfReconstructor.ReconstructionResult result)
    {
        // Build MWMO string table
        var (mwmoData, pathToOffset) = BuildMwmoTable(result.WmoNames);

        // Build MODF entries
        var modfEntries = new List<ModfEntry>();
        
        foreach (var entry in result.ModfEntries)
        {
            modfEntries.Add(new ModfEntry
            {
                NameId = pathToOffset[entry.WmoPath],
                UniqueId = entry.UniqueId,
                Position = entry.Position,
                Rotation = entry.Rotation,
                BoundsMin = entry.BoundsMin,
                BoundsMax = entry.BoundsMax,
                Flags = entry.Flags,
                DoodadSet = entry.DoodadSet,
                NameSet = entry.NameSet,
                Scale = 1024 // Always 1.0 for WMOs
            });
        }

        // Write chunks
        var mwmoChunk = WriteMwmoChunkRaw(mwmoData);
        var modfChunk = WriteModfChunk(modfEntries);

        return (mwmoChunk, modfChunk);
    }

    private byte[] WriteMwmoChunkRaw(byte[] data)
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);
        bw.Write(Encoding.ASCII.GetBytes("MWMO"));
        bw.Write(data.Length);
        bw.Write(data);
        return ms.ToArray();
    }

    /// <summary>
    /// Filter MODF entries by tile coordinates.
    /// </summary>
    public List<Pm4ModfReconstructor.ModfEntry> FilterByTile(
        List<Pm4ModfReconstructor.ModfEntry> entries, 
        int tileX, int tileY,
        float tileSize = 533.33333f)
    {
        // Calculate tile bounds in world coordinates
        // WoW coordinate system: tile (32,32) is at origin
        float worldX = (32 - tileX) * tileSize;
        float worldY = (32 - tileY) * tileSize;
        
        float minX = worldX - tileSize;
        float maxX = worldX;
        float minY = worldY - tileSize;
        float maxY = worldY;

        return entries.Where(e => 
            e.Position.X >= minX && e.Position.X < maxX &&
            e.Position.Y >= minY && e.Position.Y < maxY
        ).ToList();
    }

    // ADT coordinate constants
    private const float WorldExtent = 17066.666666666668f;
    private const float TileSize = 533.3333333333334f;

    /// <summary>
    /// Convert PM4/server coordinates to ADT world coordinates.
    /// PM4 uses a different coordinate system than ADT world coordinates.
    /// </summary>
    public static (float worldX, float worldY) ServerToWorldCoords(float serverX, float serverY)
    {
        // Transform: worldX = serverX + extent, worldY = extent - serverY
        return (serverX + WorldExtent, WorldExtent - serverY);
    }

    /// <summary>
    /// Convert ADT world coordinates to tile indices.
    /// </summary>
    public static (int tileX, int tileY) WorldToTile(float worldX, float worldY)
    {
        int tileX = (int)Math.Floor(32 - worldX / TileSize);
        int tileY = (int)Math.Floor(32 - worldY / TileSize);
        return (tileX, tileY);
    }

    /// <summary>
    /// Convert PM4/server coordinates directly to tile indices.
    /// </summary>
    public static (int tileX, int tileY) ServerToTile(float serverX, float serverY)
    {
        var (worldX, worldY) = ServerToWorldCoords(serverX, serverY);
        return WorldToTile(worldX, worldY);
    }

    /// <summary>
    /// Group MODF entries by tile.
    /// Converts PM4 server coordinates to ADT tile coordinates.
    /// </summary>
    public Dictionary<(int tileX, int tileY), List<Pm4ModfReconstructor.ModfEntry>> GroupByTile(
        List<Pm4ModfReconstructor.ModfEntry> entries)
    {
        var result = new Dictionary<(int, int), List<Pm4ModfReconstructor.ModfEntry>>();

        foreach (var entry in entries)
        {
            // Convert PM4 server coordinates to tile coordinates
            var (tileX, tileY) = ServerToTile(entry.Position.X, entry.Position.Y);

            // Clamp to valid tile range [0, 63]
            tileX = Math.Clamp(tileX, 0, 63);
            tileY = Math.Clamp(tileY, 0, 63);

            var key = (tileX, tileY);
            if (!result.ContainsKey(key))
                result[key] = new List<Pm4ModfReconstructor.ModfEntry>();
            
            result[key].Add(entry);
        }

        return result;
    }

    /// <summary>
    /// Export per-tile MODF data for ADT reconstruction.
    /// </summary>
    public void ExportPerTile(
        Pm4ModfReconstructor.ReconstructionResult result,
        string outputDir)
    {
        Directory.CreateDirectory(outputDir);

        // Group entries by tile
        var byTile = GroupByTile(result.ModfEntries);

        Console.WriteLine($"[INFO] Exporting MODF data for {byTile.Count} tiles...");

        foreach (var (tile, entries) in byTile)
        {
            var tileDir = Path.Combine(outputDir, $"tile_{tile.tileX}_{tile.tileY}");
            Directory.CreateDirectory(tileDir);

            // Build per-tile MWMO table (only WMOs used in this tile)
            var usedWmos = entries.Select(e => e.WmoPath).Distinct().ToList();
            var (mwmoData, pathToOffset) = BuildMwmoTable(usedWmos);

            // Build MODF entries with corrected NameId for this tile's MWMO table
            var modfEntries = new List<ModfEntry>();
            uint localUniqueId = 1;

            foreach (var entry in entries)
            {
                modfEntries.Add(new ModfEntry
                {
                    NameId = pathToOffset[entry.WmoPath],
                    UniqueId = localUniqueId++,
                    Position = entry.Position,
                    Rotation = entry.Rotation,
                    BoundsMin = entry.BoundsMin,
                    BoundsMax = entry.BoundsMax,
                    Flags = entry.Flags,
                    DoodadSet = entry.DoodadSet,
                    NameSet = entry.NameSet,
                    Scale = 1024
                });
            }

            // Write MWMO chunk
            var mwmoChunk = WriteMwmoChunkRaw(mwmoData);
            File.WriteAllBytes(Path.Combine(tileDir, "MWMO.bin"), mwmoChunk);

            // Write MODF chunk
            var modfChunk = WriteModfChunk(modfEntries);
            File.WriteAllBytes(Path.Combine(tileDir, "MODF.bin"), modfChunk);

            // Write human-readable summary
            using var sw = new StreamWriter(Path.Combine(tileDir, "placements.txt"));
            sw.WriteLine($"# Tile {tile.tileX}_{tile.tileY} WMO Placements");
            sw.WriteLine($"# Total: {entries.Count} placements, {usedWmos.Count} unique WMOs");
            sw.WriteLine();
            
            foreach (var entry in entries)
            {
                sw.WriteLine($"{entry.Ck24}: {Path.GetFileName(entry.WmoPath)}");
                sw.WriteLine($"  Position: ({entry.Position.X:F2}, {entry.Position.Y:F2}, {entry.Position.Z:F2})");
                sw.WriteLine($"  Rotation: ({entry.Rotation.X:F2}°, {entry.Rotation.Y:F2}°, {entry.Rotation.Z:F2}°)");
                sw.WriteLine($"  Confidence: {entry.MatchConfidence:P1}");
                sw.WriteLine();
            }

            Console.WriteLine($"  Tile {tile.tileX}_{tile.tileY}: {entries.Count} placements");
        }

        // Write global summary
        using var summaryWriter = new StreamWriter(Path.Combine(outputDir, "summary.txt"));
        summaryWriter.WriteLine($"# MODF Reconstruction Summary");
        summaryWriter.WriteLine($"# Total tiles: {byTile.Count}");
        summaryWriter.WriteLine($"# Total placements: {result.ModfEntries.Count}");
        summaryWriter.WriteLine($"# Unique WMOs: {result.WmoNames.Count}");
        summaryWriter.WriteLine();
        
        foreach (var (tile, entries) in byTile.OrderBy(x => x.Key.tileX).ThenBy(x => x.Key.tileY))
        {
            summaryWriter.WriteLine($"Tile {tile.tileX}_{tile.tileY}: {entries.Count} placements");
        }

        Console.WriteLine($"[INFO] Exported to {outputDir}");
    }
}
