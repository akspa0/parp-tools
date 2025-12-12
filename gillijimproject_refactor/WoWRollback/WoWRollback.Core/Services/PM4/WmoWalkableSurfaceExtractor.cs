using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;

namespace WoWRollback.Core.Services.PM4;

/// <summary>
/// Extracts walkable surfaces from WMO files for comparison with PM4 pathfinding data.
/// Based on MOPY flags - Flag_0x8_IsCollisionFace indicates walkable collision surfaces.
/// The PM4 navmesh data corresponds to these walkable surfaces, not the full render geometry.
/// </summary>
public sealed class WmoWalkableSurfaceExtractor
{
    /// <summary>
    /// A triangle from the WMO with its vertices and flags.
    /// </summary>
    public record WmoTriangle(
        Vector3 V0,
        Vector3 V1,
        Vector3 V2,
        ushort MaterialId,
        ushort Flags,
        bool IsCollisionFace,
        bool IsWalkable);

    /// <summary>
    /// Extracted walkable surface data from a WMO.
    /// </summary>
    public class WmoWalkableData
    {
        public string WmoPath { get; set; } = "";
        public List<WmoTriangle> AllTriangles { get; set; } = new();
        public List<WmoTriangle> WalkableTriangles { get; set; } = new();
        public List<Vector3> WalkableVertices { get; set; } = new();
        public Vector3 BoundsMin { get; set; }
        public Vector3 BoundsMax { get; set; }
        public int GroupCount { get; set; }
        public Dictionary<int, List<WmoTriangle>> TrianglesByGroup { get; set; } = new();
    }

    // MOPY flags from WoW format
    private const ushort MOPY_FLAG_NO_CAM_COLLIDE = 0x02;
    private const ushort MOPY_FLAG_NO_COLLIDE = 0x04;
    private const ushort MOPY_FLAG_IS_COLLISION_FACE = 0x08;
    private const ushort MOPY_FLAG_RENDER = 0x20;

    /// <summary>
    /// Check if a face is a collision face (walkable).
    /// </summary>
    private static bool IsCollisionFace(ushort flags) => (flags & MOPY_FLAG_IS_COLLISION_FACE) != 0;

    /// <summary>
    /// Check if a face is a render face (visible but may not be walkable).
    /// </summary>
    private static bool IsRenderFace(ushort flags) => (flags & 0x24) == 0x20;

    /// <summary>
    /// Check if a face is collidable (either collision or render face).
    /// </summary>
    private static bool IsCollidable(ushort flags) => IsCollisionFace(flags) || IsRenderFace(flags);

    /// <summary>
    /// Check if a triangle is walkable based on its normal (upward-facing).
    /// Walkable surfaces have normals pointing mostly upward (Y > threshold in WoW coords).
    /// </summary>
    private static bool IsWalkableByNormal(Vector3 v0, Vector3 v1, Vector3 v2, float threshold = 0.5f)
    {
        // Calculate face normal
        var edge1 = v1 - v0;
        var edge2 = v2 - v0;
        var normal = Vector3.Normalize(Vector3.Cross(edge1, edge2));
        
        // In WoW coordinates, Y is up. Walkable surfaces have normal.Y > threshold
        // (pointing upward, not vertical walls or ceilings)
        return normal.Y > threshold;
    }

    /// <summary>
    /// Extract walkable surfaces from WMO data bytes (v17+ format).
    /// </summary>
    public static WmoWalkableData ExtractFromBytes(byte[] rootData, string wmoPath, Func<string, byte[]?>? groupLoader = null)
    {
        var result = new WmoWalkableData { WmoPath = wmoPath };

        try
        {
            int groupCount = ParseMohdGroupCount(rootData);
            result.GroupCount = groupCount;

            Console.WriteLine($"[INFO] WMO has {groupCount} groups");

            // Load each group
            for (int i = 0; i < groupCount; i++)
            {
                // WMO group files are named: root_000.wmo, root_001.wmo, etc.
                var ext = Path.GetExtension(wmoPath);
                var basePath = wmoPath.Substring(0, wmoPath.Length - ext.Length);
                var groupPath = $"{basePath}_{i:D3}{ext}";
                
                byte[]? groupData = groupLoader?.Invoke(groupPath);
                
                if (groupData == null || groupData.Length == 0)
                    continue;

                ExtractGroupWalkableSurfaces(groupData, result, i);
            }

            // Compute bounds
            if (result.WalkableVertices.Count > 0)
            {
                result.BoundsMin = new Vector3(
                    result.WalkableVertices.Min(v => v.X),
                    result.WalkableVertices.Min(v => v.Y),
                    result.WalkableVertices.Min(v => v.Z));
                result.BoundsMax = new Vector3(
                    result.WalkableVertices.Max(v => v.X),
                    result.WalkableVertices.Max(v => v.Y),
                    result.WalkableVertices.Max(v => v.Z));
            }

            Console.WriteLine($"[INFO] Extracted {result.WalkableTriangles.Count} walkable triangles from {result.AllTriangles.Count} total");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ERROR] Failed to extract WMO: {ex.Message}");
        }

        return result;
    }

    /// <summary>
    /// Extract walkable surfaces from a WMO file (v17+ split format).
    /// </summary>
    public static WmoWalkableData ExtractFromWmoV17(string wmoRootPath)
    {
        var result = new WmoWalkableData { WmoPath = wmoRootPath };

        if (!File.Exists(wmoRootPath))
        {
            Console.WriteLine($"[ERROR] WMO root not found: {wmoRootPath}");
            return result;
        }

        try
        {
            // Parse root WMO to get group count
            var rootData = File.ReadAllBytes(wmoRootPath);
            int groupCount = ParseMohdGroupCount(rootData);
            result.GroupCount = groupCount;

            Console.WriteLine($"[INFO] WMO has {groupCount} groups");

            // Load each group file
            var basePath = Path.GetDirectoryName(wmoRootPath) ?? ".";
            var baseName = Path.GetFileNameWithoutExtension(wmoRootPath);

            for (int i = 0; i < groupCount; i++)
            {
                var groupPath = Path.Combine(basePath, $"{baseName}_{i:D3}.wmo");
                if (!File.Exists(groupPath))
                {
                    Console.WriteLine($"[WARN] Group file not found: {groupPath}");
                    continue;
                }

                var groupData = File.ReadAllBytes(groupPath);
                ExtractGroupWalkableSurfaces(groupData, result, i);
            }

            // Compute bounds
            if (result.WalkableVertices.Count > 0)
            {
                result.BoundsMin = new Vector3(
                    result.WalkableVertices.Min(v => v.X),
                    result.WalkableVertices.Min(v => v.Y),
                    result.WalkableVertices.Min(v => v.Z));
                result.BoundsMax = new Vector3(
                    result.WalkableVertices.Max(v => v.X),
                    result.WalkableVertices.Max(v => v.Y),
                    result.WalkableVertices.Max(v => v.Z));
            }

            Console.WriteLine($"[INFO] Extracted {result.WalkableTriangles.Count} walkable triangles from {result.AllTriangles.Count} total");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ERROR] Failed to extract WMO: {ex.Message}");
        }

        return result;
    }

    /// <summary>
    /// Extract walkable surfaces from a monolithic WMO v14 file.
    /// </summary>
    public static WmoWalkableData ExtractFromWmoV14(string wmoPath)
    {
        var result = new WmoWalkableData { WmoPath = wmoPath };

        if (!File.Exists(wmoPath))
        {
            Console.WriteLine($"[ERROR] WMO not found: {wmoPath}");
            return result;
        }

        try
        {
            var data = File.ReadAllBytes(wmoPath);
            
            // V14 is monolithic - all groups in one file inside MOMO container
            ExtractV14MonolithicWmo(data, result);

            // Compute bounds
            if (result.WalkableVertices.Count > 0)
            {
                result.BoundsMin = new Vector3(
                    result.WalkableVertices.Min(v => v.X),
                    result.WalkableVertices.Min(v => v.Y),
                    result.WalkableVertices.Min(v => v.Z));
                result.BoundsMax = new Vector3(
                    result.WalkableVertices.Max(v => v.X),
                    result.WalkableVertices.Max(v => v.Y),
                    result.WalkableVertices.Max(v => v.Z));
            }

            Console.WriteLine($"[INFO] Extracted {result.WalkableTriangles.Count} walkable triangles from {result.AllTriangles.Count} total");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ERROR] Failed to extract WMO v14: {ex.Message}");
        }

        return result;
    }

    /// <summary>
    /// Parse MOHD chunk to get group count.
    /// </summary>
    private static int ParseMohdGroupCount(byte[] data)
    {
        // Scan for MOHD chunk
        for (int i = 0; i < data.Length - 8; i++)
        {
            if (data[i] == 'M' && data[i + 1] == 'O' && data[i + 2] == 'H' && data[i + 3] == 'D')
            {
                // MOHD found - group count is at offset 4 (after nMaterials)
                int chunkSize = BitConverter.ToInt32(data, i + 4);
                if (i + 8 + 8 <= data.Length)
                {
                    return BitConverter.ToInt32(data, i + 8 + 4); // nGroups at offset 4 in MOHD
                }
            }
            // Also check reversed (little-endian storage)
            if (data[i] == 'D' && data[i + 1] == 'H' && data[i + 2] == 'O' && data[i + 3] == 'M')
            {
                int chunkSize = BitConverter.ToInt32(data, i + 4);
                if (i + 8 + 8 <= data.Length)
                {
                    return BitConverter.ToInt32(data, i + 8 + 4);
                }
            }
        }
        return 0;
    }

    /// <summary>
    /// Extract walkable surfaces from a WMO group file.
    /// </summary>
    private static void ExtractGroupWalkableSurfaces(byte[] data, WmoWalkableData result, int groupIndex)
    {
        // Find MOVT (vertices), MOVI (indices), MOPY (material/flags per face)
        List<Vector3> vertices = new();
        List<ushort> indices = new();
        List<(ushort flags, ushort materialId)> faceInfo = new();

        // WMO group files have structure: MVER, MOGP (container with all sub-chunks)
        // MOGP header is 68 bytes, then sub-chunks follow inside MOGP
        
        int pos = 0;
        while (pos < data.Length - 8)
        {
            string chunkId = GetChunkId(data, pos);
            int chunkSize = BitConverter.ToInt32(data, pos + 4);
            
            if (chunkSize < 0 || chunkSize > data.Length - pos - 8)
                break;

            // MOGP is a container chunk - parse its contents
            if (chunkId == "MOGP" || chunkId == "PGOM")
            {
                // MOGP header is 68 bytes, sub-chunks start after that
                const int MOGP_HEADER_SIZE = 68;
                int subChunkStart = pos + 8 + MOGP_HEADER_SIZE;
                int subChunkEnd = pos + 8 + chunkSize;
                
                int subPos = subChunkStart;
                while (subPos < subChunkEnd - 8)
                {
                    string subChunkId = GetChunkId(data, subPos);
                    int subChunkSize = BitConverter.ToInt32(data, subPos + 4);
                    
                    if (subChunkSize < 0 || subChunkSize > subChunkEnd - subPos - 8)
                        break;

                    switch (subChunkId)
                    {
                        case "MOVT":
                        case "TVOM":
                            vertices = ParseMovt(data, subPos + 8, subChunkSize);
                            break;
                        case "MOVI":
                        case "IVOM":
                            indices = ParseMovi(data, subPos + 8, subChunkSize);
                            break;
                        case "MOPY":
                        case "YPOM":
                            faceInfo = ParseMopy(data, subPos + 8, subChunkSize);
                            break;
                    }

                    subPos += 8 + subChunkSize;
                }
            }

            pos += 8 + chunkSize;
        }

        if (vertices.Count == 0 || indices.Count == 0)
        {
            Console.WriteLine($"[WARN] Group {groupIndex}: No geometry found");
            return;
        }

        if (!result.TrianglesByGroup.TryGetValue(groupIndex, out var groupTris))
        {
            groupTris = new List<WmoTriangle>();
            result.TrianglesByGroup[groupIndex] = groupTris;
        }

        // Build triangles
        int triCount = indices.Count / 3;
        int walkableCount = 0;

        for (int i = 0; i < triCount; i++)
        {
            int i0 = indices[i * 3];
            int i1 = indices[i * 3 + 1];
            int i2 = indices[i * 3 + 2];

            if (i0 >= vertices.Count || i1 >= vertices.Count || i2 >= vertices.Count)
                continue;

            var v0 = vertices[i0];
            var v1 = vertices[i1];
            var v2 = vertices[i2];

            ushort flags = 0;
            ushort materialId = 0;
            if (i < faceInfo.Count)
            {
                flags = faceInfo[i].flags;
                materialId = faceInfo[i].materialId;
            }

            bool isCollision = IsCollisionFace(flags);
            // Walkable logic preserved for reference but we extract everything now
            bool isWalkable = isCollision && IsWalkableByNormal(v0, v1, v2);

            var tri = new WmoTriangle(v0, v1, v2, materialId, flags, isCollision, isWalkable);
            result.AllTriangles.Add(tri);

            // User Request: Extract whole WMO data, not just walkable
            // We treat any collision or render face as part of the shape
            if (isCollision || IsRenderFace(flags)) 
            {
                result.WalkableTriangles.Add(tri); // Reusing this list for "Matching Candidates"
                result.WalkableVertices.Add(v0);
                result.WalkableVertices.Add(v1);
                result.WalkableVertices.Add(v2);
                walkableCount++;
                groupTris.Add(tri);
            }
        }
    }

    /// <summary>
    /// Extract from V14 monolithic WMO (all groups in MOMO container).
    /// </summary>
    private static void ExtractV14MonolithicWmo(byte[] data, WmoWalkableData result)
    {
        // V14 has MOMO container with embedded groups
        // For now, do a simple scan for MOVT/MOVI/MOPY chunks
        List<Vector3> vertices = new();
        List<ushort> indices = new();
        List<(ushort flags, ushort materialId)> faceInfo = new();

        if (!result.TrianglesByGroup.TryGetValue(0, out var groupTris))
        {
            groupTris = new List<WmoTriangle>();
            result.TrianglesByGroup[0] = groupTris;
        }

        int pos = 0;
        while (pos < data.Length - 8)
        {
            string chunkId = GetChunkId(data, pos);
            int chunkSize = BitConverter.ToInt32(data, pos + 4);

            switch (chunkId)
            {
                case "MOVT":
                case "TVOM":
                    var newVerts = ParseMovt(data, pos + 8, chunkSize);
                    vertices.AddRange(newVerts);
                    break;
                case "MOVI":
                case "IVOM":
                    var newIndices = ParseMovi(data, pos + 8, chunkSize);
                    indices.AddRange(newIndices);
                    break;
                case "MOPY":
                case "YPOM":
                    var newFaceInfo = ParseMopy(data, pos + 8, chunkSize);
                    faceInfo.AddRange(newFaceInfo);
                    break;
            }

            pos += 8 + chunkSize;
            if (chunkSize <= 0) pos++; // Safety
        }

        // Build triangles (simplified - may need group-aware processing)
        int triCount = indices.Count / 3;
        for (int i = 0; i < triCount && i < faceInfo.Count; i++)
        {
            int i0 = indices[i * 3];
            int i1 = indices[i * 3 + 1];
            int i2 = indices[i * 3 + 2];

            if (i0 >= vertices.Count || i1 >= vertices.Count || i2 >= vertices.Count)
                continue;

            var v0 = vertices[i0];
            var v1 = vertices[i1];
            var v2 = vertices[i2];

            var (flags, materialId) = faceInfo[i];
            bool isCollision = IsCollisionFace(flags);
            bool isWalkable = isCollision && IsWalkableByNormal(v0, v1, v2);

            var tri = new WmoTriangle(v0, v1, v2, materialId, flags, isCollision, isWalkable);
            result.AllTriangles.Add(tri);

            if (isWalkable)
            {
                result.WalkableTriangles.Add(tri);
                result.WalkableVertices.Add(v0);
                result.WalkableVertices.Add(v1);
                result.WalkableVertices.Add(v2);
                groupTris.Add(tri);
            }
        }
    }

    private static string GetChunkId(byte[] data, int pos)
    {
        if (pos + 4 > data.Length) return "";
        // Try forward
        char c0 = (char)data[pos];
        char c1 = (char)data[pos + 1];
        char c2 = (char)data[pos + 2];
        char c3 = (char)data[pos + 3];
        return new string(new[] { c0, c1, c2, c3 });
    }

    private static List<Vector3> ParseMovt(byte[] data, int offset, int size)
    {
        var vertices = new List<Vector3>();
        int count = size / 12; // 3 floats per vertex
        for (int i = 0; i < count; i++)
        {
            int o = offset + i * 12;
            if (o + 12 > data.Length) break;
            float x = BitConverter.ToSingle(data, o);
            float y = BitConverter.ToSingle(data, o + 4);
            float z = BitConverter.ToSingle(data, o + 8);
            vertices.Add(new Vector3(x, y, z));
        }
        return vertices;
    }

    private static List<ushort> ParseMovi(byte[] data, int offset, int size)
    {
        var indices = new List<ushort>();
        int count = size / 2; // ushort per index
        for (int i = 0; i < count; i++)
        {
            int o = offset + i * 2;
            if (o + 2 > data.Length) break;
            indices.Add(BitConverter.ToUInt16(data, o));
        }
        return indices;
    }

    private static List<(ushort flags, ushort materialId)> ParseMopy(byte[] data, int offset, int size)
    {
        var faceInfo = new List<(ushort, ushort)>();
        // MOPY is 2 bytes per face: 1 byte flags (stored as ushort with high byte), 1 byte materialId
        // Actually it's: flags (1 byte), materialId (1 byte) = 2 bytes per face
        // But in some versions it's: flags (2 bytes), materialId (2 bytes) = 4 bytes
        // Let's try 2 bytes first (v17 format)
        int entrySize = 2;
        int count = size / entrySize;
        
        for (int i = 0; i < count; i++)
        {
            int o = offset + i * entrySize;
            if (o + entrySize > data.Length) break;
            byte flags = data[o];
            byte matId = data[o + 1];
            faceInfo.Add(((ushort)flags, (ushort)matId));
        }
        return faceInfo;
    }

    /// <summary>
    /// Export walkable surfaces to OBJ file for visualization.
    /// </summary>
    public static void ExportToObj(WmoWalkableData data, string outputPath)
    {
        using var sw = new StreamWriter(outputPath);
        sw.WriteLine("# WMO Collision/Portal Geometry");
        sw.WriteLine($"# Source: {data.WmoPath}");
        sw.WriteLine($"# Total triangles: {data.AllTriangles.Count}");
        sw.WriteLine($"# Collision triangles: {data.WalkableTriangles.Count}");
        sw.WriteLine();

        // Write vertices
        foreach (var tri in data.WalkableTriangles)
        {
            sw.WriteLine($"v {tri.V0.X} {tri.V0.Y} {tri.V0.Z}");
            sw.WriteLine($"v {tri.V1.X} {tri.V1.Y} {tri.V1.Z}");
            sw.WriteLine($"v {tri.V2.X} {tri.V2.Y} {tri.V2.Z}");
        }

        sw.WriteLine();

        // Write faces (1-indexed)
        for (int i = 0; i < data.WalkableTriangles.Count; i++)
        {
            int baseIdx = i * 3 + 1;
            sw.WriteLine($"f {baseIdx} {baseIdx + 1} {baseIdx + 2}");
        }

        Console.WriteLine($"[INFO] Exported collision geometry to {outputPath}");
    }

    /// <summary>
    /// Export ALL triangles (full render geometry) to OBJ file.
    /// </summary>
    public static void ExportAllToObj(WmoWalkableData data, string outputPath)
    {
        using var sw = new StreamWriter(outputPath);
        sw.WriteLine("# WMO Full Render Geometry");
        sw.WriteLine($"# Source: {data.WmoPath}");
        sw.WriteLine($"# Total triangles: {data.AllTriangles.Count}");
        sw.WriteLine();

        // Write vertices
        foreach (var tri in data.AllTriangles)
        {
            sw.WriteLine($"v {tri.V0.X} {tri.V0.Y} {tri.V0.Z}");
            sw.WriteLine($"v {tri.V1.X} {tri.V1.Y} {tri.V1.Z}");
            sw.WriteLine($"v {tri.V2.X} {tri.V2.Y} {tri.V2.Z}");
        }

        sw.WriteLine();

        // Write faces (1-indexed)
        for (int i = 0; i < data.AllTriangles.Count; i++)
        {
            int baseIdx = i * 3 + 1;
            sw.WriteLine($"f {baseIdx} {baseIdx + 1} {baseIdx + 2}");
        }

        Console.WriteLine($"[INFO] Exported full geometry ({data.AllTriangles.Count} triangles) to {outputPath}");
    }

    /// <summary>
    /// Export upward-facing render faces (potential walkable floors) to OBJ.
    /// </summary>
    public static void ExportWalkableFloorsToObj(WmoWalkableData data, string outputPath, float normalThreshold = 0.5f)
    {
        // Filter for render faces (0x20) that are upward-facing
        var walkableFloors = data.AllTriangles
            .Where(t => (t.Flags & 0x20) != 0 && IsWalkableByNormal(t.V0, t.V1, t.V2, normalThreshold))
            .ToList();

        using var sw = new StreamWriter(outputPath);
        sw.WriteLine("# WMO Walkable Floor Surfaces");
        sw.WriteLine($"# Source: {data.WmoPath}");
        sw.WriteLine($"# Total triangles: {data.AllTriangles.Count}");
        sw.WriteLine($"# Walkable floor triangles: {walkableFloors.Count}");
        sw.WriteLine($"# Normal threshold: {normalThreshold}");
        sw.WriteLine();

        // Write vertices
        foreach (var tri in walkableFloors)
        {
            sw.WriteLine($"v {tri.V0.X} {tri.V0.Y} {tri.V0.Z}");
            sw.WriteLine($"v {tri.V1.X} {tri.V1.Y} {tri.V1.Z}");
            sw.WriteLine($"v {tri.V2.X} {tri.V2.Y} {tri.V2.Z}");
        }

        sw.WriteLine();

        // Write faces (1-indexed)
        for (int i = 0; i < walkableFloors.Count; i++)
        {
            int baseIdx = i * 3 + 1;
            sw.WriteLine($"f {baseIdx} {baseIdx + 1} {baseIdx + 2}");
        }

        Console.WriteLine($"[INFO] Exported walkable floors ({walkableFloors.Count} triangles) to {outputPath}");
    }

    /// <summary>
    /// Export aggregated geometry for each flag type across ALL groups.
    /// Creates files like: [WmoName]_flags_[FlagHex].obj
    /// </summary>
    public static void ExportPerFlag(WmoWalkableData data, string outputDir)
    {
        Directory.CreateDirectory(outputDir);

        if (data.TrianglesByGroup == null || data.TrianglesByGroup.Count == 0)
        {
            return;
        }

        var wmoBaseName = Path.GetFileNameWithoutExtension(data.WmoPath);

        // Aggregate all triangles by flag
        var trianglesByFlag = new Dictionary<ushort, List<WmoTriangle>>();

        foreach (var groupEntry in data.TrianglesByGroup)
        {
            foreach (var tri in groupEntry.Value)
            {
                if (!trianglesByFlag.TryGetValue(tri.Flags, out var list))
                {
                    list = new List<WmoTriangle>();
                    trianglesByFlag[tri.Flags] = list;
                }
                list.Add(tri);
            }
        }

        // Export one OBJ per flag
        foreach (var entry in trianglesByFlag)
        {
            ushort flags = entry.Key;
            var triangles = entry.Value;
            
            var fileName = $"{wmoBaseName}_flags_{flags:X2}.obj";
            var path = Path.Combine(outputDir, fileName);

            using var sw = new StreamWriter(path);
            sw.WriteLine("# WMO Aggregated Geometry");
            sw.WriteLine("# Source: " + wmoBaseName + ", Flags 0x" + flags.ToString("X2"));
            sw.WriteLine("# Triangles: " + triangles.Count);
            sw.WriteLine();

            foreach (var tri in triangles)
            {
                sw.WriteLine($"v {tri.V0.X} {tri.V0.Y} {tri.V0.Z}");
                sw.WriteLine($"v {tri.V1.X} {tri.V1.Y} {tri.V1.Z}");
                sw.WriteLine($"v {tri.V2.X} {tri.V2.Y} {tri.V2.Z}");
            }

            for (int i = 0; i < triangles.Count; i++)
            {
                int baseIdx = i * 3 + 1;
                sw.WriteLine($"f {baseIdx} {baseIdx + 1} {baseIdx + 2}");
            }
        }
        
        Console.WriteLine($"[INFO] Exported {trianglesByFlag.Count} aggregated flag types for {wmoBaseName}");
    }
}
