using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Linq;
using WoWRollback.Core.Services.Archive;

namespace WoWRollback.PM4Module;

/// <summary>
/// Extracts walkable pathfinding surfaces from WMO files.
/// Filters to upward-facing surfaces to match PM4 pathfinding data.
/// </summary>
public class WmoPathfindingExtractor
{
    /// <summary>
    /// Extract walkable surfaces from a WMO root file.
    /// </summary>
    public WmoStructure ExtractFromWmo(string wmoRootPath)
    {
        var structure = new WmoStructure { WmoPath = wmoRootPath };
        
        // WMO root file contains group count in MOHD
        // Each group file contains the actual geometry
        var dir = Path.GetDirectoryName(wmoRootPath) ?? ".";
        var baseName = Path.GetFileNameWithoutExtension(wmoRootPath);
        
        // Find all group files (_000.wmo, _001.wmo, etc.)
        var groupFiles = Directory.GetFiles(dir, $"{baseName}_*.wmo")
            .OrderBy(f => f)
            .ToList();
        
        if (groupFiles.Count == 0)
        {
            // Try to extract from root file directly (some WMOs are single-file)
            var gData = ExtractFromGroupFile(wmoRootPath, 0);
            gData.ComputeFingerprint();
            structure.Groups.Add(gData);
        }
        else
        {
            int groupIdx = 0;
            foreach (var groupFile in groupFiles)
            {
                var gData = ExtractFromGroupFile(groupFile, groupIdx++);
                gData.ComputeFingerprint();
                structure.Groups.Add(gData);
            }
        }
        
        structure.ComputeAggregates();
        return structure;
    }

    /// <summary>
    /// Extract walkable surfaces from WMO bytes (from MPQ).
    /// </summary>
    public WmoPathfindingData ExtractFromBytes(byte[] data, string wmoPath)
    {
        var result = new WmoPathfindingData { WmoPath = wmoPath };
        var groupData = ExtractFromData(data, wmoPath, 0);
        result.WalkableSurfaces.AddRange(groupData.WalkableSurfaces);
        result.WallSurfaces.AddRange(groupData.WallSurfaces);
        result.MopyFlags.AddRange(groupData.MopyFlags);
        result.ComputeFingerprint();
        return result;
    }

    /// <summary>
    /// Extract walkable surfaces from a WMO in an MPQ archive.
    /// </summary>
    public WmoPathfindingData ExtractFromMpq(WoWRollback.Core.Services.Archive.IArchiveSource archive, string wmoRootPath)
    {
        var result = new WmoPathfindingData { WmoPath = wmoRootPath };
        
        // 1. Read Root File
        using var rootStream = archive.OpenFile(wmoRootPath);
        if (rootStream == null) return result;

        using var ms = new MemoryStream();
        rootStream.CopyTo(ms);
        ms.Position = 0;
        using var br = new BinaryReader(ms);

        int groupCount = 0;

        // Parse Root for MOHD
        while (br.BaseStream.Position + 8 <= br.BaseStream.Length)
        {
            var sigBytes = br.ReadBytes(4); // Chunk ID
            string sig = System.Text.Encoding.ASCII.GetString(sigBytes);
            uint size = br.ReadUInt32();
            // Check for MOHD or DHOM (reversed)
            if (sig == "MOHD" || sig == "DHOM")
            {
                br.ReadUInt32(); // nTextures
                groupCount = (int)br.ReadUInt32(); // nGroups
                break;
            }
            
            br.BaseStream.Position += size;
        }

        // 2. Read Group Files
        string dir = Path.GetDirectoryName(wmoRootPath) ?? "";
        string baseName = Path.GetFileNameWithoutExtension(wmoRootPath);

        for (int i = 0; i < groupCount; i++)
        {
            string groupFile = $"{baseName}_{i:D3}.wmo";
            string groupPath = Path.Combine(dir, groupFile).Replace('\\', '/'); // Ensure consistent separators

            // Try open from MPQ
            try
            {
                using var groupStream = archive.OpenFile(groupPath);
                if (groupStream != null)
                {
                    using var gms = new MemoryStream();
                    groupStream.CopyTo(gms);
                    var groupData = ExtractFromData(gms.ToArray(), groupPath, i);
                    result.WalkableSurfaces.AddRange(groupData.WalkableSurfaces);
                    result.WallSurfaces.AddRange(groupData.WallSurfaces);
                    result.MopyFlags.AddRange(groupData.MopyFlags);
                }
            }
            catch
            {
                // Ignore missing groups (shouldn't happen in valid WMOs)
            }
        }

        result.ComputeFingerprint();
        return result;
    }

    /// <summary>
    /// Extract structural WMO data (groups separate).
    /// </summary>
    public WmoStructure ExtractStructureFromMpq(WoWRollback.Core.Services.Archive.IArchiveSource archive, string wmoRootPath)
    {
        var structure = new WmoStructure { WmoPath = wmoRootPath };
        
        // 1. Read Root File to get Group Count
        using var rootStream = archive.OpenFile(wmoRootPath);
        if (rootStream == null) return structure;

        int groupCount = 0;
        using (var ms = new MemoryStream())
        {
            rootStream.CopyTo(ms);
            ms.Position = 0;
            using var br = new BinaryReader(ms);

            while (br.BaseStream.Position + 8 <= br.BaseStream.Length)
            {
                var sigBytes = br.ReadBytes(4);
                string sig = System.Text.Encoding.ASCII.GetString(sigBytes);
                uint size = br.ReadUInt32();
                if (sig == "MOHD" || sig == "DHOM")
                {
                    br.ReadUInt32(); // nTextures
                    groupCount = (int)br.ReadUInt32(); // nGroups
                    break;
                }
                br.BaseStream.Position += size;
            }
        }

        // 2. Read Group Files
        string dir = Path.GetDirectoryName(wmoRootPath) ?? "";
        string baseName = Path.GetFileNameWithoutExtension(wmoRootPath);

        for (int i = 0; i < groupCount; i++)
        {
            string groupFile = $"{baseName}_{i:D3}.wmo";
            string groupPath = Path.Combine(dir, groupFile).Replace('\\', '/');

            try
            {
                using var groupStream = archive.OpenFile(groupPath);
                if (groupStream != null)
                {
                    using var gms = new MemoryStream();
                    groupStream.CopyTo(gms);
                    var gData = ExtractFromData(gms.ToArray(), groupPath, i);
                    gData.ComputeFingerprint();
                    structure.Groups.Add(gData);
                }
            }
            catch { }
        }

        structure.ComputeAggregates();
        return structure;
    }

    private WmoPathfindingData ExtractFromGroupFile(string groupPath, int groupIdx)
    {
        if (!File.Exists(groupPath)) return new WmoPathfindingData();
        return ExtractFromData(File.ReadAllBytes(groupPath), groupPath, groupIdx);
    }

    private WmoPathfindingData ExtractFromData(byte[] data, string wmoPath, int groupIdx)
    {
        var groupData = new WmoPathfindingData { WmoPath = wmoPath };
        if (data == null || data.Length == 0) return groupData;
        
        using var ms = new MemoryStream(data);
        using var br = new BinaryReader(ms);
        
        var vertices = new List<Vector3>();
        var indices = new List<ushort>();
        var normals = new List<Vector3>();
        var mopyFlags = new List<byte>();
        
        // Parse IFF chunks
        while (br.BaseStream.Position + 8 <= br.BaseStream.Length)
        {
            var sigBytes = br.ReadBytes(4);
            // WMO chunk IDs are stored as reversed (e.g., "REVM" on disk = "MVER")
            // Read as-is and compare to reversed signatures
            string sig = System.Text.Encoding.ASCII.GetString(sigBytes);
            uint size = br.ReadUInt32();
            long dataStart = br.BaseStream.Position;
            
            
            switch (sig)
            {
                case "MOPY":
                case "YPOM":
                    int triCount = (int)(size / 2); // 2 bytes per triangle
                    for (int i = 0; i < triCount; i++)
                    {
                        byte mopyFlagsByte = br.ReadByte();
                        byte matId = br.ReadByte();
                        mopyFlags.Add(mopyFlagsByte);
                    }
                    break;

                case "MOGP":
                case "PGOM":
                    // MOGP only matters if we are parsing a group file, but we do that in a loop.
                    // Just read the flags.
                    br.ReadUInt32(); // nameOffset
                    br.ReadUInt32(); // descOffset
                    uint mogpFlags = br.ReadUInt32();
                    groupData.GroupFlags.Add(mogpFlags);
                    br.BaseStream.Position += 56;
                    continue;

                case "MOVT":
                case "TVOM": 
                    int vertCount = (int)(size / 12);
                    for (int i = 0; i < vertCount; i++)
                    {
                        vertices.Add(new Vector3(br.ReadSingle(), br.ReadSingle(), br.ReadSingle()));
                    }
                    break;
                    
                case "MOVI": 
                case "IVOM":
                    int idxCount = (int)(size / 2);
                    for (int i = 0; i < idxCount; i++)
                    {
                        indices.Add(br.ReadUInt16());
                    }
                    break;
                    
                case "MONR": 
                case "RNOM":
                    int normCount = (int)(size / 12);
                    for (int i = 0; i < normCount; i++)
                    {
                        normals.Add(new Vector3(br.ReadSingle(), br.ReadSingle(), br.ReadSingle()));
                    }
                    break;
            }
        }
            
        // Process triangles - filter to upward-facing only
        for (int i = 0; i + 2 < indices.Count; i += 3)
        {
            int i0 = indices[i];
            int i1 = indices[i + 1];
            int i2 = indices[i + 2];
            
            if (i0 >= vertices.Count || i1 >= vertices.Count || i2 >= vertices.Count)
                continue;
            
            var v0 = vertices[i0];
            var v1 = vertices[i1];
            var v2 = vertices[i2];
            
            // Calculate face normal
            var edge1 = v1 - v0;
            var edge2 = v2 - v0;
            var faceNormal = Vector3.Normalize(Vector3.Cross(edge1, edge2));
            
            var surface = new WalkableSurface
            {
                V0 = v0,
                V1 = v1,
                V2 = v2,
                Normal = faceNormal,
                GroupIndex = groupIdx,
                MaterialFlags = (i/3 < mopyFlags.Count) ? mopyFlags[i/3] : (byte)0
            };

            if (faceNormal.Z > 0.5f)
            {
                groupData.WalkableSurfaces.Add(surface);
            }
            else if (Math.Abs(faceNormal.Z) <= 0.5f)
            {
                groupData.WallSurfaces.Add(surface);
            }
        }
        
        return groupData;
    }
}

public class WmoStructure
{
    public string WmoPath { get; set; } = "";
    public List<WmoPathfindingData> Groups { get; } = new();
    
    // Aggregate data for backward compatibility or whole-object matching
    public WmoPathfindingData Aggregate { get; private set; }
    
    public void ComputeAggregates()
    {
        Aggregate = new WmoPathfindingData { WmoPath = WmoPath };
        foreach (var g in Groups)
        {
            Aggregate.WalkableSurfaces.AddRange(g.WalkableSurfaces);
            Aggregate.WallSurfaces.AddRange(g.WallSurfaces);
            Aggregate.GroupFlags.ForEach(f => Aggregate.GroupFlags.Add(f)); // Assuming separate tracking
            Aggregate.MopyFlags.AddRange(g.MopyFlags);
        }
        Aggregate.ComputeFingerprint();
    }
}


public class WmoPathfindingData
{
    public string WmoPath { get; set; } = "";
    public List<WalkableSurface> WalkableSurfaces { get; } = new();
    public List<WalkableSurface> WallSurfaces { get; } = new();
    public List<uint> GroupFlags { get; } = new();
    public List<byte> MopyFlags { get; } = new();
    
    // Fingerprint data
    public int SurfaceCount => WalkableSurfaces.Count;
    public int WallCount => WallSurfaces.Count;
    public float DominantWallAngle { get; private set; }
    public byte DominantMopyFlag { get; private set; }
    public int VertexCount { get; private set; }
    public Vector3 BoundsMin { get; private set; }
    public Vector3 BoundsMax { get; private set; }
    public Vector3 Size => BoundsMax - BoundsMin;
    public float UpwardFacingPct { get; private set; } = 1.0f;
    
    public void ComputeFingerprint()
    {
        var allSurfaces = WalkableSurfaces.Concat(WallSurfaces).ToList();
        if (allSurfaces.Count == 0) return;
        
        var allVerts = allSurfaces
            .SelectMany(s => new[] { s.V0, s.V1, s.V2 })
            .Distinct()
            .ToList();
        
        VertexCount = allVerts.Count;
        
        BoundsMin = new Vector3(
            allVerts.Min(v => v.X),
            allVerts.Min(v => v.Y),
            allVerts.Min(v => v.Z));
        
        BoundsMax = new Vector3(
            allVerts.Max(v => v.X),
            allVerts.Max(v => v.Y),
            allVerts.Max(v => v.Z));

        // Calculate Dominant Wall Angle
        if (WallSurfaces.Count > 0)
        {
            // Bin wall areas by angle (5 degree buckets)
            var angleBins = new float[72]; 
            foreach (var wall in WallSurfaces)
            {
                // Surface Area
                var edge1 = wall.V1 - wall.V0;
                var edge2 = wall.V2 - wall.V0;
                var area = Vector3.Cross(edge1, edge2).Length() * 0.5f;

                // Angle in XY plane
                float angle = (float)Math.Atan2(wall.Normal.Y, wall.Normal.X) * (180f / (float)Math.PI);
                if (angle < 0) angle += 360f;

                int bin = (int)(angle / 5) % 72;
                angleBins[bin] += area;
            }

            int bestBin = -1;
            float maxArea = -1;
            for (int i = 0; i < 72; i++)
            {
                if (angleBins[i] > maxArea)
                {
                    maxArea = angleBins[i];
                    bestBin = i;
                }
            }
            
            DominantWallAngle = bestBin * 5f + 2.5f; // Center of bin
        }


        // Calculate Dominant MOPY Flag
        if (allSurfaces.Count > 0)
        {
            var flagCounts = new Dictionary<byte, int>();
            foreach (var s in allSurfaces)
            {
                if (!flagCounts.ContainsKey(s.MaterialFlags)) flagCounts[s.MaterialFlags] = 0;
                flagCounts[s.MaterialFlags]++;
            }
            
            DominantMopyFlag = flagCounts.OrderByDescending(kvp => kvp.Value).First().Key;
        }
    }
    
    public override string ToString()
    {
        return $"WMO Pathfinding: {SurfaceCount} walkable surfaces, {VertexCount} verts, size {Size.X:F0}x{Size.Y:F0}x{Size.Z:F0}";
    }
}

public class WalkableSurface
{
    public Vector3 V0 { get; set; }
    public Vector3 V1 { get; set; }
    public Vector3 V2 { get; set; }
    public Vector3 Normal { get; set; }
    public int GroupIndex { get; set; }
    public byte MaterialFlags { get; set; }
}
