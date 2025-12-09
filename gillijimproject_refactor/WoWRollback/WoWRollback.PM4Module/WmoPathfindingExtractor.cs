using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Linq;

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
    public WmoPathfindingData ExtractFromWmo(string wmoRootPath)
    {
        var result = new WmoPathfindingData { WmoPath = wmoRootPath };
        
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
            ExtractFromGroupFile(wmoRootPath, result, 0);
        }
        else
        {
            int groupIdx = 0;
            foreach (var groupFile in groupFiles)
            {
                ExtractFromGroupFile(groupFile, result, groupIdx++);
            }
        }
        
        result.ComputeFingerprint();
        return result;
    }

    /// <summary>
    /// Extract walkable surfaces from WMO bytes (from MPQ).
    /// </summary>
    public WmoPathfindingData ExtractFromBytes(byte[] data, string wmoPath)
    {
        var result = new WmoPathfindingData { WmoPath = wmoPath };
        ExtractFromData(data, result, 0);
        result.ComputeFingerprint();
        return result;
    }

    private void ExtractFromGroupFile(string groupPath, WmoPathfindingData result, int groupIdx)
    {
        if (!File.Exists(groupPath)) return;
        ExtractFromData(File.ReadAllBytes(groupPath), result, groupIdx);
    }

    private void ExtractFromData(byte[] data, WmoPathfindingData result, int groupIdx)
    {
        if (data == null || data.Length == 0) return;
        
        using var ms = new MemoryStream(data);
        using var br = new BinaryReader(ms);
        
        var vertices = new List<Vector3>();
        var indices = new List<ushort>();
        var normals = new List<Vector3>();
        
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
                case "TVOM": // MOVT reversed
                    int vertCount = (int)(size / 12);
                    for (int i = 0; i < vertCount; i++)
                    {
                        vertices.Add(new Vector3(br.ReadSingle(), br.ReadSingle(), br.ReadSingle()));
                    }
                    break;
                    
                case "IVOM": // MOVI reversed
                    int idxCount = (int)(size / 2);
                    for (int i = 0; i < idxCount; i++)
                    {
                        indices.Add(br.ReadUInt16());
                    }
                    break;
                    
                case "RNOM": // MONR reversed
                    int normCount = (int)(size / 12);
                    for (int i = 0; i < normCount; i++)
                    {
                        normals.Add(new Vector3(br.ReadSingle(), br.ReadSingle(), br.ReadSingle()));
                    }
                    break;
            }
            
            br.BaseStream.Position = dataStart + size;
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
            
            // Filter: only upward-facing surfaces (walkable)
            // Z is up in WoW coordinates
            if (faceNormal.Z > 0.5f)
            {
                result.WalkableSurfaces.Add(new WalkableSurface
                {
                    V0 = v0,
                    V1 = v1,
                    V2 = v2,
                    Normal = faceNormal,
                    GroupIndex = groupIdx
                });
            }
        }
    }
}

public class WmoPathfindingData
{
    public string WmoPath { get; set; } = "";
    public List<WalkableSurface> WalkableSurfaces { get; } = new();
    
    // Fingerprint data
    public int SurfaceCount => WalkableSurfaces.Count;
    public int VertexCount { get; private set; }
    public Vector3 BoundsMin { get; private set; }
    public Vector3 BoundsMax { get; private set; }
    public Vector3 Size => BoundsMax - BoundsMin;
    public float UpwardFacingPct { get; private set; } = 1.0f; // By definition, all are upward
    
    public void ComputeFingerprint()
    {
        if (WalkableSurfaces.Count == 0) return;
        
        var allVerts = WalkableSurfaces
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
}
