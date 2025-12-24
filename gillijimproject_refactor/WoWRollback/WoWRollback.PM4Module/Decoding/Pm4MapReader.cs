// PM4 Map Reader - Global PM4 scene graph loader
// Loads ALL PM4 tiles into unified global pools with tile provenance tracking
// Part of the PM4 Pipeline Refactor (Dec 2025)

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.RegularExpressions;

namespace WoWRollback.PM4Module.Decoding;

/// <summary>
/// Represents a vertex with its source tile provenance.
/// </summary>
public record GlobalVertex(Vector3 Position, int TileX, int TileY, int LocalIndex);

/// <summary>
/// Represents an index with its source tile provenance.
/// </summary>
public record GlobalIndex(uint Value, int TileX, int TileY, int LocalIndex);

/// <summary>
/// Represents a surface entry with its source tile.
/// </summary>
public record GlobalSurface(MsurChunk Surface, int TileX, int TileY);

/// <summary>
/// Represents a position reference with its source tile.
/// </summary>
public record GlobalPosRef(MprlChunk Entry, int TileX, int TileY);

/// <summary>
/// Statistics for a CK24 object group across all tiles.
/// </summary>
public record Ck24Stats(
    uint CK24,
    int SurfaceCount,
    int VertexCount,
    HashSet<(int X, int Y)> Tiles,
    Vector3 BoundsMin,
    Vector3 BoundsMax
)
{
    public bool IsCrossTile => Tiles.Count > 1;
    public string TileList => string.Join(", ", Tiles.Select(t => $"{t.X}_{t.Y}"));
}

/// <summary>
/// Global PM4 Map Reader - Loads all PM4 tiles into a unified scene graph.
/// 
/// Key insight: PM4 format is hierarchical. Objects can span multiple tiles,
/// and vertex data pools (MSVT, MSVI, MSCN) are global resources.
/// Reading tiles independently fragments cross-tile objects.
/// 
/// This reader:
/// 1. Loads ALL .pm4 files from a directory into unified global pools
/// 2. Tracks tile provenance for each entry (which tile contributed it)
/// 3. Builds a CK24 -> surfaces registry across all tiles
/// 4. Provides per-tile extraction that references global pools
/// </summary>
public class Pm4MapReader
{
    #region Global Pools
    
    /// <summary>All vertices across all tiles (MSVT)</summary>
    public List<GlobalVertex> GlobalVertices { get; } = new();
    
    /// <summary>All indices across all tiles (MSVI)</summary>
    public List<GlobalIndex> GlobalIndices { get; } = new();
    
    /// <summary>All scene nodes across all tiles (MSCN)</summary>
    public List<GlobalVertex> GlobalSceneNodes { get; } = new();
    
    /// <summary>All surfaces across all tiles (MSUR)</summary>
    public List<GlobalSurface> GlobalSurfaces { get; } = new();
    
    /// <summary>All position references across all tiles (MPRL)</summary>
    public List<GlobalPosRef> GlobalPositionRefs { get; } = new();
    
    /// <summary>CK24 -> List of surfaces (with tile info)</summary>
    public Dictionary<uint, List<GlobalSurface>> Ck24Objects { get; } = new();
    
    /// <summary>Set of tiles that have been loaded</summary>
    public HashSet<(int X, int Y)> LoadedTiles { get; } = new();
    
    /// <summary>Per-tile vertex offset (for global index remapping)</summary>
    private Dictionary<(int, int), int> _tileVertexOffsets = new();
    
    /// <summary>Per-tile index offset</summary>
    private Dictionary<(int, int), int> _tileIndexOffsets = new();
    
    /// <summary>Per-tile scene node offset</summary>
    private Dictionary<(int, int), int> _tileSceneNodeOffsets = new();
    
    #endregion
    
    #region Loading
    
    /// <summary>
    /// Load all PM4 files from a directory into global pools.
    /// </summary>
    /// <param name="pm4Directory">Directory containing PM4 files</param>
    /// <returns>Number of tiles loaded</returns>
    public int LoadDirectory(string pm4Directory)
    {
        if (!Directory.Exists(pm4Directory))
        {
            Console.WriteLine($"[ERROR] PM4 directory not found: {pm4Directory}");
            return 0;
        }
        
        var pm4Files = Directory.GetFiles(pm4Directory, "*.pm4", SearchOption.AllDirectories);
        Console.WriteLine($"[PM4MapReader] Found {pm4Files.Length} PM4 files to load...");
        
        int loaded = 0;
        foreach (var pm4Path in pm4Files)
        {
            if (LoadTile(pm4Path))
                loaded++;
        }
        
        // Build CK24 registry
        BuildCk24Registry();
        
        Console.WriteLine($"[PM4MapReader] Loaded {loaded} tiles");
        Console.WriteLine($"[PM4MapReader] Global pools: {GlobalVertices.Count} vertices, {GlobalIndices.Count} indices, {GlobalSceneNodes.Count} scene nodes, {GlobalSurfaces.Count} surfaces");
        Console.WriteLine($"[PM4MapReader] CK24 objects: {Ck24Objects.Count} unique CK24 values");
        
        // Report cross-tile objects
        var crossTile = GetCrossTileCk24s();
        if (crossTile.Any())
        {
            Console.WriteLine($"[PM4MapReader] Found {crossTile.Count} cross-tile CK24 objects");
        }
        
        return loaded;
    }
    
    /// <summary>
    /// Load a single PM4 tile into global pools.
    /// </summary>
    private bool LoadTile(string pm4Path)
    {
        var (tileX, tileY) = ParseTileCoordinates(pm4Path);
        if (tileX < 0 || tileY < 0)
        {
            Console.WriteLine($"[WARN] Could not parse tile coords from: {Path.GetFileName(pm4Path)}");
            return false;
        }
        
        if (LoadedTiles.Contains((tileX, tileY)))
        {
            Console.WriteLine($"[WARN] Tile {tileX}_{tileY} already loaded, skipping duplicate");
            return false;
        }
        
        byte[] data;
        try
        {
            data = File.ReadAllBytes(pm4Path);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[WARN] Failed to read PM4 {Path.GetFileName(pm4Path)}: {ex.Message}");
            return false;
        }
        
        var pm4 = Pm4Decoder.Decode(data);
        
        // Record offsets before adding
        _tileVertexOffsets[(tileX, tileY)] = GlobalVertices.Count;
        _tileIndexOffsets[(tileX, tileY)] = GlobalIndices.Count;
        _tileSceneNodeOffsets[(tileX, tileY)] = GlobalSceneNodes.Count;
        
        // Add vertices to global pool with provenance
        for (int i = 0; i < pm4.MeshVertices.Count; i++)
        {
            GlobalVertices.Add(new GlobalVertex(pm4.MeshVertices[i], tileX, tileY, i));
        }
        
        // Add indices to global pool with provenance
        for (int i = 0; i < pm4.MeshIndices.Count; i++)
        {
            GlobalIndices.Add(new GlobalIndex(pm4.MeshIndices[i], tileX, tileY, i));
        }
        
        // Add scene nodes to global pool with provenance
        for (int i = 0; i < pm4.SceneNodes.Count; i++)
        {
            GlobalSceneNodes.Add(new GlobalVertex(pm4.SceneNodes[i], tileX, tileY, i));
        }
        
        // Add surfaces to global pool with provenance
        foreach (var surf in pm4.Surfaces)
        {
            GlobalSurfaces.Add(new GlobalSurface(surf, tileX, tileY));
        }
        
        // Add position refs
        foreach (var posRef in pm4.PositionRefs)
        {
            GlobalPositionRefs.Add(new GlobalPosRef(posRef, tileX, tileY));
        }
        
        LoadedTiles.Add((tileX, tileY));
        return true;
    }
    
    /// <summary>
    /// Build the CK24 -> surfaces registry after loading all tiles.
    /// </summary>
    private void BuildCk24Registry()
    {
        Ck24Objects.Clear();
        
        foreach (var globalSurf in GlobalSurfaces)
        {
            uint ck24 = globalSurf.Surface.CK24;
            
            if (!Ck24Objects.TryGetValue(ck24, out var list))
            {
                list = new List<GlobalSurface>();
                Ck24Objects[ck24] = list;
            }
            
            list.Add(globalSurf);
        }
    }
    
    #endregion
    
    #region CK24 Analysis
    
    /// <summary>
    /// Get all CK24 values that span multiple tiles.
    /// </summary>
    public List<uint> GetCrossTileCk24s()
    {
        var crossTile = new List<uint>();
        
        foreach (var (ck24, surfaces) in Ck24Objects)
        {
            var tiles = surfaces.Select(s => (s.TileX, s.TileY)).Distinct().ToList();
            if (tiles.Count > 1)
                crossTile.Add(ck24);
        }
        
        return crossTile;
    }
    
    /// <summary>
    /// Get statistics for all CK24 objects.
    /// </summary>
    public List<Ck24Stats> GetCk24Statistics()
    {
        var stats = new List<Ck24Stats>();
        
        foreach (var (ck24, surfaces) in Ck24Objects)
        {
            var tiles = surfaces.Select(s => (s.TileX, s.TileY)).Distinct().ToHashSet();
            
            // Calculate bounds
            var min = new Vector3(float.MaxValue);
            var max = new Vector3(float.MinValue);
            int vertexCount = 0;
            
            foreach (var gs in surfaces)
            {
                // Get vertex bounds from this surface
                var surf = gs.Surface;
                int tileX = gs.TileX;
                int tileY = gs.TileY;
                
                if (!_tileVertexOffsets.TryGetValue((tileX, tileY), out int vertOffset))
                    continue;
                if (!_tileIndexOffsets.TryGetValue((tileX, tileY), out int idxOffset))
                    continue;
                
                // Look up indices for this surface
                for (int i = 0; i < surf.IndexCount; i++)
                {
                    int globalIdxPos = idxOffset + (int)surf.MsviFirstIndex + i;
                    if (globalIdxPos >= 0 && globalIdxPos < GlobalIndices.Count)
                    {
                        uint localVertIdx = GlobalIndices[globalIdxPos].Value;
                        int globalVertPos = vertOffset + (int)localVertIdx;
                        
                        if (globalVertPos >= 0 && globalVertPos < GlobalVertices.Count)
                        {
                            var v = GlobalVertices[globalVertPos].Position;
                            min = Vector3.Min(min, v);
                            max = Vector3.Max(max, v);
                            vertexCount++;
                        }
                    }
                }
            }
            
            if (min.X == float.MaxValue)
            {
                min = Vector3.Zero;
                max = Vector3.Zero;
            }
            
            stats.Add(new Ck24Stats(ck24, surfaces.Count, vertexCount, tiles, min, max));
        }
        
        return stats.OrderByDescending(s => s.Tiles.Count).ThenBy(s => s.CK24).ToList();
    }
    
    /// <summary>
    /// Get all CK24s that appear in a specific tile.
    /// </summary>
    public List<uint> GetTileCk24s(int tileX, int tileY)
    {
        return Ck24Objects
            .Where(kvp => kvp.Value.Any(s => s.TileX == tileX && s.TileY == tileY))
            .Select(kvp => kvp.Key)
            .ToList();
    }
    
    #endregion
    
    #region Per-Tile Extraction
    
    /// <summary>
    /// Extract WMO candidates for a specific tile.
    /// Uses global pools and includes cross-tile CK24 objects that have surfaces in this tile.
    /// </summary>
    public List<Pipeline.Pm4WmoCandidate> ExtractTileObjects(int tileX, int tileY)
    {
        var candidates = new List<Pipeline.Pm4WmoCandidate>();
        
        if (!LoadedTiles.Contains((tileX, tileY)))
            return candidates;
        
        // Get all CK24s that have surfaces in this tile
        var tileCk24s = GetTileCk24s(tileX, tileY);
        
        foreach (var ck24 in tileCk24s)
        {
            // Skip nav mesh (CK24 = 0x000000)
            if (ck24 == 0x000000)
                continue;
            
            var allSurfaces = Ck24Objects[ck24];
            
            // For per-tile extraction, only use surfaces from THIS tile
            // (Cross-tile objects will be handled separately)
            var tileSurfaces = allSurfaces.Where(s => s.TileX == tileX && s.TileY == tileY).ToList();
            
            if (tileSurfaces.Count == 0)
                continue;
            
            // Extract geometry using global pools
            var geometry = ExtractGeometryFromGlobalPools(tileSurfaces, tileX, tileY);
            
            if (geometry.Vertices.Count < 3)
                continue;
            
            // Calculate dominant angle
            float domAngle = CalculateDominantAngle(tileSurfaces.Select(s => s.Surface).ToList());
            
            // Type flags from CK24
            byte typeFlags = (byte)((ck24 >> 16) & 0xFF);
            
            // Find closest MPRL entry
            var centroid = (geometry.BoundsMin + geometry.BoundsMax) / 2f;
            var (mprlRot, mprlPos) = FindClosestMprl(centroid, tileX, tileY);
            
            // Create candidate
            var candidate = new Pipeline.Pm4WmoCandidate(
                CK24: ck24,
                InstanceId: 0, // Will be refined by MsViGapSplitter if needed
                TileX: tileX,
                TileY: tileY,
                BoundsMin: geometry.BoundsMin,
                BoundsMax: geometry.BoundsMax,
                DominantAngle: domAngle,
                SurfaceCount: tileSurfaces.Count,
                VertexCount: geometry.Vertices.Count,
                TypeFlags: typeFlags,
                MprlRotationDegrees: mprlRot,
                MprlPosition: mprlPos ?? centroid,
                DebugGeometry: geometry.Vertices,
                DebugFaces: geometry.Faces,
                DebugMscnVertices: geometry.MscnVertices
            );
            
            candidates.Add(candidate);
        }
        
        return candidates;
    }
    
    /// <summary>
    /// Extract all WMO candidates from all loaded tiles.
    /// </summary>
    public IEnumerable<Pipeline.Pm4WmoCandidate> ExtractAllTileObjects()
    {
        foreach (var (tileX, tileY) in LoadedTiles.OrderBy(t => t.X).ThenBy(t => t.Y))
        {
            foreach (var candidate in ExtractTileObjects(tileX, tileY))
            {
                yield return candidate;
            }
        }
    }
    
    #endregion
    
    #region Geometry Extraction Helpers
    
    private record ExtractedGeometry(
        List<Vector3> Vertices,
        List<int[]> Faces,
        List<Vector3> MscnVertices,
        Vector3 BoundsMin,
        Vector3 BoundsMax
    );
    
    private ExtractedGeometry ExtractGeometryFromGlobalPools(List<GlobalSurface> surfaces, int tileX, int tileY)
    {
        var vertices = new List<Vector3>();
        var faces = new List<int[]>();
        var mscnVertices = new List<Vector3>();
        var msvtToLocalMap = new Dictionary<int, int>();
        
        if (!_tileVertexOffsets.TryGetValue((tileX, tileY), out int vertOffset))
            return new ExtractedGeometry(vertices, faces, mscnVertices, Vector3.Zero, Vector3.Zero);
        if (!_tileIndexOffsets.TryGetValue((tileX, tileY), out int idxOffset))
            return new ExtractedGeometry(vertices, faces, mscnVertices, Vector3.Zero, Vector3.Zero);
        if (!_tileSceneNodeOffsets.TryGetValue((tileX, tileY), out int scnOffset))
            scnOffset = 0;
        
        // 1. Collect mesh vertices
        foreach (var gs in surfaces)
        {
            var surf = gs.Surface;
            var faceIndices = new List<int>();
            
            for (int i = 0; i < surf.IndexCount; i++)
            {
                int globalIdxPos = idxOffset + (int)surf.MsviFirstIndex + i;
                if (globalIdxPos >= 0 && globalIdxPos < GlobalIndices.Count)
                {
                    uint localVertIdx = GlobalIndices[globalIdxPos].Value;
                    int globalVertPos = vertOffset + (int)localVertIdx;
                    
                    if (!msvtToLocalMap.TryGetValue(globalVertPos, out int localIdx))
                    {
                        if (globalVertPos >= 0 && globalVertPos < GlobalVertices.Count)
                        {
                            localIdx = vertices.Count;
                            vertices.Add(GlobalVertices[globalVertPos].Position);
                            msvtToLocalMap[globalVertPos] = localIdx;
                        }
                    }
                    
                    if (msvtToLocalMap.ContainsKey(globalVertPos))
                        faceIndices.Add(msvtToLocalMap[globalVertPos]);
                }
            }
            
            if (faceIndices.Count >= 3)
                faces.Add(faceIndices.ToArray());
        }
        
        // 2. Collect MSCN vertices via MdosIndex
        foreach (var gs in surfaces)
        {
            int globalScnPos = scnOffset + (int)gs.Surface.MdosIndex;
            if (globalScnPos >= 0 && globalScnPos < GlobalSceneNodes.Count)
            {
                mscnVertices.Add(GlobalSceneNodes[globalScnPos].Position);
            }
        }
        
        // Calculate bounds
        var min = new Vector3(float.MaxValue);
        var max = new Vector3(float.MinValue);
        
        var allPoints = vertices.Concat(mscnVertices);
        if (allPoints.Any())
        {
            foreach (var v in allPoints)
            {
                min = Vector3.Min(min, v);
                max = Vector3.Max(max, v);
            }
        }
        else
        {
            min = Vector3.Zero;
            max = Vector3.Zero;
        }
        
        return new ExtractedGeometry(vertices, faces, mscnVertices, min, max);
    }
    
    private float CalculateDominantAngle(List<MsurChunk> surfaces)
    {
        float[] histogram = new float[72];
        float maxArea = 0;
        int maxBin = -1;
        
        foreach (var surf in surfaces)
        {
            if (Math.Abs(surf.Normal.Z) < 0.7f)
            {
                double angle = Math.Atan2(surf.Normal.Y, surf.Normal.X) * 180.0 / Math.PI;
                if (angle < 0) angle += 360.0;
                
                int bin = (int)(angle / 5.0) % 72;
                float weight = surf.IndexCount;
                
                histogram[bin] += weight;
                
                if (histogram[bin] > maxArea)
                {
                    maxArea = histogram[bin];
                    maxBin = bin;
                }
            }
        }
        
        return maxBin != -1 ? maxBin * 5.0f : 0f;
    }
    
    private (float? Rotation, Vector3? Position) FindClosestMprl(Vector3 centroid, int tileX, int tileY)
    {
        var tilePosRefs = GlobalPositionRefs
            .Where(p => p.TileX == tileX && p.TileY == tileY && p.Entry.Unk16 == 0)
            .ToList();
        
        if (tilePosRefs.Count == 0)
            return (null, null);
        
        var closest = tilePosRefs
            .Select(p => new { Entry = p.Entry, Dist = Vector3.Distance(p.Entry.Position, centroid) })
            .OrderBy(x => x.Dist)
            .FirstOrDefault();
        
        if (closest != null && closest.Dist < 100)
        {
            float rot = 360f * closest.Entry.Unk04 / 65536f;
            return (rot, closest.Entry.Position);
        }
        
        return (null, null);
    }
    
    #endregion
    
    #region Helpers
    
    private static (int tileX, int tileY) ParseTileCoordinates(string pm4Path)
    {
        var baseName = Path.GetFileNameWithoutExtension(pm4Path);
        var match = Regex.Match(baseName, @"(\d+)_(\d+)$");
        if (!match.Success)
            return (-1, -1);
        
        return (int.Parse(match.Groups[1].Value), int.Parse(match.Groups[2].Value));
    }
    
    #endregion
}
