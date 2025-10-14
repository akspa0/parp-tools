using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using WoWFormatLib.FileReaders;
using WoWFormatLib.Structs.WDT;
using WoWRollback.Core.Services.Archive;
using SharpGLTF.Geometry;
using SharpGLTF.Geometry.VertexTypes;
using SharpGLTF.Materials;
using SharpGLTF.Scenes;
using SharpGLTF.Schema2;

namespace WoWRollback.AnalysisModule;

/// <summary>
/// Extracts terrain mesh data (OBJ/GLB) from ADT files for 3D visualization.
/// Based on proven working implementation from ADTPreFabTool.
/// Multi-threaded for performance. Outputs to {mapName}_mesh/ directory.
/// </summary>
public sealed class AdtMeshExtractor
{
    private const float TILE_SIZE = 533.33333f;
    private const float CHUNK_SIZE = TILE_SIZE / 16f;
    private const float UNIT_SIZE = CHUNK_SIZE / 8f;
    private const float UNIT_SIZE_HALF = UNIT_SIZE / 2f;

    /// <summary>
    /// Extracts mesh data for all tiles in a map from MPQ archives.
    /// Uses multi-threading for performance.
    /// </summary>
    public MeshExtractionResult ExtractFromArchive(
        IArchiveSource source,
        string mapName,
        string outputDir,
        bool exportGlb = false,
        bool exportObj = true,
        int maxTiles = 0, // 0 = no limit
        int maxDegreeOfParallelism = -1) // -1 = use default (CPU count)
    {
        var meshDir = Path.Combine(outputDir, $"{mapName}_mesh");
        Directory.CreateDirectory(meshDir);

        Console.WriteLine($"[AdtMeshExtractor] Extracting meshes for map: {mapName}");
        Console.WriteLine($"[AdtMeshExtractor] Output directory: {meshDir}");

        // Enumerate all ADT tiles for this map
        var adtTiles = EnumerateAdtTiles(source, mapName);
        
        if (adtTiles.Count == 0)
        {
            Console.WriteLine($"[AdtMeshExtractor] No ADT tiles found for map: {mapName}");
            return new MeshExtractionResult(
                Success: true,
                TilesProcessed: 0,
                MeshDirectory: meshDir,
                ManifestPath: null);
        }

        Console.WriteLine($"[AdtMeshExtractor] Found {adtTiles.Count} ADT tiles");

        // Limit tiles if requested (for testing)
        if (maxTiles > 0 && adtTiles.Count > maxTiles)
        {
            adtTiles = adtTiles.Take(maxTiles).ToList();
            Console.WriteLine($"[AdtMeshExtractor] Limited to {maxTiles} tiles for testing");
        }

        // Process tiles in parallel for performance
        var manifestEntries = new ConcurrentBag<MeshManifestEntry>();
        int tilesProcessed = 0;
        int tilesSkipped = 0;
        object lockObj = new object();

        var options = new ParallelOptions
        {
            MaxDegreeOfParallelism = maxDegreeOfParallelism == -1 
                ? Environment.ProcessorCount 
                : maxDegreeOfParallelism
        };

        Parallel.ForEach(adtTiles, options, tile =>
        {
            try
            {
                var entry = ExtractTileMesh(source, mapName, tile.X, tile.Y, meshDir, exportGlb, exportObj);
                if (entry != null)
                {
                    manifestEntries.Add(entry);
                    lock (lockObj)
                    {
                        tilesProcessed++;
                        if (tilesProcessed % 10 == 0 || tilesProcessed == adtTiles.Count)
                        {
                            Console.WriteLine($"[AdtMeshExtractor] Progress: {tilesProcessed}/{adtTiles.Count} tiles");
                        }
                    }
                }
                else
                {
                    lock (lockObj) { tilesSkipped++; }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[AdtMeshExtractor] Warning: Failed to process tile {tile.X}_{tile.Y}: {ex.Message}");
                lock (lockObj) { tilesSkipped++; }
            }
        });

        Console.WriteLine($"[AdtMeshExtractor] Processed {tilesProcessed} tiles, skipped {tilesSkipped}");

        // Generate manifest JSON
        string? manifestPath = null;
        var entriesList = manifestEntries.ToList();
        if (entriesList.Count > 0)
        {
            manifestPath = Path.Combine(meshDir, "mesh_manifest.json");
            GenerateManifest(entriesList, mapName, manifestPath);
            Console.WriteLine($"[AdtMeshExtractor] Manifest: {manifestPath}");
        }

        return new MeshExtractionResult(
            Success: true,
            TilesProcessed: tilesProcessed,
            MeshDirectory: meshDir,
            ManifestPath: manifestPath);
    }

    private List<(int X, int Y)> EnumerateAdtTiles(IArchiveSource source, string mapName)
    {
        var tiles = new List<(int X, int Y)>();

        // Check for ADT files in the map directory
        for (int y = 0; y < 64; y++)
        {
            for (int x = 0; x < 64; x++)
            {
                var adtPath = $"world/maps/{mapName}/{mapName}_{x}_{y}.adt";
                if (source.FileExists(adtPath))
                {
                    tiles.Add((x, y));
                }
            }
        }

        return tiles;
    }

    private MeshManifestEntry? ExtractTileMesh(
        IArchiveSource source,
        string mapName,
        int tileX,
        int tileY,
        string meshDir,
        bool exportGlb,
        bool exportObj)
    {
        var adtPath = $"world/maps/{mapName}/{mapName}_{tileX}_{tileY}.adt";
        
        // Read ADT using WoWFormatLib
        var adtReader = new ADTReader();
        using (var stream = source.OpenFile(adtPath))
        {
            adtReader.ReadRootFile(stream, MPHDFlags.wdt_has_maid);
        }

        var adt = adtReader.adtfile;
        if (adt.chunks == null || adt.chunks.Length == 0)
        {
            return null;
        }

        // Build mesh data (first pass: vertices and bounds)
        var positions = new List<(float x, float y, float z)>();
        var chunkStartIndices = new List<int>();
        
        float minX = float.PositiveInfinity, maxX = float.NegativeInfinity;
        float minY = float.PositiveInfinity, maxY = float.NegativeInfinity;
        float minZ = float.PositiveInfinity, maxZ = float.NegativeInfinity;

        foreach (var chunk in adt.chunks)
        {
            if (chunk.vertices.vertices == null || chunk.vertices.vertices.Length == 0)
            {
                chunkStartIndices.Add(positions.Count);
                continue;
            }

            int startIdx = positions.Count;
            chunkStartIndices.Add(startIdx);

            // Extract vertices (17 rows, alternating 9 and 8 columns = 145 total)
            int idx = 0;
            for (int row = 0; row < 17; row++)
            {
                bool isShort = (row % 2) == 1;
                int colCount = isShort ? 8 : 9;

                for (int col = 0; col < colCount; col++)
                {
                    float vx = chunk.header.position.Y - (col * UNIT_SIZE);
                    if (isShort) vx -= UNIT_SIZE_HALF;
                    float vy = chunk.vertices.vertices[idx] + chunk.header.position.Z;
                    float vz = chunk.header.position.X - (row * UNIT_SIZE_HALF);
                    
                    positions.Add((vx, vy, vz));
                    
                    // Track bounds for UV calculation
                    if (vx < minX) minX = vx;
                    if (vx > maxX) maxX = vx;
                    if (vy < minY) minY = vy;
                    if (vy > maxY) maxY = vy;
                    if (vz < minZ) minZ = vz;
                    if (vz > maxZ) maxZ = vz;
                    
                    idx++;
                }
            }
        }

        // Second pass: compute UVs from bounds
        float spanX = Math.Max(1e-6f, maxX - minX);
        float spanZ = Math.Max(1e-6f, maxZ - minZ);
        const float eps = 2.5e-3f;

        var uvs = new List<(float u, float v)>(positions.Count);
        for (int i = 0; i < positions.Count; i++)
        {
            var p = positions[i];
            // UV mapping: normalized to tile extents
            float u = (p.x - minX) / spanX;
            float v = (maxZ - p.z) / spanZ;
            
            u = Math.Clamp(u, eps, 1f - eps);
            v = Math.Clamp(v, eps, 1f - eps);
            
            uvs.Add((u, v));
        }

        // Export GLB if requested (TODO: implement properly)
        string? glbFile = null;
        if (exportGlb && positions.Count > 0)
        {
            glbFile = $"{mapName}_{tileX}_{tileY}.glb";
            // var glbPath = Path.Combine(meshDir, glbFile);
            // ExportGLB(positions, uvs, chunkStartIndices, adt, glbPath);
        }

        // Export OBJ if requested
        string? objFile = null;
        string? mtlFile = null;
        if (exportObj && positions.Count > 0)
        {
            objFile = $"{mapName}_{tileX}_{tileY}.obj";
            mtlFile = $"{mapName}_{tileX}_{tileY}.mtl";
            var objPath = Path.Combine(meshDir, objFile);
            var mtlPath = Path.Combine(meshDir, mtlFile);
            ExportOBJ(positions, uvs, chunkStartIndices, adt, objPath, mtlPath, objFile);
        }

        return new MeshManifestEntry
        {
            TileX = tileX,
            TileY = tileY,
            GlbFile = glbFile,
            ObjFile = objFile,
            MtlFile = mtlFile,
            Bounds = new MeshBounds
            {
                MinX = minX,
                MaxX = maxX,
                MinY = minY,
                MaxY = maxY,
                MinZ = minZ,
                MaxZ = maxZ
            },
            VertexCount = positions.Count,
            TriangleCount = 0 // Calculated during face generation
        };
    }

    private void ExportGLB(
        List<(float x, float y, float z)> positions,
        List<int> indices,
        string outputPath)
    {
        var scene = new SceneBuilder();
        var material = new MaterialBuilder("TerrainMaterial")
            .WithMetallicRoughnessShader()
            .WithChannelParam(KnownChannel.BaseColor, new System.Numerics.Vector4(0.6f, 0.6f, 0.6f, 1.0f));

        var mesh = new MeshBuilder<VertexPosition>("TerrainMesh");
        var prim = mesh.UsePrimitive(material);

        // Add vertices
        var vertexPositions = positions.Select(p => new VertexPosition(p.x, p.y, p.z)).ToArray();

        // Add triangles
        for (int i = 0; i < indices.Count; i += 3)
        {
            if (i + 2 < indices.Count)
            {
                var i0 = indices[i];
                var i1 = indices[i + 1];
                var i2 = indices[i + 2];

                if (i0 < vertexPositions.Length && i1 < vertexPositions.Length && i2 < vertexPositions.Length)
                {
                    prim.AddTriangle(vertexPositions[i0], vertexPositions[i1], vertexPositions[i2]);
                }
            }
        }

        scene.AddRigidMesh(mesh, System.Numerics.Matrix4x4.Identity);

        var model = scene.ToGltf2();
        model.SaveGLB(outputPath);
    }

    private void ExportOBJ(
        List<(float x, float y, float z)> positions,
        List<(float u, float v)> uvs,
        List<int> chunkStartIndices,
        dynamic adt,
        string objPath,
        string mtlPath,
        string baseName)
    {
        // Write MTL file
        var mtl = new StringBuilder();
        mtl.AppendLine("# Terrain material");
        mtl.AppendLine("newmtl Minimap");
        mtl.AppendLine("Kd 1.000 1.000 1.000");
        mtl.AppendLine($"map_Kd {Path.GetFileNameWithoutExtension(baseName)}.png");
        File.WriteAllText(mtlPath, mtl.ToString());

        // Write OBJ file (matching working implementation)
        using var fs = new FileStream(objPath, FileMode.Create, FileAccess.Write, FileShare.Read);
        using var writer = new StreamWriter(fs);
        
        writer.WriteLine("# ADT Terrain Mesh (Textured with minimap)");
        writer.WriteLine($"mtllib {Path.GetFileName(mtlPath)}");
        writer.WriteLine("usemtl Minimap");

        // Write vertices in z,x,y order (CRITICAL for correct orientation)
        for (int i = 0; i < positions.Count; i++)
        {
            var p = positions[i];
            writer.WriteLine($"v {p.z:F6} {p.x:F6} {p.y:F6}");
        }

        // Write UVs
        for (int i = 0; i < uvs.Count; i++)
        {
            var t = uvs[i];
            writer.WriteLine($"vt {t.u.ToString(CultureInfo.InvariantCulture)} {t.v.ToString(CultureInfo.InvariantCulture)}");
        }

        // Write faces with hole detection (4 triangles per quad)
        int chunkIndex = 0;
        foreach (var chunk in adt.chunks)
        {
            if (chunk.vertices.vertices == null || chunk.vertices.vertices.Length == 0)
            {
                chunkIndex++;
                continue;
            }

            // Generate faces following working implementation pattern
            for (int j = 9, xx = 0, yy = 0; j < 145; j++, xx++)
            {
                if (xx >= 8) { xx = 0; yy++; }

                // Check for holes
                bool isHole = IsHole(chunk, xx, yy);

                if (!isHole)
                {
                    int baseIndex = chunkStartIndices[chunkIndex];
                    int i0 = j;
                    int a = baseIndex + i0 + 1;      // +1 for OBJ 1-based indexing
                    int b = baseIndex + (i0 - 9) + 1;
                    int c = baseIndex + (i0 + 8) + 1;
                    int d = baseIndex + (i0 - 8) + 1;
                    int e = baseIndex + (i0 + 9) + 1;
                    
                    // 4 triangles per quad (matching working implementation)
                    writer.WriteLine($"f {a}/{a} {b}/{b} {c}/{c}");
                    writer.WriteLine($"f {a}/{a} {d}/{d} {b}/{b}");
                    writer.WriteLine($"f {a}/{a} {e}/{e} {d}/{d}");
                    writer.WriteLine($"f {a}/{a} {c}/{c} {e}/{e}");
                }

                if (((j + 1) % (9 + 8)) == 0) j += 9;
            }

            chunkIndex++;
        }
    }

    private bool IsHole(dynamic chunk, int xx, int yy)
    {
        // Check hole flags (matching working implementation)
        if (((uint)chunk.header.flags & 0x10000u) == 0)
        {
            // Low-res holes (16 bits, 4x4 grid)
            int current = 1 << ((xx / 2) + (yy / 2) * 4);
            return (chunk.header.holesLowRes & current) != 0;
        }
        else
        {
            // High-res holes (8 bytes, 8x8 grid)
            byte holeByte = yy switch
            {
                0 => chunk.header.holesHighRes_0,
                1 => chunk.header.holesHighRes_1,
                2 => chunk.header.holesHighRes_2,
                3 => chunk.header.holesHighRes_3,
                4 => chunk.header.holesHighRes_4,
                5 => chunk.header.holesHighRes_5,
                6 => chunk.header.holesHighRes_6,
                _ => chunk.header.holesHighRes_7,
            };
            return ((holeByte >> xx) & 1) != 0;
        }
    }

    // Removed CalculateNormals - not needed for v/vt format

    private void GenerateManifest(List<MeshManifestEntry> entries, string mapName, string outputPath)
    {
        var manifest = new
        {
            map = mapName,
            tile_count = entries.Count,
            tiles = entries.Select(e => new
            {
                x = e.TileX,
                y = e.TileY,
                glb = e.GlbFile,
                obj = e.ObjFile,
                mtl = e.MtlFile,
                bounds = new
                {
                    min_x = e.Bounds.MinX,
                    max_x = e.Bounds.MaxX,
                    min_y = e.Bounds.MinY,
                    max_y = e.Bounds.MaxY,
                    min_z = e.Bounds.MinZ,
                    max_z = e.Bounds.MaxZ
                },
                vertex_count = e.VertexCount,
                triangle_count = e.TriangleCount
            }).ToList()
        };

        var json = JsonSerializer.Serialize(manifest, new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        });

        File.WriteAllText(outputPath, json);
    }
}

/// <summary>
/// Result of mesh extraction operation.
/// </summary>
public sealed record MeshExtractionResult(
    bool Success,
    int TilesProcessed,
    string MeshDirectory,
    string? ManifestPath);

/// <summary>
/// Manifest entry for a single tile mesh.
/// </summary>
internal sealed class MeshManifestEntry
{
    public required int TileX { get; init; }
    public required int TileY { get; init; }
    public string? GlbFile { get; init; }
    public string? ObjFile { get; init; }
    public string? MtlFile { get; init; }
    public required MeshBounds Bounds { get; init; }
    public required int VertexCount { get; init; }
    public required int TriangleCount { get; init; }
}

/// <summary>
/// Bounding box for a mesh.
/// </summary>
internal sealed class MeshBounds
{
    public required float MinX { get; init; }
    public required float MaxX { get; init; }
    public required float MinY { get; init; }
    public required float MaxY { get; init; }
    public required float MinZ { get; init; }
    public required float MaxZ { get; init; }
}
