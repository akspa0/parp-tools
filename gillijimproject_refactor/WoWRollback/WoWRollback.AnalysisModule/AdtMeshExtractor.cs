using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
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
/// Extracts terrain mesh data (GLB) from ADT files for 3D visualization.
/// Outputs to {mapName}_mesh/ directory with per-tile mesh files.
/// </summary>
public sealed class AdtMeshExtractor
{
    private const float UNIT_SIZE = 533.33333f / 16.0f; // ADT chunk unit size
    private const float UNIT_SIZE_HALF = UNIT_SIZE / 2.0f;

    /// <summary>
    /// Extracts mesh data for all tiles in a map from MPQ archives.
    /// </summary>
    public MeshExtractionResult ExtractFromArchive(
        IArchiveSource source,
        string mapName,
        string outputDir,
        bool exportGlb = true,
        bool exportObj = true,
        int maxTiles = 0) // 0 = no limit
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

        var manifestEntries = new List<MeshManifestEntry>();
        int tilesProcessed = 0;
        int tilesSkipped = 0;

        foreach (var (tileX, tileY) in adtTiles)
        {
            try
            {
                var entry = ExtractTileMesh(source, mapName, tileX, tileY, meshDir, exportGlb, exportObj);
                if (entry != null)
                {
                    manifestEntries.Add(entry);
                    tilesProcessed++;
                    
                    if (tilesProcessed % 10 == 0)
                    {
                        Console.WriteLine($"[AdtMeshExtractor] Progress: {tilesProcessed}/{adtTiles.Count} tiles");
                    }
                }
                else
                {
                    tilesSkipped++;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[AdtMeshExtractor] Warning: Failed to process tile {tileX}_{tileY}: {ex.Message}");
                tilesSkipped++;
            }
        }

        Console.WriteLine($"[AdtMeshExtractor] Processed {tilesProcessed} tiles, skipped {tilesSkipped}");

        // Generate manifest JSON
        string? manifestPath = null;
        if (manifestEntries.Count > 0)
        {
            manifestPath = Path.Combine(meshDir, "mesh_manifest.json");
            GenerateManifest(manifestEntries, mapName, manifestPath);
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

        // Calculate bounds
        float minX = float.MaxValue, maxX = float.MinValue;
        float minY = float.MaxValue, maxY = float.MinValue;
        float minZ = float.MaxValue, maxZ = float.MinValue;

        // Build mesh data
        var positions = new List<(float x, float y, float z)>();
        var indices = new List<int>();
        var chunkStartIndices = new List<int>();

        foreach (var chunk in adt.chunks)
        {
            if (chunk.vertices.vertices == null || chunk.vertices.vertices.Length == 0)
            {
                chunkStartIndices.Add(positions.Count);
                continue;
            }

            int startIdx = positions.Count;
            chunkStartIndices.Add(startIdx);

            // Extract vertices (17x17 grid, but stored as 145 vertices with padding)
            int idx = 0;
            for (int row = 0; row < 17; row++)
            {
                int colCount = (row % 2 == 0) ? 9 : 8;
                bool isShort = (row % 2 != 0);

                for (int col = 0; col < colCount; col++, idx++)
                {
                    float vx = chunk.header.position.Y - (col * UNIT_SIZE);
                    if (isShort) vx -= UNIT_SIZE_HALF;
                    float vy = chunk.vertices.vertices[idx] + chunk.header.position.Z;
                    float vz = chunk.header.position.X - (row * UNIT_SIZE_HALF);
                    
                    positions.Add((vx, vy, vz));
                    
                    if (vx < minX) minX = vx;
                    if (vx > maxX) maxX = vx;
                    if (vy < minY) minY = vy;
                    if (vy > maxY) maxY = vy;
                    if (vz < minZ) minZ = vz;
                    if (vz > maxZ) maxZ = vz;
                }
            }
        }

        // Build indices (triangles)
        int chunkIndex = 0;
        foreach (var chunk in adt.chunks)
        {
            if (chunk.vertices.vertices == null || chunk.vertices.vertices.Length == 0)
            {
                chunkIndex++;
                continue;
            }

            int baseIdx = chunkStartIndices[chunkIndex];
            
            // Generate triangle indices for the chunk grid
            for (int j = 9, xx = 0, yy = 0; j < 145; j++, xx++)
            {
                bool isShortRow = ((j - 9) / 17) % 2 != 0;
                if (isShortRow && xx >= 8) { xx = 0; yy++; }
                else if (!isShortRow && xx >= 9) { xx = 0; yy++; }

                if (yy >= 8) break;

                // TODO: Check for holes using appropriate field from MCNKheader
                // Skipping hole detection for now

                // Two triangles per quad
                if (isShortRow)
                {
                    if (xx < 8)
                    {
                        indices.Add(baseIdx + j);
                        indices.Add(baseIdx + j - 9);
                        indices.Add(baseIdx + j + 8);

                        indices.Add(baseIdx + j - 9);
                        indices.Add(baseIdx + j - 8);
                        indices.Add(baseIdx + j + 8);
                    }
                }
                else
                {
                    if (xx < 8)
                    {
                        indices.Add(baseIdx + j);
                        indices.Add(baseIdx + j - 8);
                        indices.Add(baseIdx + j + 9);

                        indices.Add(baseIdx + j - 8);
                        indices.Add(baseIdx + j + 1);
                        indices.Add(baseIdx + j + 9);
                    }
                }
            }

            chunkIndex++;
        }

        // Export GLB if requested
        string? glbFile = null;
        if (exportGlb && positions.Count > 0 && indices.Count > 0)
        {
            glbFile = $"tile_{tileX}_{tileY}.glb";
            var glbPath = Path.Combine(meshDir, glbFile);
            ExportGLB(positions, indices, glbPath);
        }

        // Export OBJ if requested
        string? objFile = null;
        string? mtlFile = null;
        if (exportObj && positions.Count > 0 && indices.Count > 0)
        {
            objFile = $"tile_{tileX}_{tileY}.obj";
            mtlFile = $"tile_{tileX}_{tileY}.mtl";
            var objPath = Path.Combine(meshDir, objFile);
            var mtlPath = Path.Combine(meshDir, mtlFile);
            ExportOBJ(positions, indices, objPath, mtlPath, mtlFile);
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
            TriangleCount = indices.Count / 3
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
        List<int> indices,
        string objPath,
        string mtlPath,
        string mtlFileName)
    {
        // Write MTL file
        var mtl = new StringBuilder();
        mtl.AppendLine("# WoWRollback Terrain Material");
        mtl.AppendLine("newmtl TerrainMaterial");
        mtl.AppendLine("Ka 0.6 0.6 0.6");  // Ambient color
        mtl.AppendLine("Kd 0.6 0.6 0.6");  // Diffuse color
        mtl.AppendLine("Ks 0.0 0.0 0.0");  // Specular color
        mtl.AppendLine("Ns 10.0");         // Specular exponent
        mtl.AppendLine("d 1.0");           // Dissolve (opacity)
        mtl.AppendLine("illum 2");         // Illumination model
        
        File.WriteAllText(mtlPath, mtl.ToString());

        // Write OBJ file (following wow.export format)
        var obj = new StringBuilder();
        obj.AppendLine("# Exported by WoWRollback");
        obj.AppendLine("o TerrainMesh");
        obj.AppendLine($"mtllib {mtlFileName}");
        obj.AppendLine();

        // Write vertices
        foreach (var (x, y, z) in positions)
        {
            obj.AppendLine($"v {x:F6} {y:F6} {z:F6}");
        }
        obj.AppendLine();

        // Write normals (calculate per-vertex normals)
        var normals = CalculateNormals(positions, indices);
        foreach (var (nx, ny, nz) in normals)
        {
            obj.AppendLine($"vn {nx:F6} {ny:F6} {nz:F6}");
        }
        obj.AppendLine();

        // Write UVs (simple planar mapping for now)
        for (int i = 0; i < positions.Count; i++)
        {
            var (x, _, z) = positions[i];
            float u = (x % 533.33333f) / 533.33333f; // TILE_SIZE
            float v = (z % 533.33333f) / 533.33333f;
            obj.AppendLine($"vt {u:F6} {v:F6}");
        }
        obj.AppendLine();

        // Write mesh group and faces
        obj.AppendLine("g Terrain");
        obj.AppendLine("s 1");  // Smoothing group
        obj.AppendLine("usemtl TerrainMaterial");

        // Write faces with v/vt/vn format (OBJ uses 1-based indexing)
        for (int i = 0; i < indices.Count; i += 3)
        {
            if (i + 2 < indices.Count)
            {
                var i0 = indices[i] + 1;     // +1 for 1-based indexing
                var i1 = indices[i + 1] + 1;
                var i2 = indices[i + 2] + 1;
                obj.AppendLine($"f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}");
            }
        }

        File.WriteAllText(objPath, obj.ToString());
    }

    private List<(float x, float y, float z)> CalculateNormals(
        List<(float x, float y, float z)> positions,
        List<int> indices)
    {
        // Initialize normals to zero
        var normals = new List<(float x, float y, float z)>(positions.Count);
        for (int i = 0; i < positions.Count; i++)
            normals.Add((0f, 0f, 0f));

        // Accumulate face normals
        for (int i = 0; i < indices.Count; i += 3)
        {
            if (i + 2 >= indices.Count) break;

            int i0 = indices[i];
            int i1 = indices[i + 1];
            int i2 = indices[i + 2];

            var v0 = positions[i0];
            var v1 = positions[i1];
            var v2 = positions[i2];

            // Calculate face normal using cross product
            var edge1 = (v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
            var edge2 = (v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);

            var normal = (
                edge1.Item2 * edge2.Item3 - edge1.Item3 * edge2.Item2,
                edge1.Item3 * edge2.Item1 - edge1.Item1 * edge2.Item3,
                edge1.Item1 * edge2.Item2 - edge1.Item2 * edge2.Item1
            );

            // Accumulate to vertex normals
            normals[i0] = (normals[i0].x + normal.Item1, normals[i0].y + normal.Item2, normals[i0].z + normal.Item3);
            normals[i1] = (normals[i1].x + normal.Item1, normals[i1].y + normal.Item2, normals[i1].z + normal.Item3);
            normals[i2] = (normals[i2].x + normal.Item1, normals[i2].y + normal.Item2, normals[i2].z + normal.Item3);
        }

        // Normalize all normals
        for (int i = 0; i < normals.Count; i++)
        {
            var n = normals[i];
            float length = (float)Math.Sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
            if (length > 0.0001f)
                normals[i] = (n.x / length, n.y / length, n.z / length);
            else
                normals[i] = (0f, 1f, 0f); // Default up
        }

        return normals;
    }

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
