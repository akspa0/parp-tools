using System.Globalization;
using System.IO;
using System.Text;
using System.Collections.Generic;
using System;

namespace WoWRollback.MinimapModule.Services;

/// <summary>
/// Exports ADT terrain data to OBJ mesh format for VLM training data.
/// Ports logic from WoWMapConverter's AdtMeshExporter.
/// </summary>
public static class TerrainMeshExporter
{
    private const float TILE_SIZE = 533.33333f;
    private const float CHUNK_SIZE = TILE_SIZE / 16f;
    private const float UNIT_SIZE = CHUNK_SIZE / 8f;
    private const float UNIT_SIZE_HALF = UNIT_SIZE / 2f;

    /// <summary>
    /// Export ADT heightmap data to OBJ format with minimap texture.
    /// </summary>
    /// <param name="heights">145 height values per chunk (9x9 outer + 8x8 inner interleaved)</param>
    /// <param name="chunkPositions">256 chunk base positions (X, Y, Z)</param>
    /// <param name="holes">256 hole masks</param>
    /// <param name="outputPath">Output OBJ file path</param>
    /// <param name="minimapPath">Optional minimap texture path for MTL</param>
    public static void ExportToObj(
        float[][] heights,
        (float x, float y, float z)[] chunkPositions,
        int[] holes,
        string outputPath,
        string? minimapPath = null)
    {
        if (heights.Length != 256 || chunkPositions.Length != 256)
            throw new ArgumentException("Expected 256 chunks");

        var positions = new List<(float x, float y, float z)>();
        var uvs = new List<(float u, float v)>();
        var chunkStartIndices = new List<int>();

        float minX = float.MaxValue, maxX = float.MinValue;
        float minZ = float.MaxValue, maxZ = float.MinValue;

        // First pass: collect all vertices
        for (int chunkIdx = 0; chunkIdx < 256; chunkIdx++)
        {
            var chunkHeights = heights[chunkIdx];
            var chunkPos = chunkPositions[chunkIdx];
            chunkStartIndices.Add(positions.Count);

            if (chunkHeights == null || chunkHeights.Length != 145)
            {
                // Add placeholder vertices for empty chunks
                for (int i = 0; i < 145; i++)
                    positions.Add((0, 0, 0));
                continue;
            }

            int idx = 0;
            for (int row = 0; row < 17; row++)
            {
                bool isShort = (row % 2) == 1;
                int colCount = isShort ? 8 : 9;

                for (int col = 0; col < colCount; col++)
                {
                    float vx = chunkPos.y - (col * UNIT_SIZE);
                    if (isShort) vx -= UNIT_SIZE_HALF;
                    float vy = chunkHeights[idx] + chunkPos.z;
                    float vz = chunkPos.x - (row * UNIT_SIZE_HALF);

                    positions.Add((vx, vy, vz));

                    if (vx < minX) minX = vx;
                    if (vx > maxX) maxX = vx;
                    if (vz < minZ) minZ = vz;
                    if (vz > maxZ) maxZ = vz;

                    idx++;
                }
            }
        }

        // Second pass: compute UVs
        float spanX = Math.Max(1e-6f, maxX - minX);
        float spanZ = Math.Max(1e-6f, maxZ - minZ);
        const float eps = 2.5e-3f;

        foreach (var p in positions)
        {
            float u = Math.Clamp((p.x - minX) / spanX, eps, 1f - eps);
            float v = Math.Clamp((maxZ - p.z) / spanZ, eps, 1f - eps);
            uvs.Add((u, v));
        }

        // Write MTL file
        var mtlPath = Path.ChangeExtension(outputPath, ".mtl");
        var mtlName = Path.GetFileNameWithoutExtension(outputPath);
        WriteMtlFile(mtlPath, mtlName, minimapPath);

        // Write OBJ file
        WriteObjFile(outputPath, mtlPath, positions, uvs, chunkStartIndices, holes);
    }

    private static void WriteMtlFile(string mtlPath, string materialName, string? texturePath)
    {
        var sb = new StringBuilder();
        sb.AppendLine("# ADT Terrain Material");
        sb.AppendLine($"newmtl {materialName}");
        sb.AppendLine("Kd 1.000 1.000 1.000");
        sb.AppendLine("Ka 0.200 0.200 0.200");
        
        if (!string.IsNullOrEmpty(texturePath))
        {
            // Use just the filename for the texture reference to keep it portable
            // OR use a relative path if the structure is known. 
            // For now, assuming texture is in ../images relative to meshes/, so use relative path
            // specific to the VLM structure: ../images/filename.png
            
            // However, MTL parsing can be tricky. Standardizing on just filename if in same dir, 
            // or relative path if verified.
            // The original used Path.GetFileName(texturePath).
            // Let's stick to simple filename for now, or maybe relative path from active context:
            // Output structure:
            // images/map_x_y.png
            // meshes/map_x_y.obj
            // So relative path is ../images/map_x_y.png
            
            string mtlRef = $"../images/{Path.GetFileName(texturePath)}";
            sb.AppendLine($"map_Kd {mtlRef}");
        }
        
        File.WriteAllText(mtlPath, sb.ToString());
    }

    private static void WriteObjFile(
        string objPath,
        string mtlPath,
        List<(float x, float y, float z)> positions,
        List<(float u, float v)> uvs,
        List<int> chunkStartIndices,
        int[] holes)
    {
        using var fs = new FileStream(objPath, FileMode.Create, FileAccess.Write);
        using var writer = new StreamWriter(fs);

        writer.WriteLine("# ADT Terrain Mesh");
        writer.WriteLine($"mtllib {Path.GetFileName(mtlPath)}");
        writer.WriteLine($"usemtl {Path.GetFileNameWithoutExtension(mtlPath)}");

        // Write vertices (z, x, y order for correct orientation - same as original)
        foreach (var p in positions)
        {
            writer.WriteLine(string.Format(CultureInfo.InvariantCulture, 
                "v {0:F6} {1:F6} {2:F6}", p.z, p.x, p.y));
        }

        // Write UVs
        foreach (var uv in uvs)
        {
            writer.WriteLine(string.Format(CultureInfo.InvariantCulture, 
                "vt {0:F6} {1:F6}", uv.u, uv.v));
        }

        // Write faces (4 triangles per quad, with hole detection)
        for (int chunkIdx = 0; chunkIdx < 256; chunkIdx++)
        {
            int baseIndex = chunkStartIndices[chunkIdx];
            int holeMask = holes[chunkIdx];

            for (int j = 9, xx = 0, yy = 0; j < 145; j++, xx++)
            {
                if (xx >= 8) { xx = 0; yy++; }

                // Check low-res holes (4x4 grid mapped to 8x8)
                // Note: The original used a specific bit shifting logic.
                // Holes are 16 bits for low-res (high res is 64 bits but WotLK usually meant low res holes in this context unless updated)
                // 1 << ((xx / 2) + (yy / 2) * 4)
                
                int holeBit = 1 << ((xx / 2) + (yy / 2) * 4);
                bool isHole = (holeMask & holeBit) != 0;

                if (!isHole)
                {
                    int a = baseIndex + j + 1;        // +1 for OBJ 1-based indexing
                    int b = baseIndex + (j - 9) + 1;
                    int c = baseIndex + (j + 8) + 1;
                    int d = baseIndex + (j - 8) + 1;
                    int e = baseIndex + (j + 9) + 1;

                    // Validate indices
                    if (a <= positions.Count && b > 0 && c <= positions.Count &&
                        d > 0 && e <= positions.Count)
                    {
                        writer.WriteLine($"f {a}/{a} {b}/{b} {c}/{c}");
                        writer.WriteLine($"f {a}/{a} {d}/{d} {b}/{b}");
                        writer.WriteLine($"f {a}/{a} {e}/{e} {d}/{d}");
                        writer.WriteLine($"f {a}/{a} {c}/{c} {e}/{e}");
                    }
                }

                if (((j + 1) % 17) == 0) j += 9;
            }
        }
    }
}
