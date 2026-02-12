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
    /// Exports ADT terrain data to OBJ mesh format for VLM training data.
    /// Returns the OBJ and MTL content as strings.
    /// </summary>
    /// <param name="heights">145 height values per chunk (9x9 outer + 8x8 inner interleaved)</param>
    /// <param name="chunkPositions">256 chunk base positions (X, Y, Z)</param>
    /// <param name="holes">256 hole masks</param>
    /// <param name="materialName">Name of the material (used for MTL linking)</param>
    /// <param name="minimapPath">Optional minimap texture path for MTL</param>
    /// <returns>Tuple of (ObjContent, MtlContent)</returns>
    public static (string ObjContent, string MtlContent) GenerateObjStrings(
        float[][] heights,
        (float x, float y, float z)[] chunkPositions,
        int[] holes,
        string materialName,
        string? minimapPath = null)
    {
        if (heights.Length != 256 || chunkPositions.Length != 256)
            throw new ArgumentException("Expected 256 chunks");

        var positions = new List<(float x, float y, float z)>();
        var uvs = new List<(float u, float v)>();
        var chunkStartIndices = new List<int>();

        float minX = float.MaxValue, maxX = float.MinValue;
        float minZ = float.MaxValue, maxZ = float.MinValue; // Z in OBJ corresponds to -Y in WoW

        // First pass: collect all vertices
        for (int chunkIdx = 0; chunkIdx < 256; chunkIdx++)
        {
            var chunkHeights = heights[chunkIdx];
            var chunkPosSaved = chunkPositions[chunkIdx];
            chunkStartIndices.Add(positions.Count);

            if (chunkHeights == null || chunkHeights.Length != 145)
            {
                // Add placeholder vertices for empty chunks
                for (int i = 0; i < 145; i++)
                    positions.Add((0, 0, 0));
                continue;
            }

            // Un-swizzle the "YUp" vector from Warcraft.NET back to WoW coordinates (Z-up)
            // Warcraft.NET ReadVector3(YUp): X=InX, Y=InZ, Z=-InY
            // So:
            // WoW_X = Saved.X
            // WoW_Z = Saved.Y
            // WoW_Y = -Saved.Z
            
            float baseWowX = chunkPosSaved.x;
            float baseWowY = -chunkPosSaved.z;
            float baseWowZ = chunkPosSaved.y;

            int idx = 0;
            for (int row = 0; row < 17; row++)
            {
                bool isShort = (row % 2) == 1;
                int colCount = isShort ? 8 : 9;

                for (int col = 0; col < colCount; col++)
                {
                    // Calculate WoW coordinates relative to chunk base
                    // Standard WoW ADT: X is North, Y is West. As you traverse rows/cols, you subtract from the base position.
                    
                    float currWowX = baseWowX - (row * UNIT_SIZE_HALF);
                    float currWowY = baseWowY - (col * UNIT_SIZE);
                    if (isShort) currWowY -= UNIT_SIZE_HALF;
                    
                    float currWowZ = baseWowZ + chunkHeights[idx];

                    // Convert to OBJ (Y-up)
                    // X_obj = -Y_wow (East is +X)
                    // Y_obj = Z_wow (Up)
                    // Z_obj = -X_wow (South is +Z)
                    
                    float ox = -currWowY;
                    float oy = currWowZ;
                    float oz = -currWowX;

                    positions.Add((ox, oy, oz));

                    if (ox < minX) minX = ox;
                    if (ox > maxX) maxX = ox;
                    if (oz < minZ) minZ = oz;
                    if (oz > maxZ) maxZ = oz;

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
            // UV mapping - might need flipping depending on texture
            float u = Math.Clamp((p.x - minX) / spanX, eps, 1f - eps);
            float v = Math.Clamp((maxZ - p.z) / spanZ, eps, 1f - eps); // Invert V
            uvs.Add((u, v));
        }

        // Generate MTL content
        string mtlContent = GenerateMtlContent(materialName, minimapPath);

        // Generate OBJ content
        string objContent = GenerateObjContent(materialName, positions, uvs, chunkStartIndices, holes);

        return (objContent, mtlContent);
    }
    
    // Kept for backward compatibility if needed, or redirect to new method
    public static void ExportToObj(
        float[][] heights,
        (float x, float y, float z)[] chunkPositions,
        int[] holes,
        string outputPath,
        string? minimapPath = null)
    {
         var materialName = Path.GetFileNameWithoutExtension(outputPath);
         var (obj, mtl) = GenerateObjStrings(heights, chunkPositions, holes, materialName, minimapPath);
         
         File.WriteAllText(outputPath, obj);
         File.WriteAllText(Path.ChangeExtension(outputPath, ".mtl"), mtl);
    }

    private static string GenerateMtlContent(string materialName, string? texturePath)
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
            
            string mtlRef = $"../images/{Path.GetFileName(texturePath)}";
            sb.AppendLine($"map_Kd {mtlRef}");
        }
        
        return sb.ToString();
    }

    private static string GenerateObjContent(
        string materialName,
        List<(float x, float y, float z)> positions,
        List<(float u, float v)> uvs,
        List<int> chunkStartIndices,
        int[] holes)
    {
        var sb = new StringBuilder();

        sb.AppendLine("# ADT Terrain Mesh");
        sb.AppendLine($"mtllib {materialName}.mtl");
        sb.AppendLine($"usemtl {materialName}");

        // Write vertices (z, x, y order for correct orientation - same as original)
        foreach (var p in positions)
        {
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture, 
                "v {0:F6} {1:F6} {2:F6}", p.x, p.y, p.z));
        }

        // Write UVs
        foreach (var uv in uvs)
        {
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture, 
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
                        sb.AppendLine($"f {a}/{a} {b}/{b} {c}/{c}");
                        sb.AppendLine($"f {a}/{a} {d}/{d} {b}/{b}");
                        sb.AppendLine($"f {a}/{a} {e}/{e} {d}/{d}");
                        sb.AppendLine($"f {a}/{a} {c}/{c} {e}/{e}");
                    }
                }

                if (((j + 1) % 17) == 0) j += 9;
            }
        }
        
        return sb.ToString();
    }
}
